"""A trace-to-native prototype for model-to-mechanism compilation.

The teacher model is NEVER invoked at runtime. This prototype accepts offline
input/output traces; it is not a general LLM decompiler.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from importlib.resources import files
import json
import math
from pathlib import Path
import shutil
import subprocess

ASCII_SPACE = " \t\r\n\v\f"
ASCII_LOWER = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")


def normalize(text: str) -> str:
    """Match the generated C runtime's ASCII-only input normalization."""
    return text.strip(ASCII_SPACE).translate(ASCII_LOWER)


def read_traces(path: Path) -> dict[str, str]:
    """Load traces, fail closed on contradictory labels and malformed examples."""
    entries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                source, target = record["input"], record["output"]
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"Invalid trace at line {index}") from exc
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(f"Trace {index} must contain string input/output")
            source = normalize(source)
            if not source or not target or any(x in source + target for x in ("\x00", "\n", "\r")):
                raise ValueError(f"Trace {index} is empty or contains forbidden characters")
            if source in entries and entries[source] != target:
                raise ValueError(f"Contradictory outputs for input {source!r} (line {index})")
            entries[source] = target
    if not entries:
        raise ValueError("No valid traces found")
    return entries


@dataclass
class TrieNode:
    terminal: int = -1
    children: dict[int, "TrieNode"] = field(default_factory=dict)


def minimize_dag(entries: dict[str, int]) -> tuple[list[tuple[int, tuple[tuple[int, int], ...]]], int, int]:
    """Build an acyclic byte transducer; merge equivalent suffix subgraphs."""
    root = TrieNode()
    naive_nodes = 1
    for text, label in sorted(entries.items()):
        node = root
        for byte in text.encode("utf-8"):
            if byte not in node.children:
                node.children[byte] = TrieNode()
                naive_nodes += 1
            node = node.children[byte]
        node.terminal = label

    registry: dict[tuple[int, tuple[tuple[int, int], ...]], int] = {}
    result: list[tuple[int, tuple[tuple[int, int], ...]]] = []

    def intern(node: TrieNode) -> int:
        transitions = tuple((byte, intern(child)) for byte, child in sorted(node.children.items()))
        signature = (node.terminal, transitions)
        if signature not in registry:
            registry[signature] = len(result)
            result.append(signature)
        return registry[signature]

    root_id = intern(root)
    return result, root_id, naive_nodes


MASK64 = (1 << 64) - 1


def fnv64(chunk: bytes) -> int:
    state = 0xCBF29CE484222325
    for byte in chunk:
        state = ((state ^ byte) * 0x100000001B3) & MASK64
    return state


def ngrams(text: str) -> set[int]:
    raw = text.encode("utf-8")
    return {fnv64(raw[i : i + 3]) for i in range(max(0, len(raw) - 2))}


def train_residual(entries: dict[str, int], max_features: int) -> list[tuple[int, int, float]]:
    """Distill a tiny character-trigram prototype classifier.

    This is an approximate residual classifier, not semantic reasoning.
    Classes require >=2 examples; its runtime gate abstains on weak matches.
    """
    class_docs: dict[int, int] = Counter(entries.values())
    eligible = {label for label, count in class_docs.items() if count >= 2}
    if len(eligible) < 2:
        return []

    frequency: dict[tuple[int, int], int] = Counter()
    document_frequency: dict[int, int] = Counter()
    for text, label in entries.items():
        if label not in eligible:
            continue
        features = ngrams(text)
        for feature in features:
            frequency[(feature, label)] += 1
        document_frequency.update(features)

    total_documents = sum(class_docs[label] for label in eligible)
    ranks = Counter()
    for (feature, label), count in frequency.items():
        ranks[feature] += count / class_docs[label]
    selected = {feature for feature, _ in ranks.most_common(max_features)}

    result = []
    for (feature, label), count in frequency.items():
        if feature not in selected:
            continue
        idf = math.log((1 + total_documents) / (1 + document_frequency[feature])) + 1.0
        weight = idf * (count / class_docs[label])
        result.append((feature, label, weight))
    return sorted(result)


def c_literal(text: str) -> str:
    """Escape UTF-8 data as byte escapes in C string literals."""
    return '"' + "".join(f"\\x{byte:02x}" for byte in text.encode("utf-8")) + '"'


def render_source(
    states: list[tuple[int, tuple[tuple[int, int], ...]]],
    root_id: int,
    outputs: list[str],
    ml_entries: list[tuple[int, int, float]],
    min_coverage: float,
    min_margin: float,
) -> str:
    edges = []
    nodes = []
    for terminal, transitions in states:
        nodes.append(f"    {{{len(edges)}u, {len(transitions)}u, {terminal}}},")
        edges.extend(f"    {{{byte}u, {child}u}}," for byte, child in transitions)
    ml_rows = [f"    {{UINT64_C({key}), {label}u, {weight:.8f}f}}," for key, label, weight in ml_entries]
    arrays = "\n".join(
        [
            f"#define ROOT_ID {root_id}u",
            f"#define RESPONSE_COUNT {len(outputs)}u",
            f"#define FEATURE_COUNT {len(ml_entries)}u",
            f"#define MIN_COVERAGE {min_coverage:.8f}f",
            f"#define MIN_MARGIN {min_margin:.8f}f",
            f"static const Node nodes[] = {{\n" + "\n".join(nodes) + "\n};",
            f"static const Edge edges[] = {{\n" + "\n".join(edges or ["    {0u, 0u},"]) + "\n};",
            f"static const char *responses[] = {{\n" + "\n".join(f"    {c_literal(s)}," for s in outputs) + "\n};",
            f"static const MlTerm ml_terms[] = {{\n" + "\n".join(ml_rows or ["    {UINT64_C(0), 0u, 0.0f},"]) + "\n};",
        ]
    )
    template = files("model2mechanism").joinpath("runtime.c.in").read_text(encoding="utf-8")
    return template.replace("/* GENERATED TABLES */", arrays)


def compile_traces(
    traces: Path,
    out_dir: Path,
    *,
    build: bool = False,
    cc: str = "cc",
    use_ml: bool = True,
    max_features: int = 512,
    min_coverage: float = 0.55,
    min_margin: float = 0.25,
) -> dict:
    if max_features < 1 or not (0 <= min_coverage <= 1) or not (0 <= min_margin <= 1):
        raise ValueError("Invalid ML budget or thresholds")
    observed = read_traces(traces)
    outputs = sorted(set(observed.values()))
    output_id = {output: index for index, output in enumerate(outputs)}
    labeled = {text: output_id[output] for text, output in observed.items()}
    states, root_id, naive_nodes = minimize_dag(labeled)
    ml_entries = train_residual(labeled, max_features) if use_ml else []

    out_dir.mkdir(parents=True, exist_ok=True)
    source = render_source(states, root_id, outputs, ml_entries, min_coverage, min_margin)
    source_path = out_dir / "chatbot.c"
    source_path.write_text(source, encoding="utf-8")
    ir = {
        "schema_version": 1,
        "kind": "trace_to_mechanism_prototype",
        "rules": [{"input": key, "response_id": val} for key, val in sorted(labeled.items())],
        "responses": outputs,
        "root_state": root_id,
    }
    (out_dir / "mechanism.ir.json").write_text(json.dumps(ir, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "experimental_trace_compiler_not_general_model_conversion",
        "teacher_calls_at_runtime": 0,
        "runtime": "C11 CPU-native, no runtime ML framework",
        "input_examples": len(labeled),
        "response_classes": len(outputs),
        "trie_states_before": naive_nodes,
        "dag_states_after": len(states),
        "residual_ml_features": len({feature for feature, _, _ in ml_entries}),
        "residual_ml_weights": len(ml_entries),
        "ml_gate": {"min_coverage": min_coverage, "min_margin": min_margin},
        "limits": ["offline input/output traces only", "stateless single-turn responses", "heuristic residual classifier", "abstain on unknown inputs"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if build:
        compiler = shutil.which(cc)
        if compiler is None:
            raise RuntimeError(f"C compiler not found: {cc}")
        subprocess.run(
            [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", str(source_path), "-o", str(out_dir / "chatbot")],
            check=True,
        )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model2mechanism", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("compile", help="compile offline teacher traces to native C")
    cmd.add_argument("--traces", type=Path, required=True, help="JSONL file with input/output fields")
    cmd.add_argument("--out", type=Path, required=True)
    cmd.add_argument("--build", action="store_true", help="build standalone CPU executable with C11 compiler")
    cmd.add_argument("--cc", default="cc")
    cmd.add_argument("--no-ml", action="store_true", help="compile exact decision DAG only")
    cmd.add_argument("--max-features", type=int, default=512)
    cmd.add_argument("--min-coverage", type=float, default=0.55)
    cmd.add_argument("--min-margin", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.command == "compile":
        try:
            manifest = compile_traces(args.traces, args.out, build=args.build, cc=args.cc,
                                      use_ml=not args.no_ml, max_features=args.max_features,
                                      min_coverage=args.min_coverage, min_margin=args.min_margin)
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
