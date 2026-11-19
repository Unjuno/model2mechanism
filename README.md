# model2mechanism

**Experimental model-to-mechanism compiler**: research into transforming pretrained models into lightweight, CPU-native computational mechanisms.

## Goal

Use a pretrained model as a teacher at *compile time* and automatically generate reusable decision graphs, symbolic code and selective tiny ML. Avoid running the original model, GPU inference, and transformer KV caches during execution.

**Current status: experimental trace-to-native v0.1.** This repository does not yet decompile arbitrary models or reproduce general reasoning. Its initial end-to-end tool compiles *offline input/output examples* into a minimized decision DAG and a tiny approximate character-trigram classifier, then emits a CPU-native C11 executable. Unknown cases can abstain.

## Quick start

Requires Python 3.10+ and a C11 compiler (GCC or Clang).

    python -m pip install -e .
    model2mechanism compile --traces examples/chatbot.jsonl --out build/demo --build
    ./build/demo/chatbot "hello"
    printf 'thanks\nこんにちは\nunrelated nonsense\n' | ./build/demo/chatbot

Expected outputs: "You're welcome.", "こんにちは。何をお手伝いしましょうか。", "__ABSTAIN__".

To disable the residual classifier:

    model2mechanism compile --traces examples/chatbot.jsonl --out build/rules --no-ml --build

## Trace format

The input is UTF-8 JSONL with one observation per line:

    {"input":"hello","output":"Hello. How can I help?"}

Sample traces are *handwritten test fixtures*, not evidence of successful teacher distillation. Conflicting outputs for a normalized input are rejected. ASCII case and surrounding ASCII whitespace are normalized. The prototype handles single-turn fixed responses only.

The compiler emits chatbot.c, mechanism.ir.json, manifest.json and optionally a native chatbot binary. At execution time it uses no teacher calls, ML framework, or GPU. The tiny classifier is an approximate n-gram scoring method, not a replacement for the original model's reasoning.

## Tests

    python -m unittest discover -s tests -v

See docs/ARCHITECTURE.md for scope and next milestones. Benchmarks must measure held-out fidelity, abstention, binary size, peak RAM, and p50/p95 CPU latency; **no speedup or quality retention is asserted yet**.

## Contributing and license

Research contributions via issues or pull requests are welcome. This public repository does not yet have a software license; public visibility alone does not grant reuse permissions.
