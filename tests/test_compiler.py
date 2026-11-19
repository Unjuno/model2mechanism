import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from model2mechanism.compiler import compile_traces, minimize_dag, read_traces

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "chatbot.jsonl"


class CompilerTest(unittest.TestCase):
    def test_suffix_sharing(self):
        states, root, naive = minimize_dag({"abc": 0, "xbc": 0, "ybc": 0})
        self.assertLess(len(states), naive)
        self.assertGreaterEqual(root, 0)

    def test_conflicting_teacher_examples_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "traces.jsonl"
            p.write_text('{"input":"HELLO","output":"A"}\n{"input":"hello","output":"B"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Contradictory"):
                read_traces(p)

    def test_compiled_cpu_binary(self):
        if shutil.which("cc") is None:
            self.skipTest("No C11 compiler on PATH")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output"
            manifest = compile_traces(EXAMPLES, out, build=True)
            self.assertEqual(manifest["teacher_calls_at_runtime"], 0)
            self.assertLess(manifest["dag_states_after"], manifest["trie_states_before"])
            self.assertGreater(manifest["residual_ml_features"], 0)
            self.assertTrue((out / "mechanism.ir.json").exists())
            self.assertEqual(json.loads((out / "manifest.json").read_text())["input_examples"], 10)
            binary = out / "chatbot"
            for phrase, expected in [
                (" HELLO ", "Hello. How can I help?"),
                ("thanks", "You're welcome."),
                ("こんにちは", "こんにちは。何をお手伝いしましょうか。"),
                ("unrelated nonsense", "__ABSTAIN__"),
            ]:
                run = subprocess.run([str(binary), phrase], capture_output=True, text=True, check=True)
                self.assertEqual(run.stdout.strip(), expected, phrase)

    def test_rule_only_binary_abstains_on_unseen(self):
        if shutil.which("cc") is None:
            self.skipTest("No C11 compiler on PATH")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output"
            manifest = compile_traces(EXAMPLES, out, build=True, use_ml=False)
            self.assertEqual(manifest["residual_ml_weights"], 0)
            run = subprocess.run([str(out / "chatbot")], input="hello\nnever seen this\n", capture_output=True, text=True, check=True)
            self.assertEqual(run.stdout.splitlines(), ["Hello. How can I help?", "__ABSTAIN__"])


if __name__ == "__main__":
    unittest.main()
