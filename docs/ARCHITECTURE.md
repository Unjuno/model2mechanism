# Architecture and scope

## Long-term target

A post-training compiler transforming teacher models into CPU-first symbolic programs and selective residual ML. Meaning preservation, generalization beyond seen examples, and runtime resource optimization are research goals, not demonstrated capabilities.

## Implemented trace-to-native pipeline

1. Accept offline UTF-8 teacher input/output pairs.
2. Validate input and reject contradictory examples.
3. Group identical responses into classes.
4. Construct a byte-level trie and merge equivalent suffix subgraphs, yielding a minimal acyclic output-labelled transducer for the observed input set.
5. Learn an optional, confidence-gated trigram prototype classifier with a bounded vocabulary for non-exact inputs.
6. Emit a C11 file, JSON intermediate representation, and a manifest. A platform C compiler can generate a CPU-native executable.

This implements **syntactic exact matching and approximate character-level fallback**, not semantic equivalence, arbitrary reasoning, or free-form response generation. For unknown examples, the runtime may return __ABSTAIN__ rather than guessing.

Runtime uses CPU arithmetic and static tables, with no teacher invocation, GPU requirement or transformer KV cache. Native executables are target-OS and ABI dependent. Compilation can run on a different host than the target as long as a suitable C compiler/toolchain exists.

## Research roadmap

- Typed semantic/action IR, multi-turn dialogue state, and effect-aware rule combination.
- Independent held-out and OOD evaluation, including selective prediction/abstention metrics.
- Automatic mechanism selection based on measured CPU latency, peak RSS, binary size and correctness.
- API-teacher adapters and white-box model analysis; provenance of teacher observations.
- Symbolic program synthesis, semantic abstraction and runtime extension modules.

## Evaluation

Control hardware, clock settings, thread count, input length and batch size. Report task-level correctness vs independent gold labels in addition to teacher agreement; include accuracy-confidence intervals, coverage/error trade-offs, p50/p95 latency, peak resident memory, binary size and compilation time. A structural state count alone is not a proof of lower memory or higher reasoning quality.

## Security and provenance

Treat teacher traces as untrusted and do not commit private traces. Never run generated binaries on sensitive systems without sandboxing and review. The prototype does not execute teacher-supplied source code. Licensing of the original teacher and its generated outputs must be considered independently.
