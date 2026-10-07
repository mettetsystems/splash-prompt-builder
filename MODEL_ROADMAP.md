# Remaining model work

Splash now runs three pinned diffusion candidates through a persistent local worker. The project/draft/history workflow is implemented; the remaining work is to make enrichment consistently good and broaden measured hardware support. No custom-trained or distilled Splash model is included.

## What is measured

- Compact Qwen diffusion executes on RTX 2070, RTX 5090, and CPU, but often repeats wording or answers the request instead of rewriting it.
- Dream 7B executes in BF16 on RTX 5090 and NF4 on RTX 2070. It is the most practical tested starting point for this workstation. The small multilingual audit still found a dropped testing requirement, awkward wording, and invented details.
- LLaDA 2.1 mini executes with NF4 on RTX 5090 using the bundled eager-mask and RoPE compatibility code. Its samples can be verbose and slower, with truncation at small output budgets.
- A live five-branch LLaDA synthesis omitted informed consent; the harness now retains explicit English original requirements in a labeled section.
- Literal/code/number checks and a limited multilingual test-requirement guard reject some lossy outputs. They do not verify negation, scope, translation fidelity, or all implied requirements.

The README links raw outputs, timing samples, and memory measurements. These are small execution checks, not statistically meaningful quality or latency benchmarks.

## Next qualification work

1. **Create a representative evaluation set.** Include all three agents, long coding tasks, research contradictions, several languages, uncertain requirements, exact quotations, code, negation, and references. Use independent human judgments for clarity and fidelity, not just lexical checks. Retain a held-out set.
2. **Improve constraint preservation.** Track objectives and requirements across section boundaries, verify semantic equivalence before proposing removals, identify output truncation, and evaluate a bounded repair pass. Extend multilingual checks without presenting heuristics as proof.
3. **Measure synthesis quality.** Test two through five branches for duplicate ideas, conflicting populations/time ranges/methods, source faithfulness, and preservation of the original objective. Show specific unresolved conflicts reliably.
4. **Improve useful context.** Replace the conservative single 350-character enrichment excerpt with a token-budgeted, provenance-aware context planner. Evaluate multilingual retrieval and longer research briefs. Preserve explicit exclusions and page references.
5. **Test actual deployment targets.** GTX 1050 needs an older compatible GPU stack or CPU fallback; it remains unverified. Test Windows, Intel/Apple Silicon macOS, Metal kernels, low RAM, offload, and device-removal recovery. Publish reproducible environment locks for each passing profile.
6. **Benchmark complete workloads.** Separate model load time, cold/warm generation, provider latency, end-to-end edit-to-card latency, typing responsiveness, resident RAM, allocated/reserved VRAM, and long-session storage growth. Report sample counts and p50/p95/p99, not only best-case individual timings.
7. **Consider training only after evaluation.** Build a rights-cleared prompt/rewrite dataset. Compare instruction tuning or diffusion distillation to the existing candidates. The historical QDoRA/consistency-distillation idea is a research proposal, not an implemented training pipeline or a proven memory/latency claim.

Keep the current boundary throughout: proposals and draft variants are reviewable; only an explicit, validated two-stage merge changes a main. A failed real model must never silently become mock output.
