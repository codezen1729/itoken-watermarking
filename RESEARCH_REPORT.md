# Research update: real probabilities and logical recovery

We now have a working scorer based on a trained model's token-prediction probabilities. The language-recovery experiment failed: it recovered none of the positive held-out examples and accepted one unsound argument. These results do not establish robust watermarking of general LLM-generated ideas.

Three specialist agents worked on real-model scoring, key-blind recovery, and independent benchmark/audit work. Their modules were integrated and tested together. The original symbolic simulation remains a separate result.

## What was actually run

The completed experiment uses the [MLX Qwen2.5-0.5B-Instruct 4-bit checkpoint](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3), with its stored coefficients expanded into dense FP32 CPU execution. This is an explicitly identified execution variant; it does not reconstruct the original unquantized model. Checkpoint files, transformed parameters, prompts, software versions, and frozen code are hashed. The larger 1.5B runtime attempts were stopped for performance before any development/test inference. [Runtime provenance](RESEARCH_RUNTIME.md).

The domain is still a small propositional calculus with explicit atom names and a known public problem. The benchmark is agent-authored: eight development cases and 24 held-out cases. It is not unrestricted language, an independent human corpus, or a test of arbitrary scientific ideas. [Protocol](RESEARCH_EVAL_PROTOCOL.md).

## Probabilities for predicting the next iToken

Each of nine candidate token sequences was scored with actual conditional token log probabilities, including termination. Other tokenizations that decode to the same text remain outside this list. Their masses were summed by logical step. The normalized next-iToken predictions are conditional on these nine terminated sequences. Their raw covered mass measures how much of the model's output distribution that list contains.

| Prompt | Listed probability mass | Conditional MP | Conditional AND | Conditional OR |
|---|---:|---:|---:|---:|
| open_instruction | 8.42811e-20 | 3.67764e-06 | 0.00128449 | 0.998712 |
| listed_candidates | 0.147389 | 0.973312 | 0.0070702 | 0.0196181 |
| reversed_list | 0.0626056 | 0.709752 | 0.00514626 | 0.285102 |

The open instruction's near-certain conditional OR choice applies to a vanishingly small set of sequences. Listing candidates increases coverage, but even then most output mass remains unaccounted for. Reversing the displayed list changes the model's conditional choices substantially. These measurements therefore do not recover the full distribution of possible next ideas.

Fourteen cached-versus-uncached comparisons passed; the largest observed token-log-probability difference was 2.69e-05 nats. Independent replay also verified sequence sums, grouped masses, and normalization. [All six contexts and raw checks](research_results/qwen_05b_dense_fp32_v1/probability_findings.md).

## Recovery from new wording

The first extractor generated formulas and source evidence. After it failed all six positive development cases, a separate compact interface was implemented: select public inference IDs and quote supporting text. It uses the same validator. The ID-interface designer did not inspect benchmark text; its development performance determined which interface proceeded to held-out evaluation. Both development runs are retained, and the selected interface was frozen before any test call.

| Development interface | Positive examples exactly recovered | Correct negative rejections | Gold steps recovered |
|---|---:|---:|---:|
| Formula output | 0/6 | 2/2 | 0/12 |
| Inference IDs | 1/6 | 2/2 | 1/12 |

| Held-out ID-interface measure | Result |
|---|---:|
| Complete paths exactly recovered | 0/14 |
| Partial paths exactly recovered | 0/5 |
| Correct gold steps recovered | 0/46 |
| Negative examples incorrectly accepted | 1/5 |
| Accepted steps, including incorrect ones | 1 |
| Truncated extractor outputs | 2/24 |

All 4 exact test successes were negative rejections. An abstain-all policy would reject all five negatives, so this result is not useful recovery performance. The small curated set cannot establish population error rates. [Raw held-out records and summary](research_results/qwen_05b_ids_v1/recovery_test_summary.json).

A concrete failure appears in [test22](research_results/qwen_05b_ids_v1/recovery/test22.json). The source claims that p0 establishes e0 and that e0 alone establishes p1, then explicitly says no implication connecting e0 to p1 is supplied. The model nevertheless selected the valid MP application from `(e0|p0)` and `((e0|p0)->p1)` to `p1`. Its quoted span matched the text and included the atom names, so the validator accepted it as partial evidence. Those premises were not the inference asserted in the passage. Formal admissibility and symbol overlap did not establish semantic faithfulness.

This is a false acceptance by the semantic recovery layer. The accepted step is a forced completion, not a branching choice, so it does not by itself constitute a false-positive watermark detection. The failure has been preserved without a test-specific repair.

## Native outputs and end-to-end pilot

Of 8 temperature-one native outputs, 0 exactly matched a listed terminated token sequence; 4 native outputs were truncated. None obtained an accepted single next-step label. These eight samples are a diagnostic, not a calibration study; the same unvalidated model interprets them. [Raw native calls](research_results/qwen_05b_ids_v1/native_samples/native_manifest.json).

The pilot used one unmarked and one marked planned path, each with eight branching choices. Every branch was sampled from freshly scored learned candidate probabilities, with watermark strengths 0 and 4. The model then verbalized each pair of inferences, and the key-blind ID extractor processed that prose. No synthetic-baseline likelihood ratio was applied to recovered text.

| Trace | Planned structural p-value | Recovered planned steps / planned total | Steps outside planned path | Recovered choices | Recovered structural p-value |
|---|---:|---:|---:|---:|---:|
| unmarked | 0.605 | 1/16 | 0 | 0 | abstained |
| watermarked | 0.345 | 1/16 | 0 | 0 | abstained |

Both planned proofs were valid. The bias changed one of eight sampled branches, but neither planned trace reached the detection threshold. Detection uses a 0.01 threshold. Missing evidence is an abstention, not evidence of human authorship. Two traces under one public demonstration key cannot estimate power or false-positive rates. Any planned-path likelihood ratio in the raw records applies only to the restricted planner's actual model probabilities, not to the distribution of interpreted prose. Agreement with the intended path is not an independent assessment of what the prose actually asserts. Shared-model agreement is not independent semantic verification. [Pilot records](research_results/qwen_05b_ids_v1/end_to_end/pilot_summary.json).

## What this adds, and what remains

The project now has real token-log-probability measurement, explicit candidate coverage, two language-recovery interfaces, a frozen evaluation, and an auditable end-to-end connection. The measured recovery results are negative. They identify both an inability to recover supported steps and a semantic error that survives the structural checks; neither should be hidden by the earlier symbolic simulation's success.

The next research step is to test stronger extractors and better grounding protocols on independently labeled paraphrases, then repeat the complete watermark experiment across models and domains. Estimating unrestricted next-iToken prediction probabilities also requires accounting for unlisted output mass and parser errors. [Missing-mass bounds and unresolved assumptions](RESEARCH_GAPS.md).

Validation: 100 software tests plus eight watermark-core tests passed. The saved probability/recovery records have an independent arithmetic and provenance audit; these checks do not certify semantic truth. See [scoring/development audit](research_results/qwen_05b_dense_fp32_v1/audit.json) and [ID-evaluation audit](research_results/qwen_05b_ids_v1/audit.json).
