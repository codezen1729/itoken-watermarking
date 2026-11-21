# Does the watermark survive rewording with a stronger LLM?

In this experiment, yes: every watermark that was detectable in the original text remained detectable after rewording. We recovered the same logical path from all 16 rewritten documents. But only four of the eight marked paths had a detectable watermark to begin with. So the result is **4/4 survival of initially detectable marks, and 4/8 detection across all marked rewrites**. These two numbers answer different questions.

We used fresh **GPT-6.2-astra agents with high reasoning effort** to propose logical steps, check those proposals, reword the arguments, and assess whether the rewording preserved the reasoning. The agents used the supplied source text and public inference vocabulary. Recovery agents did not receive the intended path, expected answers, watermark key, condition label, or detector score. Each role had a separate context; paired marked/unmarked documents and original/reworded versions were kept in different recovery contexts.

The model choice above records the actual requested agent routing. The agent interface does not expose a resolved model checkpoint, temperature, seed, token probabilities, or API finish reason. Tools were allowed to read assigned prompts and serialize explicit judgments; this was tool-enabled agent inference, not a standard completion API. Access restrictions were procedural, not an operating-system guarantee. The exact prompts, responses, batch membership, and file hashes are retained in [the agent artifacts](research_results/strong_recovery_v2/agent_artifacts/agent_calls.json).

We still used actual **Qwen2.5-0.5B-Instruct token probabilities** for choosing the next iToken. The stronger model supplied recovery and rewording, so this is a hybrid experiment. We did not obtain GPT-6.2-astra token probabilities or claim that Qwen's probabilities came from the stronger model.

For each of 32 local states, we scored nine complete, EOS-terminated textual expressions: three expressions for each of three next-inference choices. We summed expressions that represented the same iToken and normalized over the listed candidates. This gives probabilities for **predicting the next iToken conditional on that candidate list**. The context contained the current fact and its local premises, not the preceding full path. The same bank was therefore reused across independent task/key units under an explicitly local-state planner.

The listed expressions covered between **4.04% and 8.67%** of the model's probability mass, averaging **6.34%**. Their normalized probabilities are not the unrestricted model's complete distribution over ideas. The mean next-iToken entropy was only **0.050 nats**: the model strongly preferred one route. We tilted these probabilities by a factor of `exp(6)` for hash-preferred choices, retaining the original probability ratios within each preferred group. Concentrated starting probabilities and limited branching freedom can still produce a weak mark. The observed four nondetections were already present in the planned paths; recovery and rewording caused no additional loss.

The checkpoint was `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, dequantized into dense FP32 CPU execution. This is a specific runtime variant, not a claim of numerical identity with packed quantized execution. The 32 scoring contexts supplied 288 terminated-sequence scores. Forty cached-versus-uncached sequence checks had a maximum difference of **0.0000377 nats per token**. The bank retains token-level probabilities, contexts, checkpoint hashes, and execution metadata.

We changed logical recovery in two places. First, the model selects public inference IDs and source sentence numbers, rather than inventing formulas or copying quotations. Second, a fresh agent independently checks whether each proposed premise-to-conclusion application is actually asserted in the source. It supplies its own evidence. Deterministic checks then enforce valid IDs, source offsets, symbol anchors, sound inference rules, and consistent routes. A possible valid proof is never supplied as the recovery target. A partial explanation stays partial. These checks improve discipline, but model agreement is still not a proof of semantic entailment.

We checked the frozen protocol on nine previously exposed development cases, including the earlier unsound-argument failure. All nine were correct. We then evaluated a separately authored, frozen holdout without changing the protocol:

| Fresh recovery holdout | Result |
|---|---:|
| Complete paths recovered exactly | 12/12 |
| Partial paths recovered exactly | 8/8 |
| Negative or ambiguous cases correctly rejected | 12/12 |
| Gold inference applications recovered | 47/47 |
| Cases with explicit branching choices recovered exactly | 17/17 |
| False semantic acceptances | 0/12 negative cases |

The old model's negative results remain in [the previous report](RESEARCH_REPORT.md). Both the model and recovery protocol changed here, and the fresh holdout is a different set. This is not an ablation that isolates the contribution of model size or the second agent.

For the watermark experiment, we fixed **eight task/key units, 32 choices per path, strengths 0 and 6, one rewrite per document, a threshold of 0.01, and 199 surrogate keys** before inspecting outcomes. Each unit used independently drawn target and surrogate keys. Marked and unmarked routes shared a sampling stream within the unit, independent of the keys. No key or path was rerolled after observing detection. There are 16 planned paths and 32 original/reworded documents, but only eight paired experimental units. The controls have only two distinct canonical paths: seven are identical all-MP proofs, and the remaining one contains one AND branch. Across the controls, 255/256 branches are MP. The marked paths contain 187 MP, 41 AND, and 28 OR branches. Independent keys do not make these eight controls diverse logical tasks.

The 16 rewrites were generated once each in four isolated four-document stronger-model batches. It was asked to preserve all reasoning and atom names while changing sentence structure and wording. All 16 rewrites changed the source text and fell outside the old three-template parser. Separate fidelity assessors compared each rewrite with its source and judged all 64 inference applications per document preserved, with no additions. These assessors used the same model family; their judgments are not independent human annotation. Equality between intended and recovered paths is reported separately from this fidelity assessment.

| Watermark experiment | Marked | Unmarked controls |
|---|---:|---:|
| Planned paths detected | 4/8 | 0/8 |
| Original texts detected | 4/8 | 0/8 |
| Reworded texts detected | 4/8 | 0/8 |
| Exact original path recovery | 8/8 | 8/8 |
| Exact reworded path recovery | 8/8 | 8/8 |
| Rewrites assessed faithful | 8/8 | 8/8 |
| Detector abstentions after rewording | 0/8 | 0/8 |

Every original and rewrite yielded all 32 branching choices and 64 inference applications. Across the 16 rewrites, this is **1,024/1,024 planned steps recovered**, with no missing or extra steps. All p-values were identical before and after rewording:

| Unit | Marked: planned / original / reworded | Unmarked: planned / original / reworded |
|---|---|---|
| 0 | 0.010 / 0.010 / 0.010 | 0.800 / 0.800 / 0.800 |
| 1 | 0.005 / 0.005 / 0.005 | 0.065 / 0.065 / 0.065 |
| 2 | 0.075 / 0.075 / 0.075 | 1.000 / 1.000 / 1.000 |
| 3 | 0.005 / 0.005 / 0.005 | 0.440 / 0.440 / 0.440 |
| 4 | 0.060 / 0.060 / 0.060 | 0.470 / 0.470 / 0.470 |
| 5 | 0.050 / 0.050 / 0.050 | 0.885 / 0.885 / 0.885 |
| 6 | 0.005 / 0.005 / 0.005 | 0.660 / 0.660 / 0.660 |
| 7 | 0.015 / 0.015 / 0.015 | 0.605 / 0.605 / 0.605 |

The four initially positive marked documents all survived rewording. The primary rate still includes the four initially weak marks: **50% detection across all marked rewrites**. No failure or uncertain rewrite was removed from a denominator. Eight controls with no detections cannot establish a 1% population false-positive rate, and shared model batches can make outcomes dependent. These are descriptive pilot counts; no independent-binomial population interval is claimed.

A post hoc diagnostic found that merely checking which stage-specific symbol appears (`mN`, `dN`, or `eN`) identifies all 512 branch choices in the originals and all 512 in the rewrites. It needs no sentence-order analysis or semantic interpretation. Therefore the paired cohort does not establish that a stronger LLM, or semantic recovery itself, was necessary to recover its watermark. This shortcut does not address the negative and partial explanations in the separate holdout, and was not used by the frozen recovery pipeline. It reveals how easy the paired cohort remains despite its changed prose.

This supports a specific conclusion: **with known symbolic anchors and a fixed small calculus, stronger model-assisted recovery can preserve a structural watermark through faithful free rewording**. It does not establish reliable recovery of arbitrary ideas, invariance to renaming symbols, resistance to a rewrite that changes or omits reasoning, or robustness against an informed attacker. The holdout is agent-authored and small, the rewritten paths use a narrow repeated proof structure, and all semantic roles share a model family. General LLM-generated ideas remain an unresolved research problem.

The next useful experiments are a model/protocol ablation, independently human-annotated prose without explicit symbol names, and separately declared deletion, summarization, renaming, and route-replacement attacks. For this run, the main remaining limitation is watermark power under the concentrated planner distribution, rather than loss under faithful rewording.

All measured counts are in [summary.json](research_results/strong_recovery_v2/summary.json); individual records are in [paired_rows.json](research_results/strong_recovery_v2/paired_rows.json). The [protocol](REWORDING_PROTOCOL.md) and [recovery methods](RECOVERY_V2_METHODS.md) describe the assumptions. The saved results can be reparsed and their detector scores recomputed without an LLM:

```bash
python3 audit_strong_results.py
python3 -m unittest discover -s tests -v
python3 watermark.py
```

An [independent audit](STRONG_AUDIT.md) also recomputed all 288 sequence masses, 512 route draws, 16 planned likelihood ratios, and target/surrogate scores. The audit checks retained response bundles, frozen recovery code, exact canonical recovery, full cohort accounting, fidelity records, and detector values. It does not independently attest the hosted model checkpoint or prove semantic truth. Validation completed with 147 discovered tests and eight watermark tests passing. The previous experiment files were preserved unchanged.
