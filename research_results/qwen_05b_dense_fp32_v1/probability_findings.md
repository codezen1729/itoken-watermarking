# Measured probabilities for predicting the next iToken

We obtained learned token log probabilities from Qwen2.5-0.5B-Instruct, not verbal confidence estimates or the original finite predictor. The checkpoint is the [MLX 4-bit conversion](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3), revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, executed on the CPU in the adapter's `dense-fp32` mode. This expands the stored quantized coefficients into FP32 parameters; it does not restore the original unquantized model. [Model and runtime identity](model_metadata.json).

Each context has nine complete candidate strings: three supported expressions for each of the MP, AND-introduction, and OR-introduction next steps. We score their exact token sequences, including EOS 151645, under one fixed prompt. Each sequence probability is the product of its learned conditional token probabilities. We sum these probabilities within an iToken group and normalize over the nine candidates. No length normalization is applied. The covered mass is the model probability of the union of those nine terminated token sequences. [Methods](../../REAL_MODEL_METHODS.md), [scorer](../../real_model.py), [context construction](../../research_runner.py).

| Context | Covered probability mass | Conditional MP | Conditional AND | Conditional OR |
| --- | ---: | ---: | ---: | ---: |
| [open_instruction](scores/open_instruction.json) | 8.4281136595e-20 | 3.67763852396e-06 | 0.00128448883785 | 0.998711833524 |
| [listed_candidates](scores/listed_candidates.json) | 0.147388857915 | 0.97331167656 | 0.00707020340853 | 0.0196181200317 |
| [reversed_list](scores/reversed_list.json) | 0.0626056289142 | 0.709752047225 | 0.00514625711863 | 0.285101695657 |
| [after_mp](scores/after_mp.json) | 0.0905935723697 | 0.98995217156 | 0.00159423108134 | 0.00845359735834 |
| [after_and](scores/after_and.json) | 0.0973589017974 | 0.982309103725 | 0.00307976151856 | 0.0146111347561 |
| [after_or](scores/after_or.json) | 0.086996120593 | 0.976729884395 | 0.00312162390054 | 0.0201484917041 |

Values are rounded to twelve significant digits; [scores_summary.json](scores_summary.json) and the linked per-context records retain full precision. The last three columns sum to one conditional on the candidate set. They are not unconditional probabilities over all ways of expressing a logical step or all possible ideas.

The open-instruction prompt does not list the candidate strings. Its covered mass is only `8.43e-20`, even though conditional OR probability is approximately 99.87%. Thus this near-certain conditional choice describes a tiny selected set of token sequences. It does not establish what logical step the model usually expresses in unrestricted generation. Even when the prompt lists the nine allowed strings, their combined mass is only 14.74%.

Reversing the same displayed candidate list changes the conditioning prompt. Covered mass falls from 14.74% to 6.26%; conditional MP changes from 97.33% to 70.98%, while OR changes from 1.96% to 28.51%. This demonstrates prompt-order sensitivity in this measured choice protocol. It is not a context-independent distribution over ideas. The three history contexts also condition on different preceding logical routes; the table does not establish invariance across histories or domains.

Cached scoring with 128-token prompt chunks was compared against independent, uncached full-sequence scoring. All nine candidates were checked in `open_instruction`; the first candidate was checked in each of the other five contexts. All fourteen checks passed the predeclared `0.05` nat/token tolerance. The largest observed token-log-probability difference was `2.69412994385e-05` nats, and the largest absolute summed sequence-log-probability difference was `7.657520473e-05` nats. Independent arithmetic replay also verified sequence sums, covered mass, normalization, common prompt identity, and distinct EOS-terminated candidates. [Raw checks](scores/open_instruction.json), [runtime protocol](../../RESEARCH_RUNTIME.md).

These results establish a working restricted next-iToken probability scorer for this checkpoint and execution mode. They expose incomplete candidate coverage and sensitivity to the supplied prompt. They do not measure semantic recovery, watermark detection, false-positive rates, or robustness of unrestricted LLM-generated ideas.
