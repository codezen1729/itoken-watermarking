# Real-model experiment: what we are testing

We want to test two parts of the proposal that the original simulation did not establish: obtaining probabilities for predicting the next iToken from a trained language model, and recovering the same logical steps after their wording changes. We keep the formal proof domain fixed so that failures in these two parts can be measured separately. This document describes the method; it makes no claim about results that have not yet been observed.

## Model and reproducibility

The completed scoring experiment uses the local [MLX conversion of Qwen2.5-0.5B-Instruct with 4-bit weights](https://huggingface.co/mlx-community/Qwen2.5-0.5B-Instruct-4bit/tree/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3), pinned to revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, through [MLX-LM](https://github.com/ml-explore/mlx-lm). We use the adapter's `dense-fp32` CPU execution mode: cast the stored affine quantization coefficients to FP32, expand the packed weights, replace quantized linear and embedding modules, and cast the remaining floating parameters to FP32. The resulting 494,032,768 parameters are hashed and recorded. This does not recover the original unquantized checkpoint or establish numerical equivalence to packed quantized execution. The measured probabilities belong to this specific trained checkpoint and execution representation. See [the probability findings](research_results/qwen_05b_dense_fp32_v1/probability_findings.md) and [runtime notes](RESEARCH_RUNTIME.md).

The initial Qwen2.5-1.5B attempts were runtime checks only. Their packed or dense CPU runs were interrupted before development or held-out evaluation; they do not supply the six completed scoring contexts reported here.

The run records the checkpoint source and revision when available, file hashes, tokenizer configuration, chat template, runtime versions, device, context limit, prompt token IDs or their hash, and stopping-token identities. Generation settings and seeds are recorded separately. Model weights are kept outside the repository and are not committed. The adapter requires an existing local checkpoint and does not silently download a model.

## Phase 1: score candidate steps with actual logits

For a fixed problem and partial reasoning context, we construct nine complete candidate strings: three supported expressions for each of three possible next logical steps. Their canonical iToken labels are checked by the existing symbolic protocol.

Every candidate is scored under the same chat-prefix token sequence. The adapter checks that concatenating a candidate does not change that prefix's tokenization. It appends one declared end-of-step token and obtains each continuation token's conditional log probability from the model's learned logits. Summing these values gives the log probability of that complete terminated sequence. We do not divide by length or substitute a model's verbal confidence estimate. An independent uncached calculation can check the cached scoring path.

We then sum sequence probabilities within each canonical iToken group. Normalizing over all nine candidates produces probabilities for predicting the next iToken **conditional on the listed candidates**. We also record their raw covered probability mass, including its logarithm when the mass is very small.

This whitelist is not the model's complete support. A confidently normalized three-way distribution can still cover very little of the model's unrestricted output mass. The experiment must show that coverage rather than treating the normalized distribution as an unconditional probability over ideas.

## Phase 2: sample the model without the whitelist

We also retain native generated continuations, including unsupported or invalid outputs. These samples help reveal what the model does outside the nine listed strings. They are not replaced with a canonical answer when generation fails.

The sample's prompt, token IDs, termination, decoding settings, and raw output are preserved. Comparing sample frequencies with raw model probabilities requires matching sampling distributions, including temperature. Greedy extraction calls and transformed-temperature samples are not evidence of calibration under temperature-one token probabilities.

## Phase 3: recover logical steps from free wording

The recovery model receives the public problem, its allowed inference vocabulary, and the text to interpret. It receives neither the watermark key nor preferred labels or a gold route. The first version accepts previously unseen wording while retaining explicit logical symbols such as `p0` and `m0`. The calculus remains bounded to modus ponens, conjunction introduction, and disjunction introduction.

The model proposes structured steps and exact supporting text spans. Validation checks JSON structure, verbatim evidence, symbol anchors, admissible rules, route consistency, and proof dependencies. A recovered forced completion remains a logical step even when no branching choice is recovered; the downstream detector separately decides whether it supplies watermark evidence. Conservative lexical filters reject some negated, hypothetical, or instruction-bearing evidence. These checks do **not** prove that the prose asserts or entails the proposed step. A valid formal proof can still be an incorrect interpretation of its cited text; `semantic_entailment_verified` therefore remains false.

Two proposal interfaces were evaluated on development data. The first asks the model to write formulas. The second asks it to select public inference IDs and quote source evidence; software expands each ID into its already defined canonical step and applies the unchanged validator. This reduces the output-format burden without providing a gold route or repairing semantic mistakes. The ID interface was selected for held-out testing after development evaluation, then frozen. Both development runs and the held-out failures are retained.

## Phase 4: evaluate a frozen extractor

An independent agent prepared eight development cases and 24 held-out test cases with agent-authored gold labels. The extraction agent did not read these cases while implementing its first version. The extractor prompt and source hash are frozen before held-out evaluation. This separation reduces tuning to the examples; a small agent-authored set is still not a representative benchmark of unrestricted human language.

We report exact recovery and step-level errors, abstentions on positive and negative cases separately, and false acceptance of passages that do not assert the claimed reasoning. For paraphrases we measure preservation of the intended steps and complete-proof validity separately. Parser abstention is not a negative watermark result, and formal validity is not semantic faithfulness. Raw model responses and validation outcomes remain in the run artifacts. Gold labels are used for evaluation, never as replacement model outputs.

## Scope of any watermark pilot

A subsequent end-to-end pilot may bias actual model-derived candidate probabilities and apply the existing key-blind recovery and detector. It must report candidate coverage, recovery losses, detection, false positives, and solution validity. Reusing one model for generation, paraphrasing, and extraction is a limitation, not independent corroboration. Even a successful pilot in this domain would leave general ideas, arbitrary semantic equivalence, unseen domains, and informed removal or forgery attacks unresolved.
