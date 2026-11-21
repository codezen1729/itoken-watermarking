# Watermarking ideas through iTokens

We want to watermark the structure of an idea, so that changing its wording does not automatically remove the mark. This prototype tests that mechanism in a small symbolic proof domain. We define an iToken as an actual inference application, choose among three valid routes to the same result, and bias those choices using a keyed hash.

The original **controlled symbolic prototype** uses a finite autoregressive reference predictor with no learned parameters. Its core commands do not call an LLM, an API, or a network service. Its text parser recognizes three specified rendering templates. A separate research extension now connects a trained local model and a model-assisted extractor; it keeps the bounded calculus and explicit symbolic anchors.

Start with the [experiment report](report/report.md), or its [HTML version](report/report.html). The formal assumptions are in [THEORY.md](THEORY.md), and the closest research comparisons are in [RELATED_WORK.md](RELATED_WORK.md).

## Stronger recovery and free rewording

The [new stronger-model experiment](STRONG_RECOVERY_REPORT.md) uses GPT-6.2-astra agents for recovery, verification, rewording, and separate fidelity assessment. Actual Qwen token probabilities still supply the planner, so this is a hybrid experiment. The fresh recovery holdout was exact on all 32 cases. All 16 rewritten paths were recovered exactly; all four initially detectable marks survived. Overall detection was **4/8 marked rewrites and 0/8 unmarked controls**. The explicit symbols alone also reveal every branch in this cohort, so it does not establish that the stronger LLM was necessary. General idea watermarking remains unresolved.

Run `python3 audit_strong_results.py` to reparse saved responses and recompute the paired results without a model. The [frozen protocol](REWORDING_PROTOCOL.md), [recovery methods](RECOVERY_V2_METHODS.md), [raw artifacts](research_results/strong_recovery_v2/agent_artifacts/agent_calls.json), and [machine-readable results](research_results/strong_recovery_v2/summary.json) retain the complete cohort and model-interface limitations.

## Run the example

Use Python 3.10 or newer. The generation, detection, experiment, and test code uses the Python standard library. Run these commands from this folder:

```bash
python3 cli.py demo --out demo --stages 32
python3 cli.py detect --manifest demo/manifest.json --text demo/controlled_rewording.txt
```

The first command writes a proof, six edited versions, the original problem manifest, and detection results to `demo/`. The edits cover controlled rewording, presentation shuffling, duplicated steps, deletion, partial rerouting, and a fresh complete proof. The second command reconstructs the formal evidence from one edited text and runs the detector.

The manifest supplies the original premises, goal, public problem identifier, detector settings, and a **public demonstration key**. That key is included for reproducibility and must never be used for real attribution. Verification requires the original problem manifest; arbitrary text alone is insufficient.

## Read the detector output

- `status` distinguishes a complete proof, partial evidence, insufficient evidence, unsupported text, and paths outside or ambiguous within this domain.
- `proof_valid` says whether the supplied text contains a complete accepted proof. Partial evidence can still have a statistical score.
- `n_choices` counts recovered structural choices, ignoring exact duplicates.
- `p_value` and `detected` refer to the surrogate-key randomization test, with a default threshold of 0.01.
- `log_likelihood_ratio` and `likelihood_ratio_detected` compare against the exact reference predictor. They are supplied only when the complete path is recovered.

The randomization test assumes that an unwatermarked artifact is independent of the secret key. Its guarantee averages over key selection; it is not a universal promise for every fixed key or for content selected through a watermark oracle. A negative result does not establish human authorship, and a positive result does not establish who first conceived an idea.

## Reproduce the experiments

```bash
python3 -m unittest discover -s tests -v
python3 watermark.py
python3 experiments.py --samples 300 --calibration-keys 199 --out results
python3 build_report.py --results results --out report
```

The experiment command compares 8, 32, and 128 branching choices at watermark strengths 0, 1, 2, and 4. A separate cohort tests seven edit conditions on 64-choice proofs at strength 2. The main detector uses the surrogate-key p-value; the secondary likelihood-ratio detector uses the same fixed alternative strength of 2 across conditions.

This configuration produces 5,700 document evaluations from **1,200 independent simulated task/key units**: 900 units paired across four strengths, and 300 units paired across seven edit conditions. The 5,700 rows are not 5,700 independent trials. There are 199 surrogate keys per document, giving p-value resolution 0.005. The seed and other settings are saved in `results/config.json`.

The experiment writes individual records to `results/trials.csv`, summaries to `results/summary.json`, and example texts to `results/example_variants.json`. The report builder writes `report/report.md`, `report/report.html`, and PNG/SVG figures. Running `build_report.py` requires Matplotlib; viewing the included reports and running the core experiments do not. The report contains measured outcomes.

## Supply externally scored candidates

`scored_candidates.py` is a bridge for token log probabilities obtained elsewhere. It invokes no model. Given a JSONL file, run:

```bash
python3 scored_candidates.py candidates.jsonl --context-id fixed-context-1 --tokenizer tokenizer-name-and-version
```

Each record needs `text`, a nonempty `token_logprobs` list in natural-log units, and `includes_termination: true`. Every score must refer to a distinct terminated text sequence under the same fixed context and tokenizer. The bridge checks supported inference syntax and rule validity, groups textual realizations by canonical iToken, and sums their supplied masses.

Its normalized output is explicitly **conditional on the listed candidates**. It does not certify full support, verify the external scoring process, or recover an unrestricted LLM's complete iToken distribution. It also cannot establish that a step's premises are available in the declared context. These obligations remain with the scorer and the surrounding proof protocol.

## Research extension with a trained model

The extension scores terminated candidate texts using actual token logits, sums their mass by canonical inference, and records both candidate coverage and the resulting probabilities for predicting the next iToken conditional on that list. Its key-blind extractor proposes logical steps with supporting source spans, then applies structural and evidence checks. These checks do not establish semantic entailment.

Start with [the measured research results](RESEARCH_REPORT.md), then read [the methods](REAL_MODEL_METHODS.md), [the frozen evaluation protocol](RESEARCH_EVAL_PROTOCOL.md), and [the remaining research gaps](RESEARCH_GAPS.md). The benchmark has 8 development and 24 held-out agent-authored cases; it is not a representative corpus of unrestricted language. [Runtime setup](RESEARCH_RUNTIME.md) is separate from the standard-library prototype.

The completed scoring run uses Qwen2.5-0.5B-Instruct's 4-bit checkpoint converted to dense FP32 execution. It is a separate runtime variant, not the original unquantized model. With that local checkpoint and the research environment installed, these Bash commands reproduce the declared settings into fresh directories:

```bash
model_args=(--model /path/to/qwen2.5-0.5b-instruct-4bit
  --source-repo mlx-community/Qwen2.5-0.5B-Instruct-4bit
  --revision a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3
  --device cpu --execution-mode dense-fp32)
python research_runner.py "${model_args[@]}" --phase scores --out research_results/new_formula_run
python research_runner.py "${model_args[@]}" --phase dev --out research_results/new_formula_run
python research_id_runner.py "${model_args[@]}" --split dev --out research_results/new_id_run
python research_id_runner.py "${model_args[@]}" --split test --out research_results/new_id_run
python research_native_samples.py "${model_args[@]}" --out research_results/new_native_run
python real_model_pilot.py "${model_args[@]}" --out research_results/new_pilot --stages 8 --recovery-protocol ids
python audit_saved_results.py research_results/new_id_run
```

The runner records source and checkpoint hashes, prompts, token-level probabilities, raw responses, truncations, and failures. The formula-output extractor was evaluated on development data first. Its optional ID interface selects public inference IDs with verbatim evidence, then uses the same validator; development results determined the interface sent to held-out evaluation. Test inference requires a completed development run with the same frozen configuration. Completed evaluation cases can resume without resampling; native sampling and the end-to-end pilot require fresh output directories. The pilot's two traces demonstrate integration and cannot estimate detection power or false-positive rates.

## How the pieces fit together

| File | Role |
|---|---|
| `core.py` | Symbolic formulas, inference rules, three-route problems, controlled rendering/parser, proof checking, and exact finite text-probability aggregation. |
| `watermark.py` | Keyed preferred sets, biased sampling probabilities, likelihood-ratio calculations, surrogate-key tests, and statistical utilities. |
| `pipeline.py` | History-dependent route selection, public problem anchors, evidence extraction, edit transformations, and combined diagnostics. |
| `cli.py` | Reproducible examples and detection using a supplied manifest. |
| `scored_candidates.py` | Aggregation of externally supplied token log probabilities over a declared candidate set. |
| `experiments.py` | Paired experiments across path lengths, strengths, and edits. |
| `build_report.py` | Report and plot generation from saved results; requires Matplotlib. |
| `real_model.py` | Optional local MLX inference and untempered token-probability scoring. |
| `semantic_recovery.py` | Key-blind language-to-inference proposals with conservative validation. |
| `semantic_recovery_ids.py` | Compact inference-ID proposals expanded through the same validator. |
| `research_runner.py` | Probability coverage measurements and frozen development/test evaluation. |
| `research_id_runner.py` | Separate frozen evaluation of the ID proposal interface. |
| `research_native_samples.py` | Native temperature-one outputs, including unsupported and truncated samples. |
| `real_model_pilot.py` | Real-model-informed planning, free wording, recovery, and structural detection. |
| `audit_saved_results.py` | Independent arithmetic and freeze checks on saved scoring and recovery records. |
| `tests/` | Checks of symbolic validity, supported invariances, probability aggregation, and the end-to-end interfaces. |

## What this establishes, and what remains

The prototype makes the proposed unit concrete: a rule, its premises, and its conclusion. It can test a watermark over that unit and exact invariance under its declared rendering transformations. Each three-way branch has two inference applications, but only the route choice carries structural freedom.

The domain is engineered so that all three routes are valid and equally long. Preserving validity here does not demonstrate that watermarking preserves the usefulness or originality of real ideas. Unsupported natural prose is rejected; free variable renaming is not supported. Deletion can leave only partial evidence. Replacing the proof with a newly sampled unwatermarked path removes the original marked choices and can remove its signal even when the conclusion remains identical.

The research extension targets trained-model scoring and logical recovery separately. Full-support next-iToken prediction, independently verified semantic interpretation, larger cross-model evaluations, solution quality, and comparisons with text-token and semantic-cluster baselines remain unresolved. This package does not claim robust watermarking of general LLM-generated ideas.
