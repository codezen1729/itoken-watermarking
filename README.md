# Watermarking ideas through iTokens

We want to watermark the structure of an idea, so that changing its wording does not automatically remove the mark. This prototype tests that mechanism in a small symbolic proof domain. We define an iToken as an actual inference application, choose among three valid routes to the same result, and bias those choices using a keyed hash.

This is a **controlled symbolic prototype**. It uses a finite autoregressive reference predictor with no learned parameters. It does not call an LLM, an API, or a network service. Its text parser recognizes three specified rendering templates; it is not a general natural-language parser.

Start with the [experiment report](report/report.md), or its [HTML version](report/report.html). The formal assumptions are in [THEORY.md](THEORY.md), and the closest research comparisons are in [RELATED_WORK.md](RELATED_WORK.md).

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
| `tests/` | Checks of symbolic validity, supported invariances, probability aggregation, and the end-to-end interfaces. |

## What this establishes, and what remains

The prototype makes the proposed unit concrete: a rule, its premises, and its conclusion. It can test a watermark over that unit and exact invariance under its declared rendering transformations. Each three-way branch has two inference applications, but only the route choice carries structural freedom.

The domain is engineered so that all three routes are valid and equally long. Preserving validity here does not demonstrate that watermarking preserves the usefulness or originality of real ideas. Unsupported natural prose is rejected; free variable renaming is not supported. Deletion can leave only partial evidence. Replacing the proof with a newly sampled unwatermarked path removes the original marked choices and can remove its signal even when the conclusion remains identical.

The next substantive experiment is to use a specified trained LLM, measure candidate coverage and probability calibration, independently evaluate the parser on free paraphrases, and compare against text-token and semantic-cluster watermark baselines. The current package does not claim those results.
