# Independent replay of the stronger-recovery experiment

The saved records reproduce **4/8 marked rewrites detected, 4/4 initially detectable marked paths still detected, and 0/8 unmarked rewrites detected**. All 16 original/rewrite pairs retained exactly the same recovered choices and detector p-values. The largest qualification concerns the difficulty of the paired texts. A post hoc scan for atom names recovered every branch, and seven of the eight unmarked originals contained identical proof text.

## Evidence checked

| Quantity | Independent check |
|---|---|
| Checkpoint and sequence probabilities | The saved metadata identify the Qwen2.5-0.5B-Instruct 4-bit checkpoint and dense FP32 execution. For each of 32 contexts, I summed the saved token log probabilities of nine distinct EOS-terminated sequences, exponentiated, grouped by the parsed canonical inference, and normalized over the nine sequences. All masses and conditional inference probabilities matched the bank. |
| Probability coverage | Minimum 4.0411%, mean 6.3351%, maximum 8.6664%. Mean conditional inference entropy was 0.0499504 nats. Saved cached and uncached scores agreed within 0.000037671 nats per token in 40 sequence checks. The first context had nine checks and each of the other 31 had one. |
| Paired sampling | I independently reconstructed the HMAC labels, exponential tilt, pseudorandom uniforms, and cumulative-probability draw. All 512 recorded route draws across 16 plans matched. All 16 fixed-alternative planned log likelihood ratios and the rendered proofs matched as well. |
| Key use and detector | The eight target keys and 1,592 surrogates have distinct 256-bit values. Each unit uses one target key and 199 surrogates shared by its two conditions and textual variants. For every plan I independently scored all 200 keys and counted surrogate scores at least as large as the target's. All planned and recovered p-values matched, including the detected unit at p = 0.010. |
| Recovery and provenance | I regenerated proposal prompts from the bound public problem and source, and verification prompts from the source and actual proposal. All archived prompts matched. All 28 non-development batches have matching request/response file hashes and contain 148 assigned items in total. The saved holdout text and public problem match the frozen dataset hash. |
| End-to-end accounting | The saved-response audit reproduced 12/12 exact complete, 8/8 exact partial, and 12/12 correctly rejected negative/ambiguous holdout cases, with 47/47 gold steps. The full paired cohort contains 32 recovery records and 16 fidelity assessments. All 16 rewrites were assessed preserved by separate agent contexts. |

The sampler source draws target and surrogate keys from the operating system before drawing either route. Fixed-seed route uniforms come from a separate generator. At strength zero, probabilities are independent of the target key. Distinct saved keys and exact replay establish consistency with the recorded procedure. The files alone cannot attest the original process’s entropy source or establish a complete execution history.

All assigned cases appear in the reported denominators. The primary marked-rewrite numerator is four with eight trials. Conditioning on positive original or planned detections gives four with four trials. The corresponding control survival fractions are undefined because there were no originally or initially detectable controls. Four missed marked paths remain in the primary result. The detector deterministically returns the same p-value for identical recovered choices and a fixed key bank, so agreement across the original and rewrite supplies one paired observation per unit.

## How difficult were these paths?

At stage 7, for example, the atom name `m7` identifies modus ponens, `d7` identifies conjunction, and `e7` identifies disjunction. The same correspondence holds at the other numbered stages. I tested this observation after collection across the full texts. The scan searched for atom names at word boundaries and counted a match only if exactly one route symbol occurred at the stage and it identified the planned route. It recovered **512/512 original and 512/512 rewritten branching choices** while ignoring sentence order, inference rules, and assertions. The study asked reworders to retain each atom name. The result demonstrates that the tested marks survived this change in wording. The paired data cannot establish that semantic interpretation or a stronger model was necessary for detection. This diagnostic scores branch identity in complete paired proofs. The separate holdout requires decisions about denials, competing routes, and incomplete evidence.

The eight unmarked plans drew 255 modus-ponens branches, one conjunction, and zero disjunctions. Seven originals contained the identical all-modus-ponens proof. There were two distinct unmarked routes and eight distinct marked routes. The marked draws included 187 modus-ponens branches, 41 conjunctions, and 28 disjunctions. Independent keys give distinct randomization tests even for repeated text. Repeated proof content and four-document agent batches limit what the observed counts say about generalization across prose and logical tasks.

The hosted agent identity is the requested routing recorded by the orchestrator. Archived files preserve judgments and file hashes, while model checkpoint identity, decoding settings, and the full implementation of the hosted service are unavailable. Procedural file restrictions and fresh role contexts support blinding. Shared model-family judgments still permit common semantic errors. This replay used the saved Qwen token scores and did not rerun neural inference.

## Repeat the checks

Run these commands in the prototype directory using Python 3.10 or later with assertions enabled. The independent replay uses the public proof representation and prompt builders. Its probability, sampling, HMAC scoring, and symbol-only diagnostic are implemented separately from the production planner and detector.

```bash
python3 audit_strong_probabilities.py
python3 audit_strong_results.py
python3 -m unittest discover -s tests -q
python3 watermark.py
```

Both audit commands passed against the completed artifacts. The test suite passed 147 discovered tests and eight standalone watermark tests. The replay implementation is available in [audit_strong_probabilities.py](audit_strong_probabilities.py), and the saved-response audit is available in [audit_strong_results.py](audit_strong_results.py).
