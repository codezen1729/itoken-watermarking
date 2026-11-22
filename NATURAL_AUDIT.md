# Independent audit of the natural-language experiment

The saved records reproduce **5/8 marked rewrites detected, all 5/5 initially detectable marked paths still detected, and 0/8 unmarked rewrites detected**. These are detector results on recovered catalog labels. After requiring both the author-to-plan and rewrite-to-original fidelity assessments to pass, **3/8 marked documents remain both assessed faithful and detected**. Five marked documents passed both fidelity assessments, giving 3/5 detection within that subset. These denominators must remain separate.

The central qualification is a mismatch between label agreement and detailed meaning. Every recovered label matches the intended plan, yet separate fidelity agents identified changes or uncertainty in several originals and rewrites. A fixed catalog can be coarse enough that changed wording still maps to its nearest intended option. The recovery verifier may also overlook a qualifier that the fidelity assessor notices. Unchanged labels and p-values therefore do not certify unchanged ideas at every semantic level.

## Independent numerical replay

| Quantity | Check and observed result |
|---|---|
| Real-model scoring | Reconstructed all 384 EOS-terminated sequence scores from their stored token log probabilities across 128 distinct local contexts. Matched every candidate expression to its public catalog application and independently rebuilt conditional choice probabilities. |
| Candidate coverage | Minimum 0.3148%, mean 20.0405%, maximum 79.8875%. Mean conditional choice entropy was 0.572702 nats. These values concern three supplied expressions per decision, not all paraphrases of each idea. |
| Numerical crosschecks | Recomputed 144 cached-versus-uncached sequence comparisons. Maximum discrepancy was 0.0000553131 nats per token, below the frozen 0.05 tolerance. |
| Keyed selection | Independently reconstructed HMAC labels, exponential tilting, fixed-seed route uniforms, all 256 choices across 16 plans, and all 16 planned likelihood ratios. The marked plan changed 8, 5, 7, 9, 5, 5, 4, and 6 choices relative to its paired control. |
| Key bank and detection | Verified eight target keys and 1,592 surrogate keys, all distinct 256-bit values. Independently recomputed all planned p-values and the 64 semantic/lexical original/rewrite detector scores. |
| Raw model artifacts | All 36 recorded agent batches have matching input/output hashes and retain 168 assigned items. Author inputs match recorded selected applications; rewrite inputs match the actual originals. Author-fidelity inputs contain the selected applications and original, while rewrite-fidelity inputs contain only public decision topics and the two texts. |
| Blinding and source linkage | Regenerated every main recovery and verifier prompt from its public problem, bound source, and actual first proposal. Every request matched. No marked/unmarked pair shared these model contexts. No original/rewrite pair shared recovery history. |
| Explicit markers | Found no literal public catalog IDs or old artificial atom markers in any of the 32 original/rewrite texts. Internal answer IDs remain bookkeeping in recovery prompts and outputs. |

The sequence metadata identify `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, revision `a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3`, under dense FP32 execution. This audit used the saved token scores and did not rerun neural inference. The stronger agents supplied natural language, recovery, and fidelity judgments; their token probabilities were unavailable. This is a hybrid experiment.

The sampler draws target and surrogate keys before drawing either paired plan. Fixed-seed route uniforms use a separate generator. At strength zero, the selected path is independent of the target key. File consistency and distinct saved keys support the recorded procedure, but files alone cannot attest the original entropy source or the complete hosted execution history. Agent blinding was procedural, not an operating-system guarantee. Model routing is recorded as requested; resolved hosted checkpoint identity and decoding settings remain unavailable.

The planned marked p-values, in problem order, are `0.005, 0.005, 0.020, 0.005, 0.025, 0.005, 0.020, 0.010`. Control p-values are `0.665, 0.230, 0.890, 0.900, 0.360, 0.225, 0.265, 0.690`. Original and rewritten recovery gave the same values for every plan. The three initially weak marks remain in the eight-unit primary denominator. Equality with the 0.01 threshold counts as detected, and surrogate ties are counted conservatively.

## Meaning, labels, and fidelity

All 32 recovery records contain 16 choices matching their intended catalog labels: 512/512 label agreements, including 256/256 in the rewrites. These are **agreement with planned labels**, not 512 independently established semantic gold judgments about the actual text.

| Separate fidelity assessment | Marked | Unmarked |
|---|---:|---:|
| Original assessed faithful to intended applications | 5/8 | 7/8 |
| Original assessed changed | 1/8 | 0/8 |
| Original fidelity uncertain | 2/8 | 1/8 |
| Rewrite assessed faithful to its actual original | 6/8 | 3/8 |
| Rewrite assessed changed | 2/8 | 5/8 |
| Both assessments passed | 5/8 | 3/8 |
| Both assessments passed and rewritten mark detected | 3/8 | 0/8 |

The three fidelity-qualified positive cases are the library, cup-lending, and museum proposals. The board-game proposal and swap-fair proposal account for the other two raw positive detections. The former changed an intended eligibility condition during authoring; the latter left a required recording detail uncertain. Their unchanged detector results cannot establish faithful survival of the complete intended applications.

The recorded differences include giving a partner only to learners who request strategic help rather than giving every inexperienced player a partner with optional consultation; changing “most visits” into “most people”; changing a room a visitor enjoyed into a room that stood out; broadening a specified call into a conversation; altering who moves chairs; narrowing an audience from contributors to donors; and dropping an explicit rotating list while retaining turns. Some differences may be minor for a practical recommendation, but they matter under the frozen requirement to preserve roles, scope, timing, and consequences. The audit retains the first judgments and does not revise annotations to improve outcomes.

These fidelity judgments are themselves model assessments from the same family as the other semantic roles. They are neither independent human annotation nor formal entailment proofs. They reveal why perfect catalog-label agreement must not be presented as perfect semantic recovery.

## Lexical comparison and challenge cases

The frozen word-overlap baseline matched 116/256 planned choices in the originals and 113/256 in the rewrites, with zero exact 16-choice vectors among either set of 16 documents. Its detector flagged one original control and one marked rewrite. The semantic pipeline's detector flagged zero controls in either variant and five marked texts in either variant. Eight controls are too few to establish a population false-positive rate for either decoder.

On the separate 24-case diagnostic challenge, the semantic pipeline matched all expected choices, statuses, and ambiguity flags, recovering 28/28 annotated choices. The lexical baseline matched the exact choice sets in 7/24 cases. All eight semantic examples in the four identical-word-multiset contrast pairs were correct; the lexical baseline matched none of their exact choice sets. The challenge thus distinguishes this pipeline from the declared bag-of-words baseline on roles, order, and adoption/rejection scope. It does not prove that every lexical shortcut is absent from the main corpus or that stronger-model reasoning is necessary among all possible methods.

The public catalog supplies all admissible semantic alternatives. The authored examples are ordinary-language proposals, but recovering their selected alternatives is still a bounded classification task. It is not unrestricted discovery of a canonical logical representation for arbitrary ideas. Catalog applications describe intended planning mechanisms and effects, not machine-checked logical proofs or empirically verified causal guarantees.

## Accounting and reproducibility

All assigned documents remain in the primary denominators, including the unfaithful or uncertain ones. Before primary inference, the audit identified an empty-source accounting gap. `natural_failures.py` now keeps raw empty or whitespace-only texts as explicit `empty_source` failures without inventing a placeholder or invoking a model. No blank-source failure occurred in the completed main cohort. Nonempty requests use the frozen transport unchanged. Malformed structured bundles stop at validation instead of silently dropping a document.

Nine study/failure tests cover blank preservation, all-empty batches, assignment tampering, response completeness, condition-separated author packets, withholding intended choices from rewrite and rewrite-fidelity inputs, and instruction drift. The complete suite passed **196 discovered tests and eight standalone watermark tests, 204 total**. These tests verify software boundaries; semantic competence is assessed only by the actual retained model outputs.

The independent audit implements probability aggregation, exponential tilting, route selection, HMAC labels, and randomization p-values separately from the production planner and detector. Both complete replay commands passed against the final archived artifacts:

```bash
python3 audit_natural_probabilities.py
python3 natural_results.py
python3 -m unittest discover -s tests -q
python3 watermark.py
```

Run with assertions enabled. This pilot provides evidence that catalog-level watermark detection can survive natural rewording without explicit symbolic cues. Its stronger fidelity-qualified result is three detected cases among all eight marked units. The small curated corpus, finite catalog, shared model family, agent batching, and observed semantic drift limit any claim about general ideas or adversarial rewriting.
