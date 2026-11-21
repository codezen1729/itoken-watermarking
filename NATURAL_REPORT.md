# Natural-language ideas without explicit symbol cues

We removed the artificial atom names and route labels from the ideas. We also removed the requirement that rewording retain any particular words. This run asks whether a watermark on selected planning mechanisms can still be recovered from ordinary prose.

Among marked texts detected before rewording, **5/5 remained detected afterward**. Across all eight marked rewrites, the detection rate was **5/8**; the corresponding unmarked control count was **0/8**. The full denominators include weak initial marks, incomplete recovery, and any unfaithful text. No key, generated proposal, or rewrite was rerolled after seeing an outcome.

However, requiring independent model assessments to pass for both planned-to-original and original-to-rewrite fidelity leaves **3/8 marked attempts that are both faithfully represented and detected**. 5 marked pairs pass both fidelity checks, and 3 of those are detected. Two of the five raw positive detections come from originals whose fidelity was changed or uncertain. Unchanged detector scores therefore do not establish unchanged logical content.

The recovery pipeline uses GPT-6.2-astra agents with high reasoning effort in separate fresh contexts. A proposer identifies the mechanisms actually advocated in the text; a verifier independently checks their roles, timing, action and consequence, citing source sentences. Neither sees the intended path, watermark key, condition label, or detector score. Public options appear under shuffled opaque answer IDs in the recovery prompt; those IDs never need to occur in the source prose. No symbol or literal-keyword match is required for acceptance.

The agent interface records requested model routing, not an independently attested checkpoint. Temperature, seed, token probabilities and API finish reasons are unavailable. Agents may read their assigned packets and serialize explicit judgments; their access restrictions are procedural. Raw first responses and batch membership are retained. Different contexts can still share errors because all semantic roles use the same model family.

The separate frozen recovery challenge contains ordinary-language descriptions, partial plans, rejected and quoted suggestions, source instructions, and conflicting decisions. Four contrast pairs use exactly the same alphabetic word multiset while reversing roles, operation order, or adoption versus rejection. The model pipeline recovered **24/24 exact choice sets**, with **24/24 correct choice/status/ambiguity results**, and **28/28 gold choices**. On the eight same-word contrast cases, its strict result was **8/8**.

We froze a TF-IDF word-overlap baseline before inference. It ignores word order and assertion scope, chooses the uniquely closest option for each decision, and abstains on ties. It recovered **7/24 exact challenge choice sets** and **0/8** on the same-word contrasts. This comparison tests one specified lexical shortcut; it does not prove that no other shortcut exists.

The main cohort uses eight different everyday planning problems, each with sixteen decisions and three alternatives per decision. The alternatives share ordinary concepts and differ in who acts, what happens first, how work is allocated, or how the proposed mechanism produces its benefit. A separate author agent turns each selected plan into continuous prose. Another agent rewords only that prose, without seeing the catalog selections. Marked/unmarked pairs and original/reworded texts are kept in different inference contexts.

These are **curated semantic choices**, not machine-checked logical proofs. The complete public catalog is supplied during recovery. The model is matching expressed mechanisms to a known set, rather than discovering an unrestricted ontology from arbitrary ideas. Causal benefits are author judgments, not experimentally established facts. This is a substantial change from renaming formal atoms, but it does not settle general idea watermarking.

Fresh Qwen2.5-0.5B-Instruct token probabilities supply the planner. We scored **128 local contexts and 384 EOS-terminated natural-language expressions**, one expression per option. No symbolic probability bank was reused. The conditional probabilities are tilted by the keyed hash at strengths zero and six. The local model sees the public problem and one decision, not the preceding full plan. There are eight separately keyed problem units, sixteen choices each, a detection threshold of 0.01, and 199 independent surrogate keys; p-value resolution is 0.005. Route uniforms are paired within a problem and independent of key draws.

The listed candidate strings cover **0.00314818 to 0.798875** of total model probability mass, averaging **0.200405**. Mean conditional choice entropy is **0.572702 nats**. These are probabilities for predicting the next catalog application conditional on the three listed expressions. One wording per option does not sum over all ways to express an idea; wording, length and menu order can affect the probabilities. This is a hybrid experiment: the stronger agents do not supply these logits. The checkpoint/runtime details and all token scores are retained in the probability bank.

| Primary result | Marked | Unmarked |
|---|---:|---:|
| Planned paths detected | 5/8 | 0/8 |
| Original prose detected | 5/8 | 0/8 |
| Reworded prose detected | 5/8 | 0/8 |
| Originals with exact plan-label agreement | 8/8 | 8/8 |
| Rewrites with exact plan-label agreement | 8/8 | 8/8 |
| Original choices matching the intended plan | 128/128 | 128/128 |
| Reworded choices matching the intended plan | 128/128 | 128/128 |
| Reworded detector abstentions | 0/8 | 0/8 |
| Word-overlap baseline: reworded choices matching the plan | 61/128 | 52/128 |
| Word-overlap baseline: original detections | 0/8 | 1/8 |
| Word-overlap baseline: reworded detections | 1/8 | 0/8 |

Two further groups of fresh assessors check fidelity independently of recovery. One compares each original with its intended applications. The other compares the rewrite with the actual original, seeing decision topics but no intended selections. Marked original fidelity was `{'preserved': 5, 'changed': 1, 'uncertain': 2}`; marked rewrite fidelity was `{'preserved': 6, 'changed': 2}`. The unmarked counts were `{'preserved': 7, 'uncertain': 1}` and `{'preserved': 3, 'changed': 5}`. Across all sixteen pairs, author fidelity passed for 12 originals and rewrite fidelity passed for 9 rewrites. These are model judgments, not independent human annotation. The [fidelity review](NATURAL_FIDELITY_REVIEW.md) retains every flagged detail. Matching the recovered path to the plan is reported separately and is not used as a substitute for fidelity assessment. In particular, all 512 recovered catalog decisions match the plan even though assessors flagged changed quantifiers, roles, eligibility, and scope. This is label agreement, not 512 verified semantic inferences.

Requiring both fidelity checks to pass, the joint marked-rewrite detection count was **3/8** across all eight units; detection within that assessed-faithful subset was **3/5**. A faithfully rewritten wrong original therefore does not automatically count as faithful embedding of the intended idea. The recorded marker scan found **0** documents containing artificial atom, route-ID, or formal-operator cues.

| Problem | Marked p: planned / original / rewrite | Unmarked p: planned / original / rewrite | Rewritten choices matching plan: marked / unmarked |
|---|---|---|---|
| A calmer afternoon at the neighborhood library | 0.005 / 0.005 / 0.005 | 0.665 / 0.665 / 0.665 | 16/16 / 16/16 |
| A neighborhood café that lends reusable cups | 0.005 / 0.005 / 0.005 | 0.230 / 0.230 / 0.230 | 16/16 / 16/16 |
| A shared garden with uneven volunteer attendance | 0.020 / 0.020 / 0.020 | 0.890 / 0.890 / 0.890 | 16/16 / 16/16 |
| A small museum welcoming weekend visitors | 0.005 / 0.005 / 0.005 | 0.900 / 0.900 / 0.900 | 16/16 / 16/16 |
| A workable shared craft studio | 0.025 / 0.025 / 0.025 | 0.360 / 0.360 / 0.360 | 16/16 / 16/16 |
| A welcoming board-game social evening | 0.005 / 0.005 / 0.005 | 0.225 / 0.225 / 0.225 | 16/16 / 16/16 |
| Coordinating a community choir’s rehearsals | 0.020 / 0.020 / 0.020 | 0.265 / 0.265 / 0.265 | 16/16 / 16/16 |
| A neighborhood household-goods swap fair | 0.010 / 0.010 / 0.010 | 0.690 / 0.690 / 0.690 | 16/16 / 16/16 |

The word-overlap baseline is also scored on every main-cohort document. Good lexical performance would show that the ordinary-language corpus still contains useful surface cues; worse performance does not certify semantic reasoning. Counts use eight paired problem/key units, not 512 independent decision trials. Shared model batches and related decisions create additional dependence. Eight controls cannot establish a one-percent population false-positive rate, and no independent-binomial population interval is claimed.

This run also changes the path length from 32 symbolic branches to 16 natural planning choices, as well as the domain and candidate distribution. It is not a controlled estimate of the effect of removing symbol cues alone.

The test concerns attempts at faithful rewording of catalog-backed planning ideas. It does not test deletion, aggressive summarization, changing the chosen mechanism, inventing an alternative route to the same goal, or an informed attack. Removing visible symbols addresses the shortcut found in the previous experiment; unrestricted natural-language-to-logic recovery remains open.

The [independent audit](NATURAL_AUDIT.md), [frozen protocol](NATURAL_PROTOCOL.md), [catalog notes](NATURAL_CATALOG_NOTES.md), and [recovery methods](NATURAL_RECOVERY_METHODS.md) describe the assumptions. See the [summary](research_results/natural_language_v1/summary.json), [individual records](research_results/natural_language_v1/paired_rows.json), and [agent-call artifacts](research_results/natural_language_v1/agent_artifacts/agent_calls.json). Saved results can be replayed without invoking a model:

```bash
python3 natural_results.py
python3 audit_natural_probabilities.py
python3 -m unittest discover -s tests -q
python3 watermark.py
```

The full generated texts are also available as plain text in [the examples folder](research_results/natural_language_v1/examples/), including the first marked [original](research_results/natural_language_v1/examples/unit_00_marked_original.txt) and its [rewrite](research_results/natural_language_v1/examples/unit_00_marked_reworded.txt). These are direct copies of recorded model texts, with no labels added to their content. Solution usefulness and real-world performance were not independently evaluated.
