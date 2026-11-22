# Watermark survival after open-ended recovery of reasoning steps

The detector flagged **0/8 marked rewrites** and **0/8 control rewrites**. No initially detectable marked original met both readers’ preservation criterion. Survival conditional on confirmed preservation is **unmeasurable in this cohort**.

| Measurement | Marked | Unmarked |
|---|---:|---:|
| Planned steps detected | 0/8 | 0/8 |
| Originals detected | 1/8 | 1/8 |
| Rewrites detected | 0/8 | 0/8 |
| Rewrites detected among detected originals | 0/1 | 0/1 |
| Rewrites detected among detected plans | undefined (0 eligible) | undefined (0 eligible) |
| Pairs assessed preserved by the first reader | 4/8 | 5/8 |
| Pairs both readers assessed preserved | 3/8 | 5/8 |
| Pairs with different fidelity verdicts | 2/8 | 0/8 |
| Detected rewrites both readers assessed preserved, across all attempts | 0/8 | 0/8 |
| Survival among detected originals both readers assessed preserved | undefined (0 eligible) | 0/1 |
| Survival using the first reader’s preservation criterion | undefined (0 eligible) | 0/1 |
| Faithful pairs with detected rewrites, across all attempts | 0/8 | 0/8 |
| Survival if both extractions also passed the completeness review | undefined (0 eligible) | 0/1 |
| Detector abstentions on originals | 0/8 | 0/8 |
| Detector abstentions on rewrites | 0/8 | 0/8 |

Each of eight public problems had 24 freshly written candidate arguments. The generation step selected twelve for each original and joined their text into paragraphs. Qwen2.5-0.5B-Instruct supplied Yes and No token probabilities for a local usefulness question. Conditional Yes weights and a shared random draw determined control selection. The marked selection added four times the keyed projection score before taking the twelve highest values. This experiment embeds the watermark through the choice of which proposed arguments to include.

Readers extracted any number of advocated steps from each passage. Each step has a premise, action, consequence, and a verbatim supporting quotation. The original and rewritten readers worked in separate contexts. At detection time they saw only the public problem and prose. The detector used those open triples, the public problem, a fixed encoder and a key. The temporary generation pool played no role at detection. [Protocol](PROTOCOL.md)

The encoder read each field using the first eight transformer blocks of the local Qwen checkpoint. It pooled token representations, subtracted the public problem representation after normalization, and normalized each field difference. Each field has a separate key-derived projection. The detector averaged scores over retained triples and compared the target score with scores using 199 surrogate keys. A p-value of 0.01 or smaller counts as detection. The argument weights describe the local Yes/No inclusion policy. Native probabilities for producing whole ideas remain unknown.

Two fidelity readers separately compared the two versions directly and assessed changes to actors, causal relationships, timing and scope. Another reader checked each extraction for missing and unsupported reasoning. These are agent judgments and can share errors. A verbatim quotation verifies textual provenance. The model judgments supply the semantic interpretation. [Second-reader protocol](FIDELITY_CHECK.md)

| Problem | Condition | Planned p | Original p | Rewrite p | Original / rewrite steps | Fidelity, first / second reader | Original / rewrite extraction review |
|---|---|---:|---:|---:|---:|---|---|
| A film night at the swimming club | marked | 0.725 | 0.725 | 0.705 | 12 / 12 | preserved / changed | complete / complete |
| A film night at the swimming club | unmarked | 0.755 | 0.755 | 0.760 | 12 / 11 | preserved / preserved | partial / partial |
| Opening a neighborhood repair morning | marked | 1.000 | 1.000 | 1.000 | 12 / 12 | preserved / preserved | complete / complete |
| Opening a neighborhood repair morning | unmarked | 1.000 | 1.000 | 1.000 | 12 / 12 | preserved / preserved | partial / partial |
| A seed cabinet at the branch library | marked | 0.775 | 0.755 | 0.750 | 12 / 13 | changed / changed | complete / complete |
| A seed cabinet at the branch library | unmarked | 0.775 | 0.765 | 0.800 | 12 / 12 | preserved / preserved | partial / complete |
| Serving lunch during a school kitchen renovation | marked | 0.905 | 0.915 | 0.925 | 12 / 12 | changed / changed | complete / complete |
| Serving lunch during a school kitchen renovation | unmarked | 0.935 | 0.930 | 0.930 | 12 / 12 | changed / changed | complete / complete |
| A starwatch on the common | marked | 0.275 | 0.285 | 0.325 | 12 / 12 | preserved / preserved | partial / complete |
| A starwatch on the common | unmarked | 0.330 | 0.285 | 0.335 | 12 / 12 | changed / changed | partial / complete |
| A riverside audio walk | marked | 0.520 | 0.480 | 0.575 | 12 / 12 | changed / changed | partial / complete |
| A riverside audio walk | unmarked | 0.520 | 0.515 | 0.510 | 12 / 12 | changed / changed | partial / complete |
| A neighborhood sheet-music shelf | marked | 0.545 | 0.545 | 0.580 | 12 / 12 | preserved / preserved | partial / complete |
| A neighborhood sheet-music shelf | unmarked | 0.600 | 0.655 | 0.620 | 12 / 13 | preserved / preserved | partial / partial |
| A neighborhood jigsaw exchange | marked | 0.015 | 0.010 | 0.015 | 12 / 12 | changed / preserved | complete / complete |
| A neighborhood jigsaw exchange | unmarked | 0.015 | 0.010 | 0.015 | 12 / 12 | preserved / preserved | complete / complete |

Across the 16 pairs, the first fidelity reader judged 9 rewrites preserved, 7 changed and 0 uncertain. Across 32 passages, extraction reviews judged 20 complete, 12 partial and 0 uncertain. The detector retained 385 of the 385 proposed triples. Generation used 198 extracted triples across 192 candidate snippets. [Original passages and their selections](generation.json) · [Free extractions and assessments](calls/) · [Per-document results](results.json)

Extraction reviewers rated 7/16 original readings and 13/16 rewrite readings complete. The unmarked film-night text illustrates a partial rating. Its reader preserved the suggestion to use a striped awning as a meeting point. The source compared remembering the awning with remembering a volunteer’s name. The extracted premise compared recognizing the awning with recognizing the volunteer. [Source reading assessment](calls/review_original1/out.json)

The jigsaw exchange was the only original to meet the threshold, in both its marked and control forms. Each had p = 0.015 at planning, p = 0.010 for the original reading, and p = 0.015 after rewording. The rank moved from one surrogate scoring at least as high as the target to two. For the marked version, the first fidelity reader judged that placing children’s puzzles on the low shelf lost the requirement to reserve that space for them. The second judged the rewrite preserved. This pair illustrates sensitivity to a small rank change and the effect of reader disagreement on the preservation analysis.

A post hoc comparison found that marking replaced 2, 3, 0, 3, 2, 0, 4, 2 of the twelve selected snippets across the eight problems, respectively. The planned score under the target key rose by a mean of 0.0552 relative to its paired control. The control plans’ 199 surrogate scores had a mean sample standard deviation of 0.5601. Dividing each paired gain by its own control standard deviation and averaging gave **0.099 surrogate standard deviations**. The sampler therefore moved the score only slightly relative to the spread a detector saw across keys. This comparison uses the planned vectors before any whole-document extraction or rewriting.

Selection and detection now use a continuous projection of open text fields. The previous experiment required exact catalog IDs for its keyed hash. The difference in outcomes therefore combines the new watermark, new corpus and new recovery method. Shared topics or wording can influence this encoder, and combining independently generated arguments limits the conclusions about long chains of dependent reasoning or entirely spontaneous essays. The eight paired problems and model-based judgments support a descriptive result for this cohort.

A follow-up requested light copyedits of the initially detectable marked original and its matched control. Both fidelity readers accepted both edits. Each detector p-value moved from 0.010 to 0.015, crossing the fixed threshold by one rank. [Follow-up protocol](COPYEDIT_PROTOCOL.md) · [Copyedit results and reader judgments](COPYEDIT_REPORT.md)

Recompute the target and surrogate scores, conditional inclusion weights, shared sampling noise, selected documents, citation checks, fidelity bindings and all reported fractions with `python open_study/audit.py --check` from the repository root. The audit uses archived vectors and log probabilities from the actual model run. It does not rerun the neural encoder. Source files and raw calls accompany the results.
