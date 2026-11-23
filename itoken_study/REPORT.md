# Do independent readers recover the same iToken?

We tested a six-bit step identifier on eight fresh everyday problems. Each problem supplied 20 free-written arguments. We selected six per document under three conditions, rewrote each proposal once, and used two fresh readers for every original and rewrite. Each reader could recover any number of steps using ordinary language. The verifier received the public problem and supported reading. It never saw the generation pool or the other version. The protocol and numerical code were frozen before drawing the target keys. [Protocol](PROTOCOL.md) · [Raw packets](calls/)

The catalog-free code exactly reproduced all 144 original step IDs from a second reading. Both readers chose identical quotations for all 144. After faithful rewording, only 22/120 steps kept the same code. Both watermark rules detected three of eight originals and zero rewrites. In each arm, two initially positive originals had rewrites both judges considered faithful; neither remained detectable. The eligible cases overlap between arms.

## Watermark detection

Positive means that both independent readers met the same prespecified one-sided p ≤ 0.01 criterion for that version. The control row appears once per detector, using the same eight unmarked documents. A faithful rewrite received two independent preservation votes.

| Arm and detector | Plan positive | Original positive | Rewrite positive | Faithful rewrites | Positive among faithful rewrites | Survived among faithful rewrites of positive originals |
|---|---:|---:|---:|---:|---:|---:|
| Control, exact | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | undefined (0 eligible) |
| Control, tolerant | 0/8 | 0/8 | 0/8 | 8/8 | 0/8 | undefined (0 eligible) |
| Exact watermark | 3/8 | 3/8 | 0/8 | 6/8 | 0/6 | 0/2 |
| Bit-tolerant watermark | 3/8 | 3/8 | 0/8 | 6/8 | 0/6 | 0/2 |

The generator maximized its score using the planner’s readings. A high plan detection rate measures success on data used in selection. Fresh readers provide the independent test. Both readers use the same verifier key and represent the same document. Their votes do not multiply the statistical confidence. The machine-readable summary also reports each reader’s calls separately. [Per-reader counts](summary.json)

## Exact agreement and collisions

An exact match requires identical polarity and all five content bits for the same aligned argument. Original passages are aligned to selected source snippets by their quotations. For rewrites the two judges must agree on the source-to-quotation mapping. Missing or ambiguous readings remain in the attempted denominator. The last column compares different arguments within the same document. Counts pool correlated steps and overlapping study arms.

| Compared readings | Exact / attempted | Aligned / attempted | Same code for different aligned arguments |
|---|---:|---:|---:|
| Reader A vs B on unchanged originals | 144/144 (100.0%) | 144/144 | 54/720 (7.5%) |
| Reader A vs B on the same rewrite | 143/144 (99.3%) | 143/144 | 40/715 (5.6%) |
| Original A vs faithful rewrite B | 22/120 (18.3%) | 119/120 | 21/595 (3.5%) |
| Original B vs faithful rewrite A | 22/120 (18.3%) | 120/120 | 21/600 (3.5%) |
| Planner vs original A | 144/144 (100.0%) | 144/144 | 54/720 (7.5%) |
| Planner vs original B | 144/144 (100.0%) | 144/144 | 54/720 (7.5%) |

A fixed parser derives typed content from quoted evidence. Two readers who choose the same words get the same parse, so high agreement on one version partly measures span agreement. Cross-wording comparisons are more demanding. The encoder is MiniLM, a sentence similarity model. Public thresholds learned on six earlier problems turn its vectors into semantic bits. Distinct ideas can share one of the 64 possible codes. [Representation](canonical.py) · [Historical calibration](calibration.json) · [Encoder model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

The kite-field proposal shows the failure on a faithful rewrite. Both readers detected its exact-code watermark in the original with p = 0.005. Both gave p = 0.770 on the rewrite. The two preservation judges accepted the rewrite, while only two of the six aligned step IDs remained identical across versions. [Original and rewrite](calls/judgea1/in.json) · [First preservation judgment](calls/judgea1/out.json) · [Second preservation judgment](calls/judgeb1/out.json)

## Counterexamples with closely matched language

An independent author supplied 24 original–paraphrase–foil groups. Each foil changes a substantive relation while retaining much of the wording. The extractor saw one blinded member of each group in a separate context. A foil that receives the original’s complete code is a false match.

| Changed feature | Groups | One step for all three passages | Faithful paraphrase has identical code | Changed foil has identical code |
|---|---:|---:|---:|---:|
| All | 24 | 24 | 11/24 | 16/24 |
| Condition | 6 | 6 | 3/6 | 4/6 |
| Mechanism | 6 | 6 | 3/6 | 4/6 |
| Polarity | 6 | 6 | 1/6 | 2/6 |
| Roles | 6 | 6 | 4/6 | 6/6 |

For example, the north branch lending its scanner to the south branch and the south branch borrowing the scanner from the north branch both produced `001000`. Reversing who lent to whom also produced `001000`. All six role-swapped examples collided. A detector accepting that bit string cannot distinguish these lending directions.

The challenge tests logical sensitivity separately from ordinary within-document collisions. A poor paraphrase/foil distinction limits any claim that an iToken uniquely represents a logical step. [All challenge texts and specified changes](foils.json) · [Codes for each contrast](foil_results.json)

## Per-problem results

Each p-value pair lists the reader A and B values. The table includes every case, including fidelity disagreements and below-threshold results.

| Topic | Detector and arm | Plan p | Original p, A / B | Rewrite p, A / B | Preservation votes |
|---|---|---:|---:|---:|---|
| Kite field | Control, exact | 0.790 | 0.790 / 0.790 | 0.660 / 0.660 | preserved / preserved |
| Kite field | Control, tolerant | 0.100 | 0.100 / 0.100 | 0.115 / 0.115 | preserved / preserved |
| Kite field | Exact watermark | 0.005 | 0.005 / 0.005 | 0.770 / 0.770 | preserved / preserved |
| Kite field | Bit-tolerant watermark | 0.005 | 0.005 / 0.005 | 0.090 / 0.090 | preserved / preserved |
| Ensemble rehearsal | Control, exact | 0.225 | 0.225 / 0.225 | 0.255 / 0.255 | preserved / preserved |
| Ensemble rehearsal | Control, tolerant | 0.510 | 0.510 / 0.510 | 0.485 / 0.485 | preserved / preserved |
| Ensemble rehearsal | Exact watermark | 0.030 | 0.030 / 0.030 | 0.210 / 0.210 | preserved / preserved |
| Ensemble rehearsal | Bit-tolerant watermark | 0.150 | 0.150 / 0.150 | 0.870 / 0.870 | preserved / preserved |
| Printed bulletin | Control, exact | 0.115 | 0.115 / 0.115 | 0.260 / 0.260 | preserved / preserved |
| Printed bulletin | Control, tolerant | 0.470 | 0.470 / 0.470 | 0.555 / 0.555 | preserved / preserved |
| Printed bulletin | Exact watermark | 0.020 | 0.020 / 0.020 | 0.155 / 0.155 | changed / changed |
| Printed bulletin | Bit-tolerant watermark | 0.100 | 0.100 / 0.100 | 0.820 / 0.820 | preserved / preserved |
| Shadow puppets | Control, exact | 0.530 | 0.530 / 0.530 | 0.675 / 0.830 | preserved / preserved |
| Shadow puppets | Control, tolerant | 0.735 | 0.735 / 0.735 | 0.580 / 0.530 | preserved / preserved |
| Shadow puppets | Exact watermark | 0.060 | 0.060 / 0.060 | 0.240 / 0.240 | preserved / preserved |
| Shadow puppets | Bit-tolerant watermark | 0.220 | 0.220 / 0.220 | 0.820 / 0.820 | preserved / preserved |
| Postcard table | Control, exact | 0.965 | 0.965 / 0.965 | 0.755 / 0.755 | preserved / preserved |
| Postcard table | Control, tolerant | 0.065 | 0.065 / 0.065 | 0.055 / 0.055 | preserved / preserved |
| Postcard table | Exact watermark | 0.070 | 0.070 / 0.070 | 0.250 / 0.250 | preserved / preserved |
| Postcard table | Bit-tolerant watermark | 0.010 | 0.010 / 0.010 | 0.015 / 0.015 | preserved / preserved |
| Linen lending | Control, exact | 0.255 | 0.255 / 0.255 | 0.945 / 0.945 | preserved / preserved |
| Linen lending | Control, tolerant | 0.610 | 0.610 / 0.610 | 0.415 / 0.415 | preserved / preserved |
| Linen lending | Exact watermark | 0.060 | 0.060 / 0.060 | 0.900 / 0.900 | preserved / preserved |
| Linen lending | Bit-tolerant watermark | 0.065 | 0.065 / 0.065 | 0.670 / 0.670 | preserved / preserved |
| Birdsong walk | Control, exact | 0.155 | 0.155 / 0.155 | 0.445 / 0.445 | preserved / preserved |
| Birdsong walk | Control, tolerant | 0.560 | 0.560 / 0.560 | 0.415 / 0.415 | preserved / preserved |
| Birdsong walk | Exact watermark | 0.005 | 0.005 / 0.005 | 0.815 / 0.815 | preserved / preserved |
| Birdsong walk | Bit-tolerant watermark | 0.010 | 0.010 / 0.010 | 0.495 / 0.495 | changed / preserved |
| Fountain pen tryout | Control, exact | 0.870 | 0.870 / 0.870 | 0.025 / 0.025 | preserved / preserved |
| Fountain pen tryout | Control, tolerant | 0.255 | 0.255 / 0.255 | 0.260 / 0.260 | preserved / preserved |
| Fountain pen tryout | Exact watermark | 0.005 | 0.005 / 0.005 | 0.020 / 0.020 | changed / changed |
| Fountain pen tryout | Bit-tolerant watermark | 0.050 | 0.050 / 0.050 | 0.230 / 0.230 | changed / changed |

All detection and preservation rates refer to short proposals composed of compatible arguments from a sampled pool. The LLM’s local inclusion preference comes from real Qwen Yes/No token probabilities. The experiment does not estimate the unrestricted probability of the next logical step. Fidelity and argument alignments were assessed by models. [Study data](results.json) · [Reproduction instructions](README.md)

Across all 24 pairs the median word edit fraction was 0.66. Among the jointly faithful pairs it was 0.66. This descriptive fraction is one minus a sequence-match ratio on lowercased whitespace-delimited words. It measures wording changes, while the judges assessed the reasoning.

The independent numerical replay re-encoded all 811 step readings from the archived quotations, reproduced the 24 selections and detector ranks, and checked the agreement and fidelity-conditioned rates. [Replay record](independent_replay.json)
