# Does stronger selection make the catalog-free watermark detectable?

We marked the choice of six arguments for each of eight fresh problems. An unmarked control, a weak watermark and a strong watermark used the same candidate arguments, model probabilities, random draws and detector. The test threshold was fixed at p ≤ 0.01. Each passage received one free rewrite. Two independent model readers assessed whether each rewrite kept every substantive reasoning relation.

| Selection strength | Plans detected | Originals detected | Rewrites detected | Rewrites judged faithful by both | Detected among faithful rewrites | Survived among faithful rewrites of detected originals |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0/8 | 0/8 | 0/8 | 7/8 | 0/7 | undefined (0 eligible) |
| Weak | 0/8 | 0/8 | 0/8 | 5/8 | 0/5 | undefined (0 eligible) |
| Strong | 8/8 | 0/8 | 0/8 | 6/8 | 0/6 | undefined (0 eligible) |

Every strongly marked plan put its target ahead of all 199 surrogate keys (p = 0.005). This measures how well the selector optimized the readings it had access to. On a fresh reading of the same unchanged prose, the eight p-values ranged from 0.040 to 0.675. The strong setting changed an average of 4.25 of the six chosen arguments relative to the paired control. All 24 originals and 24 rewrites yielded six admissible triples each. There were no abstentions.

A plan is scored from the readings used when choosing arguments. For an original or rewrite, a fresh reader re-extracts its steps from the problem and the prose alone. The last column requires both an original positive and two judgments of preservation. An unconditional positive after rewriting can include a new threshold crossing, even if the original was negative.

## How much the selection changed

The new reward favors sets whose logical steps lie close together along three directions derived from a secret key. The generator evaluates every six-argument subset and adds the strength times its standardized reward to the subset’s average model-informed utility. The verifier measures the same dispersion in newly recovered steps and ranks it against 199 surrogate keys. It has no access to the candidate set or a catalog of mechanisms. The [protocol](PROTOCOL.md) specifies the rule and the [model inclusion scores](generation.json) record the two actual next-token log probabilities for each candidate. These weights express conditional Yes/No inclusion preferences. The experiment does not measure unrestricted next-iToken probabilities.

| Arm | Mean arguments replaced relative to control, of 6 | Mean conditional Yes probability of selected arguments | Median plan p | Median original p | Median rewrite p |
|---|---:|---:|---:|---:|---:|
| Control | 0.00 | 0.9949 | 0.540 | 0.440 | 0.455 |
| Weak | 1.12 | 0.9955 | 0.255 | 0.435 | 0.590 |
| Strong | 4.25 | 0.9951 | 0.005 | 0.392 | 0.153 |

The model probability is a descriptive reading of the local usefulness question. It has no independent human quality rating. A strong mark can shift which recommendations a person receives.

## Where the signal disappeared

In a post hoc check, the six triples recovered from each strong original cited six different selected snippets. All 48 selected snippets could be matched one-to-one with the new readings by their exact source quotations. Only 2 of the 144 premise, action and consequence fields matched the planner reading verbatim. The detector therefore lost the signal even when the reader located each selected argument. Differences in the readers’ interpretations and the encoder’s response to how they expressed them could both contribute.

The folding-chair proposal illustrates the distinction. Both readers recovered the advice to put the number of available chairs against each date in a shared calendar so organizers can see whether their request will fit. One expressed the premise as availability varying by date. The other expressed it as an order depending on date and quantity. For the **whole six-argument proposal**, the p-values were 0.005 for the plan, 0.355 for the unchanged original and 0.630 for the rewrite. Both fidelity readers judged this rewrite preserved. Those document scores cannot attribute the change to the calendar argument alone. [Snippet reading](calls/pool1/out.json) · [Original reading](calls/original1/out.json) · [Rewrite reading](calls/rewritten1/out.json)

## Per-problem observations

| Problem | Arm | Plan p | Original p | Rewrite p | Reader judgments | Extracted steps, original / rewrite | Word edit fraction |
|---|---|---:|---:|---:|---|---:|---:|
| [Pottery drying rack](calls/judge0batch0/in.json) | control | 0.330 | 0.245 | 0.040 | preserved / preserved | 6 / 6 | 0.65 |
| [Pottery drying rack](calls/judge0batch1/in.json) | weak | 0.095 | 0.290 | 0.675 | preserved / preserved | 6 / 6 | 0.72 |
| [Pottery drying rack](calls/judge0batch2/in.json) | strong | 0.005 | 0.040 | 0.135 | preserved / changed | 6 / 6 | 0.76 |
| [Folding-chair lending](calls/judge0batch2/in.json) | control | 0.630 | 0.190 | 0.335 | preserved / preserved | 6 / 6 | 0.69 |
| [Folding-chair lending](calls/judge0batch0/in.json) | weak | 0.025 | 0.025 | 0.445 | preserved / preserved | 6 / 6 | 0.75 |
| [Folding-chair lending](calls/judge0batch1/in.json) | strong | 0.005 | 0.355 | 0.630 | preserved / preserved | 6 / 6 | 0.82 |
| [Oral-history captions](calls/judge0batch1/in.json) | control | 0.450 | 0.125 | 0.625 | preserved / preserved | 6 / 6 | 0.79 |
| [Oral-history captions](calls/judge0batch2/in.json) | weak | 0.360 | 0.910 | 0.735 | preserved / preserved | 6 / 6 | 0.82 |
| [Oral-history captions](calls/judge0batch0/in.json) | strong | 0.005 | 0.505 | 0.105 | preserved / preserved | 6 / 6 | 0.72 |
| [Beginner dance social](calls/judge0batch0/in.json) | control | 0.835 | 0.800 | 0.830 | changed / uncertain | 6 / 6 | 0.84 |
| [Beginner dance social](calls/judge0batch1/in.json) | weak | 0.695 | 0.940 | 0.505 | preserved / changed | 6 / 6 | 0.66 |
| [Beginner dance social](calls/judge0batch2/in.json) | strong | 0.005 | 0.540 | 0.895 | preserved / preserved | 6 / 6 | 0.87 |
| [Community bread swap](calls/judge0batch2/in.json) | control | 0.985 | 0.870 | 0.440 | preserved / preserved | 6 / 6 | 0.70 |
| [Community bread swap](calls/judge0batch0/in.json) | weak | 0.325 | 0.630 | 0.955 | preserved / changed | 6 / 6 | 0.64 |
| [Community bread swap](calls/judge0batch1/in.json) | strong | 0.005 | 0.675 | 0.110 | preserved / changed | 6 / 6 | 0.78 |
| [Garden sketchbooks](calls/judge0batch1/in.json) | control | 0.235 | 0.225 | 0.200 | preserved / preserved | 6 / 6 | 0.75 |
| [Garden sketchbooks](calls/judge0batch2/in.json) | weak | 0.030 | 0.440 | 0.065 | preserved / preserved | 6 / 6 | 0.71 |
| [Garden sketchbooks](calls/judge0batch0/in.json) | strong | 0.005 | 0.260 | 0.170 | preserved / preserved | 6 / 6 | 0.68 |
| [Costume fittings](calls/judge0batch0/in.json) | control | 0.435 | 0.915 | 0.770 | preserved / preserved | 6 / 6 | 0.74 |
| [Costume fittings](calls/judge0batch1/in.json) | weak | 0.185 | 0.430 | 0.425 | preserved / changed | 6 / 6 | 0.71 |
| [Costume fittings](calls/judge0batch2/in.json) | strong | 0.005 | 0.430 | 0.495 | preserved / preserved | 6 / 6 | 0.72 |
| [Shared chess room](calls/judge0batch2/in.json) | control | 0.655 | 0.635 | 0.470 | preserved / preserved | 6 / 6 | 0.70 |
| [Shared chess room](calls/judge0batch0/in.json) | weak | 0.380 | 0.335 | 0.690 | preserved / preserved | 6 / 6 | 0.75 |
| [Shared chess room](calls/judge0batch1/in.json) | strong | 0.005 | 0.265 | 0.120 | preserved / preserved | 6 / 6 | 0.64 |

The word edit fraction is one minus the sequence-match ratio on lowercased, whitespace-delimited words. It measures surface change. Preservation is judged by comparing the reasoning directly. The [per-document results](results.json) keep all 24 pairs, including failures. The [raw packets](calls/) contain the exact proposals, rewordings, readings, source quotations and explanations of the fidelity judgments.

## Scope

These are eight authored, everyday planning problems. Each original joins six independently proposed arguments. Both argument authorship and recovery use LLM agents, and judgments of faithfulness are assessments by two models of the same family. Agents could introduce any mechanism and any number of steps. Their recovered text is an interpretation of the reasoning, and even faithful language changes can alter the projections. The measured rates concern this generator, extractor and detector. They do not establish real-world authorship attribution or performance on long dependent proofs. The earlier catalog-free experiment used a different statistic, so the within-study strong, weak and control arms provide the comparable tests.
