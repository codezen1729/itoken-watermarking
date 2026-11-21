# Protocol for the first real-model and language-recovery experiment

This protocol fixes the questions, labels, and evaluation rules before observing the extraction model's test results. It is a first research experiment over the existing proof domain. Passing it would not establish that general LLM-generated ideas can be robustly watermarked.

## What we are trying to establish

We separate three claims that need different evidence:

1. We can obtain actual next-token probabilities from a specified trained model and aggregate them into probabilities for predicting the next canonical iToken, conditional on an explicitly listed set of terminated textual candidates.
2. A model-assisted extractor can recover the same explicitly expressed logical steps from new natural wording in this domain, while recognizing missing, negated, hypothetical, ambiguous, and unrelated content.
3. Watermark detection remains useful after language transformations when generation, extraction, and verification are joined together. Good probability calculations or a good parser alone do not establish this third claim.

The previous finite reference sampler remains a separate baseline. Its known probabilities must not be relabeled as real LLM evidence.

## Fixed benchmark and provenance

`research_eval_cases.json` contains 32 examples: 8 development cases (`dev01`–`dev08`) and 24 test cases (`test01`–`test24`). The split was assigned before any extraction results were observed. The initial file SHA-256 is:

```text
449ed8f90a926f69a22be9cf344c124aecd2082c790078c5d1074f5030f0f824
```

There are 18 complete, 7 partial, and 7 abstention examples overall. The test portion has 14 complete, 5 partial, and 5 abstention examples. Each problem has one to three stages; these short examples test recovery, not reliable watermark detection.

The benchmark was newly authored by a separate project subagent. It is not human-authored data, an independently collected external corpus, or a representative sample of real writing. The author shares the project's broad context. The wording was deliberately chosen outside the old three rendering grammars, so a poor strict-parser score is expected by design; improvement over that baseline is not an unbiased estimate of real-world improvement.

The texts cover unfamiliar faithful wording, changed premise order, reversed presentation, two- and three-stage paths, repetition, irrelevant prefaces, an abandoned quoted route, a changed route with the same conclusion, missing steps, unsupported endpoint assertions, negated inferences, an unadopted hypothetical, unsound reasoning, multiple unselected routes, and instructions directed at the extractor. No watermark key, preferred-set score, or model probability was used to construct the labels.

The wording is free prose, but the formal atoms and original problem remain explicit anchors. This benchmark does not test general entity resolution, arbitrary renaming of predicates, or discovery of an unknown scientific ontology. Those are separate requirements for recovering reasoning from unrestricted real-world ideas.

## Freeze and access rules

The extraction designer may use the development cases before the first test run. Test texts, gold steps, gold evidence spans, and source routes must not be supplied to the prompt designer during that phase. The inference call itself necessarily receives the individual text under evaluation and its original public problem, but never its gold label.

Before the first test call, record the extractor source revision or hash, its complete prompt template, model identifier and resolved snapshot, tokenizer, decoding settings, output limits, parsing policy, and the benchmark hash. Freeze these settings for that run. Keep all raw model outputs, including malformed JSON, refusals, truncations, and failed extraction attempts.

Do not exclude failed outputs or repair them using gold information. If a deterministic format repair is part of the design, specify it before test inference and retain both raw and repaired outputs. If test errors motivate later improvements, retain the first-run results and label subsequent use of the same test cases exploratory. An untouched new set is then needed for another held-out claim.

## The gold contract

An accepted iToken is an explicitly asserted inference application, represented by its rule, referenced premises, and conclusion. The original problem supplies the meanings of formal symbols and the available premises. It does not authorize inventing an unexpressed intermediate inference merely because that inference would complete a valid proof.

- `complete`: the text explicitly asserts a unique full admissible path.
- `partial`: some valid applications are expressed, but the text omits part of a complete path. Recover only the expressed steps.
- `abstain`: the text supplies no unique asserted admissible path under this protocol. Do not produce a positive or negative provenance judgment from an invented path.

`expected_steps` and `expected_canonical` are deduplicated sets for scoring; presentation order is ignored. `expected_routes` contains only stages with an explicitly expressed first, branching iToken. A lone forced completion can identify a compatible route but does not make its missing choice step observed. `source_routes` is diagnostic provenance, not a label to copy: a route-changing edit can retain the conclusion and change the correct recovered path. Gold evidence spans are verbatim annotations; a matching substring alone is not a proof of semantic entailment.

Multiple explicitly presented routes require abstention from choosing one for watermark scoring. `diagnostic_mentioned_steps` can record those alternatives without treating their union as a single observed path. Negated, hypothetical, quoted-as-abandoned, or injected instructions must be interpreted in their actual linguistic scope.

## Real-model probability measurements

Use a locally loaded, trained instruction model. The planned first model is a four-bit Qwen2.5-1.5B instruction checkpoint under MLX; record the actual repository, immutable snapshot, and runtime versions used rather than treating this planned name as evidence of an executed run.

For each fixed context, score distinct candidate token sequences including the declared end-of-step termination. Record prompt token IDs, completion token IDs, per-token natural-log probabilities, and the summed sequence log probability. Confirm that completion tokenization preserves the exact model prefix boundary. Length-normalized scores, elicited numerical probabilities in a chat answer, and classifier confidence values are not substitutes for these token probabilities.

Group the disjoint scored sequences by their independently specified canonical inference. Report the raw listed mass, its logarithm when tiny, and the normalized distribution conditional on the candidate list. Do not claim that this list exhausts the model's possible expressions. An agreement between direct candidate sampling and normalized candidate masses checks the restricted sampler; it does not prove recovery of an unrestricted next-iToken distribution.

If unconstrained generations are sampled, preserve their raw text and report separately how many terminate correctly, express an admissible inference, can be extracted, and fall outside the listed candidate language. A small generation sample can reveal coverage problems; it cannot certify complete support.

## Recovery measurements

Report development and test results separately. Keep counts and denominators alongside percentages. Map implementation-specific status names to the three gold statuses using a fixed mapping, stated before scoring.

Primary measures are:

1. **Complete-path exact recovery:** among the 14 complete test cases, the recovered canonical set must equal the gold set and the output must be complete. An abstention counts as a failure.
2. **Canonical step precision and recall:** compare accepted deduplicated steps with gold steps. Include accepted steps on abstention cases as false positives. Abstaining on a case with gold steps reduces recall. Report micro totals and per-category counts.
3. **Decision-route recovery:** compare the observed branching choices with `expected_routes`, separately from forced completions. Report exact route-map matches on evidence-bearing cases and incorrect added decisions.
4. **Abstention and false acceptance:** on the five test cases labeled abstain, count any admitted path or provenance decision as false acceptance. Also report false-complete outputs on partial cases.
5. **Coverage and failure modes:** count malformed output, truncation, unsupported syntax, structural-check failure, semantic errors that pass structural checks, partial recovery, and correct abstention.

An abstain-all extractor therefore cannot obtain a good complete-path or recall score. Conversely, an extractor that always chooses a valid route cannot obtain a good false-acceptance score. A symbolic proof checker verifies inference validity and dependency availability; it cannot by itself prove that the source prose actually asserted the extracted inference. Evidence citations help auditing but do not solve that semantic problem.

The benchmark is small and curated. Any binomial interval is at most a descriptive calculation under an additional sampling assumption; it does not justify a population claim about unrestricted natural language. In particular, five rejection cases cannot tightly establish a low false-acceptance rate even if none is incorrectly accepted.

## End-to-end watermark measurements

After the probability and recovery interfaces are checked, run a separate cohort of actual-model-informed generated paths, with enough independent branch opportunities for detection. Freeze the watermark strength, path lengths, detector threshold, key sampling, and edit protocols before examining results. Preserve the distinction between an LLM-scored restricted candidate sampler and unrestricted idea generation.

Use independently sampled task/key units and pair their original and edited texts. The extractor receives neither the key nor preferred labels. Report detection on marked paths, unmarked controls, wrong keys, faithful free rewrites, deletions, alternative valid routes, and full independent rebuilding. Do not retain only rewrites that happened to parse successfully; include extraction failures in end-to-end coverage and power calculations.

Use the existing surrogate-key detector only under its stated key-independent-null contract. Count repeated structural evidence once. Unknown or incomplete paths must not inherit the full-path likelihood-ratio guarantee; an edited null can differ from the original Markov model even when the resulting proof is complete. Do not multiply evidence from repeated copies of the same short benchmark example as though it were independent.

For faithful rewrites, evaluate semantic preservation separately from detector survival. A rewrite that changes the logical path may legitimately destroy the original signal even when it preserves the final conclusion. A detector success rate on such examples is a different claim from invariance under wording changes.

## What would count as progress

Actual logged token probabilities, a reproducible restricted-candidate aggregation, and honest recovery measurements on the frozen test set are concrete progress. Failure cases—especially unsupported inferences that pass structural verification, missing choices filled in from the problem, and hypothetical routes treated as actual paths—are research findings that should be reported.

The first run can identify the next bottleneck and support a narrower feasibility claim. Robust watermarking of general LLM-generated ideas would still require larger independent datasets, multiple generators and extractors, unrestricted paraphrases, domain transfer, realistic solution-quality measurements, and attacks against a fixed deployed key.
