# Stronger-model recovery and watermark survival protocol

We want to determine whether a stronger language model can recover the same logical choices after their wording changes, and whether the resulting detector still finds the watermark. Probability estimation, semantic recovery, and statistical detection remain separate claims. This protocol fixes the comparison before the new held-out examples or watermark outcomes are evaluated.

## Fresh logical-recovery holdout

`fresh_holdout.json` contains 32 new test-only cases: 12 complete paths, 8 partial paths, and 12 examples requiring abstention. There are 47 gold inference applications. Its SHA256 is:

```text
58212368fe99fe80f3ed4a015c0c4545a11d867c6544413b6416306090d6752f
```

The previous experiment's examples can now be used for development. The new examples were authored by the benchmark agent and are withheld from the recovery designer before freeze. They retain explicit atom names and the existing public propositional problem. They use new wording, altered exposition, incomplete records, scoped denials, counterfactuals, unadopted alternatives, unsound claims, and instructions directed at the extractor. They are agent-authored research examples, not human annotations or a representative corpus of unrestricted ideas.

Gold labels identify only explicitly asserted inference applications. A valid completion inferred from the public problem is not evidence that the source expressed it. A forced completion may be a correctly recovered partial step but does not count as a recovered branching choice. Multiple explicitly unselected alternatives must not be collapsed into one chosen route. Evidence spans are annotation aids, not proofs of entailment.

Before the first held-out call, freeze the backend identity, prompt and parser source, decoding or agent settings, format constraints, verifier policy, thresholds, and benchmark hash. If an agent backend does not expose a checkpoint revision, temperature, or token probabilities, record those fields as unavailable; do not invent them. Only the current source text, public problem, and public inference vocabulary enter recovery. Categories, expected routes, gold evidence, watermark key, preferred labels, and intended paths do not.

Retain every initial response, including invalid output, explicit abstention, truncation, and infrastructure failures. A retry after a transport failure must preserve the first attempt and be distinguished from rerolling an unfavorable model answer. Changes prompted by this holdout's errors require a new heldout set for a subsequent confirmatory claim.

Report complete-path exact recovery, partial-step exact recovery, observed-branch recovery, canonical step precision/recall, abstention counts, and semantic false acceptance. Denominators include all cases. Invalid output is not quietly removed. A false semantic acceptance is distinct from a false watermark detection; compute whether it supplies any branching evidence before calling it a watermark false positive.

## Fixed minimal watermark cohort

The minimal comparison uses eight independently sampled task/key units, 32 branching choices per path, watermark strengths 0 and 6, and one free rewording for every original document. This gives 16 planned paths and 32 source/reworded documents. The two strengths are paired within a unit, and each rewrite remains paired with its own original. These are eight experimental units, not 32 independent trials. The primary detector uses alpha 0.01 and 199 independently sampled surrogate keys, giving p-value resolution 0.005.

Fix the random seeds and sampling streams in a run manifest before generating paths. Key generation, route sampling, surrogate keys, input ordering, and any rewrite sampling need distinct streams. Keys must be sampled without selecting for strong watermark outcomes. A paired common route-randomness stream is permitted between strength 0 and 6, but must be independent of the key stream. Publish demonstration keys only as reproducibility artifacts; they are unsuitable for deployed attribution.

Use actual trained-model token probabilities for the declared planner. If a finite probability bank is reused across units, its scoring prompt must deliberately condition only on the public local state represented by that bank. Save its exact prompt, candidate token sequences, per-token log probabilities, EOS convention, covered mass, and canonical grouping. Do not describe these values as predictions under a full preceding path that was never supplied to the model. Reuse across independent keys does not create additional learned-model measurements; distinguish the number of scored contexts from the number of sampled task/key units.

If the stronger model is used only for rewording/recovery while a smaller locally scored model supplies planner probabilities, call the experiment a hybrid. Do not attribute the smaller model's logits to the stronger model. In all cases, finite-list normalization remains conditional on that candidate set. Report the coverage, entropy, bias strength, and logical validity of the planned paths rather than treating strong preference among listed alternatives as complete support over ideas.

The 32-choice length gives more structural opportunities than the previous eight-choice demonstration. It does not guarantee useful power: concentrated model preferences, empty preferred sets, and incomplete recovery can still erase the signal. Keep length, strength, and sample count fixed after observing outcomes. A larger follow-up is a new experiment, not an unreported extension until success.

## Independent rewording and recovery

The reworder receives the original text and a request to preserve every asserted inference and explicit atom name while changing wording and sentence organization. It does not receive the watermark key, preferred labels, detector scores, marked/unmarked label, or extraction outputs. The extractor receives only the resulting text and public problem; it never sees the original intended path as a repair target. Preserve the exact prompts and raw outputs for both roles.

Use separate fresh model contexts for rewording, recovery, and fidelity assessment. In particular, do not put marked and unmarked documents from the same unit in a shared recovery/rewording prompt or persistent agent history. Copying between them could make the nominal unmarked artifact depend on its secret key, violating the randomization test's null. If calls are batched, record batch membership and blind condition labels. Shared batches can make recovery outcomes dependent; the pilot should report descriptive paired counts rather than unsupported independent-binomial intervals.

An assessor separate from the extractor compares original and rewritten text, without seeing detector outcomes or marked/unmarked labels. It records whether each asserted step was preserved, omitted, changed, or uncertain. If the same underlying model supplies all roles, context separation is useful but does not establish independent semantic truth. Report this shared-model limitation and retain exact evidence. Agreement between the intended path and extracted path is a consistency measurement, not a substitute for assessing what the rewritten prose actually says.

A free rewrite that changes the logical path is a fidelity failure even when the endpoint remains the same. Do not discard it and report only successful rewrites. For the primary end-to-end rate, include every assigned document, counting failed or abstained recovery as no positive detection. Also report the fidelity-preserving subset separately, with its denominator. This conditional analysis must not replace the unconditional result.

## Detection and survival measurements

For each unit and strength, retain planned canonical steps and four distinct observations: planned-path detection, original-text recovery/detection, rewritten-text recovery/detection, and independently assessed rewrite fidelity. Use the structural surrogate-key test on deduplicated recovered branching choices. A completion-only recovery yields insufficient evidence. Never transfer the old synthetic predictor's likelihood ratio to new model probabilities or to an edited prose distribution.

The primary result is the fraction of all eight marked units whose rewritten text is positively detected. Report the corresponding count on the eight unmarked units, alongside extraction coverage and fidelity. Also report:

- the fraction of planned-detected marked units still detected after rewording;
- the fraction of original-text-detected marked units still detected after rewording;
- the number of faithful, changed, and uncertain rewrites in each condition;
- exact path/branch agreement, missing steps, extra steps, and recovered choice counts;
- planned, original, and reworded p-values, including explicit abstention values;
- comparison of unmarked original and reworded false detections, without treating these eight controls as calibration of a 1% population false-positive rate.

If a survival denominator is zero, report the value as undefined, not 100% or zero. “No signal before rewriting” and “signal lost by rewriting” are different outcomes. A positive score without faithful recovery cannot establish survival of the original idea watermark. Detection of faithfully rewritten selected choices remains a bounded-domain result; it does not prove originality or identify who first conceived an idea.

Deletion and independent-path replacement are useful secondary controls, but they are not included in the primary 32-document cohort. If run, declare their count, removal/rerouting rule, seeds, and reporting labels before viewing their detector outcomes. Removing or changing logical choices tests a different property from changing their wording.

## What would justify the conclusion

An improved fresh-holdout score would establish better recovery on these examples. A high joint marked-rewrite detection count, low descriptive unmarked detections, and independently supported fidelity would establish watermark survival in this explicit-symbol experiment. These claims remain separate from recovering arbitrary ontologies, equivalence under variable renaming, detecting short ideas with little branching freedom, or resisting an informed adversary. All first-run failures and raw calls must remain available for review.
