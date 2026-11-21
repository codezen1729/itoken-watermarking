# What we mean by an iToken in this prototype

We want the watermark to survive a change in wording when the logical steps remain the same. To test that idea, we first need a setting where we can say exactly what a step is, which alternatives are valid, and which edits preserve it. This prototype provides that setting. It is a controlled symbolic experiment, not a trained LLM and not a general detector of AI-generated ideas.

## The problem and its three routes

We work in a restricted propositional language with atoms, implication, conjunction, and disjunction. A problem contains a starting fact, a target, a fixed set of premises, and a public problem identifier. At stage t, we start from p_t and must derive p_(t+1). We allow three routes, each containing two inference applications:

1. **Implication route:** from p_t and p_t -> m_t, derive m_t by modus ponens; then use m_t -> p_(t+1) to derive p_(t+1).
2. **Conjunction route:** from p_t and d_t, derive p_t AND d_t by conjunction introduction; then use (p_t AND d_t) -> p_(t+1).
3. **Disjunction route:** from p_t, derive p_t OR e_t by disjunction introduction; then use (p_t OR e_t) -> p_(t+1).

The problem supplies p_0, every required d_t, and the implications used by the routes. We join several stages to reach the final target. Every permitted route is logically valid and has the same number of inference applications. The choice changes the intermediate claim and inference rule, even though the final result is unchanged.

This is deliberately a small proof domain. We are not claiming that every scientific or creative idea is a deduction in propositional logic. A wider system would need separate operations for assumptions, hypotheses, actions, and other steps that are not deductive consequences.

## The unit and its canonical form

An iToken is one inference application:

    i = (rule, referenced premises, conclusion).

It is a symbolic object, serialized as canonical JSON. It is not an arbitrary group of adjacent words. A textual passage is a realization of this object only when the parser recognizes its rule, premises, and conclusion. A proof checker then checks whether the inference is valid and whether its premises are supplied or established elsewhere in the proof.

Each route has two iTokens. At its entry, three alternative first iTokens are available; once one is selected, the second inference is fixed by that route. The forced continuation supplies no additional independent choice. We should not count two inference applications as two watermark-bearing decisions.

Canonicalization is limited to the transformations implemented here:

- The three explicitly supported rendering templates, which state the inference rule.
- Whitespace differences and premise-order differences.
- Operand order in binary conjunctions and disjunctions.

The checker can reconstruct dependencies independently of the order in which proof lines are presented. Repeated identical evidence is not a new logical choice. We do not normalize arbitrary logical equivalence, arbitrary natural-language paraphrases, variable renaming, associativity, or every possible alternative proof of the same conclusion. Those are separate research and implementation questions.

We must preserve the difference between the three routes. If canonicalization reduced every proof to its final conclusion, it would erase the very choices that carry the watermark.

## From text probabilities to iToken probabilities

The prediction context is explicit:

    h_t = Serialize(problem, canonical partial path before step t).

The serialization is fixed. It includes the formal premises, target, public identifier, and previously selected structure needed by the predictor. A probability over iTokens therefore depends on a specified context and generation protocol; it is not an intrinsic probability of an idea independent of how we ask the model.

Let x be a complete textual realization of one step, including its termination symbol. Let C_h(x) be its parsed canonical iToken. For a specified text predictor, the induced probability is

    p_I(i | h) = sum over x with C_h(x) = i of P_text(x | h).

The strings in this sum must be disjoint outcomes of the same generation process. Including termination matters: an unfinished prefix is not a separate complete realization. The parser must assign each accepted string to at most one canonical unit.

Our reference predictor has a finite language. At the first branch, route probabilities are

    p_base = (0.55, 0.30, 0.15).

At subsequent branches, the previous route affects the next distribution:

    p(r | h) = 0.75 p_base(r) + 0.25 1{r = previous route}.

This makes the sequence depend on its history. It also lets us check that the detector does not silently assume independent steps. The three supported text realizations of a candidate have conditional style weights (0.50, 0.30, 0.20). Their complete-string probabilities are the candidate probability multiplied by the corresponding style weight.

We turn this finite distribution into an autoregressive predictor using a prefix trie and a simple whitespace tokenizer. The probability of a next token is the total mass below the extended prefix divided by the mass below its current prefix. The termination symbol is included. Multiplying those conditional probabilities recovers each complete-string probability; summing the probabilities of realizations that parse to the same iToken recovers its candidate probability.

This is an exact demonstration of the probability mapping for a small, constructed predictor. It has no learned parameters and uses no actual LLM token probabilities. It does not validate an estimator for the many possible realizations of an inference under a trained LLM.

For a real model, a finite candidate list generally covers only part of the probability mass. If we enumerate a set A of accepted terminated strings, normalize their probabilities, and group them by iToken, we obtain

    p_A(i | h) = P(C_h(X) = i | X is in A, h).

This is the distribution conditioned on the candidate set. It equals the full induced distribution only when A has full relevant support, or when the intended generator is explicitly restricted to A. Without a coverage argument, we must not present it as the model's unrestricted probability over ideas. Sampling-based estimation and its coverage, uncertainty, and calibration remain to be implemented and tested.

## Embedding the watermark

We separate prediction context h from watermark context c. The predictor may need the full partial path. The watermark uses a local anchor so that a harmless edit elsewhere does not change every subsequent label:

    c_t = canonical JSON(
        protocol = "itoken-proof-v1",
        problem_id = the supplied public identifier,
        start = "p_t",
        goal = "p_(t+1)"
    ).

The strings above illustrate the stage-specific atoms. The actual encoding uses the atoms for that stage. The public identifier is context, not a secret watermark payload. Both generation and verification need the same original problem manifest; this is not a detector that accepts arbitrary text without its problem.

A secret-keyed, domain-separated HMAC-SHA256 calculation assigns a preferred label g_K(c, i) in {0, 1} to each candidate. Encodings must be unambiguous. A preferred-set fraction gamma lies strictly between zero and one. Given the base candidate distribution p, we sample from

    q_K(i | h) = p(i | h) exp(delta g_K(c, i)) / Z_K(h),

    Z_K(h) = 1 + (exp(delta) - 1) mu_K(h),
    mu_K(h) = sum_i p(i | h) g_K(c, i),

where delta >= 0 controls the preference. The candidate set contains only the valid choices admitted by the generation protocol. Hashing decides which choices receive a preference; changing their sampling probabilities embeds the mark.

If all available choices receive the same label, or only one choice is possible, this step does not change the distribution and supplies no distinguishing evidence. A stronger preference does not create alternatives that the problem lacks. It can also change which intermediate strategy is selected, so preserving proof validity does not establish preservation of every notion of usefulness.

## Two detection settings

We distinguish two questions: whether we know the exact unwatermarked generator, and whether the observed artifact is independent of the secret key under the null hypothesis. The guarantees are different.

**Known baseline.** If the complete canonical path is recovered and the actual baseline distribution p is known, we compute its log likelihood ratio:

    L = sum_t log(q_K(i_t | h_t) / p(i_t | h_t)).

For the exact baseline process, exp(L) has expectation one under the null, or is a likelihood-ratio martingale as evidence accumulates. Consequently, declaring a watermark when L >= log(1 / alpha) gives a conservative false-positive bound alpha. This uses the actual conditional probabilities at every step; it does not require independent steps. It is a statement about this specified baseline, not every human writer or every other generator.

History matters here. If a deleted step makes the previous route unknown, we cannot substitute a guessed history and retain the same guarantee. Exact marginalization over missing paths would be another valid approach, but is not implemented in this prototype. A partial artifact must be identified as partial evidence; the full-path likelihood-ratio guarantee applies only when its required path and context are recovered.

Completeness is necessary but does not by itself justify the original null model. Selectively resampling some choices can change the distribution of even an unmarked Markov path. The reported likelihood ratios after partial rerouting are therefore descriptive scores; their original false-positive guarantee requires the actual null distribution to remain the specified baseline. The key-randomization test has its separate key-independence assumption.

**Unknown baseline.** We score the preferred labels of distinct recovered (context, candidate) pairs, and compare the true-key score with scores under R independently sampled surrogate keys:

    p_randomization = (1 + number of surrogate scores >= true-key score) / (R + 1).

Ties are treated conservatively. Counting distinct pairs prevents duplicated lines from manufacturing additional evidence. This randomization test does not need independent iTokens or an explicit baseline probability distribution. Its null assumption is instead that the artifact is independent of the randomly selected secret key, conditional on the public problem and protocol. Its calibration is over that key experiment; it is not a universal guarantee for an arbitrary fixed key and an adversary selecting content after inspecting watermark outputs. The finite number of surrogate keys also limits p-value resolution to 1 / (R + 1). We do not substitute an independent-binomial assumption for this test: HMAC is a cryptographic construction, not an exact mathematical random oracle.

The original problem manifest is required for either detector. We distinguish an invalid or unparseable proof, incomplete recoverable evidence, insufficient statistical evidence, and a detected statistical signal. A failure to parse is not evidence of human authorship.

## What “survives rewording” means here

Our exact invariance claim is conditional:

    If an edit preserves the recovered canonical inference choices,
    their local anchors, the public problem manifest, and the evidence
    required by a detector, that detector receives the same input and
    returns the same result.

For example, changing among supported renderings or reordering premises preserves the corresponding iToken. Shuffling proof presentation can preserve the dependency structure. Repeating a line should not increase the distinct-pair score. These are controlled transformations with checkable conditions.

Deletion, replacing one valid route with another, or an unsupported free paraphrase can change or remove the recoverable input. A different valid proof of the same final statement is not necessarily the same watermarked idea under this protocol. We make no unconditional invariance claim for those edits. Experiments must report them separately from exact-preservation cases.

## Capacity and limitations

With n three-way branches there are at most 3^n structural paths, hence at most n log2(3) bits of raw choice entropy. The nonuniform, history-dependent predictor has less entropy, and practical robust identification needs additional evidence. The second inference in each route and the choice among synonymous renderings do not provide additional structural capacity. Short ideas and nearly deterministic problems may contain too little usable variation for reliable detection.

We can also quantify the information introduced by the bias. At a particular state, let mu be the unmarked probability mass of preferred choices and let nu be its marked mass. Then

    nu = exp(delta) mu / [1 + (exp(delta) - 1) mu],
    KL(q || p) = delta nu - log[1 + (exp(delta) - 1) mu]
               <= delta^2 / 8.

The upper bound follows by applying the bounded-variable log-moment-generating-function bound under q to the negative tilt. It is zero when all available choices have the same label. The conditional KL chain rule adds these quantities along the path, even with history dependence. By data processing, a detector with false-positive rate at most alpha and detection power at least 1-beta must have total path KL at least

    d_Ber(1-beta || alpha).

For alpha=0.01 and power=0.80, this is about 3.186 nats. With delta=1, the loose per-choice upper bound therefore requires at least 26 non-forced choices. This is a necessary condition under the exact distributions, not a promise that 26 choices suffice. Information lost during text reconstruction can only make detection harder.

The tests must include unwatermarked paths, wrong keys, repeated evidence, presentation changes, deletions, and valid rerouting. Using fresh public problem identifiers across documents produces distinct contexts; it does not establish a guarantee for every fixed key, every prompt, or every human-authored proof.

We do not prove resistance to chosen-content attacks, watermark-query access, key inference, copying, or forgery. An adversary who can observe enough keyed outputs, search many candidate paths, or splice known marked material is outside the simple key-independent null. A hash function alone does not solve those problems. Detection also cannot establish who originally conceived an idea, or whether an observed explanation faithfully reflects a model's internal computation.

## The next experiment with a real LLM

We should keep the formal problem family and replace the constructed predictor with a specified trained model. We must record the model, tokenizer, prompt serialization, stopping rule, and decoding settings. Then we can compare a distribution over an explicitly restricted set of terminated realizations with frequencies from unconstrained sampled steps. We should report uncovered or invalid mass, not hide it through normalization.

Next, we should generate valid candidate steps, apply the keyed selection, and express the selected path in natural language. A separately evaluated parser should reconstruct the steps from held-out wording. We need independent checks of proof validity, canonicalization accuracy, false-positive calibration, detection at useful path lengths, and changes in solution quality and generation cost.

Finally, we should test genuinely free paraphrases, changes in variable names, deletion, compression, alternative valid proofs, and informed attempts to remove or imitate the mark. We should compare against a text-token watermark and a semantic-cluster baseline. Only that experiment can establish whether the controlled symbolic result extends to ideas expressed by an actual LLM.
