# What we still need to establish

We want to extrapolate an LLM's learned token-prediction probabilities to probabilities for predicting the next iToken. To make this precise, we must fix the context, tokenization, step boundary, and stopping procedure. Let X be the model's next token-sequence outcome under this procedure, with distribution P(X | h). Let C_h interpret its decoded text and map it to a canonical logical step, or to a special outcome ⊥ when it is invalid, truncated, ambiguous, or cannot be interpreted. Including this outcome prevents us from hiding failures through normalization.

The full next-iToken prediction probability is

    π(i | h) = P(C_h(X) = i | h)
             = Σ_{x: C_h(x)=i} P(x | h).

These probabilities sum to one only when the outcome space includes ⊥ and every other possible result of the stated generation procedure.

## A candidate list does not give the whole distribution

Suppose we score a finite list A of distinct terminated token sequences, using one chosen tokenization for each listed expression. Other tokenizations that decode to the same text remain outside this list. Its covered probability mass is

    M = Σ_{x∈A} P(x | h).

For M > 0, grouping and normalizing the listed strings gives

    π_A(i | h) = P(C_h(X)=i | X∈A, h).

This is a conditional distribution. It becomes the intended generation policy if we explicitly restrict the generator to A, but it need not resemble the unrestricted model's distribution.

Let L_i = M π_A(i | h), the known listed mass for i. The missing mass can be assigned to already known iTokens, previously unseen iTokens, or ⊥. Therefore

    π(i | h) = L_i + r_i,
    r_i ≥ 0,      Σ_i r_i = 1 − M,

and each individual group satisfies

    L_i ≤ π(i | h) ≤ L_i + (1 − M).

These are simultaneous constraints: every group cannot receive all the missing mass at once. If M is tiny, a sharply peaked π_A can still provide very weak information about π. If we additionally condition on valid inferences, the total valid mass is another quantity we must establish; silently dropping ⊥ changes the question being answered.

The bounds assume correctly scored, disjoint outcomes under one context and a correct canonical mapping for every listed string. They do not repair wrong semantic labels. If our parser assigns a passage to the wrong inference, its apparent lower bound for that inference may already be false.

## Native samples help, but parser errors remain

We can sample native continuations at temperature one, without additional top-k or top-p filtering, using the same boundary and stopping convention. Counting outcomes estimates the model's next-iToken probabilities under the chosen mapping. Sampling uncertainty must be reported, including uncertainty for rare steps and the abstention mass. This does not enumerate the infinitely many possible expressions.

In practice we use an estimated parser, Ĉ_h. Sample frequencies then estimate

    P(Ĉ_h(X)=i | h),

the distribution induced by that parser. They do not automatically estimate the distribution of correctly interpreted logical steps. If independently labeled samples from the relevant distribution establish a parser error rate at most ε, then the error in any one group probability is at most ε, in addition to sampling uncertainty. An error estimate from a small, different benchmark cannot simply be transferred to native model outputs.

We must therefore save raw generations, obtain independent semantic labels, and distinguish parser abstention, false acceptance, and incorrect route recovery. A logically valid reconstructed proof is not enough: it must be the proof actually expressed in the text.

## Rewording and capacity

Paraphrase invariance is conditional. An edit preserves the detector input when it preserves the recovered canonical path and the structural context used by the keyed hash. We must measure how often this condition holds on unseen wording. Compression can remove the intermediate choices entirely; recovering a valid route afterward may invent information rather than recover it.

There is also a capacity constraint. A forced next step carries no selection watermark. A choice among k alternatives has at most log₂(k) bits of raw freedom, and a strongly concentrated prediction distribution offers less useful variation. Increasing the watermark bias cannot create acceptable alternative reasoning paths that the problem lacks.

Our next experiment should therefore report three separate results: actual token-probability coverage, independently assessed logical recovery, and watermark detection after the complete generation-and-recovery process. Success at one stage does not establish the other two.
