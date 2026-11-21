# Logical recovery with sentence grounding and a second judgment

We want to recover the reasoning actually expressed by a passage. A valid step
that could solve the public problem is not enough. The previous experiment
showed both practical formatting failures and a more serious error: the model
selected a valid inference that the passage did not assert.

`recovery_v2.py` changes the recovery protocol without modifying the previous
extractors, benchmark, or saved results. It does not read a watermark key,
preferred labels, intended path, gold answers, or detector scores.

First, software divides the source into numbered sentence spans and stores
their exact character offsets. The proposer receives the source, public
problem, and all admissible inference IDs. It selects observed IDs and sentence
numbers. It does not generate formulas or copy evidence quotations. This
removes invented quotations as a formatting failure; it does not make the
selected sentences semantically sufficient.

Second, a fresh verifier call sees the entire source and the proposed inference
identities. It does not see the proposer's claimed citations, confidence, or
reasoning. The verifier must assess whether the source actually asserts each
particular premise-to-conclusion application. It chooses `supported`,
`unsupported`, or `uncertain`, cites its own sentence numbers, and states
whether a unique asserted route exists. It must account for omitted
intermediates, different premises, hypothetical or denied applications,
abandoned quotations, and instructions embedded in source text. It must not
repair an unsound argument by borrowing an available public rule.

Only explicitly supported candidates proceed to deterministic checks. Those
checks require valid source indices, the step's symbol anchors in the cited
sentences, an admitted and logically valid inference, consistent route choices,
and graph validity for a complete proof. Unsupported or uncertain candidates
are recorded and excluded. The verifier cannot add new candidates. A partial
path stays partial. Duplicate proposals are merged without adding evidence;
conflicting supported routes cause abstention. A forced completion remains a
recoverable step, but its eligibility as watermark evidence is decided later.

The old blanket lexical-risk veto is not used. Whether a denial or hypothetical
qualifier governs an inference is a discourse question for the verifier, not a
test for a word anywhere in the cited sentence. This can improve coverage, but
it also makes actual verifier accuracy essential. Sentence matches and symbol
anchors are necessary checks in this bounded experiment, not entailment tests.

Every result preserves both raw model outputs, source offsets, selected IDs,
verdicts, rejected IDs, and exact sentence citations. The convenience `evidence`
field joins selected sentences with newlines; it is not claimed to be one
contiguous source span. `source_spans` supplies the original exact substrings
and offsets. The optional `on_event` callback lets a runner persist each
completed model call before starting the next. Infrastructure errors propagate;
truncated or unterminated mapping responses fail closed. No gold repair or
silent reroll is performed.

The two calls may use the same model or separate models. A fresh call does not
establish statistically independent errors. Even agreement between strong
models is not proof of semantic entailment, so every result keeps
`semantic_entailment_verified=false`. The ontology still uses known formal
symbols and three inference rules. General entity resolution, arbitrary symbol
renaming, and unrestricted scientific ideas remain outside this protocol.

The module's 18 development tests check schema boundaries, grounding indices,
scope handling, partial recovery, conflicts, truncation, and persistence. Their
semantic verdicts are fixtures, not observations of model competence. The
fresh model experiment must measure positive recovery, omissions, semantic
false acceptance, rewording preservation, and detector survival separately.
Neither its new hold-out examples nor gold labels were consulted while writing
this module or these tests.

The main entry point is
`recover_with_model(problem, text, generator, verifier=None, on_event=None)`.
Callbacks receive a prompt string and should return a mapping containing `text`
and `finish_reason="stop"` for a normally completed response. The proposer and
verifier prompt builders and parsers are also exposed for an evaluator that
orchestrates isolated model calls itself. Parsing a proposal without a verifier
response never produces an accepted recovery.

The completed experiment uses `blinded_recovery.py` as file transport instead
of the completion callback. `prepare(cases, out, label=..., batch_group=...)`
writes a private binding and a proposal request packet. Only
`proposals/requests.json` is given to a fresh agent. Its output bundle has the
form `{"responses":[{"request_id":"...","response":"JSON string"}]}`.
`prepare_verification(out)` validates that every request has exactly one
response and exports a second packet. Only `verification/requests.json` goes
to a different fresh agent. `finalize(out)` parses and preserves both first
responses, including invalid outputs and abstentions. These functions do not
invoke an LLM themselves or turn unavailable API finish metadata into a
fabricated success value.

The actual routing was `gpt-6.2-astra`, reasoning effort `high`, and
`fork_turns="none"`. Agents could read the designated packet and serialize
their explicit judgments; extraction heuristics and additional context access
were prohibited by instruction. Model snapshot, temperature, seed, token
log-probabilities, and API finish reason were unavailable. Recorded completion
means the agent wrote its response bundle. The blinding is procedural rather
than an operating-system guarantee. The experiment's immutable request and
response files, source hashes, batch membership, and requested model settings
are retained for replay. Re-running the stochastic language stage requires
an actual agent runner or a separately documented API adapter; running the
audit only recomputes results from saved judgments.
