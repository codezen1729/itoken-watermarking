# Recovering choices from natural-language ideas

The new recovery module interprets the mechanism advocated by ordinary prose.
Its input text does not need formal symbols, option names, catalog IDs, or a
fixed vocabulary. It can use renamed actors, pronouns, changed sentence order,
and different expressions of the same reasoning. The public problem still
supplies a finite catalog of candidate mechanisms. This is recovery within a
known semantic ontology; it is not unrestricted discovery of arbitrary ideas.

Each catalog decision describes a question and its context. Each alternative
describes a mechanism, its consequence, and the premise/action/consequence
application it represents. The recovery pipeline receives the complete public
catalog, not the intended selections. The prompt builder whitelists public
fields so accidental key, preferred-label, condition, and gold annotations do
not enter its payload. Catalog identifiers are converted into opaque IDs and
all candidates are ordered by a public hash independent of any watermark key.
These opaque IDs are answer handles. They are not inserted into the source or
rewritten ideas and do not serve as textual anchors.

The proposer reads numbered source sentences and selects only choices actually
advocated by the passage. It must distinguish consideration from adoption,
rejection from support, and a prospective plan from an unselected hypothetical.
A recommended action can count as a choice even before implementation. Shared
nouns and a common goal do not establish the same causal mechanism. The prompt
asks specifically about agency, timing, allocation, direction, qualification,
and the relation between the action and its claimed consequence. It forbids
completing an omitted choice from the catalog.

A separate verifier receives the whole source, full public catalog, and the
candidate identities under review. The proposer's citations, ambiguity claims,
confidence, and reasoning are withheld. The verifier judges every proposed
option exactly once and supplies its own evidence sentences. It cannot add a
new option. Unsupported or uncertain options are excluded. A conflict between
supported options, or an ambiguity declared by either model, excludes that
decision. Other clear decisions remain usable, so local ambiguity does not
silently discard a whole document or manufacture a complete path.

Software validates the JSON schema, option and decision identities, and source
sentence references. It preserves exact character offsets and source text for
every admitted choice. There is no positive keyword test, explicit-symbol test,
or blanket negative-word veto. These deterministic checks cannot establish
that the cited sentences entail the selected mechanism. Model errors can
remain correlated across the two calls, and every result explicitly keeps
`semantic_entailment_verified=false`.

The module exposes `make_extraction_prompt(problem, text)`,
`parse_proposal(problem, text, raw)`,
`make_verification_prompt(problem, text, proposal)`, and
`parse_extraction_output(problem, text, proposal_raw, verification_raw=None)`.
`public_vocabulary(problem)` returns the shuffled public options;
`choice_bindings(problem)` and `decision_bindings(problem)` let the evaluator
map opaque answer handles back to canonical catalog IDs. No function invokes
an LLM or reads a key, expected path, or detector score.

A proposal has exactly this shape:

```json
{"status":"extracted","choices":[{"id":"OPAQUE_OPTION_ID","sentences":[1]}],"ambiguous_decisions":[]}
```

Allowed statuses are `extracted`, `abstain`, and `ambiguous`. An abstention has
no choices or ambiguities. `ambiguous` requires at least one opaque decision
ID and may retain unrelated clear choices. An empty extraction must explicitly
abstain. Ambiguities can accompany an extraction and are handled conservatively.
Repeated proposals of one option merge their sentence citations without adding
evidence or votes. Unknown IDs and malformed responses are never repaired.

A verification has exactly this shape:

```json
{"verdicts":[{"id":"OPAQUE_OPTION_ID","verdict":"supported","sentences":[1]}],"ambiguous_decisions":[]}
```

Verdicts are `supported`, `unsupported`, or `uncertain`. Every proposed option
needs one verdict; additional, omitted, or duplicate verdicts invalidate the
response. A supported verdict needs nonempty valid citations. The other two
may have an empty citation list. Ambiguity IDs refer to public decisions and
can describe unresolved competition from an unproposed alternative.

The final JSON-ready dictionary contains `choices`, mapping canonical decision
IDs to canonical option IDs; `accepted`; `complete`; `status`; unresolved and
ambiguous decisions; exact evidence spans; both raw outputs; and all proposals,
verdicts, and exclusions. `complete_choices` means every catalog decision has
one accepted option. It does not prove logical soundness of the idea. Partial
interpretations use `partial_evidence`; nonaccepting outcomes distinguish
abstention, ambiguity, malformed output, and missing verification. No choice
is accepted without the second judgment.

Sixteen tests use a small invented workshop fixture rather than the experiment
catalog or heldout gold. They check schema boundaries, opaque IDs, annotation
leakage, permutation-invariant public ordering, evidence offsets, verifier
separation, duplicate handling, local conflicts, incomplete recovery, and
failures. Semantic judgments in these tests are supplied fixtures. They do not
measure the model's understanding. The actual experiment must separately
measure recovery, rewrite fidelity, detector power, survival, and false
acceptance on natural language.
