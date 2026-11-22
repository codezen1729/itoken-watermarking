"""Render a concise measured report from the completed natural-language study."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'research_results/natural_language_v1'

def read(p):return json.loads(p.read_text())
def rate(r):return f"{r['successes']}/{r['trials']}" if r['trials'] else 'undefined (zero eligible originals)'
def pvalue(d):return 'abstain' if d['p_value'] is None else f"{d['p_value']:.3f}"

def main():
    summary=read(RUN/'summary.json');challenge=summary['challenge'];paired=summary['paired'];prob=summary['probabilities']
    marked=paired['marked'];unmarked=paired['unmarked'];rows=read(RUN/'paired_rows.json')['rows'];catalog=read(ROOT/'natural_ideas_catalog.json')
    text=f'''# Natural-language ideas without explicit symbol cues

We removed the artificial atom names and route labels from the ideas. We also removed the requirement that rewording retain any particular words. This run asks whether a watermark on selected planning mechanisms can still be recovered from ordinary prose.

Among marked texts detected before rewording, **{rate(marked['survival_from_original'])} remained detected afterward**. Across all eight marked rewrites, the detection rate was **{rate(marked['reworded_detection'])}**; the corresponding unmarked control count was **{rate(unmarked['reworded_detection'])}**. The full denominators include weak initial marks, incomplete recovery, and any unfaithful text. No key, generated proposal, or rewrite was rerolled after seeing an outcome.

However, requiring independent model assessments to pass for both planned-to-original and original-to-rewrite fidelity leaves **{rate(marked['faithful_reworded_detection_joint'])} marked attempts that are both faithfully represented and detected**. {marked['detection_given_both_fidelities']['trials']} marked pairs pass both fidelity checks, and {marked['detection_given_both_fidelities']['successes']} of those are detected. Two of the five raw positive detections come from originals whose fidelity was changed or uncertain. Unchanged detector scores therefore do not establish unchanged logical content.

The recovery pipeline uses GPT-6.2-astra agents with high reasoning effort in separate fresh contexts. A proposer identifies the mechanisms actually advocated in the text; a verifier independently checks their roles, timing, action and consequence, citing source sentences. Neither sees the intended path, watermark key, condition label, or detector score. Public options appear under shuffled opaque answer IDs in the recovery prompt; those IDs never need to occur in the source prose. No symbol or literal-keyword match is required for acceptance.

The agent interface records requested model routing, not an independently attested checkpoint. Temperature, seed, token probabilities and API finish reasons are unavailable. Agents may read their assigned packets and serialize explicit judgments; their access restrictions are procedural. Raw first responses and batch membership are retained. Different contexts can still share errors because all semantic roles use the same model family.

The separate frozen recovery challenge contains ordinary-language descriptions, partial plans, rejected and quoted suggestions, source instructions, and conflicting decisions. Four contrast pairs use exactly the same alphabetic word multiset while reversing roles, operation order, or adoption versus rejection. The model pipeline recovered **{challenge['exact_choice_sets']}/{challenge['cases']} exact choice sets**, with **{challenge['strict_successes']}/{challenge['cases']} correct choice/status/ambiguity results**, and **{challenge['correct_choices']}/{challenge['gold_choices']} gold choices**. On the eight same-word contrast cases, its strict result was **{rate(challenge['same_word_contrast_cases'])}**.

We froze a TF-IDF word-overlap baseline before inference. It ignores word order and assertion scope, chooses the uniquely closest option for each decision, and abstains on ties. It recovered **{challenge['lexical_exact_choice_sets']}/{challenge['cases']} exact challenge choice sets** and **{rate(challenge['lexical_same_word_exact_sets'])}** on the same-word contrasts. This comparison tests one specified lexical shortcut; it does not prove that no other shortcut exists.

The main cohort uses eight different everyday planning problems, each with sixteen decisions and three alternatives per decision. The alternatives share ordinary concepts and differ in who acts, what happens first, how work is allocated, or how the proposed mechanism produces its benefit. A separate author agent turns each selected plan into continuous prose. Another agent rewords only that prose, without seeing the catalog selections. Marked/unmarked pairs and original/reworded texts are kept in different inference contexts.

These are **curated semantic choices**, not machine-checked logical proofs. The complete public catalog is supplied during recovery. The model is matching expressed mechanisms to a known set, rather than discovering an unrestricted ontology from arbitrary ideas. Causal benefits are author judgments, not experimentally established facts. This is a substantial change from renaming formal atoms, but it does not settle general idea watermarking.

Fresh Qwen2.5-0.5B-Instruct token probabilities supply the planner. We scored **{prob['contexts']} local contexts and {prob['terminated_sequences']} EOS-terminated natural-language expressions**, one expression per option. No symbolic probability bank was reused. The conditional probabilities are tilted by the keyed hash at strengths zero and six. The local model sees the public problem and one decision, not the preceding full plan. There are eight separately keyed problem units, sixteen choices each, a detection threshold of 0.01, and 199 independent surrogate keys; p-value resolution is 0.005. Route uniforms are paired within a problem and independent of key draws.

The listed candidate strings cover **{prob['coverage_min']:.6g} to {prob['coverage_max']:.6g}** of total model probability mass, averaging **{prob['coverage_mean']:.6g}**. Mean conditional choice entropy is **{prob['mean_entropy_nats']:.6g} nats**. These are probabilities for predicting the next catalog application conditional on the three listed expressions. One wording per option does not sum over all ways to express an idea; wording, length and menu order can affect the probabilities. This is a hybrid experiment: the stronger agents do not supply these logits. The checkpoint/runtime details and all token scores are retained in the probability bank.

| Primary result | Marked | Unmarked |
|---|---:|---:|
| Planned paths detected | {rate(marked['planned_detection'])} | {rate(unmarked['planned_detection'])} |
| Original prose detected | {rate(marked['original_detection'])} | {rate(unmarked['original_detection'])} |
| Reworded prose detected | {rate(marked['reworded_detection'])} | {rate(unmarked['reworded_detection'])} |
| Originals with exact plan-label agreement | {marked['original_exact_paths']}/8 | {unmarked['original_exact_paths']}/8 |
| Rewrites with exact plan-label agreement | {marked['reworded_exact_paths']}/8 | {unmarked['reworded_exact_paths']}/8 |
| Original choices matching the intended plan | {marked['original_correct_decisions']}/128 | {unmarked['original_correct_decisions']}/128 |
| Reworded choices matching the intended plan | {marked['reworded_correct_decisions']}/128 | {unmarked['reworded_correct_decisions']}/128 |
| Reworded detector abstentions | {marked['reworded_abstentions']}/8 | {unmarked['reworded_abstentions']}/8 |
| Word-overlap baseline: reworded choices matching the plan | {marked['reworded_lexical_correct_decisions']}/128 | {unmarked['reworded_lexical_correct_decisions']}/128 |
| Word-overlap baseline: original detections | {rate(marked['original_lexical_detection'])} | {rate(unmarked['original_lexical_detection'])} |
| Word-overlap baseline: reworded detections | {rate(marked['reworded_lexical_detection'])} | {rate(unmarked['reworded_lexical_detection'])} |

Two further groups of fresh assessors check fidelity independently of recovery. One compares each original with its intended applications. The other compares the rewrite with the actual original, seeing decision topics but no intended selections. Marked original fidelity was `{marked['author_fidelity']}`; marked rewrite fidelity was `{marked['rewrite_fidelity']}`. The unmarked counts were `{unmarked['author_fidelity']}` and `{unmarked['rewrite_fidelity']}`. Across all sixteen pairs, author fidelity passed for {sum(paired[c]['author_fidelity'].get('preserved',0) for c in ('marked','unmarked'))} originals and rewrite fidelity passed for {sum(paired[c]['rewrite_fidelity'].get('preserved',0) for c in ('marked','unmarked'))} rewrites. These are model judgments, not independent human annotation. The [fidelity review](NATURAL_FIDELITY_REVIEW.md) retains every flagged detail. Matching the recovered path to the plan is reported separately and is not used as a substitute for fidelity assessment. In particular, all 512 recovered catalog decisions match the plan even though assessors flagged changed quantifiers, roles, eligibility, and scope. This is label agreement, not 512 verified semantic inferences.

Requiring both fidelity checks to pass, the joint marked-rewrite detection count was **{rate(marked['faithful_reworded_detection_joint'])}** across all eight units; detection within that assessed-faithful subset was **{rate(marked['detection_given_both_fidelities'])}**. A faithfully rewritten wrong original therefore does not automatically count as faithful embedding of the intended idea. The recorded marker scan found **{paired['explicit_marker_documents']}** documents containing artificial atom, route-ID, or formal-operator cues.

| Problem | Marked p: planned / original / rewrite | Unmarked p: planned / original / rewrite | Rewritten choices matching plan: marked / unmarked |
|---|---|---|---|
'''
    for unit,problem in enumerate(catalog['problems']):
        values={}
        for cond in ('marked','unmarked'):
            a=next(r for r in rows if r['unit_id']==unit and r['condition']==cond and r['variant']=='original')
            b=next(r for r in rows if r['unit_id']==unit and r['condition']==cond and r['variant']=='reworded')
            values[cond]=(f"{pvalue(a['planned_detection'])} / {pvalue(a['detection'])} / {pvalue(b['detection'])}",b['agreement']['correct'])
        text+=f"| {problem['title']} | {values['marked'][0]} | {values['unmarked'][0]} | {values['marked'][1]}/16 / {values['unmarked'][1]}/16 |\n"
    text+='''
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
'''
    text+='\nThe full generated texts are also available as plain text in [the examples folder](research_results/natural_language_v1/examples/), including the first marked [original](research_results/natural_language_v1/examples/unit_00_marked_original.txt) and its [rewrite](research_results/natural_language_v1/examples/unit_00_marked_reworded.txt). These are direct copies of recorded model texts, with no labels added to their content. Solution usefulness and real-world performance were not independently evaluated.\n'
    examples=RUN/'examples';examples.mkdir(exist_ok=True)
    for number in range(1,9):
        variant='original' if number<=4 else 'reworded'
        for record in read(RUN/'agent_artifacts'/f'recovery_{number:02d}'/'results.json')['rows']:
            row=next(r for r in rows if r['document_id']==record['id'] and r['variant']==variant)
            name=f"unit_{row['unit_id']:02d}_{row['condition']}_{variant}.txt"
            (examples/name).write_text(record['text'])
    problems={p['id']:p for p in catalog['problems']}
    review='# Fidelity issues retained from the first assessments\n\nThese are model-assessed differences and uncertainties, not human-adjudicated semantic ground truth. They remain in the primary denominators; no original or rewrite was repaired or replaced. Recovery label agreement is a separate measurement.\n\n'
    for row in rows:
        if row['variant']!='reworded':continue
        problem=problems[row['problem_id']];questions={d['id']:d['question'] for d in problem['decisions']}
        for field,title in [('author_fidelity','Planned application to original'),('rewrite_fidelity','Original to rewrite')]:
            assessment=row[field]
            issues=[d for d in assessment['decisions'] if d['verdict']!='preserved']
            if not issues and not assessment['added_conflicting_decisions']:continue
            review+=f"## {problem['title']} — {row['condition']}, {title.lower()}\n\nOverall assessment: **{assessment['status']}**.\n\n"
            for issue in issues:
                review+=f"- **{questions[issue['decision_id']]}** — {issue['verdict']}. {issue.get('reason','')}\n"
            review+='\n'
    (ROOT/'NATURAL_FIDELITY_REVIEW.md').write_text(review.rstrip()+'\n')
    (ROOT/'NATURAL_REPORT.md').write_text(text)

if __name__=='__main__':main()
