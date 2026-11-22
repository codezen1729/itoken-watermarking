"""Recompute the open-step trial from its retained calls, logits and vectors."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parent
FIELDS = ("premise", "action", "consequence")


def read(path):
    return json.loads(path.read_text())


def require(test, message):
    if not test:
        raise ValueError(message)


def equal(left, right, message):
    require(left == right, message)


def close(left, right, message):
    require(np.allclose(left, right, rtol=1e-9, atol=1e-10), message)


def rows_by_id(rows):
    result = {row["id"]: row for row in rows}
    equal(len(result), len(rows), "Repeated ID within model packet")
    return result


def frozen_instructions():
    freeze = read(ROOT / "freeze.json")
    expected = {"study.py", "PROTOCOL.md", "sources.json", "sources_b.json", "FIDELITY_CHECK.md", "../real_model.py"}
    equal(set(freeze), expected, "Frozen source set mismatch")
    for filename, sha in freeze.items():
        equal(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(), sha,
              f"Frozen file changed: {filename}")
    syntax = ast.parse((ROOT / "study.py").read_text())
    return {statement.targets[0].id: ast.literal_eval(statement.value)
            for statement in syntax.body if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id in {"EXTRACT", "REWRITE", "REVIEW", "FIDELITY"}}


def read_call(name, expected_documents):
    folder = ROOT / "calls" / name
    packet = read(folder / "in.json")
    equal(set(packet), {"instruction", "documents"}, f"Unexpected fields in {name}")
    instructions = frozen_instructions()
    family = ("REVIEW" if name.startswith("review_") else
              "REWRITE" if name.startswith("rewrite") else
              "FIDELITY" if name.startswith(("fidelity", "second")) else "EXTRACT")
    equal(packet["instruction"], instructions[family], f"Prompt changed in {name}")
    equal(packet["documents"], expected_documents, f"Input document or isolation violation in {name}")
    result = rows_by_id(read(folder / "out.json")["documents"])
    equal(set(result), set(rows_by_id(expected_documents)), f"Missing or unassigned response in {name}")
    return result


def retained_steps(output, text):
    """A literal quotation and de-duplication establish eligibility for scoring."""
    eligible = []
    seen = set()
    for step in output["steps"]:
        if not isinstance(step, dict):
            continue
        values = [step.get(field) for field in (*FIELDS, "quote")]
        if not all(isinstance(value, str) and value.strip() for value in values):
            continue
        if step["quote"] not in text:
            continue
        identity = tuple(" ".join(s.lower().split()) for s in values[:3])
        if identity not in seen:
            eligible.append(step)
            seen.add(identity)
    return eligible


def projection_matrix(keys, problem, size):
    """Independent bit decoding, most-significant bit first within each byte."""
    require(size > 0 and size % len(FIELDS) == 0, "Invalid field-vector dimensions")
    width = size // len(FIELDS)
    matrix = np.empty((len(keys), size), dtype=float)
    for key_index, key in enumerate(keys):
        for field_index, field in enumerate(FIELDS):
            frame = b"\0".join([b"open-ideas-v1", bytes.fromhex(key), problem.encode(), field.encode()])
            raw = np.frombuffer(hashlib.shake_256(frame).digest((width + 7) // 8), dtype=np.uint8)
            bits = ((raw[:, None] >> np.arange(7, -1, -1)) & 1).ravel()[:width]
            matrix[key_index, field_index * width:(field_index + 1) * width] = 2.0 * bits - 1.0
    return matrix


def replay_test(vector, matrix, reported, name):
    if vector is None:
        equal([reported.get(k) for k in ("p", "score", "detected")], [None] * 3, name)
        return
    require(np.isfinite(vector).all(), f"Nonfinite detector input in {name}")
    # Independent vectorized contraction. For floating-point ties use the same
    # one-dimensional dot product as scoring only at the exact tie boundary.
    values = np.einsum("ij,j->i", matrix, vector) / len(FIELDS)
    near = np.isclose(values, values[0], rtol=1e-12, atol=1e-12)
    values[near] = [np.dot(row, vector) / len(FIELDS) for row in matrix[near]]
    p = (1 + int(np.count_nonzero(values[1:] >= values[0]))) / len(values)
    close(values[0], reported["score"], f"Wrong target score for {name}")
    equal(p, reported["p"], f"Wrong surrogate rank for {name}")
    equal(p <= 0.01, reported["detected"], f"Wrong detection for {name}")


def generation_replay():
    problems = read(ROOT / "sources.json")["problems"] + read(ROOT / "sources_b.json")["problems"]
    keysets = read(ROOT / "keys.json")
    dataset = read(ROOT / "generation.json")
    units = dataset["units"]
    equal((len(problems), len(keysets), len(units)), (8, 8, 8), "Wrong cohort size")
    equal(len({p["id"] for p in problems}), 8, "Repeated public problem")
    equal(len({k for keys in keysets for k in keys}), 1600, "Repeated or missing demonstration keys")
    equal(dataset["model"]["revision"], "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3", "Model revision changed")
    matrices = {}
    selected_ids = set()
    step_counts = []
    with np.load(ROOT / "pool_vectors.npz", allow_pickle=False) as vectors:
        equal(set(vectors.files), {f"c{i}" for i in range(8)}, "Pool vector binding mismatch")
        for i, (problem, unit, keys) in enumerate(zip(problems, units, keysets, strict=True)):
            equal(unit["unit"], i, "Unit ordering changed")
            equal(unit["problem"], problem["prompt"], "Problem binding changed")
            equal(len(keys), 200, "Target needs 199 surrogates")
            require(all(len(bytes.fromhex(k)) == 32 for k in keys), "Bad key length")
            texts = [r["text"] for r in problem["ideas"]]
            equal(len(texts), 24, "Expected 24 open candidate arguments")
            docs = [dict(id=f"s{j}", problem=problem["prompt"], text=s) for j, s in enumerate(texts)]
            pool = read_call(f"pool{i + 1}", docs)
            counts = np.array([len(retained_steps(pool[d["id"]], d["text"])) for d in docs])
            equal(counts.tolist(), unit["pool_n_steps"], "Retained pool count mismatch")
            originals = vectors[f"c{i}"]
            equal(originals.shape[0], 24, "Missing candidate vector")
            require(np.isfinite(originals).all(), "Nonfinite pool features")
            matrix = projection_matrix(keys, unit["problem"], originals.shape[1])
            matrices[i] = matrix
            key_scores = np.einsum("ij,j->i", originals, matrix[0]) / len(FIELDS)
            close(key_scores, unit["key_scores"], "Candidate target scores changed")
            rng = random.Random(5290 + i)
            noise = [-math.log(-math.log(rng.random())) for _ in texts]
            close(noise, unit["noise"], "Paired sampling deviates from published seed")
            base = []
            for s, weight in zip(texts, unit["weights"], strict=True):
                expected = f"Problem: {problem['prompt']}\nProposed step: {s}\nWould this step help achieve the objective? Answer Yes or No."
                equal(weight["prompt"], expected, "Inclusion prompt mismatch")
                yes, no = weight["logprobs"]
                require(all(math.isfinite(v) and v <= 0 for v in (yes, no)), "Invalid log probabilities")
                require(len(set(weight["token_ids"])) == 2, "Yes and No must be distinct tokens")
                high = max(yes, no)
                lp = (yes - high) - math.log(math.exp(yes - high) + math.exp(no - high))
                close(lp, weight["log_weight"], "Conditional Yes weight differs")
                base.append(lp)
            for condition, strength in (("unmarked", 0), ("marked", 4)):
                variant = unit["variants"][condition]
                require(variant["id"] not in selected_ids, "Duplicate main-study document")
                selected_ids.add(variant["id"])
                utility = np.array(base) + noise + strength * key_scores
                chosen = sorted(np.argsort(-utility, kind="stable")[:12].tolist())
                equal(chosen, variant["selected"], "Keyed draw differs")
                sequence = [texts[j] for j in chosen]
                expected_text = "\n\n".join(" ".join(sequence[j:j + 4]) for j in (0, 4, 8))
                equal(expected_text, variant["text"], "Original does not match the draw")
                number = int(sum(counts[chosen]))
                planned = None if not number else np.average(originals[chosen], weights=counts[chosen], axis=0)
                replay_test(planned, matrix, variant["planned"], f"planned {condition} on {i}")
            step_counts.extend(counts.tolist())
    return problems, units, matrices, step_counts


def full_replay():
    problems, units, matrices, pool_counts = generation_replay()
    records = read(ROOT / "results.json")
    observed = {(r["unit"], r["condition"]): r for r in records}
    equal(len(observed), 16, "Missing documents in primary cohort")
    equal(len(records), 16, "Duplicate documents in primary cohort")
    file_keys = set()
    with np.load(ROOT / "document_vectors.npz", allow_pickle=False) as document_vectors:
        for group, cond in (("1", "unmarked"), ("2", "marked")):
            originals = [dict(id=u["variants"][cond]["id"], problem=u["problem"], text=u["variants"][cond]["text"]) for u in units]
            rewrites = read_call("rewrite" + group, originals)
            revised = [{**d, "text": rewrites[d["id"]]["text"]} for d in originals]
            comparisons = [dict(id=d["id"], original=d["text"], rewrite=r["text"]) for d, r in zip(originals, revised, strict=True)]
            fidelity = read_call("fidelity" + group, comparisons)
            second = read_call("second" + group, comparisons)
            for var, docs in (("original", originals), ("rewritten", revised)):
                batch = var + group
                extracted = read_call(batch, docs)
                requests = [{**d, "steps": retained_steps(extracted[d["id"]], d["text"])} for d in docs]
                reviews = read_call("review_" + batch, requests)
                for i, (doc, request) in enumerate(zip(docs, requests, strict=True)):
                    ident = doc["id"]
                    row = observed[(i, cond)]
                    equal(ident, row["id"], "Document ID mismatch")
                    equal(row["planned"], units[i]["variants"][cond]["planned"], "Baseline mismatch")
                    equal(row["fidelity"], fidelity[ident], "Fidelity response mismatch")
                    row["second_fidelity"] = second[ident]
                    require(second[ident]["status"] in {"preserved", "changed", "uncertain"}, "Unknown second fidelity status")
                    require(fidelity[ident]["status"] in {"preserved", "changed", "uncertain"}, "Unknown fidelity status")
                    result = row[var]
                    equal(result["proposed"], len(extracted[ident]["steps"]), "Raw proposal count mismatch")
                    equal(result["retained"], len(request["steps"]), "Lost or extra admitted step")
                    equal(result["review"], reviews[ident], "Independent extraction assessment mismatch")
                    require(reviews[ident]["status"] in {"complete", "partial", "uncertain"}, "Unknown review status")
                    key = batch + ident
                    vector = None
                    if request["steps"]:
                        file_keys.add(key)
                        vector = document_vectors[key]
                    replay_test(vector, matrices[i], result, key)
        equal(set(document_vectors.files), file_keys, "Unbound document vectors")
    return problems, records, pool_counts


def pos(row, var):
    return row[var]["detected"] is True


def fraction(successes, n):
    return f"{successes}/{n}" if n else "undefined (0 eligible)"


def preserve(row):
    return row["fidelity"]["status"] == "preserved"


def summary(rows):
    planned = [r for r in rows if pos(r,"planned")]
    originals = [r for r in rows if pos(r,"original")]
    faithful = [r for r in rows if preserve(r)]
    eligible = [r for r in originals if preserve(r)]
    both_faithful = [r for r in faithful if r["second_fidelity"]["status"] == "preserved"]
    confirmed = [r for r in both_faithful if pos(r, "original")]
    complete = [r for r in eligible if all(r[v]["review"]["status"] == "complete" for v in ("original","rewritten"))]
    return {
        "both_preserved": fraction(len(both_faithful), len(rows)),
        "both_positive": fraction(sum(pos(r, "rewritten") for r in both_faithful), len(rows)),
        "both_eligible": len(confirmed),
        "both_survival": fraction(sum(pos(r, "rewritten") for r in confirmed), len(confirmed)),
        "disagreements": fraction(sum(r["fidelity"]["status"] != r["second_fidelity"]["status"] for r in rows), len(rows)),
        "planned": fraction(len(planned), len(rows)),
        "original": fraction(len(originals), len(rows)),
        "rewrite": fraction(sum(pos(r,"rewritten") for r in rows), len(rows)),
        "survival": fraction(sum(pos(r,"rewritten") for r in originals), len(originals)),
        "planned_survival": fraction(sum(pos(r,"rewritten") for r in planned), len(planned)),
        "faithful": fraction(len(faithful), len(rows)),
        "conditional": fraction(sum(pos(r,"rewritten") for r in eligible), len(eligible)),
        "faithful_detected": fraction(sum(pos(r,"rewritten") for r in faithful), len(rows)),
        "both_reviews": fraction(sum(pos(r,"rewritten") for r in complete), len(complete)),
        "abstain_orig": fraction(sum(r["original"]["p"] is None for r in rows), len(rows)),
        "abstain_rewrite": fraction(sum(r["rewritten"]["p"] is None for r in rows), len(rows)),
    }


def fmt_p(r):
    return "abstain" if r["p"] is None else f"{r['p']:.3f}"


def paired_selection_effect():
    """Compare planned target gain to the control plan's surrogate spread."""
    units = read(ROOT / "generation.json")["units"]
    keysets = read(ROOT / "keys.json")
    swaps, gains, spreads = [], [], []
    with np.load(ROOT / "pool_vectors.npz", allow_pickle=False) as vectors:
        for i, unit in enumerate(units):
            variants = unit["variants"]
            selection = variants["unmarked"]["selected"]
            swaps.append(len(set(variants["marked"]["selected"]) - set(selection)))
            values = vectors[f"c{i}"][selection]
            counts = np.asarray(unit["pool_n_steps"])[selection]
            control = np.average(values, weights=counts, axis=0)
            matrix = projection_matrix(keysets[i][1:], unit["problem"], control.size)
            surrogate_scores = (matrix @ control) / len(FIELDS)
            spreads.append(float(np.std(surrogate_scores, ddof=1)))
            gains.append(variants["marked"]["planned"]["score"] - variants["unmarked"]["planned"]["score"])
    require(all(sd > 0 for sd in spreads), "Zero surrogate spread in effect-size diagnostic")
    return swaps, np.mean(gains), np.mean(spreads), np.mean(np.asarray(gains) / spreads)


def render(problems, records, pool_counts):
    groups = {c:[r for r in records if r["condition"]==c] for c in ("marked","unmarked")}
    stat = {c:summary(rows) for c,rows in groups.items()}
    marked, controls = stat["marked"], stat["unmarked"]
    if marked["both_eligible"]:
        main = f"Among initially detectable marked originals that both blinded readers judged faithfully reworded, **{marked['both_survival']}** remained detectable."
    else:
        main = "No initially detectable marked original met both readers’ preservation criterion. Survival conditional on confirmed preservation is **unmeasurable in this cohort**."
    lines = [
        "# Watermark survival after open-ended recovery of reasoning steps",
        "",
        f"The detector flagged **{marked['rewrite']} marked rewrites** and **{controls['rewrite']} control rewrites**. {main}",
        "",
        "| Measurement | Marked | Unmarked |", "|---|---:|---:|",
    ]
    for label, key in (("Planned steps detected","planned"),("Originals detected","original"),("Rewrites detected","rewrite"),
                       ("Rewrites detected among detected originals","survival"),("Rewrites detected among detected plans","planned_survival"),
                       ("Pairs assessed preserved by the first reader","faithful"),
                       ("Pairs both readers assessed preserved","both_preserved"),
                       ("Pairs with different fidelity verdicts","disagreements"),
                       ("Detected rewrites both readers assessed preserved, across all attempts","both_positive"),
                       ("Survival among detected originals both readers assessed preserved","both_survival"),("Survival using the first reader’s preservation criterion","conditional"),
                       ("Faithful pairs with detected rewrites, across all attempts","faithful_detected"),
                       ("Survival if both extractions also passed the completeness review","both_reviews"),
                       ("Detector abstentions on originals","abstain_orig"),("Detector abstentions on rewrites","abstain_rewrite")):
        lines.append(f"| {label} | {marked[key]} | {controls[key]} |")
    lines += [
        "", "Each of eight public problems had 24 freshly written candidate arguments. The generation step selected twelve for each original and joined their text into paragraphs. Qwen2.5-0.5B-Instruct supplied Yes and No token probabilities for a local usefulness question. Conditional Yes weights and a shared random draw determined control selection. The marked selection added four times the keyed projection score before taking the twelve highest values. This experiment embeds the watermark through the choice of which proposed arguments to include.",
        "", "Readers extracted any number of advocated steps from each passage. Each step has a premise, action, consequence, and a verbatim supporting quotation. The original and rewritten readers worked in separate contexts. At detection time they saw only the public problem and prose. The detector used those open triples, the public problem, a fixed encoder and a key. The temporary generation pool played no role at detection. [Protocol](PROTOCOL.md)",
        "", "The encoder read each field using the first eight transformer blocks of the local Qwen checkpoint. It pooled token representations, subtracted the public problem representation after normalization, and normalized each field difference. Each field has a separate key-derived projection. The detector averaged scores over retained triples and compared the target score with scores using 199 surrogate keys. A p-value of 0.01 or smaller counts as detection. The argument weights describe the local Yes/No inclusion policy. Native probabilities for producing whole ideas remain unknown.",
        "", "Two fidelity readers separately compared the two versions directly and assessed changes to actors, causal relationships, timing and scope. Another reader checked each extraction for missing and unsupported reasoning. These are agent judgments and can share errors. A verbatim quotation verifies textual provenance. The model judgments supply the semantic interpretation. [Second-reader protocol](FIDELITY_CHECK.md)",
        "", "| Problem | Condition | Planned p | Original p | Rewrite p | Original / rewrite steps | Fidelity, first / second reader | Original / rewrite extraction review |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(records,key=lambda r:(r["unit"],r["condition"])):
        a,b=row["original"],row["rewritten"]
        lines.append(f"| {problems[row['unit']]['title']} | {row['condition']} | {fmt_p(row['planned'])} | {fmt_p(a)} | {fmt_p(b)} | {a['retained']} / {b['retained']} | {row['fidelity']['status']} / {row['second_fidelity']['status']} | {a['review']['status']} / {b['review']['status']} |")
    swaps, gain, spread, standardized = paired_selection_effect()
    counts = Counter(r["fidelity"]["status"] for r in records)
    review_counts = Counter(r[v]["review"]["status"] for r in records for v in ("original","rewritten"))
    complete_by_variant = {v: sum(r[v]["review"]["status"] == "complete" for r in records)
                           for v in ("original","rewritten")}
    accepted = sum(r[v]["retained"] for r in records for v in ("original","rewritten"))
    proposed = sum(r[v]["proposed"] for r in records for v in ("original","rewritten"))
    lines += ["",f"Across the 16 pairs, the first fidelity reader judged {counts['preserved']} rewrites preserved, {counts['changed']} changed and {counts['uncertain']} uncertain. Across 32 passages, extraction reviews judged {review_counts['complete']} complete, {review_counts['partial']} partial and {review_counts['uncertain']} uncertain. The detector retained {accepted} of the {proposed} proposed triples. Generation used {sum(pool_counts)} extracted triples across 192 candidate snippets. [Original passages and their selections](generation.json) · [Free extractions and assessments](calls/) · [Per-document results](results.json)",
              "", f"Extraction reviewers rated {complete_by_variant['original']}/16 original readings and {complete_by_variant['rewritten']}/16 rewrite readings complete. The unmarked film-night text illustrates a partial rating. Its reader preserved the suggestion to use a striped awning as a meeting point. The source compared remembering the awning with remembering a volunteer’s name. The extracted premise compared recognizing the awning with recognizing the volunteer. [Source reading assessment](calls/review_original1/out.json)",
              "", "The jigsaw exchange was the only original to meet the threshold, in both its marked and control forms. Each had p = 0.015 at planning, p = 0.010 for the original reading, and p = 0.015 after rewording. The rank moved from one surrogate scoring at least as high as the target to two. For the marked version, the first fidelity reader judged that placing children’s puzzles on the low shelf lost the requirement to reserve that space for them. The second judged the rewrite preserved. This pair illustrates sensitivity to a small rank change and the effect of reader disagreement on the preservation analysis.",
              "", f"A post hoc comparison found that marking replaced {', '.join(map(str,swaps))} of the twelve selected snippets across the eight problems, respectively. The planned score under the target key rose by a mean of {gain:.4f} relative to its paired control. The control plans’ 199 surrogate scores had a mean sample standard deviation of {spread:.4f}. Dividing each paired gain by its own control standard deviation and averaging gave **{standardized:.3f} surrogate standard deviations**. The sampler therefore moved the score only slightly relative to the spread a detector saw across keys. This comparison uses the planned vectors before any whole-document extraction or rewriting.",
              "", "Selection and detection now use a continuous projection of open text fields. The previous experiment required exact catalog IDs for its keyed hash. The difference in outcomes therefore combines the new watermark, new corpus and new recovery method. Shared topics or wording can influence this encoder, and combining independently generated arguments limits the conclusions about long chains of dependent reasoning or entirely spontaneous essays. The eight paired problems and model-based judgments support a descriptive result for this cohort.",
              "", "A follow-up requested light copyedits of the initially detectable marked original and its matched control. Both fidelity readers accepted both edits. Each detector p-value moved from 0.010 to 0.015, crossing the fixed threshold by one rank. [Follow-up protocol](COPYEDIT_PROTOCOL.md) · [Copyedit results and reader judgments](COPYEDIT_REPORT.md)",
              "", "Recompute the target and surrogate scores, conditional inclusion weights, shared sampling noise, selected documents, citation checks, fidelity bindings and all reported fractions with `python open_study/audit.py --check` from the repository root. The audit uses archived vectors and log probabilities from the actual model run. It does not rerun the neural encoder. Source files and raw calls accompany the results.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-only",action="store_true",help="audit pool and selections before document readings finish")
    parser.add_argument("--check",action="store_true",help="check report against replay without writing it")
    args=parser.parse_args()
    if args.generation_only:
        generation_replay()
        print("Generation replay passed for eight units.")
    else:
        problems, rows, counts = full_replay()
        report = render(problems, rows, counts)
        if args.check:
            equal(report,(ROOT/"REPORT.md").read_text(),"Report differs from replay")
        else:
            (ROOT/"REPORT.md").write_text(report)
        print(json.dumps({k:summary([r for r in rows if r["condition"]==k]) for k in ("marked","unmarked")}, indent=2))
