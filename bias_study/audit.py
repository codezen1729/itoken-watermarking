"""Replay the three-arm trial from archived model vectors, log probabilities and raw calls.

Run python3 audit.py for the full replay, or use --generation-only to audit the draw.
Requires NumPy. The replay reads the archive without executing the experiment code.
"""
import argparse
import ast
from collections import Counter
from difflib import SequenceMatcher
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random

import numpy as np


ROOT = Path(__file__).resolve().parent
FIELDS = ("premise", "action", "consequence")
ARMS = {"control": 0.0, "weak": 0.2, "strong": 12.0}
STAGES = ("plan", "original", "rewrite")
KEY_COUNT = 200
WIDTH = 896
# Recorded from the frozen manifest while generation was in progress.
MANIFEST_DIGEST = "e80c716817f4a77abd881acbdc4a164e4e48e15477499657253cb2e542f019b5"


def read(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompts_and_freeze():
    manifest = read(ROOT / "freeze.json")
    check(digest(ROOT / "freeze.json") == MANIFEST_DIGEST, "Freeze manifest digest differs")
    expected = {"PROTOCOL.md", "selection.py", "experiment.py", "sources_a.json",
                "sources_b.json", "../open_study/study.py", "../real_model.py"}
    expected.update(f"calls/pool{i}/{side}.json" for i in range(8) for side in ("in", "out"))
    check(set(manifest) == expected, "Unexpected frozen file set")
    for name, recorded in manifest.items():
        check(digest(ROOT / name) == recorded, f"Frozen file changed: {name}")

    # Recover literal instructions only. Prompt checks must not import the scoring code.
    instructions = {}
    for name in ("../open_study/study.py", "experiment.py"):
        for node in ast.parse((ROOT / name).read_text()).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                continue
            key = node.targets[0].id
            if key in {"EXTRACT", "FIDELITY", "REWORD"}:
                instructions[key] = ast.literal_eval(node.value)
            if key == "JUDGE":
                check(isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add)
                      and node.value.left.id == "FIDELITY", "Unexpected judge prompt construction")
                instructions[key] = instructions["FIDELITY"] + ast.literal_eval(node.value.right)
    return instructions


def packet(label, expected, instruction):
    directory = ROOT / "calls" / label
    actual = read(directory / "in.json")
    check(actual == {"instruction": instruction, "documents": expected},
          f"Input, blinding fields or prompt differs: {label}")
    rows = read(directory / "out.json")["documents"]
    answers = {row["id"]: row for row in rows}
    check(len(answers) == len(rows) == len(expected)
          and set(answers) == {doc["id"] for doc in expected}, f"Response IDs differ: {label}")
    return answers


def retained(row, text):
    seen = set()
    steps = []
    for step in row.get("steps", []):
        if not isinstance(step, dict):
            continue
        fields = [step.get(field) for field in (*FIELDS, "quote")]
        if not all(isinstance(value, str) and value.strip() for value in fields):
            continue
        signature = tuple(" ".join(value.lower().split()) for value in fields[:3])
        if fields[3] in text and signature not in seen:
            seen.add(signature)
            steps.append(step)
    return steps


def check_vectors(matrix, steps, problem, seen_fields, label):
    check(matrix.shape == (len(steps), len(FIELDS) * WIDTH), f"Wrong vector dimensions: {label}")
    check(np.isfinite(matrix).all(), f"Nonfinite embedding: {label}")
    for row, step in zip(matrix, steps, strict=True):
        for field, vector in zip(FIELDS, np.split(row, len(FIELDS)), strict=True):
            norm = np.linalg.norm(vector)
            check(norm == 0 or math.isclose(norm, 1, abs_tol=1e-9), f"Field norm differs: {label}")
            # A field string has the same vector wherever it recurs in this public problem.
            identity = (problem, step[field])
            if identity in seen_fields:
                check(np.allclose(vector, seen_fields[identity], rtol=1e-7, atol=1e-9),
                      f"Inconsistent embedding for repeated field: {label}")
            else:
                seen_fields[identity] = vector


def directions(problem, keys):
    """Independently decode SHAKE output, consuming the most significant bit first."""
    signs = np.array([[1 if value & (1 << shift) else -1 for shift in range(7, -1, -1)]
                      for value in range(256)], dtype=np.int8)
    axes = np.empty((len(keys), 3, 3 * WIDTH), dtype=np.int8)
    for k, key in enumerate(keys):
        for axis in range(3):
            seed = hashlib.sha256(b"idea-concentration-v1\x00" + bytes.fromhex(key)
                                  + axis.to_bytes(1, "big")).digest()
            for j, field in enumerate(FIELDS):
                frame = b"\x00".join((b"open-ideas-v1", seed, problem.encode(), field.encode()))
                stream = hashlib.shake_256(frame).digest((WIDTH + 7) // 8)
                field_signs = signs[np.frombuffer(stream, dtype=np.uint8)].ravel()[:WIDTH]
                axes[k, axis, j * WIDTH:(j + 1) * WIDTH] = field_signs
    return axes


def choose_all(matrices, target_axes, base):
    """Use pairwise squared distances instead of the generator's moment formula."""
    points = [matrix @ target_axes.T / 3 for matrix in matrices]
    eligible = [j for j, p in enumerate(points) if len(p)]
    check(len(eligible) >= 6, "Insufficient eligible candidates")
    sets = np.array(list(combinations(eligible, 6)))
    within = np.zeros(len(points))
    between = np.zeros((len(points), len(points)))
    for j in eligible:
        for k in eligible:
            if k < j:
                continue
            delta = points[j][:, None, :] - points[k][None, :, :]
            distances = np.square(delta).mean(axis=2).sum()
            if j == k:
                within[j] = distances / 2
            else:
                between[j, k] = distances
    # Population variance equals the sum of squared distances over unordered
    # observation pairs divided by n^2. Every eligible triple is one observation.
    pair_sums = within[sets].sum(axis=1)
    for i, j in combinations(range(6), 2):
        pair_sums += between[sets[:, i], sets[:, j]]
    counts = np.array([len(p) for p in points])[sets].sum(axis=1)
    raw_reward = -pair_sums / np.square(counts)
    sd = raw_reward.std()
    rewards = (raw_reward - raw_reward.mean()) / sd if sd else np.zeros(len(sets))
    utilities = base[sets].mean(axis=1)
    return {arm: sets[np.argmax(utilities + strength * rewards)].tolist()
            for arm, strength in ARMS.items()}


def rank_test(matrix, axes):
    if len(matrix) < 2:
        return dict(score=None, p=None, detected=None)
    # Project step contrasts directly. Subtracting a shared centroid commutes
    # with projection and gives the population variance with no correction.
    centered = matrix - matrix.mean(axis=0)
    dots = centered @ axes.reshape(-1, matrix.shape[1]).T
    scores = -np.square(dots.reshape(len(matrix), KEY_COUNT, 3) / 3).mean(axis=(0, 2))
    rank = 1 + np.count_nonzero(scores[1:] >= scores[0])
    p = int(rank) / KEY_COUNT
    return dict(score=float(scores[0]), p=p, detected=p <= .01)


def check_score(replayed, saved, label):
    check(set(replayed) == set(saved), f"Statistic fields differ: {label}")
    if replayed["score"] is None:
        check(saved == replayed, f"Wrong abstention: {label}")
        return
    check(math.isclose(replayed["score"], saved["score"], rel_tol=1e-9, abs_tol=1e-10)
          and replayed["p"] == saved["p"] and replayed["detected"] is saved["detected"],
          f"Score, rank or detection differs: {label}")


def weights(unit, passages):
    weights = []
    for passage, row in zip(passages, unit["weights"], strict=True):
        prompt = (f"Problem: {unit['problem']}\nProposed step: {passage}\n"
                  "Would this step help achieve the objective? Answer Yes or No.")
        check(prompt == row["prompt"] and row["token_ids"] == [9454, 2753],
              "Usefulness prompt or pinned Qwen Yes/No token IDs differ")
        yes, no = row["logprobs"]
        check(all(math.isfinite(x) and x <= 0 for x in (yes, no)), "Invalid log probability")
        high = max(yes, no)
        weight = (yes - high) - math.log(math.exp(yes - high) + math.exp(no - high))
        check(math.isclose(weight, row["log_weight"], abs_tol=1e-12), "Conditional weight differs")
        weights.append(weight)
    generator = random.Random(77801 + unit["unit"])
    noise = [-math.log(-math.log(generator.random())) for _ in passages]
    check(noise == unit["noise"], "Shared Gumbel values differ")
    return np.array(weights) + noise


def generation(prompts, seen_fields):
    groups = [read(ROOT / name) for name in ("sources_a.json", "sources_b.json")]
    check([len(g) for g in groups] == [4, 4], "Source cohort differs")
    sources = groups[0] + groups[1]
    check(len({s["problem"] for s in sources}) == 8, "Repeated public problem")
    keys = read(ROOT / "keys.json")
    check([len(k) for k in keys] == [KEY_COUNT] * 8, "Key cohort differs")
    decoded = [bytes.fromhex(k) for row in keys for k in row]
    check(len(set(decoded)) == len(decoded) and all(len(k) == 32 for k in decoded),
          "Wrong key length or repeated key")
    data = read(ROOT / "generation.json")
    model = data["model"]
    check(model["source_repo"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
          and model["revision"] == "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
          and model["execution_mode"] == "dense-fp32" and model["model_invoked"] is True,
          "Model metadata differs")
    units = data["units"]
    cohort = []
    with np.load(ROOT / "pool_vectors.npz", allow_pickle=False) as archive:
        check(set(archive.files) == {f"u{i}c{j}" for i in range(8) for j in range(24)},
              "Wrong pool archive members")
        for i, (source, unit, keyset) in enumerate(zip(sources, units, keys, strict=True)):
            problem, passages = source["problem"], source["passages"]
            check(unit["unit"] == i and unit["problem"] == problem, f"Wrong source for unit {i}")
            check(len(passages) == len(set(passages)) == 24, f"Wrong candidate pool for unit {i}")
            docs = [dict(id=f"s{j:02}", problem=problem, text=t) for j, t in enumerate(passages)]
            raw = packet(f"pool{i}", docs, prompts["EXTRACT"])
            matrices, pool_readings = [], []
            for j, d in enumerate(docs):
                name = f"u{i}c{j}"
                matrix = archive[name]
                steps = retained(raw[d["id"]], d["text"])
                check_vectors(matrix, steps, problem, seen_fields, name)
                matrices.append(matrix)
                pool_readings.append(steps)
            axes = directions(problem, keyset)
            choices = choose_all(matrices, axes[0], weights(unit, passages))
            check(set(unit["variants"]) == set(ARMS), "Missing arm")
            plans = {}
            for arm, choice in choices.items():
                v = unit["variants"][arm]
                check(v["selected"] == choice, f"Argmax differs: unit {i}, {arm}")
                selected_text = [passages[j] for j in choice]
                document = " ".join(selected_text[:4]) + "\n\n" + " ".join(selected_text[4:])
                check(v["text"] == document, f"Selected passages differ: unit {i}, {arm}")
                plan = np.concatenate([matrices[j] for j in choice])
                plans[arm] = (rank_test(plan, axes), len(plan))
            cohort.append(dict(unit=unit, axes=axes, plans=plans, pool_steps=sum(map(len, matrices)),
                               passages=passages, pool_readings=pool_readings))
    ids = [v["id"] for c in cohort for v in c["unit"]["variants"].values()]
    check(len(set(ids)) == 24, "Duplicate document IDs")
    return cohort


def quote_alignment(unit, arm, steps):
    """Post hoc comparisons of original and planning fields, aligned by verbatim source quotes."""
    chosen = unit["unit"]["variants"][arm]["selected"]
    hits = Counter()
    comparable = exact_triples = exact_fields = 0
    for step in steps:
        matches = [j for j in chosen if step["quote"] in unit["passages"][j]]
        if len(matches) != 1:
            continue
        candidate = matches[0]
        hits[candidate] += 1
        prior = unit["pool_readings"][candidate]
        if len(prior) == 1:
            comparable += 1
            same = sum(prior[0][field] == step[field] for field in FIELDS)
            exact_fields += same
            exact_triples += int(same == len(FIELDS))
    return dict(documents=1, selected_snippets=len(chosen), original_triples=len(steps),
                triples_with_unique_snippet_match=sum(hits.values()), snippets_covered=len(hits),
                documents_with_one_to_one_coverage=int(len(steps) == len(chosen)
                                                      and all(hits[j] == 1 for j in chosen)),
                aligned_to_single_plan_triple=comparable, exact_triples=exact_triples,
                compared_fields=comparable * len(FIELDS), exact_fields=exact_fields)


def document_results(cohort, prompts, seen_fields):
    records = read(ROOT / "results.json")
    saved = {(r["unit"], r["arm"]): r for r in records}
    check(len(saved) == len(records) == 24, "Missing or duplicate results")
    rows = []
    bound_arrays = set()
    arms = list(ARMS)
    with np.load(ROOT / "document_vectors.npz", allow_pickle=False) as archive:
        for b in range(3):
            arm_list = [arms[(b + i) % 3] for i in range(8)]
            variants = [c["unit"]["variants"][arm] for c, arm in zip(cohort, arm_list, strict=True)]
            originals = [dict(id=v["id"], problem=c["unit"]["problem"], text=v["text"])
                         for c, v in zip(cohort, variants, strict=True)]
            words = packet(f"reword{b}", originals, prompts["REWORD"])
            rewrites = [{**d, "text": words[d["id"]]["text"]} for d in originals]
            pairs = [dict(id=o["id"], original=o["text"], rewrite=r["text"])
                     for o, r in zip(originals, rewrites, strict=True)]
            readers = [packet(f"original{b}", originals, prompts["EXTRACT"]),
                       packet(f"rewritten{b}", rewrites, prompts["EXTRACT"])]
            judges = [packet(f"judge{j}batch{b}", pairs, prompts["JUDGE"]) for j in range(2)]
            for i, (arm, pair) in enumerate(zip(arm_list, pairs, strict=True)):
                uid, old, new = pair["id"], pair["original"], pair["rewrite"]
                check(isinstance(new, str) and new.strip() and old.strip() != new.strip(),
                      f"Missing new wording for {uid}")
                votes = [j[uid]["status"] for j in judges]
                check(set(votes) <= {"preserved", "changed", "uncertain"}, f"Unknown vote for {uid}")
                edit = 1 - SequenceMatcher(None, old.lower().split(), new.lower().split(),
                                           autojunk=False).ratio()
                observed = saved[(i, arm)]
                check(observed["id"] == uid and observed["fidelity"] == votes
                      and observed["faithful"] is all(v == "preserved" for v in votes)
                      and math.isclose(edit, observed["word_edit_fraction"], abs_tol=1e-12),
                      f"ID, votes or edit fraction differs: {uid}")
                plan, plan_count = cohort[i]["plans"][arm]
                check_score(plan, observed["plan"], f"plan {uid}")
                row = dict(unit=i, arm=arm, faithful=all(v == "preserved" for v in votes),
                           votes=votes, edit=edit, plan=plan, plan_count=plan_count)
                for stage, text, prefix, outputs in zip(("original", "rewrite"), (old, new),
                                                        ("o", "r"), readers, strict=True):
                    name = prefix + uid
                    bound_arrays.add(name)
                    steps = retained(outputs[uid], text)
                    check(len(steps) == observed[stage + "_retained"], f"Retained count differs: {name}")
                    if stage == "original":
                        row["alignment"] = quote_alignment(cohort[i], arm, steps)
                    matrix = archive[name]
                    check_vectors(matrix, steps, originals[i]["problem"], seen_fields, name)
                    row[stage] = rank_test(matrix, cohort[i]["axes"])
                    row[stage + "_count"] = len(steps)
                    check_score(row[stage], observed[stage], name)
                rows.append(row)
        check(set(archive.files) == bound_arrays, "Unbound or missing document vectors")
    return rows


def summarize(cohort, rows):
    summary = {"pool_retained_triples": sum(c["pool_steps"] for c in cohort), "arms": {}}
    for arm in ARMS:
        group = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["unit"])
        faithful = [r for r in group if r["faithful"]]
        initial = [r for r in faithful if r["original"]["detected"]]
        counts = {stage: sum(r[stage]["detected"] is True for r in group) for stage in STAGES}
        def fraction(eligible):
            return [sum(r["rewrite"]["detected"] is True for r in eligible), len(eligible)]
        overlaps = []
        model_weights = []
        for c in cohort:
            u = c["unit"]
            chosen = u["variants"][arm]["selected"]
            overlaps.append(len(set(chosen) & set(u["variants"]["control"]["selected"])))
            model_weights.extend(math.exp(u["weights"][j]["log_weight"]) for j in chosen)
        alignment = {key: sum(r["alignment"][key] for r in group) for key in group[0]["alignment"]}
        summary["arms"][arm] = dict(
            n=len(group), detections=counts, faithful=len(faithful),
            rewrite_positive_given_faithful=fraction(faithful),
            rewrite_positive_given_faithful_positive_original=fraction(initial),
            joint_original_faithful_rewrite=[fraction(initial)[0], len(group)],
            faithful_and_rewrite_positive=[fraction(faithful)[0], len(group)],
            votes=[dict(Counter(r["votes"][j] for r in group)) for j in range(2)],
            disagreements=sum(r["votes"][0] != r["votes"][1] for r in group),
            retained_steps={s: [r[s + "_count"] for r in group] for s in STAGES},
            abstentions={s: sum(r[s]["p"] is None for r in group) for s in STAGES},
            mean_word_edit_fraction=float(np.mean([r["edit"] for r in group])),
            selected_overlap_with_control=overlaps, mean_conditional_yes=float(np.mean(model_weights)),
            post_hoc_original_vs_plan_quote_alignment=alignment)
    paired = {}
    for arm in ("weak", "control"):
        other = {r["unit"]: r["original"]["detected"] is True for r in rows if r["arm"] == arm}
        strong = {r["unit"]: r["original"]["detected"] is True for r in rows if r["arm"] == "strong"}
        paired[f"strong_vs_{arm}"] = dict(
            gained=sum(strong[i] and not other[i] for i in other),
            lost=sum(other[i] and not strong[i] for i in other))
    summary["paired_original_detections"] = paired
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="run the default read-only verification")
    args = parser.parse_args()
    prompts = prompts_and_freeze()
    seen_fields = {}
    cohort = generation(prompts, seen_fields)
    if args.generation_only:
        print("Verified freeze, raw pool bindings, 8 key sets, 192 inclusion weights and 24 selections.")
    else:
        rows = document_results(cohort, prompts, seen_fields)
        print(json.dumps(summarize(cohort, rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
