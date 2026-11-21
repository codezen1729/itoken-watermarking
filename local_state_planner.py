"""Actual model-scored local-state policy for the paired rewording study.

The generator deliberately supplies only the current fact and local premises,
not its preceding route. This makes a recorded per-stage probability bank the
actual generation policy across independent tasks/keys. It is not a claim about
the model's probabilities under full-history or unrestricted generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import secrets

from core import And, Or, Implies, STYLES, make_problem, candidate_pairs, render_step, render_proof, parse_step
from pipeline import context, draw, steps_for_routes
from real_model import MLXModel
from research_runner import save, digest
from real_model_pilot import detect_structural
from watermark import bias_distribution

SOURCE = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
REVISION = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
PROTOCOL = "local-state-real-probability-rewording-v2"
DEPENDENCIES = ("local_state_planner.py", "core.py", "real_model.py", "pipeline.py",
                "watermark.py", "real_model_pilot.py", "research_runner.py")


def request(stage):
    problem = make_problem(stage + 1, "local-state-bank")
    p, nxt, m, d, e = f"p{stage}", f"p{stage+1}", f"m{stage}", f"d{stage}", f"e{stage}"
    premises = {p, d, Implies(p, m), Implies(m, nxt),
                Implies(And(p, d), nxt), Implies(Or(p, e), nxt)}
    pairs = candidate_pairs(problem, stage)
    candidates = [{"text": render_step(pair[0], style), "itoken_id": pair[0].canonical(), "route": route}
                  for route, pair in pairs.items() for style in STYLES]
    state = {"current_fact": p, "local_given_premises": sorted(premises), "next_goal": nxt}
    prompt = ("Choose exactly one next inference toward next_goal using the supplied local state. "
              "Return exactly one listed statement and stop, without explanation.\n"
              + "\n".join(c["text"] for c in candidates)
              + "\nLOCAL STATE (the entire model context):\n" + json.dumps(state, sort_keys=True))
    return candidates, [{"role": "user", "content": prompt}]


def build_bank(model, out, stages=32):
    if isinstance(stages, bool) or not isinstance(stages, int) or stages < 1:
        raise ValueError("Stages must be a positive integer")
    root = Path(__file__).resolve().parent
    freeze = {"protocol": PROTOCOL, "stages": stages, "model_metadata": model.metadata,
              "source_sha256": {n: digest(root/n) for n in DEPENDENCIES},
              "context_policy": "current fact plus local premises only; no path history, task ID, or key",
              "scope": "conditional on nine listed terminated token sequences",
              "numerical_tolerance_nats_per_token": .05}
    manifest = out / "bank_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != freeze:
        raise ValueError("Bank configuration changed; choose a new directory")
    save(manifest, freeze)
    rows = []
    for stage in range(stages):
        path = out / f"stage_{stage:02d}.json"
        candidates, messages = request(stage)
        if path.exists():
            row = json.loads(path.read_text())
            if row["messages"] != messages or row["freeze"] != freeze:
                raise ValueError("Saved bank context changed")
        else:
            scores = model.score_candidates(messages, candidates, verify_first_uncached=True,
                                             verify_all_uncached=stage == 0)
            max_difference = max(x["max_absolute_token_logprob_difference"] for x in scores["uncached_crosschecks"])
            if max_difference > .05:
                save(out/f"failed_numeric_stage_{stage:02d}.json", scores)
                raise ValueError("Probability numerics failed the predeclared tolerance")
            for c in candidates:
                if parse_step(c["text"]).canonical() != c["itoken_id"]:
                    raise AssertionError("Invalid candidate label")
            row = {"stage": stage, "freeze": freeze, "messages": messages, "scores": scores,
                   "max_checked_logprob_difference": max_difference}
            save(path, row)
        rows.append(row)
        print("bank",stage,"mass",row["scores"]["covered_probability_mass"],flush=True)
    save(out/"bank_summary.json", {"freeze": freeze, "stages": [{"stage": r["stage"],
         "itoken_probabilities": r["scores"]["itoken_probabilities"],
         "covered_probability_mass": r["scores"]["covered_probability_mass"],
         "max_checked_logprob_difference": r["max_checked_logprob_difference"]} for r in rows]})


def sample_cohort(bank_path, out, *, units=8, stages=32, delta=6.0, seed=20261122):
    """Draw keys once; never select tasks by detection or recovery outcomes."""
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (units, stages)):
        raise ValueError("Units and stages must be positive integers")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
        raise ValueError("Delta must be finite and nonnegative")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Cohort destination must be empty; no outcome-dependent rerolls")
    bank = json.loads((bank_path/"bank_summary.json").read_text())
    manifest = json.loads((bank_path/"bank_manifest.json").read_text())
    if bank["freeze"] != manifest or manifest["protocol"] != PROTOCOL or manifest["stages"] != stages:
        raise ValueError("Bank manifest or protocol mismatch")
    root = Path(__file__).resolve().parent
    if manifest["source_sha256"] != {n: digest(root/n) for n in DEPENDENCIES}:
        raise ValueError("Bank source dependencies changed")
    if len(bank["stages"]) != stages:
        raise ValueError("Bank does not match declared path length")
    shape = make_problem(stages)
    for stage, entry in enumerate(bank["stages"]):
        if isinstance(entry["stage"], bool) or entry["stage"] != stage:
            raise ValueError("Bank stage order mismatch")
        expected = {pair[0].canonical() for pair in candidate_pairs(shape, stage).values()}
        if set(entry["itoken_probabilities"]) != expected:
            raise ValueError("Bank candidate support mismatch")
        bias_distribution(entry["itoken_probabilities"], b"validation-only", "validation", delta=0)
        difference = entry["max_checked_logprob_difference"]
        if (isinstance(difference, bool) or not isinstance(difference, (int, float))
                or not math.isfinite(difference) or not 0 <= difference <= .05):
            raise ValueError("Bank numerical check did not pass")
        raw = json.loads((bank_path/f"stage_{stage:02d}.json").read_text())
        if (raw["freeze"] != manifest or raw["stage"] != stage or raw["messages"] != request(stage)[1]
                or raw["scores"]["itoken_probabilities"] != entry["itoken_probabilities"]
                or raw["max_checked_logprob_difference"] != difference
                or raw["scores"]["covered_probability_mass"] != entry["covered_probability_mass"]):
            raise ValueError("Bank summary differs from raw stage record")
    spec = {"protocol": PROTOCOL, "units": units, "stages": stages, "marked_delta": delta,
            "unmarked_delta": 0.0, "seed": seed, "alpha": .01, "surrogate_count": 199,
            "bank_sha256": digest(bank_path/"bank_summary.json"),
            "key_sampling": "target and surrogate keys independently drawn with secrets.token_bytes(32), recorded once",
            "pairing": "same sampling uniforms per unit across marked/unmarked; independent keys across units",
            "source_sha256": {n: digest(root/n) for n in DEPENDENCIES}}
    save(out/"cohort_config.json", spec)
    documents = []
    for unit in range(units):
        problem_id = "study-" + hashlib.sha256(f"PUBLIC-PROBLEM:{seed}:{unit}".encode()).hexdigest()[:16]
        problem = make_problem(stages, problem_id)
        key = secrets.token_bytes(32)
        surrogates = [secrets.token_bytes(32) for _ in range(199)]
        unit_spec = {"unit_id": unit, "problem_id": problem_id, "public_demonstration_key_hex": key.hex(),
                     "surrogate_keys_hex": [k.hex() for k in surrogates], "sampling_seed": seed + unit}
        # Record the random key before drawing either path.
        save(out/f"unit_{unit:02d}_key.json", unit_spec)
        for condition, strength in (("unmarked", 0.0), ("marked", delta)):
            rng = random.Random(seed + unit)
            routes, policy_rows = [], []
            log_lr = 0.0
            for stage, entry in enumerate(bank["stages"]):
                p = entry["itoken_probabilities"]
                q = bias_distribution(p, key, context(problem, stage), delta=strength)
                alt = bias_distribution(p, key, context(problem, stage), delta=delta)
                selected = draw(q, rng)
                route = next(r for r,pair in candidate_pairs(problem, stage).items()
                             if pair[0].canonical() == selected)
                routes.append(route)
                log_lr += math.log(alt[selected]/p[selected])
                policy_rows.append({"stage": stage, "p": p, "q": q, "selected": selected, "route": route})
            steps = steps_for_routes(problem, routes)
            opaque_id = secrets.token_hex(8)
            row = {"document_id": opaque_id, **unit_spec, "condition": condition,
                   "delta": strength, "n_stages": stages, "routes": routes,
                   "planned_canonical": [s.canonical() for s in steps], "policy_rows": policy_rows,
                   "text": render_proof(steps), "planned_detection": detect_structural(problem, steps, key, surrogates),
                   "planned_log_likelihood_ratio": log_lr,
                   "likelihood_ratio_scope": "fixed-delta alternative against the explicit local-state planner only"}
            save(out/"private_plans"/(opaque_id+".json"), row)
            documents.append(row)
    save(out/"cohort_index.json", {"config": spec, "documents": [{k:r[k] for k in
         ("document_id","unit_id","problem_id","condition","planned_detection")} for r in documents]})
    return documents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--stages", type=int, default=32)
    args = parser.parse_args()
    if args.model:
        model = MLXModel(args.model, source_repo=SOURCE, revision=REVISION, device="cpu",
                         execution_mode="dense-fp32", max_context_tokens=8192)
        build_bank(model, args.bank, args.stages)
    if args.cohort:
        sample_cohort(args.bank, args.cohort, stages=args.stages)
