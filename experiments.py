"""Reproducible controlled experiments; run python experiments.py --out results."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

from core import make_problem, make_step_predictor, candidate_pairs, render_proof
from pipeline import generate, diagnose, transform, surrogate_keys
from watermark import auroc, wilson_interval

ATTACKS = ("clean", "controlled_rewording", "presentation_shuffle", "duplicate_steps",
           "delete_25_percent", "reroute_25_percent", "full_rewrite")


def trial(n, delta, sample, seed, calibration_count):
    trial_seed = seed + n * 1_000_000 + sample
    rng = random.Random(trial_seed)
    key = rng.getrandbits(256).to_bytes(32, "big")
    problem = make_problem(n, f"trial-{trial_seed}")
    produced = generate(problem, key, delta, rng)
    calibration = surrogate_keys(trial_seed + 10_000_000_000, calibration_count)
    return problem, key, produced, calibration, trial_seed


def flatten(result, *, cohort, n, delta, sample, trial_seed, attack, routes):
    return {"cohort": cohort, "n_stages": n, "delta": delta, "sample": sample,
            "trial_seed": trial_seed, "attack": attack,
            "generated_routes": " ".join(routes), **result}


def summarize(rows):
    n = len(rows)
    hits = sum(row["detected"] is True for row in rows)
    lo, hi = wilson_interval(hits, n)
    lr_rows = [row for row in rows if row["likelihood_ratio_detected"] is not None]
    lr_hits = sum(row["likelihood_ratio_detected"] for row in lr_rows)
    lr_ci = wilson_interval(lr_hits, len(lr_rows)) if lr_rows else (None, None)
    return {"trials": n, "detections": hits, "detection_rate": hits / n,
            "ci95_low": lo, "ci95_high": hi,
            "mean_recovered_choices": sum(row["n_choices"] for row in rows) / n,
            "proof_valid_rate": sum(row["proof_valid"] for row in rows) / n,
            "abstentions": sum(row["detected"] is None for row in rows),
            "lr_eligible_trials": len(lr_rows),
            "lr_detection_rate": lr_hits / len(lr_rows) if lr_rows else None,
            "lr_ci95_low": lr_ci[0], "lr_ci95_high": lr_ci[1]}


def exact_aggregation_check():
    problem = make_problem(1, "aggregation-check")
    predictor = make_step_predictor(problem, 0)
    actual = predictor.itoken_probabilities()
    pairs = candidate_pairs(problem, 0)
    expected = {"mp": .55, "and": .30, "or": .15}
    errors = [abs(actual[pairs[r][0]] - p) for r, p in expected.items()]
    return {"textual_outcomes": len(predictor.text_probabilities),
            "canonical_next_itokens": len(actual),
            "expected": expected,
            "recovered": {r: actual[pairs[r][0]] for r in expected},
            "max_absolute_error": max(errors),
            "token_chain_total_mass": sum(predictor.sequence_probability(x)
                                           for x in predictor.text_probabilities)}


def run(out: Path, samples=200, calibration_count=199, seed=20261121):
    if samples < 2 or calibration_count < 99:
        raise ValueError("Use at least two samples and 99 calibration keys (1% resolution)")
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    config = {"seed": seed, "samples_per_condition": samples,
              "surrogate_keys_per_document": calibration_count,
              "alpha": .01, "lengths": [8, 32, 128], "strengths": [0, 1, 2, 4],
              "likelihood_ratio_alternative_delta": 2,
              "attack_length": 64, "attack_strength": 2,
              "model": "exact finite autoregressive reference distribution; not an LLM",
              "baseline": "first stage (0.55,0.30,0.15); later 0.75*base+0.25*previous-route indicator",
              "experimental_unit": "independent simulated key/task pair; paired across strengths or attacks",
              "null_scope": "key-randomization guarantee averages over keys; not universal fixed-key protection",
              "python": sys.version.split()[0], "platform": platform.platform()}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    all_rows, conditions = [], []
    for n in config["lengths"]:
        null_scores = None
        for delta in config["strengths"]:
            rows = []
            for sample in range(samples):
                problem, key, produced, bank, trial_seed = trial(n, delta, sample, seed, calibration_count)
                result = diagnose(problem, render_proof(produced["steps"]), key, bank, delta=2)
                if not result["proof_valid"]:
                    raise AssertionError("Generator produced an invalid proof")
                if delta == 2 and abs(result["log_likelihood_ratio"] - produced["log_lr"]) > 1e-9:
                    raise AssertionError("Generation and detection likelihoods disagree")
                row = flatten(result, cohort="length_strength", n=n, delta=delta,
                              sample=sample, trial_seed=trial_seed, attack="clean",
                              routes=produced["routes"])
                rows.append(row)
            scores = [-math.log(r["p_value"]) for r in rows]
            if delta == 0:
                null_scores = scores
            summary = {"n_stages": n, "delta": delta, **summarize(rows)}
            summary["auroc_vs_unmarked"] = None if delta == 0 else auroc(scores, null_scores)
            conditions.append(summary)
            all_rows.extend(rows)
            print(f"length={n:3d} delta={delta} detection={summary['detection_rate']:.3f} "
                  f"valid={summary['proof_valid_rate']:.3f}", flush=True)
    attack_rows = {name: [] for name in ATTACKS}
    examples = None
    for sample in range(samples):
        problem, key, produced, bank, trial_seed = trial(64, 2, sample, seed + 1_000_000_000,
                                                        calibration_count)
        original = None
        current_examples = {}
        for j, attack in enumerate(ATTACKS):
            text = transform(problem, produced, attack, random.Random(trial_seed + j * 10_000))
            result = diagnose(problem, text, key, bank, delta=2)
            if attack == "clean":
                original = result
            if attack in {"controlled_rewording", "presentation_shuffle", "duplicate_steps"}:
                if any(result[k] != original[k] for k in ("p_value", "n_choices", "proof_valid")):
                    raise AssertionError(f"Required invariance failed under {attack}")
            row = flatten(result, cohort="attacks", n=64, delta=2, sample=sample,
                          trial_seed=trial_seed, attack=attack, routes=produced["routes"])
            attack_rows[attack].append(row)
            if sample == 0:
                current_examples[attack] = {"text": text, "detection": result}
        if sample == 0:
            examples = {"problem": {"n_stages": 64, "problem_id": problem.problem_id,
                                    "premises": sorted(problem.premises), "goal": problem.goal},
                        "public_demo_key_hex": key.hex(), "trial_seed": trial_seed,
                        "variants": current_examples}
        if (sample + 1) % 25 == 0:
            print(f"attack cohort: {sample + 1}/{samples}", flush=True)
    attacks = [{"attack": name, **summarize(rows)} for name, rows in attack_rows.items()]
    for rows in attack_rows.values():
        all_rows.extend(rows)
    with (out / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {"config": config, "aggregation_check": exact_aggregation_check(),
               "conditions": conditions, "attacks": attacks,
               "wall_seconds": time.monotonic() - start,
               "total_document_evaluations": len(all_rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "example_variants.json").write_text(json.dumps(examples, indent=2) + "\n")
    print(f"Saved {len(all_rows)} evaluations to {out}; {summary['wall_seconds']:.1f}s", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--calibration-keys", type=int, default=199)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    run(args.out, args.samples, args.calibration_keys, args.seed)
