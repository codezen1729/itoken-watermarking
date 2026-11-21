"""Run auditable real-model scoring and blinded semantic-recovery pilots.

Requires a local checkpoint and the optional MLX research environment. Model
weights are never downloaded by this runner. Raw model outputs are retained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from core import candidate_pairs, make_problem, parse_step, render_step, render_proof, STYLES
from pipeline import context
from real_model import MLXModel
from semantic_recovery import make_extraction_prompt, parse_extraction_output, PROMPT_VERSION, RecoveryResult
from watermark import bias_distribution, wilson_interval

ROOT = Path(__file__).resolve().parent
SOURCE_REPO = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
SCORING_CROSSCHECK_TOLERANCE = 0.05  # Natural-log units, per token; declared before model runs.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def context_cases():
    cases = []
    for mode in ("open_instruction", "listed_candidates", "reversed_list"):
        cases.append({"id": mode, "n_stages": 1, "stage": 0, "previous_route": None, "mode": mode})
    for previous in ("mp", "and", "or"):
        cases.append({"id": "after_" + previous, "n_stages": 2, "stage": 1,
                      "previous_route": previous, "mode": "listed_candidates"})
    return cases


def scoring_request(case):
    problem = make_problem(case["n_stages"], "real-score-" + case["id"])
    pairs = candidate_pairs(problem, case["stage"])
    candidates = [{"text": render_step(pairs[route][0], style),
                   "itoken_id": pairs[route][0].canonical(), "route": route, "style": style}
                  for route in ("mp", "and", "or") for style in STYLES]
    preceding = []
    if case["previous_route"]:
        preceding = list(candidate_pairs(problem, 0)[case["previous_route"]])
    payload = {"given_premises": sorted(problem.premises),
               "steps_already_taken": [json.loads(s.canonical()) for s in preceding],
               "current_fact": f"p{case['stage']}", "next_goal": f"p{case['stage']+1}"}
    instruction = ("Take exactly one next valid inference toward next_goal. "
                   "Use MP, AND_INTRO, or OR_INTRO. State its premises and conclusion. "
                   "Stop after that one inference; do not complete the whole proof.\n")
    if case["mode"] != "open_instruction":
        menu = [c["text"] for c in candidates]
        if case["mode"] == "reversed_list":
            menu.reverse()
        instruction += "Return exactly one of these complete statements, without extra text:\n" + "\n".join(menu) + "\n"
    prompt = instruction + "Problem and current reasoning state:\n" + json.dumps(payload, indent=2)
    return problem, candidates, [{"role": "user", "content": prompt}]


def run_scores(model, out):
    summaries = []
    for case in context_cases():
        target = out / "scores" / (case["id"] + ".json")
        if target.exists():
            raise ValueError(f"Refusing to overwrite scored context {case['id']}; use a fresh output directory")
        problem, candidates, messages = scoring_request(case)
        result = model.score_candidates(messages, candidates, verify_first_uncached=True,
                                        verify_all_uncached=case["id"] == "open_instruction")
        discrepancy = max(check["max_absolute_token_logprob_difference"] for check in result["uncached_crosschecks"])
        result["numerical_crosscheck_tolerance"] = SCORING_CROSSCHECK_TOLERANCE
        result["probability_numerics_status"] = ("passed" if discrepancy <= SCORING_CROSSCHECK_TOLERANCE
                                                 else "needs_investigation")
        result.update(case=case, messages=messages,
                      labels_verified_by="core.parse_step on known controlled candidate strings")
        for item in result["candidates"]:
            if parse_step(item["text"]).canonical() != item["itoken_id"]:
                raise AssertionError("Candidate logical label mismatch")
        mapping = {item["itoken_id"]: item["route"] for item in candidates}
        p = result["itoken_probabilities"]
        key = hashlib.sha256(b"PUBLIC-REAL-MODEL-DEMO").digest()
        q = bias_distribution(p, key, context(problem, case["stage"]), delta=2)
        entropy = -math.fsum(v * math.log2(v) for v in p.values() if v)
        kl = math.fsum(q[k] * math.log(q[k] / p[k]) for k in p if q[k])
        brief = {"id": case["id"], "covered_probability_mass": result["covered_probability_mass"],
                 "log_covered_probability_mass": result["log_covered_probability_mass"],
                 "conditional_route_probabilities": {mapping[k]: v for k, v in p.items()},
                 "watermarked_route_probabilities": {mapping[k]: v for k, v in q.items()},
                 "conditional_route_entropy_bits": entropy, "watermark_kl_nats": kl,
                 "uncached_crosscheck": result["uncached_crosscheck"],
                 "uncached_crosscheck_scope": result["uncached_crosscheck_scope"],
                 "max_checked_token_logprob_difference": discrepancy,
                 "probability_numerics_status": result["probability_numerics_status"],
                 "wall_seconds": result["wall_seconds"],
                 "distribution_scope": result["distribution_scope"]}
        save(target, result)
        summaries.append(brief)
        print("scored", case["id"], "covered mass", f"{brief['covered_probability_mass']:.3g}",
              "routes", brief["conditional_route_probabilities"], flush=True)
    save(out / "scores_summary.json", {"contexts": summaries, "model_metadata": model.metadata,
                                       "model_invoked": True})


def result_metrics(case, recovered):
    gold = set(case["expected_canonical"])
    pred = {step.canonical() for step in recovered.steps} if recovered.accepted else set()
    if case["expected_status"] == "complete":
        status_correct = recovered.complete
    elif case["expected_status"] == "partial":
        status_correct = recovered.accepted and not recovered.complete
    else:
        status_correct = not recovered.accepted
    predicted_routes = {}
    if recovered.accepted:
        for stage in range(case["n_stages"]):
            pairs = candidate_pairs(make_problem(case["n_stages"], case["problem_id"]), stage)
            for route, pair in pairs.items():
                if pair[0].canonical() in pred:
                    predicted_routes[str(stage)] = route
    return {"exact_steps": pred == gold, "status_correct": status_correct,
            "strict_success": pred == gold and status_correct,
            "gold_steps": len(gold), "predicted_steps": len(pred),
            "correct_steps": len(gold & pred), "extra_steps": sorted(pred - gold),
            "missing_steps": sorted(gold - pred),
            "predicted_routes": predicted_routes,
            "exact_explicit_routes": predicted_routes == case["expected_routes"],
            "false_acceptance": case["expected_status"] == "abstain" and recovered.accepted}


def freeze_record(out, model, dataset, max_tokens):
    current = {"extractor_sha256": digest(ROOT / "semantic_recovery.py"),
               "core_sha256": digest(ROOT / "core.py"),
               "adapter_sha256": digest(ROOT / "real_model.py"),
               "runner_sha256": digest(Path(__file__)), "dataset_sha256": digest(dataset),
               "prompt_version": PROMPT_VERSION, "temperature": 0.0,
               "max_generation_tokens": max_tokens, "seed": 0,
               "model_checkpoint_sha256": model.metadata["checkpoint"]["manifest_sha256"],
               "model_metadata_sha256": hashlib.sha256(json.dumps(model.metadata, sort_keys=True).encode()).hexdigest(),
               "model_revision": model.metadata["revision"]}
    target = out / "evaluation_freeze.json"
    if target.exists() and json.loads(target.read_text()) != current:
        raise ValueError("Evaluation configuration changed; use a new output directory and report a new run")
    save(target, current)
    return current


def recovery_summary(rows):
    successes = sum(r["metrics"]["strict_success"] for r in rows)
    positive = [r for r in rows if r["gold"]["expected_status"] != "abstain"]
    negative = [r for r in rows if r["gold"]["expected_status"] == "abstain"]
    complete_gold = [r for r in rows if r["gold"]["expected_status"] == "complete"]
    partial_gold = [r for r in rows if r["gold"]["expected_status"] == "partial"]
    route_gold = [r for r in rows if r["gold"]["expected_routes"]]
    tp = sum(r["metrics"]["correct_steps"] for r in rows)
    predicted = sum(r["metrics"]["predicted_steps"] for r in rows)
    gold = sum(r["metrics"]["gold_steps"] for r in rows)
    return {"cases": len(rows), "strict_successes": successes,
            "strict_accuracy": successes / len(rows), "strict_accuracy_ci95": wilson_interval(successes, len(rows)),
            "positive_cases": len(positive),
            "positive_exact_recoveries": sum(r["metrics"]["strict_success"] for r in positive),
            "complete_gold_cases": len(complete_gold),
            "complete_exact_recoveries": sum(r["metrics"]["strict_success"] for r in complete_gold),
            "partial_gold_cases": len(partial_gold),
            "partial_exact_recoveries": sum(r["metrics"]["strict_success"] for r in partial_gold),
            "partial_falsely_completed": sum(r["recovery"]["complete"] for r in partial_gold),
            "cases_with_explicit_branch_choices": len(route_gold),
            "exact_route_recoveries": sum(r["metrics"]["exact_explicit_routes"] for r in route_gold),
            "negative_cases": len(negative),
            "false_acceptances": sum(r["metrics"]["false_acceptance"] for r in negative),
            "accepted_cases": sum(r["recovery"]["accepted"] for r in rows),
            "complete_recovered_proofs": sum(r["recovery"]["complete"] for r in rows),
            "step_precision": tp / predicted if predicted else None,
            "step_recall": tp / gold if gold else None,
            "correct_steps": tp, "predicted_steps": predicted, "gold_steps": gold,
            "statuses": dict(Counter(r["recovery"]["status"] for r in rows)),
            "truncated_generations": sum(r["generation"]["finish_reason"] == "length" for r in rows),
            "wall_seconds": sum(r["generation"]["wall_seconds"] for r in rows)}


def run_recovery(model, out, split, max_tokens=1024):
    dataset = ROOT / "research_eval_cases.json"
    freeze = freeze_record(out, model, dataset, max_tokens)
    if split == "test" and not (out / "recovery_dev_summary.json").exists():
        raise ValueError("Run the declared dev split before opening the test evaluation")
    data = json.loads(dataset.read_text())
    cases = [case for case in data["cases"] if case["split"] == split]
    rows = []
    for case in cases:
        target = out / "recovery" / (case["id"] + ".json")
        if target.exists():
            row = json.loads(target.read_text())
            if row["freeze"] != freeze:
                raise ValueError("Existing case comes from a different frozen configuration")
            rows.append(row)
            continue
        problem = make_problem(case["n_stages"], case["problem_id"])
        prompt = make_extraction_prompt(problem, case["text"])
        generated = model.generate([{"role": "user", "content": prompt}], max_tokens=max_tokens,
                                   temperature=0.0, seed=0)
        recovered = (parse_extraction_output(problem, case["text"], generated["text"])
                     if generated["finish_reason"] == "stop"
                     else RecoveryResult(status="truncated_generation", reason="Model did not produce its stop token",
                                         model_output=generated["text"]))
        row = {"id": case["id"], "split": split, "category": case["category"], "freeze": freeze,
               "input_text": case["text"], "extraction_prompt": prompt,
               "generation": generated, "recovery": recovered.to_dict(),
               "gold": {key: case[key] for key in ("expected_status", "expected_canonical", "expected_routes")},
               "metrics": result_metrics(case, recovered)}
        save(target, row)
        rows.append(row)
        print(split, case["id"], recovered.status, "exact", row["metrics"]["strict_success"],
              "tokens", generated["generated_tokens"], flush=True)
    summary = {"split": split, "freeze": freeze, "model_metadata": model.metadata,
               **recovery_summary(rows)}
    save(out / f"recovery_{split}_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("split", "strict_accuracy", "step_precision", "step_recall", "false_acceptances")}), flush=True)


def run_open_samples(model, out, count=8):
    case = context_cases()[0]
    problem, candidates, messages = scoring_request(case)
    listed = {c["text"] for c in candidates}
    rows = []
    for index in range(count):
        generated = model.generate(messages, max_tokens=128, temperature=1.0, seed=20261121 + index)
        try:
            strict = parse_step(generated["text"]).canonical()
        except ValueError:
            strict = None
        extraction_prompt, extraction_generation = None, None
        if generated["text"].strip():
            extraction_prompt = make_extraction_prompt(problem, generated["text"])
            extraction_generation = model.generate([{"role": "user", "content": extraction_prompt}],
                                                    max_tokens=1024, temperature=0.0, seed=0)
            recovered = (parse_extraction_output(problem, generated["text"], extraction_generation["text"])
                         if extraction_generation["finish_reason"] == "stop"
                         else RecoveryResult(status="truncated_generation", reason="Extraction was truncated",
                                             model_output=extraction_generation["text"]))
        else:
            recovered = RecoveryResult(status="empty_generation", reason="Model produced no source text")
        first = {pair[0].canonical(): route for route, pair in candidate_pairs(problem, 0).items()}
        recovered_route = None
        if recovered.accepted and len(recovered.steps) == 1:
            recovered_route = first.get(recovered.steps[0].canonical())
        rows.append({"index": index, "generation": generated, "strict_itoken": strict,
                     "exact_listed_text": generated["text"] in listed,
                     "extraction_prompt": extraction_prompt, "extraction_generation": extraction_generation,
                     "recovery": recovered.to_dict(), "single_first_step_route": recovered_route})
        print("open sample", index, "listed", rows[-1]["exact_listed_text"],
              "recovery", recovered.status, "single step", recovered_route, flush=True)
    save(out / "open_samples.json", {"model_metadata": model.metadata, "messages": messages,
                                     "sampling_temperature": 1.0, "samples": rows,
                                     "scope": "Small diagnostic sample; no full-distribution calibration claim"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--out", type=Path, default=Path("research_results"))
    parser.add_argument("--phase", choices=("smoke", "scores", "dev", "test", "open-samples"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("quantized", "dense-bf16", "dense-fp32"), default="quantized")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    model = MLXModel(args.model, source_repo=args.source_repo, revision=args.revision,
                     device=args.device, execution_mode=args.execution_mode, max_context_tokens=8192)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out / "model_metadata.json", model.metadata)
    if args.phase == "smoke":
        result = model.generate([{"role": "user", "content": 'Return exactly this JSON: {"ok":true}'}],
                                max_tokens=64, temperature=0.0)
        save(args.out / "smoke.json", result)
        print(json.dumps({k: result[k] for k in ("text", "finish_reason", "generated_tokens", "wall_seconds")}), flush=True)
    elif args.phase == "scores":
        run_scores(model, args.out)
    elif args.phase in {"dev", "test"}:
        run_recovery(model, args.out, args.phase, args.max_tokens)
    else:
        run_open_samples(model, args.out)


if __name__ == "__main__":
    main()
