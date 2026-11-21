#!/usr/bin/env python3
"""Read-only audit of saved research_runner results; never invokes an LLM.

Usage: python audit_saved_results.py RESULT_DIRECTORY [--project PATH]
Exit 0: stored evidence agrees. Exit 1: discrepancies. Exit 2: no scored or
recovery records. A successful arithmetic audit is not a semantic or model
execution certification. Partial runs are identified, not silently completed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


DEFAULT_PROJECT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(obj, *, compact=False):
    kwargs = {"separators": (",", ":")} if compact else {"sort_keys": True}
    return hashlib.sha256(json.dumps(obj, **kwargs).encode()).hexdigest()


def logsum(values):
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


class Audit:
    def __init__(self, project):
        self.project = Path(project)
        sys.path.insert(0, str(self.project))
        import core
        self.core = core
        self.errors, self.warnings = [], []
        self.checks = 0

    def check(self, condition, where, detail):
        self.checks += 1
        if not condition:
            self.errors.append({"where": where, "error": detail})

    def equal(self, actual, expected, where):
        if isinstance(expected, float):
            okay = (isinstance(actual, (int, float)) and math.isfinite(actual)
                    and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10))
            self.check(okay, where, "numeric mismatch")
        elif isinstance(expected, dict):
            self.check(isinstance(actual, dict), where, "expected object")
            if isinstance(actual, dict):
                self.check(set(actual) == set(expected), where, "object keys differ")
                for key in expected.keys() & actual.keys():
                    self.equal(actual[key], expected[key], where + "." + key)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)), where, "expected sequence")
            if isinstance(actual, (list, tuple)):
                self.check(len(actual) == len(expected), where, "sequence lengths differ")
                for index, (a, b) in enumerate(zip(actual, expected)):
                    self.equal(a, b, f"{where}[{index}]")
        else:
            self.check(actual == expected, where, "value mismatch")

    def subset(self, actual, expected, where):
        for key, value in expected.items():
            self.check(key in actual, where, f"missing field {key}")
            if key in actual:
                self.equal(actual[key], value, where + "." + key)

    def guarded(self, where, fn, *args):
        try:
            return fn(*args)
        except (KeyError, ValueError, TypeError, IndexError, OSError, AttributeError) as exc:
            self.errors.append({"where": where, "error": f"{type(exc).__name__}: {exc}"})
            return None

    def metadata(self, obj, where):
        manifest = obj["checkpoint"]
        digest = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        self.equal(manifest["manifest_sha256"], digest, where + ".manifest_sha256")
        self.check(obj.get("model_invoked") is True, where, "model_invoked is not true")

    def score(self, row, where):
        self.metadata(row["metadata"], where + ".metadata")
        self.equal(row["context_id"], json_sha(row["prompt_token_ids"], compact=True),
                   where + ".context_id")
        logs, grouped, seen = [], defaultdict(list), set()
        for index, candidate in enumerate(row["candidates"]):
            loc = f"{where}.candidates[{index}]"
            ids, token_logs = candidate["token_ids"], candidate["token_logprobs"]
            self.check(bool(ids) and len(ids) == len(token_logs), loc, "invalid token lengths")
            self.check(tuple(ids) not in seen, loc, "duplicated token event")
            seen.add(tuple(ids))
            self.check(all(math.isfinite(x) and x <= 1e-6 for x in token_logs), loc,
                       "nonfinite or positive token log probability")
            value = math.fsum(token_logs)
            logs.append(value)
            grouped[candidate["itoken_id"]].append(value)
            self.equal(candidate["sequence_log_probability"], value, loc + ".sequence_log_probability")
            self.equal(candidate["context_id"], row["context_id"], loc + ".context_id")
            self.equal(candidate["tokenizer_id"], row["metadata"]["checkpoint"]["manifest_sha256"], loc + ".tokenizer_id")
            self.check(candidate["includes_termination"] is True and ids[-1] == candidate["termination_token_id"],
                       loc, "missing declared termination")
            if "scoring_eos_token_id" in row["metadata"]:
                self.equal(candidate["termination_token_id"], row["metadata"]["scoring_eos_token_id"], loc + ".eos")
                stops = set(row["metadata"]["generation_stop_token_ids"])
                self.check(not (set(ids[:-1]) & stops), loc, "candidate body includes an earlier stop token")
            self.equal(self.core.parse_step(candidate["text"]).canonical(), candidate["itoken_id"], loc + ".logical_label")
        normalizer = logsum(logs)
        self.check(normalizer <= math.log1p(1e-5), where, "disjoint raw mass exceeds one")
        group_logs = {key: logsum(values) - normalizer for key, values in grouped.items()}
        calculated = {
            "covered_probability_mass": math.exp(normalizer),
            "log_covered_probability_mass": normalizer,
            "candidate_probabilities": [math.exp(value - normalizer) for value in logs],
            "itoken_probabilities": {key: math.exp(value) for key, value in group_logs.items()},
            "itoken_log_probabilities": group_logs,
            "distribution_scope": "conditional_on_listed_terminated_token_sequences",
            "support_status": "not_certified_complete",
        }
        self.subset(row, calculated, where)
        crosschecks = row.get("uncached_crosschecks", [])
        if crosschecks:
            self.equal(row["uncached_crosscheck"], crosschecks[0], where + ".uncached_crosscheck_alias")
        for check in crosschecks:
            index = check["checked_candidate_index"]
            cached = row["candidates"][index]["token_logprobs"]
            uncached = check["uncached_token_logprobs"]
            self.check(len(cached) == len(uncached), where, "crosscheck token counts differ")
            differences = [a - b for a, b in zip(cached, uncached)]
            self.subset(check, {
                "max_absolute_token_logprob_difference": max(map(abs, differences)),
                "absolute_sequence_logprob_difference": abs(math.fsum(differences)),
                "sum_absolute_token_logprob_differences": math.fsum(map(abs, differences)),
                "uncached_sequence_log_probability": math.fsum(uncached),
                "scored_tokens": len(cached),
            }, where + f".crosscheck[{index}]")
        if row.get("uncached_crosscheck_scope") == "all_candidates":
            self.equal(sorted(c["checked_candidate_index"] for c in crosschecks),
                       list(range(len(logs))), where + ".crosscheck_coverage")
        if "probability_numerics_status" in row:
            discrepancy = max(c["max_absolute_token_logprob_difference"] for c in crosschecks)
            status = "passed" if discrepancy <= row["numerical_crosscheck_tolerance"] else "needs_investigation"
            self.equal(row["probability_numerics_status"], status, where + ".probability_numerics_status")
            if status != "passed":
                self.warnings.append(where + ": numerical crosscheck exceeds declared tolerance")
        stage = row.get("case", {}).get("stage", 0)
        problem = self.core.make_problem(stage + 1, "audit-scoring")
        labels = {pair[0].canonical(): route for route, pair in self.core.candidate_pairs(problem, stage).items()}
        return {"candidates": len(logs), "canonical_groups": len(grouped),
                "covered_probability_mass": calculated["covered_probability_mass"],
                "log_covered_probability_mass": normalizer,
                "conditional_route_probabilities": {labels[k]: p for k, p in calculated["itoken_probabilities"].items()},
                "conditional_probabilities_sum": math.fsum(calculated["itoken_probabilities"].values())}

    def freeze(self, frozen, model, where):
        sources = {"extractor_sha256": "semantic_recovery.py", "core_sha256": "core.py",
                   "adapter_sha256": "real_model.py", "runner_sha256": "research_runner.py",
                   "dataset_sha256": "research_eval_cases.json"}
        protocol = frozen.get("extractor_protocol", "formulas")
        self.check(protocol in {"formulas", "inference_ids"}, where, "unknown extractor protocol")
        if protocol == "inference_ids":
            sources.update(extractor_sha256="semantic_recovery_ids.py",
                           validator_sha256="semantic_recovery.py",
                           runner_sha256="research_id_runner.py",
                           metrics_runner_sha256="research_runner.py")
            self.equal(frozen["prompt_version"], "symbol-grounded-inference-ids-v1", where + ".prompt_version")
        else:
            self.equal(frozen["prompt_version"], "symbol-grounded-extraction-v1", where + ".prompt_version")
        for key, name in sources.items():
            self.equal(frozen[key], sha(self.project / name), where + "." + key)
        self.equal(frozen["model_metadata_sha256"], json_sha(model), where + ".model_metadata_sha256")
        self.equal(frozen["model_checkpoint_sha256"], model["checkpoint"]["manifest_sha256"], where + ".checkpoint")
        self.equal(frozen["model_revision"], model["revision"], where + ".revision")
        self.equal(frozen["temperature"], 0.0, where + ".temperature")
        self.equal(frozen["seed"], 0, where + ".seed")

    def recovery(self, row, case, frozen, where):
        self.equal(row["freeze"], frozen, where + ".freeze")
        self.subset(row, {"id": case["id"], "split": case["split"], "input_text": case["text"],
                          "category": case["category"]}, where)
        self.equal(row["gold"], {k: case[k] for k in ("expected_status", "expected_canonical", "expected_routes")}, where + ".gold")
        recovered, generated = row["recovery"], row["generation"]
        self.equal(recovered["prompt_version"], frozen["prompt_version"], where + ".prompt_version")
        accepted = recovered["status"] in {"complete_proof", "partial_evidence"}
        self.equal(recovered["accepted"], accepted, where + ".accepted")
        self.equal(recovered["complete"], recovered["status"] == "complete_proof", where + ".complete")
        self.check(not recovered["semantic_entailment_verified"], where, "unsupported semantic entailment certification")
        self.equal(recovered["model_output"], generated["text"], where + ".raw_output")
        self.equal(generated["temperature"], frozen["temperature"], where + ".temperature")
        self.equal(generated["seed"], frozen["seed"], where + ".seed")
        self.equal(generated["model_checkpoint_sha256"], frozen["model_checkpoint_sha256"], where + ".checkpoint")
        self.check(generated["finish_reason"] == "stop" or not accepted, where, "truncated generation accepted")
        problem = self.core.make_problem(case["n_stages"], case["problem_id"])
        steps = [self.core.IToken(s["rule"], tuple(s["premises"]), s["conclusion"]) for s in recovered["steps"]]
        allowed = {step.canonical() for stage in range(case["n_stages"])
                   for pair in self.core.candidate_pairs(problem, stage).values() for step in pair}
        for step in steps:
            self.check(self.core.valid_inference(step) and step.canonical() in allowed, where, "invalid or out-of-domain recovered step")
        if accepted:
            self.equal(len(recovered["evidence"]), len(steps), where + ".evidence_count")
            self.check(all(isinstance(e, str) and e and e in case["text"] for e in recovered["evidence"]),
                       where, "evidence is not verbatim source text")
        else:
            self.check(not steps, where, "abstention retained accepted steps")
        if recovered["complete"]:
            self.check(self.core.validate_proof(problem, steps), where, "complete proof fails validation")
        pred = {step.canonical() for step in steps} if accepted else set()
        gold = set(case["expected_canonical"])
        if case["expected_status"] == "complete":
            status_correct = recovered["complete"]
        elif case["expected_status"] == "partial":
            status_correct = accepted and not recovered["complete"]
        else:
            status_correct = not accepted
        routes = {}
        for stage in range(case["n_stages"]):
            for route, pair in self.core.candidate_pairs(problem, stage).items():
                if pair[0].canonical() in pred:
                    self.check(str(stage) not in routes, where, "multiple recovered routes at one stage")
                    routes[str(stage)] = route
        metrics = {"exact_steps": pred == gold, "status_correct": status_correct,
                   "strict_success": pred == gold and status_correct,
                   "gold_steps": len(gold), "predicted_steps": len(pred), "correct_steps": len(gold & pred),
                   "extra_steps": sorted(pred - gold), "missing_steps": sorted(gold - pred),
                   "predicted_routes": routes, "exact_explicit_routes": routes == case["expected_routes"],
                   "false_acceptance": case["expected_status"] == "abstain" and accepted}
        self.equal(row["metrics"], metrics, where + ".metrics")
        return dict(row, metrics=metrics)

    def summary(self, rows):
        successes = sum(r["metrics"]["strict_success"] for r in rows)
        n = len(rows)
        positive = [r for r in rows if r["gold"]["expected_status"] != "abstain"]
        negative = [r for r in rows if r["gold"]["expected_status"] == "abstain"]
        complete = [r for r in rows if r["gold"]["expected_status"] == "complete"]
        partial = [r for r in rows if r["gold"]["expected_status"] == "partial"]
        route_gold = [r for r in rows if r["gold"]["expected_routes"]]
        correct, predicted, gold = [sum(r["metrics"][key] for r in rows)
                                    for key in ("correct_steps", "predicted_steps", "gold_steps")]
        z, p = 1.959963984540054, successes / n
        denominator = 1 + z*z/n
        center = (p + z*z/(2*n))/denominator
        half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
        return {"cases": n, "strict_successes": successes, "strict_accuracy": p,
                "strict_accuracy_ci95": [max(0., center-half), min(1., center+half)],
                "positive_cases": len(positive), "positive_exact_recoveries": sum(r["metrics"]["strict_success"] for r in positive),
                "complete_gold_cases": len(complete), "complete_exact_recoveries": sum(r["metrics"]["strict_success"] for r in complete),
                "partial_gold_cases": len(partial), "partial_exact_recoveries": sum(r["metrics"]["strict_success"] for r in partial),
                "partial_falsely_completed": sum(r["recovery"]["complete"] for r in partial),
                "cases_with_explicit_branch_choices": len(route_gold),
                "exact_route_recoveries": sum(r["metrics"]["exact_explicit_routes"] for r in route_gold),
                "negative_cases": len(negative), "false_acceptances": sum(r["metrics"]["false_acceptance"] for r in negative),
                "accepted_cases": sum(r["recovery"]["accepted"] for r in rows),
                "complete_recovered_proofs": sum(r["recovery"]["complete"] for r in rows),
                "step_precision": correct/predicted if predicted else None,
                "step_recall": correct/gold if gold else None,
                "correct_steps": correct, "predicted_steps": predicted, "gold_steps": gold,
                "statuses": dict(Counter(r["recovery"]["status"] for r in rows)),
                "truncated_generations": sum(r["generation"]["finish_reason"] == "length" for r in rows),
                "wall_seconds": sum(r["generation"]["wall_seconds"] for r in rows)}

    def run(self, directory):
        directory = Path(directory)
        cases = {c["id"]: c for c in read(self.project / "research_eval_cases.json")["cases"]}
        scored, recovered, scoring_metadata = {}, defaultdict(list), {}
        for path in sorted((directory / "scores").glob("*.json")):
            row = self.guarded(path.name, read, path)
            if row is None:
                continue
            result = self.guarded(path.name, self.score, row, "scores/" + path.name)
            if result is not None:
                scored[path.stem] = result
                scoring_metadata[path.stem] = row["metadata"]
        score_summary_path = directory / "scores_summary.json"
        if score_summary_path.exists():
            score_summary = self.guarded(score_summary_path.name, read, score_summary_path)
            if score_summary is not None:
                contexts = score_summary["contexts"]
                self.check({r["id"] for r in contexts} == set(scored), score_summary_path.name, "summary contexts differ from raw contexts")
                self.check(len(contexts) == 6, score_summary_path.name, "summary does not contain six contexts")
                for item in contexts:
                    if item["id"] not in scored:
                        continue
                    self.equal(scoring_metadata[item["id"]], score_summary["model_metadata"], score_summary_path.name + ".model_metadata." + item["id"])
                    expected = scored[item["id"]]
                    self.subset(item, {key: expected[key] for key in ("covered_probability_mass", "log_covered_probability_mass", "conditional_route_probabilities")}, score_summary_path.name + "." + item["id"])
                    p, q = expected["conditional_route_probabilities"], item["watermarked_route_probabilities"]
                    self.equal(item["conditional_route_entropy_bits"], -math.fsum(v*math.log2(v) for v in p.values() if v), score_summary_path.name + ".entropy")
                    self.equal(math.fsum(q.values()), 1.0, score_summary_path.name + ".watermarked_normalizer")
                    self.equal(item["watermark_kl_nats"], math.fsum(q[k]*math.log(q[k]/p[k]) for k in p if q[k]), score_summary_path.name + ".kl")
        rows = sorted((directory / "recovery").glob("*.json"))
        frozen = self.guarded("evaluation_freeze.json", read, directory / "evaluation_freeze.json") if rows else None
        model_path = directory / "model_metadata.json"
        model = self.guarded(model_path.name, read, model_path) if (rows or model_path.exists()) else None
        if model:
            for ident, metadata in scoring_metadata.items():
                self.equal(metadata, model, "scores/" + ident + ".model_metadata")
        if frozen and model:
            self.guarded("freeze", self.freeze, frozen, model, "freeze")
            self.guarded("model_metadata", self.metadata, model, "model_metadata")
        for path in rows:
            row = self.guarded(path.name, read, path)
            if row is None:
                continue
            self.check(row.get("id") in cases, path.name, "unknown benchmark case")
            if row.get("id") not in cases:
                continue
            result = self.guarded(path.name, self.recovery, row, cases[row["id"]], frozen, path.name)
            if result is not None:
                recovered[row["split"]].append(result)
        summaries = {}
        for split, split_rows in recovered.items():
            calculated = self.summary(split_rows)
            summaries[split] = {key: calculated[key] for key in ("cases", "strict_successes", "complete_exact_recoveries", "partial_exact_recoveries", "false_acceptances", "step_precision", "step_recall", "statuses")}
            expected_ids = {c["id"] for c in cases.values() if c["split"] == split}
            actual_ids = {r["id"] for r in split_rows}
            self.check(len(actual_ids) == len(split_rows), split, "duplicate recovery cases")
            path = directory / f"recovery_{split}_summary.json"
            if path.exists():
                self.check(actual_ids == expected_ids, split, "summary exists with incomplete split")
                stored = self.guarded(path.name, read, path)
                if stored is not None:
                    self.subset(stored, calculated, path.name)
                    self.equal(stored["freeze"], frozen, path.name + ".freeze")
                    self.equal(stored["model_metadata"], model, path.name + ".model_metadata")
            else:
                self.warnings.append(f"{split}: partial or unfinished run; {len(actual_ids)}/{len(expected_ids)} records, no summary")
        if "test" in recovered:
            self.check((directory / "recovery_dev_summary.json").exists(), "test", "test has no prior dev summary")
        if scored and len(scored) != 6:
            self.warnings.append(f"scoring: partial context set ({len(scored)}/6)")
        has_results = bool(scored or recovered)
        return {"status": "failed" if self.errors else "consistent" if has_results else "no_results",
                "result_directory": directory.name, "audit_script_sha256": sha(__file__), "checks": self.checks,
                "scoring": scored, "recovery": summaries, "errors": self.errors,
                "warnings": self.warnings,
                "limits": ["Checks stored arithmetic and declared hashes; does not rerun the model or verify weight bytes.",
                           "Does not certify tokenization, textual entailment, representative data, or general idea-watermark robustness.",
                           "Candidate probabilities are conditional on listed terminated sequences; pilot/open-sample outputs are outside this audit."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    args = parser.parse_args()
    audit = Audit(args.project)
    report = audit.guarded("audit", audit.run, args.result_directory)
    if report is None:
        report = {"status": "failed", "errors": audit.errors}
    print(json.dumps(report, separators=(",", ":"), allow_nan=False))
    return {"consistent": 0, "failed": 1, "no_results": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
