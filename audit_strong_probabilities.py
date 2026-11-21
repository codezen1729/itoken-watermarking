"""Replay the archived probability, route and key arithmetic without loading a model.

Run with ordinary Python assertions enabled. The audit reads the saved experiment
without writing to it. The branch-symbol diagnostic was devised after collection.
"""
from collections import Counter, defaultdict
import hashlib
import hmac
from itertools import accumulate
import json
import math
from pathlib import Path
import random
import re
import statistics

from core import candidate_pairs, make_problem, parse_proof, parse_step, validate_proof
from recovery_v2 import make_extraction_prompt, make_verification_prompt, parse_proposal


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "research_results/strong_recovery_v2"
ARTIFACTS = RUN / "agent_artifacts"
ROUTE_SYMBOLS = {"m": "mp", "d": "and", "e": "or"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=2e-13, abs_tol=1e-20), (actual, expected)


def replay_bank():
    bank = RUN / "probability_bank"
    summary = read(bank / "bank_summary.json")
    manifest = read(bank / "bank_manifest.json")
    assert summary["freeze"] == manifest
    metadata = manifest["model_metadata"]
    assert metadata["revision"] == "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
    assert metadata["source_repo"] == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    assert metadata["execution_mode"] == "dense-fp32"
    eos = metadata["scoring_eos_token_id"]
    assert eos == 151645 and len(summary["stages"]) == 32
    probabilities, crosschecks, coverage, entropy = [], [], [], []
    for stage, record in enumerate(summary["stages"]):
        raw = read(bank / f"stage_{stage:02d}.json")
        scores = raw["scores"]
        candidates = scores["candidates"]
        assert raw["stage"] == record["stage"] == stage and raw["freeze"] == manifest
        assert scores["metadata"] == metadata
        assert len({tuple(c["token_ids"]) for c in candidates}) == len(candidates) == 9
        masses = defaultdict(float)
        sequence_masses = []
        for candidate in candidates:
            ids = candidate["token_ids"]
            logprobs = candidate["token_logprobs"]
            assert len(ids) == len(logprobs) and ids[-1] == eos and ids.count(eos) == 1
            assert candidate["includes_termination"] and candidate["termination_token_id"] == eos
            assert all(math.isfinite(x) and x <= 0 for x in logprobs)
            assert candidate["context_id"] == scores["context_id"]
            token = parse_step(candidate["text"]).canonical()
            assert token == candidate["itoken_id"]
            total = math.fsum(logprobs)
            close(total, candidate["sequence_log_probability"])
            mass = math.exp(total)
            sequence_masses.append(mass)
            masses[token] += mass
        covered = math.fsum(sequence_masses)
        for value in (scores["covered_probability_mass"], record["covered_probability_mass"]):
            close(covered, value)
        for mass, value in zip(sequence_masses, scores["candidate_probabilities"], strict=True):
            close(mass / covered, value)
        p = {token: mass / covered for token, mass in masses.items()}
        assert len(p) == 3
        for table in (scores["itoken_probabilities"], record["itoken_probabilities"]):
            assert p.keys() == table.keys()
            for token, value in p.items():
                close(value, table[token])
        checks = scores["uncached_crosschecks"]
        expected_indices = set(range(9)) if stage == 0 else {0}
        assert {c["checked_candidate_index"] for c in checks} == expected_indices
        differences = []
        for check in checks:
            cached = candidates[check["checked_candidate_index"]]["token_logprobs"]
            uncached = check["uncached_token_logprobs"]
            difference = max(abs(x - y) for x, y in zip(cached, uncached, strict=True))
            close(difference, check["max_absolute_token_logprob_difference"])
            differences.append(difference)
        close(max(differences), record["max_checked_logprob_difference"])
        crosschecks.extend(differences)
        probabilities.append(p)
        coverage.append(covered)
        entropy.append(-math.fsum(v * math.log(v) for v in p.values()))
    print(f"32 contexts, 288 sequences, {len(crosschecks)} crosschecks, "
          f"maximum per-token discrepancy {max(crosschecks):.8g} nats")
    print(f"Coverage min/mean/max {min(coverage):.8f}/{statistics.mean(coverage):.8f}/"
          f"{max(coverage):.8f}, mean conditional entropy {statistics.mean(entropy):.8f} nats")
    return probabilities


def public_context(problem_id, stage):
    value = {"protocol": "itoken-proof-v1", "problem_id": problem_id,
             "start": f"p{stage}", "goal": f"p{stage + 1}"}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def preferred(key, context, token):
    message = json.dumps([context, token], ensure_ascii=False, separators=(",", ":")).encode()
    return hmac.digest(key, b"itoken-watermark:v1\x00" + message, "sha256")[0] < 128


def replay_cohort(probabilities):
    cohort = RUN / "cohort"
    config = read(cohort / "cohort_config.json")
    index = read(cohort / "cohort_index.json")
    assert config == index["config"]
    assert config["bank_sha256"] == digest(RUN / "probability_bank/bank_summary.json")
    assert (config["units"], config["stages"], config["seed"]) == (8, 32, 20261122)
    assert (config["marked_delta"], config["unmarked_delta"], config["alpha"]) == (6, 0, .01)
    for name, sha in config["source_sha256"].items():
        assert digest(ROOT / name) == sha
    keys_used = []
    paired = read(RUN / "paired_rows.json")["rows"]
    assert len(paired) == 32
    totals = defaultdict(Counter)
    paths = {}
    plans = {}
    p_values = defaultdict(list)
    for unit in range(8):
        key_record = read(cohort / f"unit_{unit:02d}_key.json")
        target = bytes.fromhex(key_record["public_demonstration_key_hex"])
        surrogates = list(map(bytes.fromhex, key_record["surrogate_keys_hex"]))
        assert len(surrogates) == config["surrogate_count"] == 199
        keys_used.extend([target, *surrogates])
        seed = config["seed"]
        expected_id = hashlib.sha256(f"PUBLIC-PROBLEM:{seed}:{unit}".encode()).hexdigest()[:16]
        assert key_record["problem_id"] == "study-" + expected_id
        assert key_record["sampling_seed"] == seed + unit
        problem = make_problem(32, key_record["problem_id"])
        entries = [d for d in index["documents"] if d["unit_id"] == unit]
        assert sorted(e["condition"] for e in entries) == ["marked", "unmarked"]
        for entry in entries:
            ident, condition = entry["document_id"], entry["condition"]
            plan = read(cohort / "private_plans" / f"{ident}.json")
            assert ident not in plans and plan["document_id"] == ident
            plans[ident] = plan
            assert all(plan[k] == v for k, v in key_record.items())
            assert plan["condition"] == condition and plan["delta"] == config[condition + "_delta"]
            rng = random.Random(key_record["sampling_seed"])
            steps, labels, routes = [], [], []
            log_lr = 0
            for stage, (p, row) in enumerate(zip(probabilities, plan["policy_rows"], strict=True)):
                context = public_context(problem.problem_id, stage)
                weights = {t: v * math.exp(6 * preferred(target, context, t)) for t, v in p.items()}
                alt = {t: w / math.fsum(weights.values()) for t, w in weights.items()}
                q = alt if condition == "marked" else p
                assert row["stage"] == stage and p.keys() == row["p"].keys() == row["q"].keys()
                for token in p:
                    close(p[token], row["p"][token])
                    close(q[token], row["q"][token])
                u = rng.random()
                choice = next(t for t, limit in zip(q, accumulate(q.values())) if u < limit)
                assert choice == row["selected"]
                options = candidate_pairs(problem, stage)
                route, pair = next((r, s) for r, s in options.items() if s[0].canonical() == choice)
                assert route == row["route"] == plan["routes"][stage]
                routes.append(route)
                steps.extend(pair)
                labels.append((context, choice))
                log_lr += math.log(alt[choice] / p[choice])
            close(log_lr, plan["planned_log_likelihood_ratio"])
            assert validate_proof(problem, steps)
            assert [t.canonical() for t in steps] == plan["planned_canonical"]
            assert parse_proof(plan["text"]) == steps
            paths[unit, condition] = tuple(routes)
            totals[condition].update(routes)
            observed = sum(preferred(target, ctx, t) for ctx, t in labels)
            null = [sum(preferred(key, ctx, t) for ctx, t in labels) for key in surrogates]
            p_value = (1 + sum(s >= observed for s in null)) / 200
            p_values[condition].append(p_value)
            variants = [r for r in paired if r["document_id"] == ident]
            assert sorted(r["variant"] for r in variants) == ["original", "reworded"]
            for record in (plan["planned_detection"], entry["planned_detection"],
                           *(r["detection"] for r in variants)):
                assert record["preferred"] == observed and record["p_value"] == p_value
                assert record["n_choices"] == 32 and record["detected"] == (p_value <= .01)
    assert len(keys_used) == len(set(keys_used)) == 1600 and all(len(k) == 32 for k in keys_used)
    for condition, counts in totals.items():
        unique = len({p for (_, c), p in paths.items() if c == condition})
        print(f"{condition}: {dict(counts)}, {unique} distinct paths, p-values {p_values[condition]}")
    return plans


def check_agent_records(plans):
    holdout = ROOT / "fresh_holdout.json"
    assert digest(holdout) == "58212368fe99fe80f3ed4a015c0c4545a11d867c6544413b6416306090d6752f"
    gold = {c["id"]: c for c in read(holdout)["cases"]}
    calls = read(ARTIFACTS / "agent_calls.json")["calls"]
    assert len({c["agent"] for c in calls}) == len(calls) == 28
    for call in calls:
        assert digest(ARTIFACTS / call["input"]) == call["input_sha256"]
        assert digest(ARTIFACTS / call["output"]) == call["output_sha256"]
        assert call["attempt"] == 1 and call["fork_turns"] == "none"
    recovered_documents = defaultdict(set)
    symbol_matches = Counter()
    for folder in sorted([*ARTIFACTS.glob("recovery_*"), *ARTIFACTS.glob("holdout_batch_*")]):
        bindings = read(folder / "private_manifest.json")["bindings"]
        proposals = {r["request_id"]: r for r in read(folder / "proposals/requests.json")["requests"]}
        answers = {r["request_id"]: r["response"] for r in read(folder / "proposals/responses.json")["responses"]}
        verifiers = {r["request_id"]: r for r in read(folder / "verification/requests.json")["requests"]}
        expected_verifiers = set()
        assert set(proposals) == set(answers) == {b["request_id"] for b in bindings}
        for binding in bindings:
            rid, case = binding["request_id"], binding["case"]
            ident, text = case["id"], case["text"]
            problem = make_problem(case["n_stages"], case["problem_id"])
            assert proposals[rid]["prompt"] == make_extraction_prompt(problem, text)
            decision = parse_proposal(problem, text, answers[rid])
            if decision["status"] == "extracted" and decision["steps"]:
                expected_verifiers.add(rid)
                assert verifiers[rid]["prompt"] == make_verification_prompt(problem, text, decision)
            if folder.name.startswith("holdout_"):
                assert all(gold[ident][k] == case[k] for k in ("text", "n_stages", "problem_id"))
                continue
            number = int(folder.name.split("_")[1])
            variant = "original" if number <= 4 else "reworded"
            plan = plans[ident]
            assert plan["problem_id"] == problem.problem_id and plan["n_stages"] == problem.n_stages
            if variant == "original":
                assert text == plan["text"]
            assert ident not in recovered_documents[variant]
            recovered_documents[variant].add(ident)
            symbols = set(re.findall(r"\b([mde])(\d+)\b", text))
            for stage, route in enumerate(plan["routes"]):
                guesses = {ROUTE_SYMBOLS[s] for s, n in symbols if int(n) == stage}
                symbol_matches[variant] += guesses == {route}
        assert set(verifiers) == expected_verifiers
    for variant in ("original", "reworded"):
        assert recovered_documents[variant] == set(plans)
        print(f"Post hoc symbol-only branch matches ({variant}): {symbol_matches[variant]}/512")
    print(f"{len(calls)} agent batches with matching hashes, {sum(c['items'] for c in calls)} items")


if __name__ == "__main__":
    check_agent_records(replay_cohort(replay_bank()))
