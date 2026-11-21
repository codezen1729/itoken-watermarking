"""Small end-to-end demonstration, not a detection-power benchmark.

Every branch uses freshly scored real-model probabilities under the actual
preceding symbolic path. Selected inferences are verbalized by that model,
then recovered from its prose by a key-blind extraction call. Raw calls remain
available for review. Shared-model agreement is not independent semantic truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from core import candidate_pairs, make_problem, render_step, render_proof, validate_proof, STYLES
from pipeline import context, draw, extract, surrogate_keys
from real_model import MLXModel
from research_runner import save, digest, SOURCE_REPO, REVISION
from semantic_recovery import make_extraction_prompt, parse_extraction_output, RecoveryResult
from watermark import bias_distribution, key_randomization_pvalue


def detect_structural(problem, steps, key, bank):
    if not steps:
        return {"status": "insufficient_evidence", "n_choices": 0, "p_value": None, "detected": None}
    recovered = extract(problem, render_proof(steps))
    if not recovered.choices:
        return {"status": recovered.status, "n_choices": 0, "p_value": None, "detected": None}
    score, p, n = key_randomization_pvalue(recovered.choices, key, bank)
    return {"status": recovered.status, "n_choices": n, "p_value": p,
            "preferred": score, "detected": p <= .01,
            "detector": "key_randomization_only; no synthetic-baseline likelihood ratio"}


def run(model, out, stages=8, seed=20261121, recovery_protocol="steps"):
    if recovery_protocol not in {"steps", "ids"}:
        raise ValueError("Recovery protocol must be steps or ids")
    make_prompt, parse_output = make_extraction_prompt, parse_extraction_output
    if recovery_protocol == "ids":
        from semantic_recovery_ids import make_extraction_prompt as make_prompt
        from semantic_recovery_ids import parse_extraction_output as parse_output
        from semantic_recovery_ids import PROMPT_VERSION as recovery_prompt_version
    else:
        from semantic_recovery import PROMPT_VERSION as recovery_prompt_version
    if stages < 1:
        raise ValueError("Pilot needs at least one stage")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Pilot output is not empty; choose a fresh output directory")
    key = hashlib.sha256(f"PUBLIC-REAL-PILOT-KEY:{seed}".encode()).digest()
    problem = make_problem(stages, f"real-pilot-{seed}")
    bank = surrogate_keys(seed + 800, 199)
    root = Path(__file__).resolve().parent
    save(out / "pilot_manifest.json", {
        "seed": seed, "stages": stages, "model_metadata": model.metadata,
        "recovery_protocol": recovery_protocol, "recovery_prompt_version": recovery_prompt_version,
        "problem": {"problem_id": problem.problem_id, "premises": sorted(problem.premises),
                    "goal": problem.goal},
        "public_demonstration_key_hex": key.hex(),
        "surrogate_keys_seed": seed + 800, "surrogate_keys_count": 199,
        "detection_threshold": .01, "planner_deltas": [0, 4],
        "likelihood_ratio_alternative_delta": 4,
        "verbalization": {"temperature": .7, "max_tokens": 192, "seed_rule": "seed + stage"},
        "extraction": {"temperature": 0, "max_tokens": 768, "seed": 0},
        "source_sha256": {name: digest(root / name) for name in (
            "real_model_pilot.py", "research_runner.py", "real_model.py", "semantic_recovery.py", "semantic_recovery_ids.py",
            "core.py", "pipeline.py", "watermark.py")}})
    summaries = []
    for mode, delta in (("unmarked", 0), ("watermarked", 4)):
        destination = out / mode
        if (destination / "summary.json").exists():
            raise ValueError("Pilot output already exists; choose a new output directory")
        rng = random.Random(seed)
        planned, recovered_steps, paragraphs, rows = [], [], [], []
        planned_log_lr = 0.0
        for stage in range(stages):
            pair_map = candidate_pairs(problem, stage)
            candidates = [{"text": render_step(pair[0], style), "itoken_id": pair[0].canonical()}
                          for pair in pair_map.values() for style in STYLES]
            state = {"given_premises": sorted(problem.premises),
                     "steps_already_taken": [json.loads(s.canonical()) for s in planned],
                     "next_goal": f"p{stage+1}"}
            prompt = ("Choose exactly one next inference toward next_goal. Return exactly one of the following "
                      "statements and stop, without further explanation.\n"
                      + "\n".join(c["text"] for c in candidates)
                      + "\nCurrent problem and path:\n" + json.dumps(state, indent=2))
            scores = model.score_candidates([{"role": "user", "content": prompt}], candidates)
            p = scores["itoken_probabilities"]
            q = bias_distribution(p, key, context(problem, stage), delta=delta)
            detector_q = bias_distribution(p, key, context(problem, stage), delta=4)
            selected = draw(q, rng)
            planned_log_lr += math.log(detector_q[selected]) - math.log(p[selected])
            route = next(name for name, pair in pair_map.items() if pair[0].canonical() == selected)
            pair = pair_map[route]
            planned.extend(pair)
            row = {"stage": stage, "route": route, "delta": delta,
                   "scoring_prompt": prompt, "scores": scores, "biased_distribution": q,
                   "planned_steps": [json.loads(s.canonical()) for s in pair],
                   "phase": "scored"}
            stage_path = destination / f"stage_{stage:02d}.json"
            save(stage_path, row)
            verbalize_prompt = (
                "Write a short natural-language paragraph explaining exactly these two logical inferences. "
                "Keep every atom name (such as p0 or m0) unchanged. State the premises supporting each conclusion. "
                "Assert the reasoning actually taken; do not offer hypothetical alternatives. "
                "Use your own wording rather than a rigid numbered template. Do not add or omit an inference.\n"
                + json.dumps([json.loads(s.canonical()) for s in pair], indent=2))
            verbalization = model.generate([{"role": "user", "content": verbalize_prompt}],
                                           max_tokens=192, temperature=.7, seed=seed + stage)
            row.update(verbalization_prompt=verbalize_prompt, verbalization=verbalization,
                       phase="verbalized")
            save(stage_path, row)
            text = verbalization["text"]
            extraction_prompt, extraction = None, None
            if text.strip():
                extraction_prompt = make_prompt(problem, text)
                extraction = model.generate([{"role": "user", "content": extraction_prompt}],
                                            max_tokens=768, temperature=0, seed=0)
                recovered = (parse_output(problem, text, extraction["text"])
                             if extraction["finish_reason"] == "stop"
                             else RecoveryResult(status="truncated_generation", reason="Extractor did not stop",
                                                 model_output=extraction["text"], prompt_version=recovery_prompt_version))
            else:
                recovered = RecoveryResult(status="empty_verbalization", reason="Verbalizer produced no source text",
                                           prompt_version=recovery_prompt_version)
            if recovered.accepted:
                recovered_steps.extend(recovered.steps)
            paragraphs.append(text)
            row.update(extraction_prompt=extraction_prompt, extraction_generation=extraction,
                       recovery=recovered.to_dict(), planned_recovered_agreement=set(pair) == set(recovered.steps),
                       phase="complete")
            save(stage_path, row)
            rows.append(row)
            print(mode, "stage", stage, "route", route, "recovery", recovered.status,
                  "agreement", row["planned_recovered_agreement"], flush=True)
        planned_set, recovered_set = set(planned), set(recovered_steps)
        summary = {"mode": mode, "stages": stages, "delta": delta,
                   "public_demonstration_key_hex": key.hex(),
                   "model_metadata": model.metadata,
                   "planned_proof_valid": validate_proof(problem, planned),
                   "recovered_proof_valid": validate_proof(problem, recovered_steps),
                   "exact_representation_agreement": planned_set == recovered_set,
                   "correct_recovered_steps": len(planned_set & recovered_set),
                   "planned_steps": len(planned_set), "recovered_steps": len(recovered_set),
                   "stage_agreements": sum(r["planned_recovered_agreement"] for r in rows),
                   "verbalization_truncations": sum(r["verbalization"]["finish_reason"] == "length" for r in rows),
                   "extraction_truncations": sum(bool(r["extraction_generation"]) and
                                                  r["extraction_generation"]["finish_reason"] == "length" for r in rows),
                   "planned_detection": detect_structural(problem, planned, key, bank),
                   "planned_policy_log_likelihood_ratio": planned_log_lr,
                   "planned_policy_likelihood_ratio_detected": planned_log_lr >= math.log(100),
                   "likelihood_ratio_scope": "Fixed delta4 alternative against the actual candidate-restricted model policy; planned path only",
                   "recovered_detection": detect_structural(problem, recovered_steps, key, bank),
                   "limitations": ["One trace per setting, not an effectiveness estimate.",
                                   "A real model scores a restricted candidate set; uncovered mass remains.",
                                   "Same model verbalizes and extracts; agreement is not independent semantic verification.",
                                   "Explicit symbolic anchors and the original problem are required."]}
        save(destination / "summary.json", summary)
        (destination / "free_prose.txt").write_text("\n\n".join(paragraphs) + "\n")
        (destination / "planned_proof.txt").write_text(render_proof(planned) + "\n")
        (destination / "recovered_proof.txt").write_text(render_proof(recovered_steps) + "\n")
        summaries.append(summary)
    save(out / "pilot_summary.json", {"traces": summaries, "power_estimate": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--out", type=Path, default=Path("research_results/end_to_end"))
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("quantized", "dense-bf16", "dense-fp32"), default="quantized")
    parser.add_argument("--recovery-protocol", choices=("steps", "ids"), default="steps")
    args = parser.parse_args()
    model = MLXModel(args.model, source_repo=args.source_repo, revision=args.revision,
                     device=args.device, execution_mode=args.execution_mode, max_context_tokens=16384)
    run(model, args.out, args.stages, recovery_protocol=args.recovery_protocol)
