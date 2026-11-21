"""Retain unfiltered native samples and blinded inference-ID extractions.

This is a small diagnostic of output outside the listed candidate strings,
not a calibration study or an independent assessment of semantic truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

from core import candidate_pairs, parse_step
from real_model import MLXModel, continuation_tokens
from research_runner import context_cases, scoring_request
from semantic_recovery import RecoveryResult
from semantic_recovery_ids import PROMPT_VERSION, make_extraction_prompt, parse_extraction_output


PROTOCOL_VERSION = "native-open-step-sampling-ids-v1"
NATIVE_SEED = 20261121


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _single_first_route(problem, recovered) -> tuple[str | None, str]:
    if not recovered.accepted:
        return None, "extraction_not_accepted"
    if len(recovered.steps) != 1:
        return None, "multiple_or_zero_recovered_steps"
    first = {pair[0].canonical(): route for route, pair in candidate_pairs(problem, 0).items()}
    route = first.get(recovered.steps[0].canonical())
    return route, "one_first_step" if route else "recovered_step_is_not_a_first_step"


def run(model, out, count: int = 8) -> dict:
    """Run fixed native/ID-extraction settings, reusing one loaded model.

    Each native output is saved before extraction starts; each extraction is
    saved before parsing. An interruption therefore leaves completed raw calls.
    A new, empty output directory is required. No gold cases or keys are read.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Native-sampling output must be a new or empty directory")
    problem, candidates, messages = scoring_request(context_cases()[0])
    listed_texts = {item["text"] for item in candidates}
    prompt_text, prompt_ids, context_id = model._prompt(messages)
    listed_sequences = set()
    for candidate in candidates:
        prefix, tokens = continuation_tokens(model.tokenizer, prompt_text, candidate["text"],
                                              model.eos_token_id, model.stop_token_ids)
        if prefix != prompt_ids:
            raise AssertionError("Listed-event prompt differs from the native prompt")
        listed_sequences.add(tuple(tokens))
    root = Path(__file__).resolve().parent
    manifest = {
        "protocol_version": PROTOCOL_VERSION, "count_requested": count,
        "model_metadata": model.metadata,
        "model_metadata_sha256": hashlib.sha256(json.dumps(model.metadata, sort_keys=True).encode()).hexdigest(),
        "problem": {"problem_id": problem.problem_id, "premises": sorted(problem.premises),
                    "goal": problem.goal, "n_stages": problem.n_stages},
        "native_prompt_messages": messages, "native_prompt_token_ids": prompt_ids,
        "native_context_id": context_id, "listed_candidates_for_analysis_only": candidates,
        "native_generation": {"temperature": 1.0, "max_tokens": 128,
                              "seed_rule": "20261121 + sample_index", "unfiltered": True},
        "extraction": {"protocol": "ids", "prompt_version": PROMPT_VERSION,
                       "temperature": 0.0, "max_tokens": 1024, "seed": 0,
                       "watermark_key_provided": False, "gold_labels_provided": False},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("research_native_samples.py", "research_runner.py", "real_model.py",
                                       "core.py", "semantic_recovery_ids.py", "semantic_recovery.py")},
        "scope": "Small native-output diagnostic; no full-distribution calibration or semantic-accuracy claim",
    }
    _save(out / "native_manifest.json", manifest)
    rows = []
    for index in range(count):
        path = out / f"sample_{index:02d}.json"
        row = {"index": index, "protocol_version": PROTOCOL_VERSION,
               "native_seed": NATIVE_SEED + index, "phase": "native_generation_pending"}
        _save(path, row)
        try:
            generated = model.generate(messages, max_tokens=128, temperature=1.0,
                                       seed=NATIVE_SEED + index)
        except BaseException as exc:
            row.update(phase="native_generation_failed_or_interrupted", exception_type=type(exc).__name__)
            _save(path, row)
            raise
        row.update(generation=generated, phase="native_generation_saved",
                   exact_listed_text=generated["text"] in listed_texts,
                   exact_listed_terminated_sequence=(generated["finish_reason"] == "stop"
                                                    and tuple(generated["token_ids"]) in listed_sequences))
        _save(path, row)
        try:
            strict_itoken = parse_step(generated["text"]).canonical()
        except ValueError:
            strict_itoken = None
        row.update(strict_itoken=strict_itoken, extraction_prompt=None, extraction_generation=None)
        if generated["text"].strip():
            prompt = make_extraction_prompt(problem, generated["text"])
            row.update(extraction_prompt=prompt, phase="extraction_pending")
            _save(path, row)
            try:
                extraction = model.generate([{"role": "user", "content": prompt}],
                                            max_tokens=1024, temperature=0.0, seed=0)
            except BaseException as exc:
                row.update(phase="extraction_failed_or_interrupted", exception_type=type(exc).__name__)
                _save(path, row)
                raise
            row.update(extraction_generation=extraction, phase="extraction_saved")
            _save(path, row)
            if extraction["finish_reason"] == "stop":
                recovered = parse_extraction_output(problem, generated["text"], extraction["text"])
            else:
                recovered = RecoveryResult(status="truncated_generation", reason="Extractor did not produce a stop token",
                                           model_output=extraction["text"], prompt_version=PROMPT_VERSION)
        else:
            recovered = RecoveryResult(status="abstain", model_status="abstain",
                                       reason="Native generation produced no non-whitespace source text",
                                       prompt_version=PROMPT_VERSION)
        prefix_route, prefix_status = _single_first_route(problem, recovered)
        source_stopped = generated["finish_reason"] == "stop"
        route = prefix_route if source_stopped else None
        route_status = prefix_status if source_stopped else "native_generation_truncated"
        row.update(recovery=recovered.to_dict(), single_first_step_route=route,
                   prefix_single_first_step_route=None if source_stopped else prefix_route,
                   prefix_route_recovery_status=None if source_stopped else prefix_status,
                   route_recovery_status=route_status, phase="complete")
        _save(path, row)
        rows.append(row)
        print("native", index, "listed text", row["exact_listed_text"], "recovery", recovered.status,
              "single first route", route, flush=True)
    summary = {
        "protocol_version": PROTOCOL_VERSION, "model_metadata": model.metadata,
        "count_requested": count, "count_completed": len(rows),
        "exact_listed_text_matches": sum(row["exact_listed_text"] for row in rows),
        "exact_listed_terminated_sequence_matches": sum(row["exact_listed_terminated_sequence"] for row in rows),
        "native_generation_truncations": sum(row["generation"]["finish_reason"] != "stop" for row in rows),
        "extraction_truncations": sum(row["extraction_generation"] is not None and
                                      row["extraction_generation"]["finish_reason"] != "stop" for row in rows),
        "recovery_status_counts": dict(Counter(row["recovery"]["status"] for row in rows)),
        "single_first_step_route_counts": dict(Counter(row["single_first_step_route"] or "unknown_or_multiple"
                                                       for row in rows)),
        "scope": manifest["scope"],
        "limitations": ["Eight samples by default; not a calibration or frequency estimate over ideas.",
                        "The same model generates and interprets; agreement is not independent semantic verification.",
                        "Text matches are distinguished from matching terminated token sequences.",
                        "Truncated native prefixes are retained but receive no primary route; truncated extractions are rejected.",
                        "Only one recovered first step receives a route label; multi-step outputs remain unknown."],
    }
    _save(out / "native_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("quantized", "dense-bf16", "dense-fp32"), default="quantized")
    args = parser.parse_args()
    model = MLXModel(args.model, source_repo=args.source_repo, revision=args.revision, device=args.device,
                     execution_mode=args.execution_mode, max_context_tokens=8192)
    run(model, args.out, args.count)


if __name__ == "__main__":
    main()
