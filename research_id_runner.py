"""Frozen evaluation of the optional inference-ID proposal protocol.

Gold is used only after generation. The original formula-generating extractor
and its first development run remain unchanged for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from core import make_problem
from real_model import MLXModel
from research_runner import ROOT, digest, save, result_metrics, recovery_summary
from semantic_recovery import RecoveryResult
from semantic_recovery_ids import PROMPT_VERSION, make_extraction_prompt, parse_extraction_output


def freeze_record(out, model, dataset, max_tokens):
    current = {
        "extractor_protocol": "inference_ids",
        "extractor_sha256": digest(ROOT / "semantic_recovery_ids.py"),
        "validator_sha256": digest(ROOT / "semantic_recovery.py"),
        "core_sha256": digest(ROOT / "core.py"),
        "adapter_sha256": digest(ROOT / "real_model.py"),
        "runner_sha256": digest(Path(__file__)),
        "metrics_runner_sha256": digest(ROOT / "research_runner.py"),
        "dataset_sha256": digest(dataset),
        "prompt_version": PROMPT_VERSION, "temperature": 0.0,
        "max_generation_tokens": max_tokens, "seed": 0,
        "model_checkpoint_sha256": model.metadata["checkpoint"]["manifest_sha256"],
        "model_metadata_sha256": hashlib.sha256(json.dumps(model.metadata, sort_keys=True).encode()).hexdigest(),
        "model_revision": model.metadata["revision"]}
    target = out / "evaluation_freeze.json"
    if target.exists() and json.loads(target.read_text()) != current:
        raise ValueError("ID evaluation configuration changed; use a new output directory")
    save(target, current)
    return current


def run_recovery(model, out, split, max_tokens=1024):
    if split not in {"dev", "test"}:
        raise ValueError("Split must be dev or test")
    dataset = ROOT / "research_eval_cases.json"
    freeze = freeze_record(out, model, dataset, max_tokens)
    save(out / "model_metadata.json", model.metadata)
    if split == "test" and not (out / "recovery_dev_summary.json").exists():
        raise ValueError("Run the declared dev split before test evaluation")
    cases = [c for c in json.loads(dataset.read_text())["cases"] if c["split"] == split]
    rows = []
    for case in cases:
        target = out / "recovery" / (case["id"] + ".json")
        if target.exists():
            row = json.loads(target.read_text())
            if row["freeze"] != freeze:
                raise ValueError("Saved case has another frozen configuration")
            rows.append(row)
            continue
        problem = make_problem(case["n_stages"], case["problem_id"])
        prompt = make_extraction_prompt(problem, case["text"])
        generated = model.generate([{"role": "user", "content": prompt}],
                                   max_tokens=max_tokens, temperature=0, seed=0)
        recovered = (parse_extraction_output(problem, case["text"], generated["text"])
                     if generated["finish_reason"] == "stop"
                     else RecoveryResult(status="truncated_generation", reason="Model did not produce its stop token",
                                         model_output=generated["text"], prompt_version=PROMPT_VERSION))
        row = {"id": case["id"], "split": split, "category": case["category"], "freeze": freeze,
               "input_text": case["text"], "extraction_prompt": prompt,
               "generation": generated, "recovery": recovered.to_dict(),
               "gold": {k: case[k] for k in ("expected_status", "expected_canonical", "expected_routes")},
               "metrics": result_metrics(case, recovered)}
        save(target, row)
        rows.append(row)
        print("ids", split, case["id"], recovered.status,
              "exact", row["metrics"]["strict_success"], "tokens", generated["generated_tokens"], flush=True)
    summary = {"split": split, "freeze": freeze, "model_metadata": model.metadata,
               **recovery_summary(rows)}
    save(out / f"recovery_{split}_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("split", "strict_accuracy", "step_precision", "step_recall", "false_acceptances")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("quantized", "dense-bf16", "dense-fp32"), default="quantized")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    model = MLXModel(args.model, source_repo=args.source_repo, revision=args.revision,
                     device=args.device, execution_mode=args.execution_mode, max_context_tokens=8192)
    run_recovery(model, args.out, args.split, args.max_tokens)
