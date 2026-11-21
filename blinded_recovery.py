"""File transport for isolated stronger-agent proposal and verification calls.

This records tool-enabled LLM judgments, not a token-log-probability API. The
caller must actually run fresh agents against the exported requests. Gold labels,
keys, conditions and previous results are never placed in a request packet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets

from core import make_problem
from recovery_v2 import (make_extraction_prompt, make_verification_prompt,
                         parse_proposal, parse_extraction_output, RecoveryV2Result)
from research_runner import save, digest, result_metrics

ROOT = Path(__file__).resolve().parent
MODEL = {"requested_model": "gpt-6.2-astra", "requested_reasoning_effort": "high",
         "transport": "fresh isolated tool-enabled agent; tools only read prompts and serialize explicit judgments",
         "resolved_model_snapshot_available": False, "token_logprobs_available": False,
         "temperature_and_seed_available": False, "api_finish_reason_available": False,
         "blinding": "procedural file-access restrictions, not an operating-system access guarantee"}
DEPENDENCIES = ("blinded_recovery.py", "recovery_v2.py", "semantic_recovery.py",
                "semantic_recovery_ids.py", "core.py", "research_runner.py")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _response_map(path, requests):
    raw = json.loads(Path(path).read_text())
    if set(raw) != {"responses"} or not isinstance(raw["responses"], list):
        raise ValueError("Response bundle must contain only responses")
    expected = {r["request_id"] for r in requests}
    found = {}
    for row in raw["responses"]:
        if not isinstance(row, dict) or set(row) != {"request_id", "response"}:
            raise ValueError("Each response requires only request_id and response")
        key = row["request_id"]
        if key not in expected or key in found or not isinstance(row["response"], str):
            raise ValueError("Unknown, repeated, or invalid response")
        found[key] = row["response"]
    if set(found) != expected:
        raise ValueError("Missing responses; never silently omit failed cases")
    return found


def prepare(cases, out, *, label, batch_group):
    """Export only source/public problem as prompts; keep this directory private.

    Only the proposals/requests.json file is sent to the proposal agent. For a
    watermark study, caller must separate condition AND variant across contexts.
    """
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh batch directory")
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Cases must be nonempty with unique IDs")
    requests, bindings = [], []
    for c in cases:
        rid = secrets.token_hex(12)
        prompt = make_extraction_prompt(make_problem(c["n_stages"], c["problem_id"]), c["text"])
        requests.append({"request_id": rid, "prompt": prompt,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        bindings.append({"request_id": rid, "case": c})
    freeze = {"protocol": "blinded-agent-recovery-v2", "label": label, "batch_group": batch_group,
              "model": MODEL, "cases_sha256": _hash(cases),
              "source_sha256": {name: digest(ROOT/name) for name in DEPENDENCIES},
              "proposal_requests_sha256": _hash(requests)}
    save(out/"private_manifest.json", {"freeze": freeze, "bindings": bindings})
    save(out/"proposals/requests.json", {"protocol": "blind-agent-inference-v1", "requests": requests})
    return freeze


def _load(out):
    out = Path(out)
    manifest = json.loads((out/"private_manifest.json").read_text())
    if manifest["freeze"]["source_sha256"] != {n: digest(ROOT/n) for n in DEPENDENCIES}:
        raise ValueError("Frozen recovery sources changed")
    req = json.loads((out/"proposals/requests.json").read_text())["requests"]
    if _hash(req) != manifest["freeze"]["proposal_requests_sha256"]:
        raise ValueError("Proposal packet changed")
    if _hash([b["case"] for b in manifest["bindings"]]) != manifest["freeze"]["cases_sha256"]:
        raise ValueError("Private case binding changed")
    return manifest, req


def prepare_verification(out):
    out = Path(out)
    manifest, req = _load(out)
    responses = _response_map(out/"proposals/responses.json", req)
    target = out/"verification/requests.json"
    if target.exists():
        raise ValueError("Verifier packet already exists; no silent rerolls")
    prompts, early = [], {}
    for b in manifest["bindings"]:
        rid, c = b["request_id"], b["case"]
        problem = make_problem(c["n_stages"], c["problem_id"])
        try:
            proposal = parse_proposal(problem, c["text"], responses[rid])
        except (ValueError, TypeError):
            early[rid] = parse_extraction_output(problem, c["text"], responses[rid]).to_dict()
            continue
        if proposal["status"] == "extracted" and proposal["steps"]:
            prompt = make_verification_prompt(problem, c["text"], proposal)
            prompts.append({"request_id": rid, "prompt": prompt,
                            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        else:
            early[rid] = parse_extraction_output(problem, c["text"], responses[rid]).to_dict()
    save(target, {"protocol": "blind-agent-inference-v1", "requests": prompts})
    save(out/"verification/private_binding.json", {"requests_sha256": _hash(prompts),
         "proposal_responses_sha256": digest(out/"proposals/responses.json"), "early_results": early})
    return len(prompts)


def finalize(out):
    out = Path(out)
    manifest, req = _load(out)
    if (out/"results.json").exists():
        raise ValueError("Results already finalized")
    proposals = _response_map(out/"proposals/responses.json", req)
    vreq = json.loads((out/"verification/requests.json").read_text())["requests"]
    binding = json.loads((out/"verification/private_binding.json").read_text())
    if (binding["requests_sha256"] != _hash(vreq)
            or binding["proposal_responses_sha256"] != digest(out/"proposals/responses.json")):
        raise ValueError("Verifier/proposal packet drift")
    verifications = (_response_map(out/"verification/responses.json", vreq) if vreq else {})
    rows = []
    for b in manifest["bindings"]:
        rid, c = b["request_id"], b["case"]
        result = parse_extraction_output(make_problem(c["n_stages"], c["problem_id"]),
                                         c["text"], proposals[rid], verifications.get(rid))
        row = {"id": c["id"], "request_id": rid, "n_stages": c["n_stages"],
               "problem_id": c["problem_id"], "text": c["text"], "recovery": result.to_dict(),
               "proposal_response": proposals[rid], "verification_response": verifications.get(rid)}
        if "expected_canonical" in c:
            row["metrics"] = result_metrics(c, result)
            row["gold"] = {k: c[k] for k in ("expected_status", "expected_canonical", "expected_routes")}
        rows.append(row)
    save(out/"results.json", {"freeze": manifest["freeze"], "rows": rows,
         "proposal_response_artifact_sha256": digest(out/"proposals/responses.json"),
         "verification_response_artifact_sha256": digest(out/"verification/responses.json") if vreq else None})
    return rows
