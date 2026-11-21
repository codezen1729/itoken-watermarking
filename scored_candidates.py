"""Aggregate externally scored textual steps into conditional iToken masses.

No model is invoked.  JSONL records have ``text``, ``token_logprobs`` (natural
logs), and ``includes_termination: true``.  The log probabilities must include
the explicit end-of-step token, under one fixed conditioning context and
tokenizer.  Context/tokenizer identities are declarations, not verification of
the external scoring process.  Texts must describe supported, sound inferences.

The output is CONDITIONAL ON THE LISTED CANDIDATES.  It never certifies complete
support or claims to recover an unrestricted LLM's full iToken distribution.
Candidates with duplicate or whitespace-equivalent text are rejected
conservatively.  Their scores are never added together or silently transferred
between whitespace variants; actual model tokenization is not performed here.

Usage: python scored_candidates.py candidates.jsonl --context-id task-1 \
           --tokenizer tokenizer-name-and-version
Use '-' for stdin.  The JSON result is written to stdout.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
import math
from numbers import Real
import sys

from core import parse_step, valid_inference


_MASS_TOLERANCE = 1e-9


def _logsumexp(values: list[float]) -> float:
    largest = max(values)
    return largest + math.log(math.fsum(math.exp(value - largest) for value in values))


def _identity(records: list[Mapping], field: str, declared: str | None) -> str:
    if declared is not None and (not isinstance(declared, str) or not declared.strip()):
        raise ValueError(f"{field} must be a nonempty string")
    if declared is None:
        if any(field not in record for record in records):
            raise ValueError(f"Declare one {field} globally or on every record")
        declared = records[0][field]
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"{field} must be a nonempty string")
    if any(field in record and record[field] != declared for record in records):
        raise ValueError(f"All records must share the declared {field}")
    return declared


def aggregate_scored_candidates(
    records: Iterable[Mapping], *, context_id: str | None = None,
    tokenizer_id: str | None = None,
) -> dict:
    """Sum sequence masses by canonical inference, then condition on the list.

    Each token log probability must be finite and <=0.  Zero-probability
    sequences (containing -infinity) are excluded from this supported-path
    format.  Very small positive masses may underflow in the raw probability
    field; log masses and normalized conditional probabilities remain useful.
    """
    records = list(records)
    if not records or any(not isinstance(record, Mapping) for record in records):
        raise ValueError("Supply at least one JSON object describing a candidate")
    context_id = _identity(records, "context_id", context_id)
    tokenizer_id = _identity(records, "tokenizer_id", tokenizer_id)
    seen_texts: set[str] = set()
    groups: dict[str, dict] = {}
    sequence_logs: list[float] = []
    for index, record in enumerate(records, start=1):
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record {index}: text must be a nonempty string")
        collision_key = " ".join(text.split())
        if collision_key in seen_texts:
            raise ValueError(f"Record {index}: duplicate or whitespace-equivalent text")
        seen_texts.add(collision_key)
        if record.get("includes_termination") is not True:
            raise ValueError(f"Record {index}: includes_termination must explicitly be true")
        values = record.get("token_logprobs")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Record {index}: token_logprobs must be a nonempty list")
        logprobs = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Record {index}: token log probabilities must be numbers")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError(f"Record {index}: token log probabilities must be finite") from exc
            if not math.isfinite(value) or value > 0:
                raise ValueError(f"Record {index}: token log probabilities must be finite and <=0")
            logprobs.append(value)
        try:
            sequence_log = math.fsum(logprobs)
        except OverflowError as exc:
            raise ValueError(f"Record {index}: total log probability exceeds floating-point range") from exc
        if not math.isfinite(sequence_log):
            raise ValueError(f"Record {index}: total log probability must be finite")
        step = parse_step(text)
        if not valid_inference(step):
            raise ValueError(f"Record {index}: the stated inference is not logically valid")
        canonical = step.canonical()
        if canonical not in groups:
            groups[canonical] = {"step": step, "logs": []}
        groups[canonical]["logs"].append(sequence_log)
        sequence_logs.append(sequence_log)

    # Normalize in a shared shifted coordinate system.  Subtracting separately
    # computed absolute log masses would lose log(2), for example, when two
    # equally likely candidates both have a log mass of -1e300.
    reference_log_mass = max(sequence_logs)
    log_scaled_total = _logsumexp([value - reference_log_mass for value in sequence_logs])
    covered_log_mass = reference_log_mass + log_scaled_total
    if covered_log_mass > math.log1p(_MASS_TOLERANCE):
        raise ValueError(
            "Listed sequence probabilities exceed one; scores are incompatible "
            "with disjoint terminated outputs under one fixed context/tokenizer"
        )
    output_groups = []
    for canonical, group in sorted(groups.items()):
        step = group["step"]
        log_scaled_group = _logsumexp([value - reference_log_mass for value in group["logs"]])
        log_mass = reference_log_mass + log_scaled_group
        log_probability = log_scaled_group - log_scaled_total
        output_groups.append({
            "canonical": canonical, "rule": step.rule,
            "premises": list(step.premises), "conclusion": step.conclusion,
            "n_text_candidates": len(group["logs"]),
            "log_listed_probability_mass": log_mass,
            "probability": math.exp(log_probability),
            "log_probability": log_probability,
        })
    return {
        "source": "externally_supplied_token_logprobs", "model_invoked": False,
        "distribution_scope": "conditional_on_listed_candidates",
        "support_status": "not_certified_complete",
        "context_id": context_id, "tokenizer_id": tokenizer_id,
        "n_candidates": len(records), "n_itokens": len(groups),
        "covered_probability_mass": math.exp(covered_log_mass),
        "log_covered_probability_mass": covered_log_mass,
        "itokens": output_groups,
        "assumptions": [
            "External scores use the same fixed context and declared tokenizer.",
            "Every score includes an explicit end-of-step token.",
            "Listed texts represent distinct scored token sequences.",
            "These assertions and the actual tokenization are not independently verified.",
            "Logical validity checks the inference rule, not availability of premises in the context.",
        ],
    }


def _read_jsonl(stream) -> list[dict]:
    records = []
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="JSONL file, or '-' for stdin")
    parser.add_argument("--context-id", required=True, help="Identity of the shared fixed conditioning context")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer identity/version used by the external scorer")
    args = parser.parse_args(argv)
    try:
        if args.input == "-":
            records = _read_jsonl(sys.stdin)
        else:
            with open(args.input, encoding="utf-8") as stream:
                records = _read_jsonl(stream)
        report = aggregate_scored_candidates(
            records, context_id=args.context_id, tokenizer_id=args.tokenizer
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
