"""Actual local MLX model scoring and generation, with explicit probability scope.

The scorer uses learned next-token logits at temperature one. Each candidate
ends with one declared EOS token; scores are sums of token log probabilities,
never length-normalized scores. The resulting iToken distribution is conditional
on the supplied finite candidate set. No full semantic-support claim is made.

MLX and mlx-lm are imported only when MLXModel is constructed. A local checkpoint
directory is required, so importing or constructing this adapter never silently
downloads model weights. Unit tests of numerical utilities use no model; actual
model runs record checkpoint hashes, versions, token IDs and token logprobs.

Primary implementation references (consulted 2026-11-21):
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/evaluate.py
https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py
https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Mapping, Sequence


def _logsumexp(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Supply nonempty finite log probabilities")
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def selected_logprobs(logits: Sequence[Sequence[float]], token_ids: Sequence[int],
                      prefix_length: int) -> list[float]:
    """Reference causal indexing: row j predicts token j+1.

    This stdlib numerical oracle is used by unit tests, not for production
    inference. token_ids includes the prefix and the terminated continuation.
    """
    if isinstance(prefix_length, bool) or not 1 <= prefix_length < len(token_ids):
        raise ValueError("A nonempty prefix and continuation are required")
    if len(logits) != len(token_ids) - 1:
        raise ValueError("Expected one logits row per input token except the last")
    scores = []
    for position in range(prefix_length, len(token_ids)):
        row = logits[position - 1]
        target = token_ids[position]
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < len(row):
            raise ValueError("Target token outside vocabulary")
        scores.append(float(row[target]) - _logsumexp(row))
    return scores


def continuation_tokens(tokenizer, prompt_text: str, text: str,
                        eos_token_id: int, stop_token_ids: set[int]) -> tuple[list[int], list[int]]:
    """Tokenize concatenated text and reject a shifting prefix boundary.

    Encoding prefix and continuation independently can assign the wrong
    conditional event to a BPE model. We verify the complete sequence starts
    with the exact fixed-prefix token sequence before slicing its continuation.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Candidate text must be a nonempty string")
    prefix = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    full = list(tokenizer.encode(prompt_text + text, add_special_tokens=False))
    if not prefix or full[:len(prefix)] != prefix:
        raise ValueError("Candidate changes the fixed prompt tokenization boundary")
    body = full[len(prefix):]
    if not body or any(token in stop_token_ids for token in body):
        raise ValueError("Candidate body is empty or contains an early stopping token")
    return prefix, body + [eos_token_id]


def aggregate_sequence_scores(records: Sequence[Mapping]) -> dict:
    """Aggregate disjoint scored token sequences by caller-supplied iToken ID.

    iToken labels must be checked by the surrounding logical protocol. The
    scorer does not establish that a free textual candidate expresses its label.
    """
    if not records:
        raise ValueError("At least one scored candidate is required")
    seen_tokens, logs, groups = set(), [], {}
    for record in records:
        ids = tuple(record["token_ids"])
        if not ids or ids in seen_tokens:
            raise ValueError("Empty or duplicate terminated token sequence")
        seen_tokens.add(ids)
        values = list(record["token_logprobs"])
        if len(values) != len(ids) or any(not math.isfinite(x) or x > 1e-6 for x in values):
            raise ValueError("Every token needs a finite nonpositive model log probability")
        label = record["itoken_id"]
        if not isinstance(label, str) or not label:
            raise ValueError("iToken IDs must be nonempty strings")
        value = math.fsum(values)
        if not math.isfinite(value):
            raise ValueError("Sequence log probability exceeds floating-point range")
        logs.append(value)
        groups.setdefault(label, []).append(value)
    shift = max(logs)
    scaled_total = _logsumexp([x - shift for x in logs])
    covered_log_mass = shift + scaled_total
    if covered_log_mass > math.log1p(1e-5):
        raise ValueError("Disjoint candidate mass exceeds one; scores or termination are inconsistent")
    group_logs = {label: _logsumexp([x - shift for x in group]) - scaled_total
                  for label, group in groups.items()}
    return {
        "distribution_scope": "conditional_on_listed_terminated_token_sequences",
        "support_status": "not_certified_complete",
        "covered_probability_mass": math.exp(covered_log_mass),
        "log_covered_probability_mass": covered_log_mass,
        "candidate_probabilities": [math.exp((x - shift) - scaled_total) for x in logs],
        "itoken_probabilities": {label: math.exp(value) for label, value in group_logs.items()},
        "itoken_log_probabilities": group_logs,
        "candidate_labels_verified_by_scorer": False,
    }


def checkpoint_manifest(path: Path, *, hash_weights: bool = True) -> dict:
    """Record source bytes without exporting the user's absolute local path."""
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
             "merges.txt", "special_tokens_map.json", "added_tokens.json",
             "generation_config.json", "model.safetensors.index.json", "download_manifest.json")
    paths = [path / name for name in names if (path / name).is_file()]
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise ValueError("Local checkpoint contains no safetensors weights")
    if hash_weights:
        paths += weights
    files = []
    for item in paths:
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        files.append({"name": item.name, "size_bytes": item.stat().st_size,
                      "sha256": digest.hexdigest()})
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"files": files, "weights_hashed": hash_weights,
            "weights_size_bytes": sum(x.stat().st_size for x in weights),
            "manifest_sha256": hashlib.sha256(serialized).hexdigest()}


class MLXModel:
    """One locally loaded trained model, reused for scoring and generation."""

    def __init__(self, local_path: str | Path, *, source_repo: str | None = None,
                 revision: str | None = None, max_context_tokens: int = 4096,
                 hash_weights: bool = True, device: str = "auto",
                 execution_mode: str = "quantized", prefill_chunk_size: int = 128):
        path = Path(local_path).expanduser().resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError("Supply an existing local model directory with config.json")
        if max_context_tokens < 2:
            raise ValueError("max_context_tokens must be at least two")
        if device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device must be auto, cpu, or gpu")
        if execution_mode not in {"quantized", "dense-bf16", "dense-fp32"}:
            raise ValueError("execution_mode must be quantized, dense-bf16, or dense-fp32")
        if isinstance(prefill_chunk_size, bool) or not isinstance(prefill_chunk_size, int) or prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be a positive integer")
        self.prefill_chunk_size = prefill_chunk_size
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as exc:
            raise RuntimeError(f"Cannot import the MLX/MLX-LM runtime: {exc}") from exc
        if device == "cpu":
            mx.set_default_device(mx.cpu)
        elif device == "gpu":
            mx.set_default_device(mx.gpu)
        elif not mx.metal.is_available():
            mx.set_default_device(mx.cpu)
        self.mx, self.nn, self.make_prompt_cache = mx, nn, make_prompt_cache
        self.model, self.tokenizer = load(str(path), tokenizer_config={"trust_remote_code": False})
        representation = self._execution_representation(execution_mode)
        self.max_context_tokens = max_context_tokens
        config = json.loads((path / "config.json").read_text())
        eos = self.tokenizer.eos_token_id
        if not isinstance(eos, int):
            raise ValueError("Tokenizer must declare one primary EOS token ID")
        self.eos_token_id = eos
        self.stop_token_ids = set(getattr(self.tokenizer, "eos_token_ids", None) or [eos]) | {eos}
        packages = {}
        for name in ("mlx", "mlx-lm", "transformers", "tokenizers"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "unknown"
        self.metadata = {
            "source_repo": source_repo, "revision": revision,
            "local_directory_name": path.name, "model_type": config.get("model_type"),
            "quantization": config.get("quantization", config.get("quantization_config")),
            "packages": packages, "python": platform.python_version(),
            "device": str(mx.default_device()),
            "max_context_tokens": self.max_context_tokens,
            "prefill_chunk_size": self.prefill_chunk_size,
            "prefill_strategy": "evaluate cache states in chunks, then final prompt token logits",
            "execution_mode": execution_mode, "execution_representation": representation,
            "checkpoint": checkpoint_manifest(path, hash_weights=hash_weights),
            "scoring_temperature": 1.0, "scoring_eos_token_id": eos,
            "generation_stop_token_ids": sorted(self.stop_token_ids),
            "model_invoked": True,
        }

    def _execution_representation(self, mode: str) -> dict:
        """Explicitly derive dense runtime parameters from the packed checkpoint.

        This changes arithmetic/rounding and is a separately identified model
        execution variant. It does not recover the original unquantized weights.
        """
        if mode == "quantized":
            return {"transform": "none; use checkpoint quantized modules",
                    "parameter_sha256": None}
        import numpy as np
        from mlx.utils import tree_flatten, tree_unflatten
        mx, nn = self.mx, self.nn
        dtype = mx.float32 if mode == "dense-fp32" else mx.bfloat16
        names = [name for name, module in self.model.named_modules()
                 if isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding))]
        if not names:
            raise ValueError("Dense execution requires supported quantized layers")
        for name in names:
            module = self.model
            for component in name.split("."):
                module = module[int(component)] if isinstance(module, list) else module[component]
            if getattr(module, "mode", "affine") != "affine":
                raise ValueError("Dense execution currently supports affine quantization only")
            weight = mx.dequantize(
                module.weight, module.scales.astype(dtype), module.biases.astype(dtype),
                group_size=module.group_size, bits=module.bits, mode="affine")
            mx.eval(weight)
            if isinstance(module, nn.QuantizedEmbedding):
                replacement = nn.Embedding(weight.shape[0], weight.shape[1])
            else:
                replacement = nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in module)
                if "bias" in module:
                    replacement.bias = module.bias.astype(dtype)
            replacement.weight = weight
            self.model.update_modules(tree_unflatten([(name, replacement)]))
        self.model.set_dtype(dtype)
        self.model.eval()
        parameters = sorted(tree_flatten(self.model.parameters()))
        mx.eval([value for _, value in parameters])
        digest, count, byte_count = hashlib.sha256(), 0, 0
        for name, value in parameters:
            descriptor = json.dumps({"name": name, "dtype": str(value.dtype),
                                     "shape": list(value.shape)},
                                    sort_keys=True, separators=(",", ":")).encode()
            digest.update(len(descriptor).to_bytes(8, "little"))
            digest.update(descriptor)
            # Raw byte views avoid converting BF16 values or materializing a
            # second full-size float32 copy merely to identify the parameters.
            raw = np.asarray(value.view(mx.uint8))
            digest.update(memoryview(raw).cast("B"))
            count += value.size
            byte_count += value.nbytes
        mx.clear_cache()
        return {"transform": "cast affine scales/biases to execution dtype, dequantize packed weights, replace QuantizedLinear/QuantizedEmbedding, cast all floating parameters",
                "execution_dtype": str(dtype), "dequantized_layers": len(names),
                "parameter_sha256": digest.hexdigest(), "parameter_count": count,
                "parameter_bytes": byte_count, "hash_byte_order": sys.byteorder,
                "probability_equivalence_to_quantized_execution_claimed": False}

    def _prompt(self, messages: Sequence[Mapping], chat_template_kwargs: dict | None = None):
        if not messages or any(m.get("role") not in {"system", "user", "assistant"}
                               or not isinstance(m.get("content"), str) for m in messages):
            raise ValueError("Supply chat messages with explicit roles and string content")
        kwargs = dict(chat_template_kwargs or {})
        if {"tokenize", "add_generation_prompt", "return_dict"} & kwargs.keys():
            raise ValueError("Reserved chat-template setting")
        text = self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, **kwargs)
        ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        if not ids or len(ids) >= self.max_context_tokens:
            raise ValueError("Prompt is empty or exceeds the declared context limit")
        digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        return text, ids, digest

    def _prefill(self, prefix: list[int]):
        cache = self.make_prompt_cache(self.model)
        # Match mlx_lm.generate.generate_step: evaluating only cache states lets
        # MLX skip discarded vocabulary-head logits for earlier prompt tokens.
        # Keep the final token separate to obtain exactly the next-token logits.
        for start in range(0, len(prefix) - 1, self.prefill_chunk_size):
            end = min(start + self.prefill_chunk_size, len(prefix) - 1)
            self.model(self.mx.array(prefix[start:end])[None], cache=cache)
            self.mx.eval([item.state for item in cache])
            self.mx.clear_cache()
        logits = self.model(self.mx.array(prefix[-1:])[None], cache=cache)
        last = self.nn.log_softmax(logits[0, -1, :].astype(self.mx.float32))
        self.mx.eval(last, [item.state for item in cache])
        return last, cache

    def _score_continuation(self, continuation: list[int], first_logprobs, prefix_cache):
        result = [float(first_logprobs[continuation[0]].item())]
        if len(continuation) > 1:
            cache = copy.deepcopy(prefix_cache)
            inputs = self.mx.array(continuation[:-1])[None]
            targets = self.mx.array(continuation[1:])[None, :, None]
            logits = self.model(inputs, cache=cache).astype(self.mx.float32)
            probabilities = self.nn.log_softmax(logits, axis=-1)
            selected = self.mx.take_along_axis(probabilities, targets, axis=-1)[0, :, 0]
            self.mx.eval(selected)
            result += selected.tolist()
        return result

    def _uncached_scores(self, prefix, continuation):
        complete = prefix + continuation
        logits = self.model(self.mx.array(complete[:-1])[None])
        selected_logits = logits[0, len(prefix) - 1:, :].astype(self.mx.float32)
        logprobs = self.nn.log_softmax(selected_logits, axis=-1)
        targets = self.mx.array(continuation)[:, None]
        selected = self.mx.take_along_axis(logprobs, targets, axis=-1)[:, 0]
        self.mx.eval(selected)
        return selected.tolist()

    def score_candidates(self, messages: Sequence[Mapping], candidates: Sequence[Mapping], *,
                         chat_template_kwargs: dict | None = None,
                         verify_first_uncached: bool = False,
                         verify_all_uncached: bool = False) -> dict:
        """Score candidate {text, itoken_id} objects under one fixed chat prefix.

        Uncached verification independently computes complete-sequence logits.
        ``uncached_crosscheck`` retains the first checked candidate's record;
        ``uncached_crosschecks`` contains every checked candidate. With
        verify_all_uncached, callers must evaluate that list to bound all forks.
        """
        start = time.monotonic()
        prompt_text, prefix, context_id = self._prompt(messages, chat_template_kwargs)
        if not candidates:
            raise ValueError("Supply candidate textual steps with iToken labels")
        prepared, seen = [], set()
        for candidate in candidates:
            label = candidate.get("itoken_id")
            if not isinstance(label, str) or not label:
                raise ValueError("Each candidate requires a nonempty itoken_id")
            checked_prefix, continuation = continuation_tokens(
                self.tokenizer, prompt_text, candidate.get("text"), self.eos_token_id,
                self.stop_token_ids)
            if checked_prefix != prefix:
                raise AssertionError("Candidate scoring contexts differ")
            if len(prefix) + len(continuation) > self.max_context_tokens:
                raise ValueError("Candidate exceeds the declared context limit; no truncation is performed")
            if tuple(continuation) in seen:
                raise ValueError("Candidate list duplicates a terminated token sequence")
            seen.add(tuple(continuation))
            prepared.append((candidate, continuation))
        first_logprobs, cache = self._prefill(prefix)
        records = []
        verifications = []
        for index, (candidate, continuation) in enumerate(prepared):
            scores = self._score_continuation(continuation, first_logprobs, cache)
            if verify_all_uncached or (verify_first_uncached and index == 0):
                independent = self._uncached_scores(prefix, continuation)
                if len(scores) != len(independent):
                    raise AssertionError("Cached and uncached scorers selected different token counts")
                differences = [a - b for a, b in zip(scores, independent)]
                verifications.append({
                    "max_absolute_token_logprob_difference": max(map(abs, differences)),
                    "absolute_sequence_logprob_difference": abs(math.fsum(differences)),
                    "sum_absolute_token_logprob_differences": math.fsum(map(abs, differences)),
                    "uncached_sequence_log_probability": math.fsum(independent),
                    "uncached_token_logprobs": independent,
                    "scored_tokens": len(scores),
                    "checked_candidate_index": index,
                    "scope": "first_candidate_only" if index == 0 else "individual_candidate",
                })
            records.append({"text": candidate["text"], "itoken_id": candidate["itoken_id"],
                            "token_ids": continuation, "token_logprobs": scores,
                            "sequence_log_probability": math.fsum(scores),
                            "includes_termination": True, "termination_token_id": self.eos_token_id,
                            "context_id": context_id,
                            "tokenizer_id": self.metadata["checkpoint"]["manifest_sha256"]})
        aggregation = aggregate_sequence_scores(records)
        self.mx.clear_cache()
        return {"metadata": self.metadata, "context_id": context_id,
                "prompt_token_ids": prefix, "serialized_prompt": prompt_text,
                "candidates": records, **aggregation,
                "uncached_crosscheck": verifications[0] if verifications else None,
                "uncached_crosschecks": verifications,
                "uncached_crosscheck_scope": ("all_candidates" if verify_all_uncached else
                                              "first_candidate_only" if verify_first_uncached else "disabled"),
                "wall_seconds": time.monotonic()-start}

    def generate(self, messages: Sequence[Mapping], *, max_tokens: int = 256,
                 temperature: float = 0.0, seed: int = 0,
                 chat_template_kwargs: dict | None = None) -> dict:
        """Generate unrestricted text; report actual stop or length truncation.

        token_logprobs always refer to the untempered base model. Sampling log
        probabilities are separate because greedy/tempered decoding is a
        different distribution. No constraint grammar or top-k filtering is used.
        """
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        start = time.monotonic()
        _, prefix, context_id = self._prompt(messages, chat_template_kwargs)
        if len(prefix) + max_tokens > self.max_context_tokens:
            raise ValueError("Generation would exceed the declared context limit")
        self.mx.random.seed(seed)
        logprobs, cache = self._prefill(prefix)
        tokens, model_scores, sampling_scores = [], [], []
        reason = "length"
        for index in range(max_tokens):
            if temperature == 0:
                token = int(self.mx.argmax(logprobs).item())
                sampling_score = 0.0
            else:
                sampling_logprobs = self.nn.log_softmax(logprobs / temperature)
                token = int(self.mx.random.categorical(sampling_logprobs).item())
                sampling_score = float(sampling_logprobs[token].item())
            tokens.append(token)
            model_scores.append(float(logprobs[token].item()))
            sampling_scores.append(sampling_score)
            if token in self.stop_token_ids:
                reason = "stop"
                break
            if index + 1 < max_tokens:
                logits = self.model(self.mx.array([[token]]), cache=cache)
                logprobs = self.nn.log_softmax(logits[0, -1, :].astype(self.mx.float32))
                self.mx.eval(logprobs)
        body = tokens[:-1] if reason == "stop" else tokens
        text = self.tokenizer.decode(body, skip_special_tokens=False)
        self.mx.clear_cache()
        return {"text": text, "token_ids": tokens, "token_logprobs": model_scores,
                "sampling_token_logprobs": sampling_scores,
                "model_sequence_log_probability": math.fsum(model_scores),
                "sampling_sequence_log_probability": math.fsum(sampling_scores),
                "finish_reason": reason, "includes_termination": reason == "stop",
                "temperature": temperature, "seed": seed, "context_id": context_id,
                "prompt_tokens": len(prefix), "generated_tokens": len(tokens),
                "wall_seconds": time.monotonic()-start,
                "model_checkpoint_sha256": self.metadata["checkpoint"]["manifest_sha256"],
                "model_invoked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="JSON with messages and candidates, or generation settings")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("score", "generate"), default="score")
    parser.add_argument("--source-repo")
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--execution-mode", choices=("quantized", "dense-bf16", "dense-fp32"), default="quantized")
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    args = parser.parse_args()
    request = json.loads(args.input.read_text())
    model = MLXModel(args.model, source_repo=args.source_repo, revision=args.revision,
                     device=args.device, execution_mode=args.execution_mode,
                     prefill_chunk_size=args.prefill_chunk_size)
    if args.mode == "score":
        result = model.score_candidates(request["messages"], request["candidates"], verify_first_uncached=True,
                                        chat_template_kwargs=request.get("chat_template_kwargs"))
    else:
        result = model.generate(request["messages"], max_tokens=request.get("max_tokens", 256),
                                temperature=request.get("temperature", 0.0), seed=request.get("seed", 0),
                                chat_template_kwargs=request.get("chat_template_kwargs"))
        result["metadata"] = model.metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "model_invoked": True,
                      "mode": args.mode, "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
