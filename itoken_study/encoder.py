"""Local 384-D sentence embeddings from a pinned, unquantized MiniLM checkpoint."""
from collections import defaultdict
from hashlib import file_digest
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
FILE_HASHES = {
    "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
}
DIMENSION = 384
MAX_TOKENS = 256
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "work/models/all-MiniLM-L6-v2"


class SentenceEncoder:
    """Mean-pool contextual token states, then L2 normalize, as in the released model."""

    def __init__(self, path=None):
        directory = Path(path or os.environ.get("ITOKEN_MINILM_DIR", DEFAULT_PATH))
        for name, expected in FILE_HASHES.items():
            with (directory / name).open("rb") as source:
                if file_digest(source, "sha256").hexdigest() != expected:
                    raise ValueError(f"Unexpected checksum for {directory / name}")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.no_padding()
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 4
        settings.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "onnx/model.onnx"), sess_options=settings,
            providers=["CPUExecutionProvider"],
        )
        self.metadata = {
            "repository": REPOSITORY, "revision": REVISION, "sha256": dict(FILE_HASHES),
            "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider",
            "max_tokens": MAX_TOKENS, "dimension": DIMENSION,
        }

    def encode(self, texts):
        if isinstance(texts, str):
            raise TypeError("encode expects a sequence of strings")
        texts = list(texts)
        result = np.empty((len(texts), DIMENSION), dtype=np.float32)
        if not texts:
            return result
        # Sentence length fixes padding independently of other items in the corpus.
        buckets = defaultdict(list)
        for i, tokenization in enumerate(self.tokenizer.encode_batch(texts)):
            ids = tokenization.ids
            width = ((len(ids) + 7) // 8) * 8
            buckets[width].append((i, ids))
        for width, group in buckets.items():
            for start in range(0, len(group), 32):
                batch = group[start:start + 32]
                ids = np.zeros((len(batch), width), dtype=np.int64)
                mask = np.zeros_like(ids)
                for row, (_, sequence) in enumerate(batch):
                    ids[row, :len(sequence)] = sequence
                    mask[row, :len(sequence)] = 1
                embeddings, = self.session.run(["last_hidden_state"], {
                    "input_ids": ids, "attention_mask": mask,
                    "token_type_ids": np.zeros_like(ids),
                })
                totals = np.sum(embeddings * mask[..., None].astype(np.float32), axis=1)
                vectors = totals / mask.sum(axis=1, keepdims=True).astype(np.float32)
                indices = [i for i, _ in batch]
                result[indices] = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return result
