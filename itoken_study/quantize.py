"""Public, key-independent semantic iToken quantizer.

Use Hash().codes(embeddings) at inference. Running this module with --fit
rebuilds the planes from previously published readings, without fresh examples.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "bias_study"
BITS = 12
TRAIN = tuple(range(6))
HOLDOUT = (6, 7)
FIELDS = ("quote", "premise", "action", "consequence")
COEFFICIENTS = (0.65, 0.35, 0.50, 0.35)
ARTIFACT = HERE / "public_axes.npz"

# Each row is a permutation from rewritten reader order to original reader order.
# These were mapped by comparing the published actions, before fitting any planes.
REWRITE_TO_ORIGINAL = (
    ("012345", "014253", "210345", "401325", "012345", "051234", "203451", "120345"),
    ("031245", "015432", "041253", "012435", "045132", "031425", "023145", "401325"),
    ("012345", "012345", "102453", "012453", "021345", "031245", "012345", "012345"),
)


def normalize(matrix):
    values = np.asarray(matrix, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def embed_steps(encoder, steps):
    """Embed the cited evidence with roles derived by a fixed, public parser."""
    from canonical import canonical_step
    return _embed_fields(encoder, [canonical_step(s) for s in steps])


def _embed_fields(encoder, steps):
    """Use distinct fixed transforms for evidence, premise, action and consequence."""
    if not steps:
        return np.empty((0, 384), dtype=np.float64)
    texts = list(dict.fromkeys(step[field] for step in steps for field in FIELDS))
    lookup = dict(zip(texts, normalize(encoder.encode(texts)), strict=True))
    total = np.zeros((len(steps), 384), dtype=np.float64)
    for i, (field, coeff) in enumerate(zip(FIELDS, COEFFICIENTS, strict=True)):
        values = np.stack([lookup[s[field]] for s in steps])
        if i:
            rng = np.random.default_rng(70931 + i)
            values = values[:, rng.permutation(values.shape[1])] * rng.choice((-1, 1), values.shape[1])
        total += coeff * values
    return normalize(total)


class Hash:
    """A public fixed binary map; no proposal bank, examples or key at inference."""
    def __init__(self, path=ARTIFACT):
        with np.load(path, allow_pickle=False) as fitted:
            self.axes = fitted["axes"].astype(np.float64)
            self.thresholds = fitted["thresholds"].astype(np.float64)
        if self.axes.shape != (384, BITS) or self.thresholds.shape != (BITS,):
            raise ValueError("Unexpected frozen quantizer dimensions")

    def codes(self, matrix):
        margins = np.asarray(matrix) @ self.axes - self.thresholds
        return ["".join("1" if bit else "0" for bit in row) for row in margins >= 0]


def old_readings():
    """Collect keyed-off step groups using citations and independently assigned alignment."""
    hashes = {}

    def load(rel):
        path = PREVIOUS / rel
        raw = path.read_bytes()
        hashes[str(rel)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    groups = defaultdict(list)
    result = {r["id"]: r for r in load("results.json")}
    sources = load("sources_a.json") + load("sources_b.json")
    for unit in range(8):
        for candidate, row in enumerate(load(f"calls/pool{unit}/out.json")["documents"]):
            if row["id"] != f"s{candidate:02d}":
                raise ValueError("Historical pool identifiers have changed")
            if len(row["steps"]) == 1:
                groups[(unit, candidate)].append(("pool", row["steps"][0]))
    for batch in range(3):
        original = load(f"calls/original{batch}/out.json")["documents"]
        by_id = {d["id"]: d for d in load(f"calls/rewritten{batch}/out.json")["documents"]}
        for doc in original:
            row = result[doc["id"]]
            unit = row["unit"]
            passages = sources[unit]["passages"]
            sigs = []
            for step in doc["steps"]:
                matches = [j for j, passage in enumerate(passages) if step["quote"] in passage]
                if len(matches) != 1:
                    raise ValueError("Unaligned original quotation")
                sigs.append((unit, matches[0]))
            perm = [int(k) for k in REWRITE_TO_ORIGINAL[batch][unit]]
            if sorted(perm) != list(range(len(sigs))) or len(by_id[doc['id']]['steps']) != len(sigs):
                raise ValueError("Invalid historical step alignment")
            for sig, step in zip(sigs, doc["steps"], strict=True):
                if sig in groups:
                    groups[sig].append((doc["id"] + "_original", step))
            if row["faithful"]:
                for i, step in zip(perm, by_id[doc["id"]]["steps"], strict=True):
                    if sigs[i] in groups:
                        groups[sigs[i]].append((doc["id"] + "_rewrite", step))
    return groups, hashes


def select_planes(representatives, alternatives, labels):
    """Choose stable, within-topic discriminative cuts from a seeded public family."""
    n = representatives.shape[0]
    others, owner = alternatives
    rng = np.random.default_rng(42137)
    pool = rng.normal(size=(384, 4096))
    # Discount historical nuisance directions where independent reads disagree.
    diffs = others - representatives[owner]
    _, strength, basis = np.linalg.svd(diffs, full_matrices=False)
    attenuation = strength**2 / (strength**2 + 0.75)
    pool -= basis.T @ (attenuation[:, None] * (basis @ pool))
    pool = normalize(pool.T).T
    scores = representatives @ pool
    thresholds = np.median(scores, axis=0)
    bits = scores >= thresholds
    other_bits = others @ pool >= thresholds
    # Distinct arguments each contribute one vote, including ones with several readings.
    observed = np.flatnonzero(np.bincount(owner, minlength=n))
    error = np.zeros(pool.shape[1])
    for i in observed:
        error += (other_bits[owner == i] != bits[i]).mean(axis=0) / len(observed)
    separation = np.mean([4 * bits[labels == u].mean(0) * (1-bits[labels == u].mean(0))
                          for u in set(labels)], axis=0)
    # Thin-margin planes can flip from a small paraphrase even when sampled bits agree.
    margin = np.sort(np.abs(scores - thresholds), axis=0)[n // 4]
    quality = separation - 3 * error + 2 * margin
    chosen = []
    signed = bits.astype(np.float64)*2 - 1
    max_correlation = np.zeros(pool.shape[1])
    for _ in range(BITS):
        objective = quality - 0.6 * max_correlation
        objective[chosen] = -np.inf
        idx = int(np.argmax(objective))
        chosen.append(idx)
        corr = np.abs(signed[:, idx] @ signed) / n
        max_correlation = np.maximum(max_correlation, corr)
    return pool[:, chosen], thresholds[chosen]


def summarize(matrix, labels, fitted):
    output = {}
    all_codes = fitted.codes(matrix)
    for fold, topics in (("fit", TRAIN), ("held_out", HOLDOUT)):
        counts = defaultdict(int)
        seen = defaultdict(dict)
        for i, (unit, candidate, reading) in enumerate(labels):
            if unit in topics:
                seen[(unit, candidate)][reading] = all_codes[i]
        for (unit, candidate), views in seen.items():
            for reading, code in views.items():
                if reading == "pool":
                    continue
                phase = "rewrite" if reading.endswith("_rewrite") else "original"
                counts[phase + "_pairs"] += 1
                counts[phase + "_same"] += views["pool"] == code
                counts[phase + "_bits_agree"] += sum(a == b for a,b in zip(code,views["pool"]))
                for (u, j), wrong in seen.items():
                    if u == unit and j != candidate:
                        counts[phase + "_wrong_pairs"] += 1
                        counts[phase + "_wrong_same"] += wrong["pool"] == code
        output[fold] = dict(counts)
    return output


def fit(reader_fields=False):
    from encoder import SentenceEncoder
    readings, source_hashes = old_readings()
    labels, inputs = [], []
    for (unit, candidate), values in sorted(readings.items()):
        for reading, step in values:
            labels.append((unit, candidate, reading))
            inputs.append(step)
    make_vectors = _embed_fields if reader_fields else embed_steps
    matrix = make_vectors(SentenceEncoder(), inputs)
    axes_file = HERE / "reader_fields_axes.npz" if reader_fields else ARTIFACT
    report_file = HERE / ("calibration_reader_fields.json" if reader_fields else "calibration.json")
    input_roles = ("Historical reader-authored premise, action and consequence fields; development baseline."
                   if reader_fields else "Deterministic clauses and directed roles from the cited evidence.")
    primary = [i for i,(unit,_,r) in enumerate(labels) if unit in TRAIN and r == "pool"]
    others = [i for i,(unit,_,r) in enumerate(labels) if unit in TRAIN and r != "pool"]
    index = {labels[i][:2]:k for k,i in enumerate(primary)}
    owner = np.array([index[labels[i][:2]] for i in others])
    axes, thresholds = select_planes(matrix[primary], (matrix[others], owner),
                                     np.array([labels[i][0] for i in primary]))
    np.savez_compressed(axes_file, axes=axes, thresholds=thresholds)
    report = dict(train_topics=TRAIN, held_out_topics=HOLDOUT, bits=BITS, fit_scope="bias_study only", input_roles=input_roles,
                  paired_readings="All singly recovered historical pool snippets; selected original readings are matched by source quotation. Rewrites are included only when both archived fidelity readers judged the entire text preserved. Reordered arguments are matched by manual action-based permutations.",
                  denominators="Counts are reading pairs, not independent problems. An old snippet selected in several arms can contribute several readouts. Wrong-step pairs compare each readout to every different single-triple pool candidate within its own problem.",
                  matching="Exact signatures compare the 12 quantizer bits alone. Any additional logical operator fields used by a deployed detector are separate.",
                  source_sha256=source_hashes, counts=summarize(matrix, labels, Hash(axes_file)),
                  quantizer_sha256=hashlib.sha256(axes_file.read_bytes()).hexdigest())
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", required=True)
    parser.add_argument("--reader-fields", action="store_true", help="rebuild the historical reader-wording development baseline")
    fit(reader_fields=parser.parse_args().reader_fields)
