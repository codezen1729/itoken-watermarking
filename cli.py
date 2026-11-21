"""Generate, reword and detect a controlled proof. Python standard library only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from core import make_problem, render_proof
from pipeline import demo_key, diagnose, generate, surrogate_keys, transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Create a proof plus edits and detector results")
    demo.add_argument("--out", type=Path, default=Path("demo"))
    demo.add_argument("--stages", type=int, default=32)
    demo.add_argument("--delta", type=float, default=2)
    demo.add_argument("--seed", type=int, default=20261121)
    detect = commands.add_parser("detect", help="Detect using the supplied original problem and key")
    detect.add_argument("--manifest", type=Path, required=True)
    detect.add_argument("--text", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        problem = make_problem(args.stages, f"demo-{args.seed}")
        key = demo_key(args.seed)
        calibration_seed = args.seed + 700
        bank = surrogate_keys(calibration_seed, 199)
        produced = generate(problem, key, args.delta, random.Random(args.seed))
        args.out.mkdir(parents=True, exist_ok=True)
        manifest = {"n_stages": args.stages, "problem_id": problem.problem_id,
                    "premises": sorted(problem.premises), "goal": problem.goal,
                    "delta": args.delta, "key_hex": key.hex(),
                    "calibration_seed": calibration_seed, "calibration_keys": 199,
                    "warning": "Public reproducibility key; this is a controlled reference model, not an LLM"}
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        results = {}
        for j, attack in enumerate(("clean", "controlled_rewording", "presentation_shuffle",
                                    "duplicate_steps", "delete_25_percent", "reroute_25_percent", "full_rewrite")):
            text = transform(problem, produced, attack, random.Random(args.seed + j + 1))
            (args.out / f"{attack}.txt").write_text(text + "\n")
            results[attack] = diagnose(problem, text, key, bank, delta=args.delta)
        (args.out / "detection.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
    else:
        manifest = json.loads(args.manifest.read_text())
        problem = make_problem(manifest["n_stages"], manifest["problem_id"])
        # The manifest must name this domain exactly; silently changing premises
        # while retaining a watermark context is not accepted.
        if set(manifest.get("premises", [])) != set(problem.premises) or manifest.get("goal") != problem.goal:
            parser.error("Manifest premises/goal do not match the controlled problem")
        result = diagnose(problem, args.text.read_text(), bytes.fromhex(manifest["key_hex"]),
                          surrogate_keys(manifest["calibration_seed"], manifest["calibration_keys"]),
                          delta=manifest["delta"])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
