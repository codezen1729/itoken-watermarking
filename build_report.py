"""Build a reproducible static research report from existing experiment results.

Usage: python build_report.py --results results --out report
Matplotlib is needed only for reporting; the experimental core uses stdlib.
This script never runs experiments or invokes a model.  The HTML is standalone
and its example selector displays precomputed data, not a live detector.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path

from watermark import wilson_interval


LABELS = {
    "clean": "Original proof", "controlled_rewording": "Controlled rewording",
    "presentation_shuffle": "Presentation shuffle", "duplicate_steps": "Repeated steps",
    "delete_25_percent": "Delete 25% of lines",
    "reroute_25_percent": "Resample 25% of choices", "full_rewrite": "Full unmarked rewrite",
}
SHORT_LABELS = {
    "clean": "Original\nproof", "controlled_rewording": "Controlled\nrewording",
    "presentation_shuffle": "Presentation\nshuffle", "duplicate_steps": "Repeated\nsteps",
    "delete_25_percent": "Delete 25%\nof lines",
    "reroute_25_percent": "Resample 25%\nof choices", "full_rewrite": "Full unmarked\nrewrite",
}


def pct(value, digits=1):
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def interval(row, prefix=""):
    rate = row.get(prefix + "detection_rate")
    if rate is None:
        return "Not eligible"
    return f"{pct(rate)} [{pct(row[prefix + 'ci95_low'])}, {pct(row[prefix + 'ci95_high'])}]"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def html_table(headers, rows):
    return ('<div class="table-wrap"><table><thead><tr>'
            + "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def make_plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#26364a", "text.color": "#18273a",
                         "axes.edgecolor": "#cbd5e1", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.0), gridspec_kw={"width_ratios": [1, 1.28]})
    fig.patch.set_facecolor("white")
    colors = {0: "#8491a3", 1: "#337ac8", 2: "#087f83", 4: "#8356b0"}
    for delta in sorted({row["delta"] for row in summary["conditions"]}):
        rows = sorted((row for row in summary["conditions"] if row["delta"] == delta), key=lambda row: row["n_stages"])
        y = [row["detection_rate"] for row in rows]
        errors = [[max(0.0, row["detection_rate"] - row["ci95_low"]) for row in rows],
                  [max(0.0, row["ci95_high"] - row["detection_rate"]) for row in rows]]
        axes[0].errorbar([row["n_stages"] for row in rows], y, yerr=errors,
                         label="Unmarked (δ = 0)" if delta == 0 else f"δ = {delta}",
                         color=colors.get(delta, "#337ac8"), marker="o", markersize=5,
                         linewidth=2, linestyle="--" if delta == 0 else "-", capsize=4)
    lengths = sorted({row["n_stages"] for row in summary["conditions"]})
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(lengths, [str(value) for value in lengths])
    axes[0].set_xlabel("Structural choices per proof")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_title("More choices provide more detectable signal", loc="left", fontsize=12, pad=14)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)

    attacks = summary["attacks"]
    x = list(range(len(attacks)))
    errors = [[max(0.0, row["detection_rate"] - row["ci95_low"]) for row in attacks],
              [max(0.0, row["ci95_high"] - row["detection_rate"]) for row in attacks]]
    axes[1].bar(x, [row["detection_rate"] for row in attacks], width=.65,
                color=["#087f83"] * 4 + ["#d9a043", "#b77950", "#8491a3"],
                yerr=errors, capsize=4, error_kw={"elinewidth": 1, "ecolor": "#334155"})
    axes[1].set_xticks(x, [SHORT_LABELS[row["attack"]] for row in attacks], rotation=28, ha="right", fontsize=8.6)
    for i, row in enumerate(attacks):
        axes[1].text(i, row["ci95_high"] + .025, pct(row["detection_rate"]), ha="center", fontsize=9)
    cfg = summary["config"]
    axes[1].set_title(f"Editing experiment · {cfg['attack_length']} choices, δ = {cfg['attack_strength']}", loc="left", fontsize=12, pad=14)
    for ax in axes:
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e9edf2", linewidth=.8)
        ax.axhline(cfg["alpha"], color="#adb5c0", linewidth=.8, linestyle=":")
    fig.suptitle("Structural watermarking in a controlled finite model", x=.06, ha="left", fontsize=16, fontweight="bold", y=.98)
    fig.text(.06, .035, f"Key-randomization test · α = {pct(cfg['alpha'], 0)} · {cfg['samples_per_condition']} documents per condition · 95% Wilson intervals · No real LLM", fontsize=9, color="#526379")
    fig.subplots_adjust(left=.06, right=.99, bottom=.25, top=.84, wspace=.22)
    fig.savefig(out / "results.png", dpi=190, facecolor="white")
    fig.savefig(out / "results.svg", facecolor="white")
    plt.close(fig)


def build(results: Path, out: Path):
    summary = json.loads((results / "summary.json").read_text())
    example = json.loads((results / "example_variants.json").read_text())
    cfg, conditions, attacks = summary["config"], summary["conditions"], summary["attacks"]
    out.mkdir(parents=True, exist_ok=True)
    make_plot(summary, out)
    root = Path(__file__).resolve().parent
    link = lambda path: os.path.relpath(path, out).replace(os.sep, "/")
    null = [row for row in conditions if row["delta"] == 0]
    null_hits, null_n = sum(row["detections"] for row in null), sum(row["trials"] for row in null)
    null_ci = wilson_interval(null_hits, null_n)
    by_attack = {row["attack"]: row for row in attacks}
    by_condition = {(row["n_stages"], row["delta"]): row for row in conditions}
    units = cfg["samples_per_condition"] * (len(cfg["lengths"]) + 1)
    evaluations = summary.get("total_document_evaluations", sum(row["trials"] for row in conditions + attacks))
    alternative = cfg.get("likelihood_ratio_alternative_delta")
    secondary = (f"The secondary likelihood-ratio detector uses the same preselected alternative δ = {alternative} for every condition. "
                 if alternative is not None else "This legacy result file uses generation-matched likelihood-ratio alternatives; its δ = 0 secondary test is uninformative. ")
    secondary += ("It is reported only for complete proofs and assumes the exact specified Markov baseline. "
                  "Its guarantee does not extend to arbitrary unknown text distributions. "
                  "The selectively rerouted rows show descriptive likelihood-ratio scores: resampling some choices can change the null path distribution, "
                  "so proof completeness alone does not preserve the original false-positive guarantee.")
    setup = (f"The experiment evaluates an exact finite autoregressive reference distribution, not an LLM. "
             f"There are {cfg['samples_per_condition']} documents per condition, {cfg['surrogate_keys_per_document']} independent surrogate keys per document, "
             f"and a detection threshold α = {pct(cfg['alpha'], 0)}. "
             f"The {evaluations:,} document evaluations come from {units:,} independent task/key units: conditions are paired across strengths, "
             "and editing variants are paired within the attack cohort. They are not 5,700 independent trials.")
    # Avoid carrying a literal evaluation count into reports for different runs.
    setup = setup.replace("5,700 independent trials", f"{evaluations:,} independent trials")
    baseline = ("Each stage offers three sound two-step routes to the same next proposition: an intermediate modus-ponens route, "
                "conjunction introduction followed by modus ponens, or disjunction introduction followed by modus ponens. "
                "Only the first step is a choice; the forced completion adds no likelihood-ratio signal. "
                "Initial route probabilities are (0.55, 0.30, 0.15); later probabilities mix 75% of that baseline with 25% on the preceding route.")
    a = summary["aggregation_check"]
    aggregation = (f"The numerical aggregation check recovers {a['canonical_next_itokens']} canonical iToken masses from {a['textual_outcomes']} "
                   f"terminated textual outcomes. Their token-product mass is {a['token_chain_total_mass']:.12g}; "
                   f"the maximum absolute discrepancy from the analytical route probabilities is {a['max_absolute_error']:.3g}. "
                   "This verifies the finite construction, not probability recovery from an unrestricted language model.")
    low = by_condition.get((min(cfg["lengths"]), 2))
    high = by_condition.get((max(cfg["lengths"]), 2))
    findings = []
    if low and high:
        findings.append(f"At δ = 2, detection rises from {pct(low['detection_rate'])} with {low['n_stages']} choices to {pct(high['detection_rate'])} with {high['n_stages']} choices.")
    findings.append(f"In the {cfg['attack_length']}-choice editing cohort, original, controlled-rewording, shuffled, and repeated-step versions each detect at {pct(by_attack['clean']['detection_rate'])}. This invariance is built into the declared parser and canonicalization.")
    findings.append(f"Deleting 25% of lines retains keyed evidence in {pct(by_attack['delete_25_percent']['detection_rate'])} of documents, while {pct(by_attack['delete_25_percent']['proof_valid_rate'])} remain complete valid proofs. Evidence recovery is not proof validity.")
    findings.append(f"Resampling 25% of choices yields {pct(by_attack['reroute_25_percent']['detection_rate'])} detection; fully resampling from the unmarked model yields {pct(by_attack['full_rewrite']['detection_rate'])}. A resampled choice can equal its original value.")
    null_note = (f"Across the unmarked length conditions, {null_hits}/{null_n} documents detect ({pct(null_hits/null_n, 2)}; "
                 f"95% Wilson interval {pct(null_ci[0], 2)}–{pct(null_ci[1], 2)}). By length: "
                 + "; ".join(f"{row['n_stages']} choices: {row['detections']}/{row['trials']} ({pct(row['detection_rate'], 2)})" for row in null)
                 + ". The pooled interval is a descriptive summary across lengths, not a guarantee for each length or fixed key.")
    validity = ("All full-proof conditions and all non-deletion editing variants have 100% valid proofs in this run. "
                "Deletion variants are incomplete and excluded from the likelihood-ratio analysis.")
    if not all(row["proof_valid_rate"] == 1 for row in conditions + [r for r in attacks if r["attack"] != "delete_25_percent"]):
        validity = "Proof completeness and validity are reported by condition in the tables; incomplete variants are excluded from the likelihood-ratio analysis."
    inference = ("The primary test scores distinct (semantic context, selected iToken) pairs and compares the real-key score with surrogate-key scores. "
                 "The p-value is (1 + number of surrogate scores at least as large)/(R + 1); ties are conservative. "
                 "Repeated steps do not add evidence. With 199 surrogates, p-values have a minimum of 0.005. "
                 "Intervals are 95% Wilson intervals across independent task/key units within each row. Paired rows are correlated, and the intervals do not estimate paired differences.")
    inference = inference.replace("199 surrogates", f"{cfg['surrogate_keys_per_document']} surrogates").replace("0.005", f"{1/(cfg['surrogate_keys_per_document']+1):.6g}")
    limitations = [
        "No real LLM was used. The scorer bridge accepts externally supplied token log probabilities but has only synthetic analytic tests; its output is conditional on the listed candidates.",
        "Only three controlled rendering grammars are recognized. Arbitrary paraphrases, general semantic equivalence, creative ideas, and natural-language extraction are not tested.",
        "Detection requires the original public problem and its stable identifier. Unsupported text, ambiguous routes, and text outside the admitted domain trigger abstention.",
        "The randomization guarantee assumes the document is independent of the randomly selected secret key under the null. It averages over keys and gives no universal fixed-key or adaptive adversarial guarantee.",
        "Strong bias and longer proofs can increase detectability, but idea quality, novelty, and realistic generation costs have not been measured. Independent structural rewriting can erase the signal.",
        "The example key and reproducible simulation seeds are public. This package demonstrates a protocol; its examples cannot provide private provenance or ownership evidence.",
    ]
    condition_headers = ["Choices", "δ", "Detected / n", "Primary rate [95% CI]", "Secondary LLR [95% CI]", "Valid proof"]
    condition_rows = [[row["n_stages"], row["delta"], f"{row['detections']}/{row['trials']}", interval(row), interval(row, "lr_"), pct(row["proof_valid_rate"])] for row in conditions]
    attack_headers = ["Variant", "Detected / n", "Primary rate [95% CI]", "Mean choices", "Valid proof", "Secondary LLR [95% CI]"]
    attack_rows = [[LABELS[row["attack"]], f"{row['detections']}/{row['trials']}", interval(row), f"{row['mean_recovered_choices']:.2f}", pct(row["proof_valid_rate"]), interval(row, "lr_")] for row in attacks]
    resources = [("README and commands", root / "README.md"), ("Formal framework", root / "THEORY.md"),
                 ("Related work", root / "RELATED_WORK.md"), ("Summary data", results / "summary.json"),
                 ("Trial data", results / "trials.csv"), ("Example variants", results / "example_variants.json"),
                 ("External-score bridge", root / "scored_candidates.py")]
    if (root / "validation.txt").exists():
        resources.append(("Validation log", root / "validation.txt"))
    resources.extend([("PNG figure", out / "results.png"), ("SVG figure", out / "results.svg")])
    title = "Structural watermarking in a finite proof domain"
    md = [f"# {title}", "A reproducible research prototype; no real LLM results.", setup,
          "![Detection rates and controlled editing results](results.png)", "## Findings",
          "\n".join(f"- {text}" for text in findings), null_note, validity,
          "## Construction and detection", baseline, aggregation, inference, secondary,
          "## Length and strength results", md_table(condition_headers, condition_rows),
          "## Editing results", md_table(attack_headers, attack_rows),
          "## Example browser", "Open [the standalone HTML report](report.html) to select an editing variant and inspect its proof text and precomputed detection result. The selector does not run a detector.",
          "## Limits", "\n".join(f"- {text}" for text in limitations),
          "## Reproduce and inspect", "\n".join(f"- [{label}]({link(path)})" for label, path in resources)]
    (out / "report.md").write_text("\n\n".join(md) + "\n")
    plot64 = base64.b64encode((out / "results.png").read_bytes()).decode()
    example_payload = json.dumps({"variants": example["variants"], "labels": LABELS}, ensure_ascii=True).replace("<", "\\u003c")
    options = "".join(f'<option value="{html.escape(key)}">{html.escape(LABELS.get(key, key))}</option>' for key in example["variants"])
    p = lambda text: f"<p>{html.escape(text)}</p>"
    lists = lambda texts: "<ul>" + "".join(f"<li>{html.escape(text)}</li>" for text in texts) + "</ul>"
    css = """*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#18273a;font:16px/1.62 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}main{max-width:1180px;margin:auto;background:#fff;padding:48px 54px 60px}h1{font-size:38px;line-height:1.15;letter-spacing:-.8px;max-width:850px;margin:12px 0 16px}h2{font-size:23px;margin:34px 0 12px}p{max-width:1050px}.eyebrow{font-size:12px;letter-spacing:1.8px;text-transform:uppercase;color:#087f83;font-weight:700}.subtle{color:#63738a;font-size:14px}.figure{width:100%;height:auto;margin:18px 0}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}.metric{padding:17px;border:1px solid #dce4ed;border-radius:9px}.metric strong{display:block;font-size:27px}.metric span{font-size:13px;color:#63738a}a{color:#126a9b}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:11px 9px;border-bottom:1px solid #e4eaf0;white-space:nowrap}th{background:#eef3f7;font-size:12px;color:#4c6077}li{margin:8px 0}select{width:100%;max-width:400px;font:inherit;padding:10px 12px;border:1px solid #bac8d8;border-radius:6px;background:white}pre{font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:420px;overflow:auto;padding:18px;background:#f3f6f9;border:1px solid #dce4ed;border-radius:7px;white-space:pre-wrap;overflow-wrap:anywhere}.example{padding:22px;background:#f8fafc;border:1px solid #dce4ed;border-radius:9px}.example .metrics{grid-template-columns:repeat(4,1fr);margin:16px 0}.example .metric{background:#fff;padding:12px}.example .metric strong{font-size:19px}footer{border-top:1px solid #dce4ed;margin-top:30px;padding-top:18px;font-size:13px;color:#63738a}@media(max-width:720px){main{padding:27px 18px}h1{font-size:29px}.metrics,.example .metrics{grid-template-columns:repeat(2,1fr)}th,td{padding:9px 7px}}"""
    js = """const data=JSON.parse(document.getElementById('example-data').textContent);const select=document.getElementById('variant');function show(){const entry=data.variants[select.value],d=entry.detection;document.getElementById('proof').textContent=entry.text;document.getElementById('example-p').textContent=d.p_value==null?'Unavailable':d.p_value.toFixed(3);document.getElementById('example-n').textContent=d.n_choices;document.getElementById('example-valid').textContent=d.proof_valid?'Complete':'Incomplete';document.getElementById('example-decision').textContent=d.detected==null?'Abstained':d.detected?'Detected':'Not detected';document.getElementById('example-status').textContent='Stored extraction status: '+d.status.replaceAll('_',' ');}select.addEventListener('change',show);show();"""
    body = (f'<div class="eyebrow">Research prototype · controlled experiment</div><h1>{title}</h1>'
            + p("A structural signal survives supported changes to presentation; independent rewriting can remove it. These results use a finite reference model, not an LLM.")
            + '<div class="metrics">' + ''.join(f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>' for value, label in [(f"{evaluations:,}", "Document evaluations"), (f"{units:,}", "Independent task/key units"), (pct(null_hits/null_n, 2), "Pooled unmarked detection"), (pct(by_attack['clean']['detection_rate']), "Original editing-cohort detection")]) + '</div>'
            + p(setup) + f'<img class="figure" src="data:image/png;base64,{plot64}" alt="Detection rates by proof length and watermark strength, and across seven editing conditions; all include 95 percent Wilson intervals.">'
            + '<h2>Findings</h2>' + lists(findings) + p(null_note) + p(validity)
            + '<h2>Construction and detection</h2>' + ''.join(map(p, [baseline, aggregation, inference, secondary]))
            + '<h2>Length and strength results</h2>' + html_table(condition_headers, condition_rows)
            + '<h2>Editing results</h2>' + html_table(attack_headers, attack_rows)
            + '<h2>Inspect one example</h2><div class="example"><p><strong>Precomputed example browser.</strong> Selecting a variant displays saved proof text and test results. This page does not run a detector.</p>'
            + f'<label for="variant">Editing variant</label><br><select id="variant">{options}</select>'
            + '<div class="metrics">' + ''.join(f'<div class="metric"><strong id="{ident}">—</strong><span>{label}</span></div>' for ident,label in [("example-p","Stored p-value"),("example-n","Distinct choices"),("example-valid","Proof completeness"),("example-decision","Stored decision")]) + '</div>'
            + '<p id="example-status" class="subtle"></p><pre id="proof" aria-label="Selected proof text"></pre><p class="subtle">The example key is public for reproducibility. This is a protocol demonstration, not a private provenance credential.</p></div>'
            + '<h2>Limits</h2>' + lists(limitations) + '<h2>Reproduce and inspect</h2><ul>'
            + ''.join(f'<li><a href="{html.escape(link(path))}">{html.escape(label)}</a></li>' for label,path in resources) + '</ul>'
            + '<footer>Generated from the supplied summary and example files. Figures and example data are embedded; the report requires no network connection.</footer>')
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>{css}</style></head><body><main>{body}</main><script type="application/json" id="example-data">{example_payload}</script><script>{js}</script></body></html>'
    (out / "report.html").write_text(document)
    return {"report": str(out / "report.html"), "markdown": str(out / "report.md"),
            "figures": [str(out / "results.png"), str(out / "results.svg")],
            "likelihood_ratio_alternative_delta": alternative}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("report"))
    args = parser.parse_args()
    print(json.dumps(build(args.results.resolve(), args.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
