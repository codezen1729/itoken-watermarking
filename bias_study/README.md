# Replaying the stronger catalog-free watermark

[Read the experimental results](REPORT.md) or inspect the [frozen design](PROTOCOL.md). The complete fresh cohort contains eight problems and 192 candidate arguments. The calls directory contains the source packets as actually presented and the agents’ first submitted responses. The demonstration keys are public for reproducibility. They provide no attribution security once published.

From the repository root, run the archived numerical audit with Python and NumPy.

```sh
python3 bias_study/audit.py
```

To score your own prose, use the encoder in [the runtime notes](../RESEARCH_RUNTIME.md). Give the first command's JSON to a fresh reader and save its response as an object containing a steps array. The verifier's JSON key file is the list for a single problem, with the target key first and all 199 surrogates following it.

```sh
python3 bias_study/detect.py prompt --problem "$public_problem" --text proposal.txt
python3 bias_study/detect.py score --problem "$public_problem" --text proposal.txt \
  --steps steps.json --keys verifier_keys.json --model "$qwen_checkpoint"
```

A full model replay of the submitted document readings uses `python3 bias_study/experiment.py evaluate --model "$qwen_checkpoint"`. It regenerates the document vectors and the result table from those readings. A new blind reader can give a different extraction. To reproduce the original numerical ranks without model inference, the audit checks the stored vectors, keys and counts.
