# A catalog-free iToken for open reasoning steps

[Results](REPORT.md) · [Frozen design](PROTOCOL.md) · [Numerical outcomes](summary.json) · [Blind readers’ inputs and outputs](calls/)

The verifier uses the public question, prose, a blind reader’s supported steps, public code and coefficients, and the verifier’s keys. It can score a previously unseen step. A complete ID contains one polarity bit and five semantic bits. The exact detector keys the whole ID. The tolerant detector keys overlapping three-bit portions of the **same ID**.

From repository root, generate a request for a new blind reader. Save the reader’s reply as an object with a `steps` array. Use a JSON list containing the verifier's target key followed by 199 surrogate keys. For reproduction, the lists in `keys.json` are public demonstration keys for the eight problems in source order.

```sh
python itoken_study/detect.py prompt --problem "$public_problem" --text proposal.txt
python itoken_study/detect.py score --problem "$public_problem" --text proposal.txt \
  --steps reading.json --keys verifier_keys.json --mode exact
```

For neural re-encoding, set `ITOKEN_MINILM_DIR` to a local directory containing the pinned `tokenizer.json` and `onnx/model.onnx`. The loader verifies both hashes. The run used ONNX Runtime 1.23.2 on CPU, NumPy and Hugging Face Tokenizers. The sparse keyed detector and archived-ID replay do not require a hosted API. The independent research readers were LLM agents. New readings may differ. [Pinned MiniLM checkpoint](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

The optional training command `python itoken_study/quantize.py --fit` refits the public thresholds on earlier research materials. Deployment only loads the saved axes. The Qwen weights came from the pinned local model used in the previous experiment. [Qwen runtime](../RESEARCH_RUNTIME.md)

Recompute the outcome table from the archived readings with `python itoken_study/endpoints.py`. Run `python itoken_study/audit.py` for an independent numerical replay using NumPy. Add `--reencode` to audit every ID from its quotation using the local sentence encoder. [Full replay record](independent_replay.json)
