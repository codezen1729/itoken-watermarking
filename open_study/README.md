# Detecting a watermark in freely described reasoning

Start with [the measured results](REPORT.md). [The protocol](PROTOCOL.md) records the generation rule, how readers recover reasoning steps, and which pairs qualify for the faithfulness analysis. `calls/` contains the inputs and first outputs of the readers, rewriters, and assessors.

To score new prose, the verifier needs the public problem, text, reader output, encoder checkpoint, target key and 199 surrogate keys. A reader can introduce a mechanism or consequence that never appeared in generation. Source citations allow a verifier to inspect what the reader interpreted. The statistical detector checks that a citation is present verbatim. The separate source review evaluates whether it supports the interpretation.

```sh
python3 open_study/detect_text.py prompt --problem "$public_problem" --text proposal.txt
# Give the printed request to a fresh reader and save the returned object in steps.json.
python3 open_study/detect_text.py score --problem "$public_problem" --text proposal.txt \
  --steps steps.json --keys verifier_keys.json --model "$qwen_checkpoint"
```

The first element of `verifier_keys.json` is the target and the other 199 elements are surrogate keys. Use the exact public problem string. The archived demonstration key sets appear in problem order in `keys.json`. Model installation details are in [the runtime notes](../RESEARCH_RUNTIME.md). Neural re-encoding uses the dense FP32 CPU execution of the pinned Qwen2.5-0.5B-Instruct checkpoint. The stored vector and score audit needs Python and NumPy and runs with `python open_study/audit.py`.
