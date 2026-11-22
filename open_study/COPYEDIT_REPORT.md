# A lighter rewrite of the detectable original

Two readers independently judged that the copyedit of the marked puzzle proposal preserved its substantive reasoning. A fresh reader recovered twelve reasoning steps from the edit. Its detector p-value rose from 0.010 to 0.015, crossing the fixed 0.01 threshold. The equally detectable unmarked puzzle control followed the same pattern.

| Puzzle-exchange text | Original p | Primary free rewrite p | Lighter copyedit p | Both readers judged copyedit preserved |
|---|---:|---:|---:|---|
| Marked | 0.010 | 0.015 | 0.015 | Yes |
| Unmarked control | 0.010 | 0.015 | 0.015 | Yes |

In the primary marked rewrite one reader flagged the change from reserving the bottom shelf for children's puzzles to placing the puzzles there. The other reader accepted that rewrite. In the light edit both readers found the shelf restriction intact. They also checked who handles damp boxes, how missing pieces are identified, and why the exchange keeps the manufacturer's piece count alongside contributors' descriptions. The full [marked pair](calls/editjudge2/in.json), [first assessment](calls/editjudge2/out.json), [second assessment](calls/editcheck2/out.json) and [fresh extraction](calls/editread2/out.json) are available for inspection.

We selected the only positive original and its paired control after seeing the primary original scores. The [follow-up procedure](COPYEDIT_PROTOCOL.md) was fixed before creating these copyedits. Both 0.010 → 0.015 changes represent one rank in a test with 199 surrogate keys. The identical control behavior and the weak original signal limit the inference to this observed threshold crossing. It supplies one fidelity-confirmed example and no estimate of a population survival rate. All [follow-up scores and judgments](copyedit_results.json) are retained.
