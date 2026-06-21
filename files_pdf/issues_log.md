# Issues Log (equation + text extraction)

Recorded during dev on the first sample(s). Group by status so we can optimize after the
next 10-paper batch.

## Fixed
- Align block "(no context found)": multi-row align equations only registered the first
  row, leaving later rows with no context. Fixed by registering every numbered row.
  (was: 2406.03179 E10, 2407.12116 E7)
- physics-package macros rendered red in preview (\expectationvalue, \norm,
  \absolutevalue, \innerproduct, \ket, \bra, ...). Fixed by defining them in the MathJax
  config of review.html. PREVIEW ONLY, extracted LaTeX untouched. (2409.07516, 2402.02373)
- Inline-math % line-wrap artifacts in context: cleaned in context builder (_clean_tex).
- Skip-and-log added with 3 reasons: no_html, degraded_html, no_enumerated_eqs.
- Extraction capped at 10 per paper (dataset stays first 7 + small review buffer).

- Unresolved-reference junk leaking into context text (FIXED): undefined macros
  (\added/\deleted, ltx_ERROR undefined), unresolved \cite (ltx_missing_citation, raw
  keys like patel2024curriculum) and \ref (ltx_missing_label, LABEL:fig:...). Removed at
  the DOM level in the context builder; resolved citations "[23]" kept; prose, math and
  equation LaTeX untouched. (was: 2402.02373, 2411.00230, 2503.02436, 2409.18916)

## Pending decisions
- HTML equation numbers diverge from the PDF (3 forms seen):
  - gap: HTML 1,2,3,4,7,8,11 vs PDF 1-7 (2505.02445, same v2).
  - offset: HTML 5-11 vs PDF 1-7, clean +4 (2406.03179, same v2 -> true HTML/PDF divergence).
  - scheme/version: HTML plain 1,2,3 vs PDF section-based 1.1 (2504.02706; our HTML is v1,
    likely a version difference). User confirmed: extraction itself is correct, only the
    PDF numbering differed.
  In every case the equation CONTENT and ORDER are correct; only the printed NUMBER is
  unreliable as a key. Proposed handling: for plain-integer papers, if the numbers are not
  contiguous from 1, either skip+log or renumber by document order (renumber matches the
  PDF for the confirmed cases). Leave section/subequation/appendix schemes (1.1, 4a, E.1)
  as-is. NOT YET IMPLEMENTED.
- Change-tracking macro leakage: \added / \deleted (and glued forms like \addedthe,
  \deletedof) leak into the context text. Cleanup is context-only, proposed, not done.
  (2402.02373)
- Version strategy for fetching: cache holds a mix of v1 and v2. Decide whether to fetch a
  specific/latest version and record it, so numbers match what graders read.

## Accepted limitations (noted, no action now)
- Run-on paragraphs: some papers pack many equations in one block with almost no full
  stops, so those equations share a wide context with several [EQ] markers; the window
  does not tighten. The [TARGET] marker disambiguates and each definition is still present,
  so rule-based symbol extraction is fine. Later refinement: also cut at equation markers.
  (2406.01298, 2412.06345, 2404.06210 eq 7)
- % line-wrap comment inside equation LaTeX (e.g. \left%): MathJax treats % as a comment
  and renders correctly. Left as-is (do not touch equation data).
- Legitimate non-integer numbering captured correctly (section 1.1, subequations 4a/4b,
  appendix E.1). Not errors.

## Reference notes
- Prof. clarification: dataset JSON must be valid escaped JSON (\\phi). Build with
  json.dump (auto-escapes), verify with json.loads. Never hand-write JSON.
- To draw a fresh 10-paper batch: change SEED in run.py (e.g. 29 -> 30) and rerun.
