# HANDOVER — NLP Equations Knowledge Graph project

New chat: read this first, then pipeline_plan.md and issues_log.md. Folder: /Users/hemanthmg/Documents/NLP

## What the project is
Build a deterministic, NON-generative Python pipeline that reads quantum-physics arXiv
papers and extracts their numbered equations into a JSON "equations knowledge graph", with
a per-method audit trail. Exam ID 29. Deadline 25.06.2026 12:00 via Moodle. No prompting /
no text generation anywhere. Use only paper_list_29.txt, in order, for the final dataset.
Per equation the JSON needs: equation (LaTeX), meaning, symbols {sym: desc}, relations
{other: {grade, description}}, audit-trail {method: msg}. Dataset target 350-356 equations
(first 7 per paper). JSON must be valid escaped JSON (\\phi) — build with json.dump, verify
with json.loads. (Prof. confirmed the spec example is mis-escaped.)

## Working method (important)
- Go step by step. Do NOT write or run code until the user says so. Explain first.
- Develop/tune on a small random sample, not the whole list (nothing is trained, so no need
  to make all 350 early). Final dataset = paper_list_29.txt in order, produced once at the end.
- User is a beginner; keep explanations simple and concise.

## Decisions locked
- Source: fetch from arXiv only, flat 6s delay, respect robots.txt, cache. HTML is primary
  (LaTeXML). PDF/LaTeX-source NOT used (robots disallows /src; PDF math extraction is lossy).
- Pipeline A (deterministic core) is the chosen build. No model training in A.
- Equations: extract only enumerated (numbered) equations from arXiv HTML. First 7 per paper
  = dataset; extraction capped at 10 (7 + small review buffer).
- Context window: enclosing paragraph + 1 neighbour each side, trimmed ~2 sentences around
  the target equation, target marked [TARGET], other equations [EQ], inline math kept as $..$.

## Status — DONE and working (offline on cache/)
- robot_fetch.py: fetch (6s, robots, cache, HTML-first, PDF fallback).
- review_equations.py: extract_equations (number, latex, eq_id, in_dataset), classify_paper
  (ok / no_html / degraded_html / no_enumerated_eqs), build_review -> review.html + review.csv.
- context_extract.py: get_contexts -> per-equation context window. Cleans inline-math % wraps
  AND drops unresolved-reference junk (ltx_missing_citation/label, ltx_ERROR undefined like
  \added). Align multi-row fix in place.
- context_review.py: build_review -> combined context_review.txt (first 7 per paper).
- run.py: pick SAMPLE_SIZE seeded-random USABLE papers (walks list, logs skips to
  skipped.csv), builds review.html/csv + context_review.txt. SEED currently 30.
- Reviewed 2 batches (20 papers): equation extraction + context text are high quality and
  generalize. review.csv is paper-wise, id written as "arXiv:..." so spreadsheets keep it text.
- review.html MathJax config defines physics-package macros (\ket, \expectationvalue, ...) for
  PREVIEW only. Extracted LaTeX is kept verbatim — DO NOT modify equation LaTeX.

## NEXT STEP
Symbols (Stage 4). For each equation: (a) list its identifiers from the LaTeX/MathML,
normalise to no-backslash keys (phi not \phi), drop standard operators; (b) find each
symbol's definition in the cleaned context using Hearst-style patterns ("where X is the",
"X denotes") + Schwartz-Hearst abbreviations. Rule-based, non-generative. Then meaning,
relations, audit-trail, JSON assembly.

## Pending decisions (see issues_log.md)
- Equation numbering divergence (HTML number != PDF number): gap (2505.02445), offset
  (2406.03179), scheme/version (2504.02706). Content/order always correct, only the number
  label unreliable. Proposed: for plain-integer papers, skip+log or renumber by document
  order if not contiguous from 1; leave section/subequation/appendix schemes (1.1, 4a, E.1).
  NOT decided/implemented yet.
- Version strategy for fetching (cache holds a mix of v1/v2).

## Reference docs in this folder
- pipeline_plan.md (plan + stages), issues_log.md (all issues by status),
  research_notes.pdf (literature: symbols, relations, audit-trail methods),
  files_pdf/nlp_task.pdf (the assignment).
