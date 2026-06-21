## Imported Claude Cowork project instructions

PROJECT: Equations Knowledge Graph (NLP Project Work, OTH Amberg-Weiden, Summer 2026, Prof. Levi)
Exam ID: 29. Deadline 25.06.2026 12:00 via Moodle.

GOAL
Build a Python system that extracts enumerated equations from arbitrary quantum physics arxiv papers and builds an equations knowledge graph as JSON, with a per method audit trail.

HARD RULES (violation = fail or downgrade)
1. Use only paper_list_29.txt. Process papers in exact listed order. No reordering, no substitutions.
2. Extraction sources: arxiv PDF and HTML only. LaTeX source (/src, /e-print) is forbidden because arxiv robots.txt disallows it and robots.txt has priority over the task text. Do not download or parse the e-print tarball.
3. No prompting and no text generation, at all, even if the output is a single word. No ChatGPT, Claude, Perplexity, external APIs, or any local or self hosted (L)LM used via prompting. Non generative LM use is allowed: embeddings, vector similarity, token classification, encoders that emit labels or vectors. Any such model must run locally on DC1.07 for reproducibility.
4. Respect robots.txt. Use sleep delays between requests. Reuse robot_fetch.py.
5. Code in Python, runs on GPU lab DC 1.07.

DATASET SCOPE
Target 350 to 356 equations. Per paper take the first 7 enumerated equations (fewer if paper has fewer). Keep adding papers while total < 350. Once total reaches 350, finish the current paper fully then stop, so final count can exceed 350.
Only enumerated equations count (number in brackets, e.g. (1)). Ignore unnumbered equations.

INPUT STRATEGY (revised after robots.txt clarification)
Primary source: arxiv HTML version (LaTeXML rendered, contains MathML and printed equation numbers). Use it to get the equation LaTeX/MathML and the canonical printed number.
Fallback: PDF (PyMuPDF, pdfplumber) when no HTML version exists. Recover the printed number from the trailing "(n)" via a right margin regex.
Do not infer equation numbers by counting (\tag, subequations and manual numbering break the count).
Do not use neural PDF to markup transcribers (Nougat, im2markup): they are seq2seq decoders that generate markup, which violates rule 3.

JSON SCHEMA
Top level key: arxiv ID. Every processed paper gets a key even if it has no enumerated equations (then its equation dict is empty).
Per equation key (the equation number as printed in the paper):
  equation: LaTeX code of the equation
  meaning: short description of what it expresses, plus its name if any
  symbols: dict {symbol: description}. Symbol keys use LaTeX command without backslash (phi for phi). Do not explain standard operators (+, -, nabla, etc).
  relations: dict over every other equation in the same paper. Each entry has:
      grade: "none" | "strong" | "potential"
      description: required if "strong" (e.g. equivalent, special case, negation), recommended if "potential"
  audit-trail: dict {method_name: short precise output of what it extracted}

RELATION GRADING
Design and document an explicit classification concept for none / potential / strong. Apply it consistently.
Suggested non generative fusion of logged channels: structural similarity (tree edit distance over expression trees, zss/APTED), symbol overlap (Jaccard of symbol sets), textual context similarity (cosine of local sentence embeddings from a local encoder), explicit cross references (\ref, \eqref, cue phrases like "substituting (3)", "combining (1) and (2)"), and a relation type label from a cue phrase lexicon (special case, generalisation, substitution, equivalent, negation, limit). Monotone decision rule: explicit reference or high structural similarity -> strong; moderate textual plus symbol overlap, no explicit link -> potential; below threshold -> none. Pairwise within paper (<=7 eqs), O(n^2) is cheap. Calibrate thresholds on a small hand labelled dev split, report precision/recall per grade as evidence.

SYMBOL AND MEANING EXTRACTION
Symbol enumeration: parse the equation (from HTML MathML or LaTeX) and collect identifier leaves, normalise to the no backslash key form, drop standard operators via a stop list.
Definition linking: search the introducing sentence and an n sentence window using Hearst style templates ("where X is the", "X denotes", "let X be"), dependency parse arcs (spaCy/scispaCy appositions and copular nsubj-attr), and the Schwartz-Hearst matcher for parenthetical abbreviations (in scispaCy).
Optional: BIO definition span classification with a local encoder (SciBERT), label prediction only, no decoding.

AUDIT TRAIL
Keys are the actual extraction methods in code. Values are short precise records of what each method found per item. Treat as first class: per equation AuditLog plus a method decorator emitting one record (method, rule that fired, short evidence snippet, output, confidence), collapsed to {method_name: short_message}. Be deterministic (fixed seeds, sorted iteration). Truncate evidence, no paragraph dumps. Emit compliance markers (source = HTML or PDF, model = encoder/classifier) so the trail proves the no generation and arxiv only rules. Must let a grader verify quality, trace success and failure, prove rule compliance. Verbose or unclear trails are downgraded.

APPROACH CONSTRAINTS
meaning, symbols, and relations must be derived from the paper text itself using non generative NLP (definition mining around symbols, surrounding text parsing, similarity, embeddings, rules, classifiers). No generative model output anywhere.

CODE QUALITY
Well structured (packages, classes, methods). Sufficient comments (uncommented code is downgraded). Every function has a NumPy style docstring. Use existing Python packages where possible. Efficient and not error prone. CODE QUALITY
Well structured (packages, classes, methods). Sufficient comments (uncommented code is downgraded). Every function has a NumPy style docstring. Use existing Python packages where possible. Efficient and not error prone.
Execution rules (bypass generic AI patterns):
- Style: direct, human readable logic. Banned: redundant wrappers, over-engineered abstractions, placeholders (pass/TODO).
- Integrity: isolate data pipelines to prevent leakage. Handle parameters and regularization explicitly to mitigate overfitting. Optimize state updates and loops.
- Documentation: explain why a choice was made, not what the syntax does. Minimal functional docstrings.
- Formatting: banned: emojis, ASCII art, decorative comment dividers (# --- or # ===). Standard text for all logs and prints.
- Output: contiguous code blocks, then a concise explanation of architecture and execution flow.

DOCUMENTATION
PDF only, proper report format. LaTeX article class, no paper template. Arial 11 pt, one column, single line spacing. Pure text max 1200 words (references, captions, tables excluded). Readable by an AI master student who is not a domain expert. Justify every step. Proper scientific citation. Declare AI tool use.
Focus on three points: key solution ideas and why (not code description), generalization to arbitrary quantum physics papers with evidence, critical quality discussion with evidence including what could not be improved and why. Analyze success and error cases.

DELIVERABLES
1. Dataset as JSON
2. Documentation as PDF
3. Source code in a separate folder, zipped (zip only, no rar or tar)
