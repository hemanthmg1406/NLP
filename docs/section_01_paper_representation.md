# Section 1: Paper representation

## Locked scope

This section converts one cached arXiv LaTeXML HTML paper into a deterministic
intermediate representation. It does not select the first seven equations and does not
extract meanings, symbol definitions, relations, audit messages, or final JSON.

## Representation

`PaperDocument` contains:

- arXiv-only source provenance and the paper title;
- hierarchical sections;
- readable paragraphs with inline LaTeX and character spans;
- every enumerated equation with its printed label and verbatim source LaTeX;
- equation blocks that preserve align/subequation grouping;
- paragraph and equation-block nodes in DOM reading order;
- resolved hyperlinks from prose to known equation ids.

This structure supports later local context windows, paper-wide definition lookup, and
explicit equation-relation evidence without repeatedly parsing the HTML.

## Decisions from the assignment

- Python is used throughout and every function has a focused docstring.
- Provenance records that the source is cached arXiv HTML.
- Printed labels are retained because the dataset key must match the paper.
- The complete paper is represented because the seven-equation cutoff applies to the
  dataset, while definitions and references may occur later in the document.
- No language model, prompting, external API, or generated text is used.

## Completion checks

The real cached paper `2412.06345.html` verifies that:

- the first labels are `1`, `2`, `3`, `4a`, `4b`, `5`, and `6`;
- `4a` and `4b` share one display block;
- equation LaTeX is unchanged;
- inline MathML is reduced to one readable LaTeX occurrence;
- later references to equations `3`, `4b`, and `6` resolve;
- document order places the paragraph defining `T` directly after equation `5`.
