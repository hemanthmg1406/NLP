"""Quick end-to-end test: run the pipeline on 5 cached papers and write test_output.json.

Usage (on DC 1.07):
    python run_test5.py

Bypasses the interactive main() so no prompts are needed.  Reads from the
existing cache — no network calls made.  Output goes to test_output.json so
it does not overwrite the production output.json.
"""

import json
from pathlib import Path

# Import the processing function directly — main() is intentionally skipped.
from build_json import process_paper, OUTPUT_FILE

TEST_PAPERS = [
    "2409.02921",
    "2409.04026",
    "2409.07516",
    "2409.15184",
    "2409.15203",
]

OUTPUT = Path("test_output.json")


def run():
    results = {}
    for arxiv_id in TEST_PAPERS:
        print(f"\n{'='*60}")
        print(f"Processing {arxiv_id}")
        print('='*60)
        paper_result = process_paper(arxiv_id)
        results[arxiv_id] = paper_result

    OUTPUT.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT}")

    # Quick summary: print meaning + rule for every equation
    print("\n=== MEANING SUMMARY ===")
    for arxiv_id, eqs in results.items():
        print(f"\n-- {arxiv_id} ({len(eqs)} equations) --")
        for eq_num, eq_data in eqs.items():
            meaning = eq_data.get("meaning", "")
            rule = eq_data.get("audit-trail", {}).get("meaning_rule", "")
            print(f"  ({eq_num}) [{rule}] {meaning}")


if __name__ == "__main__":
    run()
