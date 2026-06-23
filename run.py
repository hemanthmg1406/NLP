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
from pipeline import process_paper, OUTPUT_FILE

TEST_PAPERS = [
    "2409.02921",
    "2409.04026",
    "2409.07516",
    "2409.15184",
    "2409.15203",
    "2409.18916",
    "2410.00650",
    "2410.03482",
    "2410.11839",
    "2410.21900",
    "2411.00230",
    "2411.03995",
    "2411.09350",
    "2411.18434",
    "2412.06345",
    "2412.07653",
    "2412.15568",
    "2502.07886",
    "2502.10115",
    "2503.02436",
    "2503.12870",
    "2503.14798",
    "2504.02706",
    "2504.04671",
    "2504.07019",
    "2504.07341",
    "2504.18149",
    "2505.02445",
    "2505.04321",
    "2505.05058",
    "2505.09857",
    "2505.12985",
    "2505.20204",
    "2505.20373",
    "2506.04298",
    "2506.12906",
    "2506.21707",
    "2506.22684",
    "2507.04159",
    "2507.05160",
    "2507.07900",
    "2507.09604",
    "2507.14691",
    "2508.02514",
    "2508.04880",
    "2508.06441",
    "2508.15410",
    "2508.20116",
    "2509.11585",
    "2509.13406",
    "2509.24666",
    "2510.00203",
    "2510.01035",
    "2510.07461",
    "2510.07587",
    "2510.11831",
    "2510.12677",
    "2510.18689",
    "2510.21101",
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
