"""Score the pipeline against labelled ground truth.

    python eval.py                          # uses examples/ground_truth.json
    python eval.py --no-semantic            # skip embedding-based summary scoring
    python eval.py --out eval_results.json
"""

import argparse
import json
import logging

from dotenv import load_dotenv

from src import evaluate_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate metadata extraction quality.")
    parser.add_argument("--ground-truth", default="examples/ground_truth.json")
    parser.add_argument("--out", help="write full results as JSON")
    parser.add_argument("--no-semantic", action="store_true",
                        help="skip embedding-based summary scoring")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )

    report = evaluate_sync(args.ground_truth, use_semantic=not args.no_semantic)
    print(report)

    failures = report.failures()
    if failures:
        print(f"\nFields scoring below 0.5 ({len(failures)}):")
        for f in failures[:15]:
            print(f"  {f.field_name:<16} expected {f.expected!r:<40} got {f.predicted!r}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
