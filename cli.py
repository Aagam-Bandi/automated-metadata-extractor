"""Command-line entry point: python cli.py path/to/document.pdf"""

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

from src import generate_metadata_sync


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate metadata for a document.")
    parser.add_argument("path", help="path to a .pdf, .docx or .txt file")
    parser.add_argument("-o", "--output", help="write JSON here instead of stdout")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s %(message)s",
    )

    try:
        result = generate_metadata_sync(args.path).to_dict()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    payload = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(payload)
        print(f"wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
