#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PAGES = ROOT / "pages"
TEMPLATE = Path(__file__).with_name("report-template.html")
OUT = PAGES / "index.html"


def load_datasets() -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {}
    for path in sorted(RESULTS.glob("*-99900.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        data[rows[0]["dataset"]] = rows
    return data


def main() -> None:
    data = load_datasets()
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html)
    print(f"wrote {OUT} with {len(data)} datasets: {', '.join(data)}")


if __name__ == "__main__":
    main()
