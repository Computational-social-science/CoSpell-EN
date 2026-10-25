#!/usr/bin/env python3
"""
从 Google Books Ngram API 获取指定词形时序（1800–2022），并合并进时序 JSON。

默认获取 here（根词）及其 RWSE 变体 herd、herr（见 rwse_probabilities.csv）。
默认合并目标：
  - rwse_advs_timeseries.json
  - final_adversarial_advs_timeseries.json

用法:
    python 6-compute_word_series.py
    python 6-compute_word_series.py --from-json here_herd_herr_timeseries.json --no-fetch
    python 6-compute_word_series.py --target-timeseries final_adversarial_advs_timeseries.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.parse
from typing import Any

try:
    import requests
except ImportError:
    print("[ERROR] 需要 requests：pip install requests", file=sys.stderr)
    sys.exit(1)


DEFAULT_ROOT = "here"
DEFAULT_VARIANTS = ["herd", "herr"]

DEFAULT_MERGE_TARGETS = [
    "rwse_advs_timeseries.json",
    "final_adversarial_advs_timeseries.json",
]


def load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] File not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        sys.exit(f"[ERROR] Expected JSON object in {path}")
    return data


def save_json_atomic(data: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(os.path.abspath(path)) or ".",
        prefix=".tmp_",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def fetch_ngram_batch(
    phrases: list[str],
    *,
    start_year: int = 1800,
    end_year: int = 2022,
    corpus: str = "english",
    smoothing: int = 3,
    max_retries: int = 4,
    retry_delay: int = 10,
) -> tuple[list[dict[str, Any]], list[int] | None]:
    query = ",".join(phrases)
    encoded = urllib.parse.quote(query)
    url = (
        f"https://books.google.com/ngrams/json?content={encoded}"
        f"&year_start={start_year}&year_end={end_year}"
        f"&corpus={corpus}&smoothing={smoothing}"
    )

    for attempt in range(max_retries):
        try:
            print(f"[API] Fetch {phrases} (attempt {attempt + 1}/{max_retries})")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return [], None

            years_axis: list[int] | None = None
            for item in data:
                ts = item.get("timeseries")
                if ts:
                    n = len(ts)
                    years_axis = list(range(start_year, start_year + n))
                    if len(years_axis) > end_year - start_year + 1:
                        years_axis = list(range(start_year, end_year + 1))
                    break
            return data, years_axis
        except requests.RequestException as exc:
            print(f"[WARN] API error: {exc}", file=sys.stderr)
            if attempt + 1 < max_retries:
                time.sleep(retry_delay)
            else:
                raise
    return [], None


def build_timeseries_json(
    root: str,
    variants: list[str],
    *,
    start_year: int = 1800,
    end_year: int = 2022,
    corpus: str = "english",
    smoothing: int = 3,
) -> dict[str, Any]:
    words = [root] + [v for v in variants if v != root]
    api_data, years_axis = fetch_ngram_batch(
        words,
        start_year=start_year,
        end_year=end_year,
        corpus=corpus,
        smoothing=smoothing,
    )
    if not years_axis:
        sys.exit("[ERROR] API returned no valid timeseries / years_axis")

    by_ngram: dict[str, dict[str, Any]] = {}
    for item in api_data:
        ngram = (item.get("ngram") or "").strip()
        ts = item.get("timeseries")
        if ngram and isinstance(ts, list) and len(ts) == len(years_axis):
            by_ngram[ngram] = item

    ngrams_out: dict[str, Any] = {}
    missing: list[str] = []

    if root in by_ngram:
        ngrams_out[root] = {
            "ngram": root,
            "parent_ngram": None,
            "timeseries": [float(x) for x in by_ngram[root]["timeseries"]],
            "type": "SPELLING",
        }
    else:
        missing.append(root)

    for v in variants:
        if v == root:
            continue
        if v in by_ngram:
            ngrams_out[v] = {
                "ngram": v,
                "parent_ngram": root,
                "timeseries": [float(x) for x in by_ngram[v]["timeseries"]],
                "type": "MISSPELLING",
            }
        else:
            missing.append(v)

    if missing:
        print(f"[WARN] Missing from API response: {', '.join(missing)}", file=sys.stderr)

    return {
        "years_axis": years_axis,
        "ngrams_data": ngrams_out,
        "meta": {
            "root": root,
            "variants_requested": variants,
            "corpus": corpus,
            "smoothing": smoothing,
            "source": "Google Books Ngram API",
        },
    }


def merge_into_target(
    target: dict[str, Any],
    fetched: dict[str, Any],
    *,
    overwrite: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """
    Merge fetched ngrams_data into target timeseries JSON.

    Returns:
        added, updated, skipped (already present, not overwritten)
    """
    if "years_axis" not in target or "ngrams_data" not in target:
        sys.exit("[ERROR] Target JSON must contain years_axis and ngrams_data")

    src_years = [int(y) for y in fetched["years_axis"]]
    tgt_years = [int(y) for y in target["years_axis"]]
    if src_years != tgt_years:
        sys.exit(
            f"[ERROR] years_axis mismatch: fetched {src_years[0]}–{src_years[-1]} "
            f"({len(src_years)}), target {tgt_years[0]}–{tgt_years[-1]} ({len(tgt_years)})"
        )

    ngrams_data = target["ngrams_data"]
    if not isinstance(ngrams_data, dict):
        sys.exit("[ERROR] target ngrams_data must be a JSON object")

    added: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []

    for key, entry in fetched.get("ngrams_data", {}).items():
        if not isinstance(entry, dict):
            continue
        ngram = str(entry.get("ngram", key)).strip()
        ts = entry.get("timeseries")
        if not ngram or not isinstance(ts, list) or len(ts) != len(tgt_years):
            print(f"[WARN] Skip invalid entry for {key!r}", file=sys.stderr)
            continue

        payload = {
            "ngram": ngram,
            "parent_ngram": entry.get("parent_ngram"),
            "timeseries": [float(x) for x in ts],
            "type": entry.get("type", "MISSPELLING"),
        }

        if ngram in ngrams_data:
            if overwrite:
                ngrams_data[ngram] = payload
                updated.append(ngram)
            else:
                skipped.append(ngram)
        else:
            ngrams_data[ngram] = payload
            added.append(ngram)

    return added, updated, skipped


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Fetch Ngram timeseries and merge into project timeseries JSON file(s)",
    )
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Canonical root word (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=None,
        help="All words to fetch (overrides --root/--variants if set)",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=DEFAULT_VARIANTS,
        help=f"Variant words under root (default: {DEFAULT_VARIANTS})",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(here, "here_herd_herr_timeseries.json"),
        help="Snapshot JSON path for fetched data",
    )
    parser.add_argument(
        "--target-timeseries",
        nargs="+",
        default=[os.path.join(here, p) for p in DEFAULT_MERGE_TARGETS],
        help="One or more timeseries JSON files to merge into",
    )
    parser.add_argument(
        "--from-json",
        default=None,
        help="Load fetched data from existing JSON instead of calling API",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not call API; use --from-json or existing --output file",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Only save snapshot, do not merge into target timeseries",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ngrams in target when keys collide",
    )
    parser.add_argument("--start-year", type=int, default=1800)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--corpus", default="english")
    parser.add_argument("--smoothing", type=int, default=3)
    args = parser.parse_args()

    root = args.root
    if args.words:
        words = [w.strip() for w in args.words if w.strip()]
        if not words:
            sys.exit("[ERROR] --words is empty")
        root = words[0]
        variants = words[1:]
    else:
        variants = [v.strip() for v in args.variants if v.strip()]

    if args.from_json:
        fetched = load_json(args.from_json)
    elif args.no_fetch:
        if not os.path.isfile(args.output):
            sys.exit(f"[ERROR] --no-fetch but snapshot not found: {args.output}")
        fetched = load_json(args.output)
    else:
        fetched = build_timeseries_json(
            root,
            variants,
            start_year=args.start_year,
            end_year=args.end_year,
            corpus=args.corpus,
            smoothing=args.smoothing,
        )
        save_json_atomic(fetched, args.output)
        print(f"[OK] Wrote snapshot {args.output}")

    n_ok = len(fetched.get("ngrams_data", {}))
    years = fetched.get("years_axis", [])
    if years:
        print(
            f"[OK] Fetched/loaded {n_ok} ngram(s): "
            f"{', '.join(fetched.get('ngrams_data', {}))}"
        )
        print(f"[OK] Years: {years[0]}–{years[-1]} ({len(years)} points)")

    if args.no_merge:
        return

    for target_path in args.target_timeseries:
        target = load_json(target_path)
        added, updated, skipped = merge_into_target(
            target, fetched, overwrite=args.overwrite
        )
        save_json_atomic(target, target_path)

        print(f"[OK] Merged into {target_path}")
        print(f"[OK] Added ({len(added)}): {', '.join(added) if added else '(none)'}")
        if updated:
            print(f"[OK] Updated ({len(updated)}): {', '.join(updated)}")
        if skipped:
            print(f"[INFO] Skipped existing ({len(skipped)}): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
