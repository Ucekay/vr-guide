"""Score a results CSV against your own answer key.

Answer key = a 2-column CSV (UTF-8), one row per expected poster:

    query,ref
    438100_38.jpg,era1/train_arrival_03.jpg
    438100_50.jpg,era1/factory_gate_01.jpg
    438100_50.jpg,era1/factory_gate_02.jpg
    438100_54.jpg,none

Use `none` for a screenshot that contains no poster at all. A screenshot may
appear on several rows if it shows several posters.

Usage:
    python score.py --results out/fast.csv --labels labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# Windows consoles default to a legacy codepage (cp936 here), which mangles the
# Chinese labels below. Force UTF-8 so the report is readable anywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def dwidth(s: str) -> int:
    """Display width in terminal columns: CJK characters occupy two."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


LANG = {
    "zh": {
        "total": "截图总数", "perfect": "完全正确", "counts": "命中 / 误报 / 漏检",
        "prec_hint": "(报出来的有多少是对的)", "rec_hint": "(该找到的找到了多少)",
        "problems": "有问题的 {n} 张：", "miss": "漏", "extra": "错",
    },
    "en": {
        "total": "Screenshots", "perfect": "Fully correct", "counts": "Hit / FP / Miss",
        "prec_hint": "(how many reported answers are right)",
        "rec_hint": "(how many of the expected ones were found)",
        "problems": "{n} with problems:", "miss": "missing", "extra": "wrong",
    },
}


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="out/fast.csv or out/results.csv")
    ap.add_argument("--labels", required=True, help="your answer key CSV")
    ap.add_argument("--top1-only", action="store_true",
                    help="judge only the single best guess per screenshot")
    ap.add_argument("--lang", default="zh", choices=["zh", "en"],
                    help="language of the report")
    args = ap.parse_args()
    L = LANG[args.lang]

    for p in (args.results, args.labels):
        if not Path(p).exists():
            raise SystemExit(f"File not found: {p}")

    truth: dict[str, set[str]] = defaultdict(set)
    for r in read_csv(args.labels):
        q = (r.get("query") or "").strip()
        ref = (r.get("ref") or "").strip()
        if not q:
            continue
        if ref.lower() not in ("none", "", "-"):
            truth[q].add(ref)
        else:
            truth[q]                      # touch: known screenshot, expects nothing

    pred: dict[str, set[str]] = defaultdict(set)
    seen: set[str] = set()
    for r in read_csv(args.results):
        q = r["query"]
        seen.add(q)
        if r["detected"] == "1" and not (args.top1_only and r["rank"] != "1"):
            pred[q].add(r["ref"])

    missing = sorted(set(truth) - seen)
    if missing:
        print(f"[warn] {len(missing)} labelled screenshot(s) absent from results, "
              f"e.g. {missing[:3]}")

    tp = fp = fn = 0
    perfect = 0
    problems: list[str] = []
    for q in sorted(set(truth) | seen):
        exp, got = truth.get(q, set()), pred.get(q, set())
        hit, extra, miss = exp & got, got - exp, exp - got
        tp += len(hit); fp += len(extra); fn += len(miss)
        if not extra and not miss:
            perfect += 1
        else:
            bits = []
            if miss:
                bits.append(L["miss"] + " " + ", ".join(sorted(miss)))
            if extra:
                bits.append(L["extra"] + " " + ", ".join(sorted(extra)))
            problems.append(f"  {q:<22}{' | '.join(bits)}")

    n = len(set(truth) | seen)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    labels = [L["total"], L["perfect"], L["counts"], "precision", "recall", "F1"]
    w = max(dwidth(s) for s in labels) + 2

    def row(label: str, value: str) -> None:
        print(label + " " * (w - dwidth(label)) + value)

    print()
    row(L["total"], f"{n}")
    if n:
        row(L["perfect"], f"{perfect}/{n}  ({perfect / n * 100:.1f}%)")
    row(L["counts"], f"{tp} / {fp} / {fn}")
    row("precision", f"{prec:.3f}   {L['prec_hint']}")
    row("recall", f"{rec:.3f}   {L['rec_hint']}")
    row("F1", f"{f1:.3f}")
    if problems:
        print("\n" + L["problems"].format(n=len(problems)))
        print("\n".join(problems))


if __name__ == "__main__":
    main()
