#!/usr/bin/env python
"""
Calibrate the Jev decision threshold on labelled verdicts (E10-F2).

Jev returns P(yes) and is under-confident at 0.5, so the threshold that turns
P(yes) into a correct / wrong verdict must be fitted on labelled answers.

Labels come either from a file of human verdicts or, when no file is given,
from the LLM judge verdicts stored in the same result files.

Usage:
    # human labels: JSON {"q1": 1, ...}, JSONL {"question_id": .., "label": ..} or CSV question_id,label
    python scripts/calibrate_jev_threshold.py results/vanilla/run_*.json --labels data/evaluation/jev_labels.csv

    # no labels file: use the LLM judge verdicts as reference
    python scripts/calibrate_jev_threshold.py results/vanilla/run_*.json

Then set ``evaluation.jev_threshold`` in config/benchmarks/user-config.yaml
(or in the Evaluation section of the parameter modal).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.correctness_judge import calibrate_threshold  # noqa: E402


def _to_label(value) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if value >= 0.5 else 0.0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "yes", "true", "correct", "y"}:
            return 1.0
        if v in {"0", "no", "false", "wrong", "incorrect", "n"}:
            return 0.0
    return None


def load_labels(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    labels: dict[str, float] = {}
    if path.suffix == ".csv":
        for row in csv.DictReader(text.splitlines()):
            label = _to_label(row.get("label"))
            if row.get("question_id") and label is not None:
                labels[row["question_id"]] = label
    elif path.suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                label = _to_label(row.get("label"))
                if label is not None:
                    labels[str(row["question_id"])] = label
    else:
        data = json.loads(text)
        for qid, value in data.items():
            label = _to_label(value)
            if label is not None:
                labels[str(qid)] = label
    return labels


def collect(result_files: list[Path], labels: dict[str, float] | None):
    probs, refs = [], []
    for path in result_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("per_question_full", []):
            corr = entry.get("correctness") or {}
            prob = (corr.get("jev") or {}).get("prob_yes")
            if prob is None:
                continue
            if labels is not None:
                ref = labels.get(str(entry.get("question_id")))
            else:
                ref = (corr.get("llm") or {}).get("correct")
            if ref is None:
                continue
            probs.append(float(prob))
            refs.append(float(ref))
    return probs, refs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="benchmark result JSON files")
    parser.add_argument("--labels", type=Path, help="human verdicts (CSV, JSON or JSONL); default: LLM judge verdicts")
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else None
    probs, refs = collect(args.results, labels)
    if not probs:
        print("No question has both a Jev P(yes) and a reference verdict.", file=sys.stderr)
        return 1

    out = calibrate_threshold(probs, refs)
    source = f"human labels ({args.labels})" if labels is not None else "LLM judge verdicts"
    at05 = out["at_0_5"]
    print(f"Reference          : {source}")
    print(f"Questions          : {out['n']}  (positives: {int(sum(refs))})")
    print(f"At threshold 0.5   : balanced acc {at05['balanced_accuracy']:.3f}, acc {at05['accuracy']:.3f}, kappa {at05['kappa'] if at05['kappa'] is None else round(at05['kappa'], 3)}")
    kappa = out["kappa"]
    print(f"Best threshold     : {out['threshold']}  (balanced acc {out['balanced_accuracy']:.3f}, acc {out['accuracy']:.3f}, kappa {kappa if kappa is None else round(kappa, 3)})")
    print("\nSet it with:\n  evaluation:\n    jev_threshold: " + str(out["threshold"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
