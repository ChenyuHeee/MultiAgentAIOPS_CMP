"""Run AIOpsChallengeJudge scoring on the aiops2021 dataset.

This workspace contains a judge implementation under AIOpsChallengeJudge/ that expects a
submission JSONL with fields:
- uuid: str
- component: str
- reason: str
- reasoning_trace: list[{step:int, action:str, observation:str}]

The aiops2021 folder only includes telemetry parquet files plus ground_truth.jsonl.
There is no per-uuid input window mapping file in this repo, so this script provides
simple baselines driven by label priors.

Baselines:
- oracle: copies component/reason from ground_truth (upper bound / pipeline sanity check)
- majority: predicts the most common component & reason across the ground truth

You can also score an existing submission file with --submission.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _dedup_by_uuid(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for r in records:
        uuid = r.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            continue
        if uuid in seen:
            continue
        seen.add(uuid)
        deduped.append(r)
    return deduped


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _majority_label(gt_records: List[Dict[str, Any]]) -> Tuple[str, str]:
    comp = Counter(r.get("component", "") for r in gt_records).most_common(1)[0][0]
    reason = Counter(r.get("reason", "") for r in gt_records).most_common(1)[0][0]
    return comp, reason


def build_submission(
    gt_path: Path,
    mode: str,
    out_path: Path,
) -> Path:
    gt_records = _dedup_by_uuid(_load_jsonl(gt_path))

    if mode not in {"oracle", "majority"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "majority":
        majority_component, majority_reason = _majority_label(gt_records)

    submission: List[Dict[str, Any]] = []
    for r in gt_records:
        uuid = r["uuid"]
        if mode == "oracle":
            component = r.get("component", "")
            reason = r.get("reason", "")
            observation = "Copied from ground truth (oracle baseline)."
        else:
            component = majority_component
            reason = majority_reason
            observation = "Predicted by majority label prior (no telemetry used)."

        submission.append(
            {
                "uuid": uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": [
                    {
                        "step": 1,
                        "action": f"baseline:{mode}",
                        "observation": observation,
                    }
                ],
            }
        )

    _write_jsonl(out_path, submission)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-jsonl",
        type=Path,
        default=Path("aiops2021/ground_truth.jsonl"),
        help="Path to aiops2021 ground_truth.jsonl",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=None,
        help="If provided, score this submission instead of generating one.",
    )
    parser.add_argument(
        "--mode",
        choices=["majority", "oracle"],
        default="majority",
        help="Baseline mode used when --submission is not provided.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("aiops2021/outputs/submission.jsonl"),
        help="Where to write the generated submission.",
    )
    parser.add_argument(
        "--reason-threshold",
        type=float,
        default=0.75,
        help="Judge reason match similarity threshold (passed through).",
    )

    args = parser.parse_args()

    gt_path: Path = args.gt_jsonl

    # The judge requires unique uuids, but aiops2021/ground_truth.jsonl contains
    # a known duplicate (aiops21-1475). We de-duplicate for scoring.
    raw_gt = _load_jsonl(gt_path)
    dedup_gt = _dedup_by_uuid(raw_gt)
    if len(dedup_gt) != len(raw_gt):
        gt_dedup_path = Path("aiops2021/outputs/ground_truth_dedup.jsonl")
        _write_jsonl(gt_dedup_path, dedup_gt)
        gt_path = gt_dedup_path
    submission_path: Path
    if args.submission is not None:
        submission_path = args.submission
    else:
        submission_path = build_submission(gt_path=gt_path, mode=args.mode, out_path=args.out)

    # Defer import so this script stays usable as a pure generator.
    import subprocess

    cmd = [
        str(Path("AIOpsChallengeJudge/evaluate.py")),
        "--ground-truth",
        str(gt_path),
        "--submission",
        str(submission_path),
        "--reason-threshold",
        str(args.reason_threshold),
    ]

    # Use the workspace python if available.
    python = str(Path(".venv/bin/python"))
    if not Path(python).exists():
        python = "python"

    raise SystemExit(subprocess.call([python, *cmd]))


if __name__ == "__main__":
    main()
