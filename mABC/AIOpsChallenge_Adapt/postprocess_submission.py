from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _merge_jsonl(inputs: List[Path], out_path: Path, require_unique_uuid: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: Set[str] = set()
    dups = 0
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for p in inputs:
            for obj in _read_jsonl(p):
                uid = str(obj.get("uuid", "")).strip()
                if not uid:
                    continue
                if uid in seen:
                    dups += 1
                    if require_unique_uuid:
                        raise ValueError(f"Duplicate uuid {uid} found while merging: {p}")
                    continue
                seen.add(uid)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

    print(f"merged_inputs={len(inputs)} written={written} unique_uuids={len(seen)} dups={dups} out={out_path}")


def _expand_glob(pattern: str) -> List[Path]:
    paths = sorted(Path().glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return paths


def _run_judge(ground_truth: Path, submission: Path, report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    judge_py = Path("AIOpsChallengeJudge/evaluate.py")
    if not judge_py.exists():
        raise FileNotFoundError(f"Missing judge script: {judge_py}")

    cmd = [sys.executable, str(judge_py), "-g", str(ground_truth), "-s", str(submission), "-o", str(report)]
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge submission shards and/or run AIOpsChallengeJudge")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--shard-glob", type=str, help="Glob for shard JSONLs to merge, e.g. outputs/submissions/*_shard*.jsonl")
    g.add_argument("--inputs", type=str, help="Comma-separated list of input JSONLs to merge")

    p.add_argument("--out-submission", type=Path, required=True, help="Merged submission JSONL path")
    p.add_argument("--allow-duplicate-uuid", action="store_true", help="Do not fail on duplicate uuid; keep first")

    p.add_argument("--ground-truth", type=Path, default=None, help="Ground truth JSONL path; if set, also run judge")
    p.add_argument("--out-report", type=Path, default=None, help="Judge report JSON path (required if --ground-truth is set)")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.shard_glob:
        inputs = _expand_glob(args.shard_glob)
    else:
        parts = [p.strip() for p in (args.inputs or "").split(",") if p.strip()]
        inputs = [Path(p) for p in parts]

    _merge_jsonl(inputs, args.out_submission, require_unique_uuid=not args.allow_duplicate_uuid)

    if args.ground_truth:
        if not args.out_report:
            raise ValueError("--out-report is required when --ground-truth is set")
        _run_judge(args.ground_truth, args.out_submission, args.out_report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
