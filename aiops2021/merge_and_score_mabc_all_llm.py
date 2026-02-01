"""Merge sharded mABC(all) LLM outputs and score with AIOpsChallengeJudge.

This repo runs mABC(all) in 4 shards and writes:
  aiops2021/outputs/submission_mabc_all_llm.part{0,1,2,3}.jsonl

The runner is append/resumable; shards may be restarted. This script:
  1) Merges all parts into a single JSONL (dedup by uuid)
  2) Validates coverage against aiops2021/inputs/all_input_clean.json
  3) Runs AIOpsChallengeJudge/evaluate.py for zh/en GT

Usage (after shards complete):
  .venv/bin/python aiops2021/merge_and_score_mabc_all_llm.py --cleanup-parts

Notes:
- This script never uses GT during method inference; it only uses GT for scoring.
- Dedup policy: keep the last-seen record for each uuid (so re-runs win).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(obj, dict):
                continue
            out.append(obj)
    return out


def _iter_part_paths(outputs_dir: Path, part_prefix: str, shard_count: int) -> Iterable[Path]:
    prefix = (part_prefix or "submission_mabc_all_llm").strip()
    for i in range(int(shard_count)):
        yield outputs_dir / f"{prefix}.part{i}.jsonl"


def _merge_parts(part_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for p in part_paths:
        for obj in _load_jsonl(p):
            uuid = str(obj.get("uuid", "")).strip()
            if not uuid:
                continue
            merged[uuid] = obj  # last write wins
    return merged


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _run_judge(python_exe: str, submission: Path, gt: Path, report: Path) -> None:
    cmd = [
        python_exe,
        str(Path("AIOpsChallengeJudge") / "evaluate.py"),
        "--ground-truth",
        str(gt),
        "--submission",
        str(submission),
        "--report",
        str(report),
        "--reason-threshold",
        "0.65",
    ]
    print("[judge]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-json",
        type=Path,
        default=Path("aiops2021/inputs/all_input_clean.json"),
        help="AIOpsChallenge-style input.json used to define expected uuids",
    )
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("aiops2021/outputs"),
        help="Directory containing sharded part*.jsonl files",
    )
    p.add_argument(
        "--part-prefix",
        type=str,
        default="submission_mabc_all_llm",
        help="Prefix for sharded jsonl files (default: submission_mabc_all_llm)",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=4,
        help="Number of shards/part files to merge (default: 4)",
    )
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("aiops2021/outputs/submission_mabc_all_allstats_clean.jsonl"),
        help="Merged submission output path",
    )
    p.add_argument(
        "--gt-zh",
        type=Path,
        default=Path("aiops2021/outputs/ground_truth_all.jsonl"),
    )
    p.add_argument(
        "--gt-en",
        type=Path,
        default=Path("aiops2021/outputs/ground_truth_all_en.jsonl"),
    )
    p.add_argument(
        "--report-zh",
        type=Path,
        default=Path("aiops2021/outputs/report_mabc_all_zh.json"),
    )
    p.add_argument(
        "--report-en",
        type=Path,
        default=Path("aiops2021/outputs/report_mabc_all_en.json"),
    )
    p.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use when running judge",
    )
    p.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete part*.jsonl after a successful merge+score",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not args.input_json.exists():
        raise FileNotFoundError(f"Missing input json: {args.input_json}")

    entries = json.loads(args.input_json.read_text(encoding="utf-8"))
    expected = [str(it.get("uuid", "")).strip() for it in entries if isinstance(it, dict)]
    expected_set = {u for u in expected if u}

    part_paths = list(_iter_part_paths(args.outputs_dir, args.part_prefix, int(args.shard_count)))
    missing_parts = [p for p in part_paths if not p.exists()]
    if missing_parts:
        raise FileNotFoundError(
            "Missing part files: " + ", ".join(str(p) for p in missing_parts)
        )

    merged = _merge_parts(part_paths)

    got_set = set(merged.keys())
    missing = sorted(expected_set - got_set)
    extra = sorted(got_set - expected_set)

    print(f"[merge] expected={len(expected_set)} got={len(got_set)}", flush=True)
    if missing:
        print(f"[merge][WARN] missing uuids: {len(missing)} (showing up to 10)")
        for u in missing[:10]:
            print(" -", u)
    if extra:
        print(f"[merge][WARN] extra uuids: {len(extra)} (showing up to 10)")
        for u in extra[:10]:
            print(" -", u)

    # Stable order: follow input order
    merged_list = [merged[u] for u in expected if u in merged]
    _write_jsonl(args.out_jsonl, merged_list)
    print(f"[merge] wrote: {args.out_jsonl} lines={len(merged_list)}", flush=True)

    # Score (always run; evaluate.py will warn+fill blanks if missing)
    _run_judge(args.python, args.out_jsonl, args.gt_zh, args.report_zh)
    _run_judge(args.python, args.out_jsonl, args.gt_en, args.report_en)

    if args.cleanup_parts and not missing:
        for p in part_paths:
            p.unlink(missing_ok=True)
        print("[cleanup] deleted part*.jsonl", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
