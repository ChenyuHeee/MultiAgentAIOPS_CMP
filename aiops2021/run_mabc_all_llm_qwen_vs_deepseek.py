"""Run mABC(all) twice: Qwen vs DeepSeek (separate runs) and score both.

This script keeps the same constraints as the rest of this workspace:
- GT-free inference (ground truth is only used for scoring)
- Uses the no-leak input json (window only)
- Runs in 4 shards in parallel per provider

It relies on mABC/.env to provide provider keys:
- QWEN_API_KEY (optionally QWEN_MODEL, QWEN_BASE_URL)
- DEEPSEEK_API_KEY (optionally DEEPSEEK_MODEL, DEEPSEEK_BASE_URL)

Usage:
  .venv/bin/python aiops2021/run_mabc_all_llm_qwen_vs_deepseek.py

Outputs:
  aiops2021/outputs/submission_mabc_all_allstats_clean_qwen.jsonl
  aiops2021/outputs/report_mabc_all_qwen_{zh,en}.json

  aiops2021/outputs/submission_mabc_all_allstats_clean_deepseek.jsonl
  aiops2021/outputs/report_mabc_all_deepseek_{zh,en}.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _load_env_file(env_path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env.setdefault(key, value)
    return env


def _require_keys(provider: str, env: Dict[str, str]) -> None:
    if provider == "qwen" and not (env.get("QWEN_API_KEY") or os.environ.get("QWEN_API_KEY")):
        raise SystemExit("Missing QWEN_API_KEY (set in mABC/.env or environment).")
    if provider == "deepseek" and not (
        env.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    ):
        raise SystemExit("Missing DEEPSEEK_API_KEY (set in mABC/.env or environment).")


def _spawn_shards(
    *,
    python_exe: Path,
    runner_py: Path,
    mabc_root: Path,
    input_json: Path,
    out_prefix: str,
    provider: str,
    shard_count: int,
    per_uuid_timeout_seconds: int,
    capture_output: bool,
    ground_truth_jsonl: Path,
    base_env: Dict[str, str],
) -> List[subprocess.Popen]:
    procs: List[subprocess.Popen] = []

    for shard_index in range(shard_count):
        out_jsonl = Path(f"{out_prefix}.part{shard_index}.jsonl")
        cmd = [
            str(python_exe),
            str(runner_py),
            "--mabc-root",
            str(mabc_root),
            "--input-json",
            str(input_json),
            "--out-jsonl",
            str(out_jsonl),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
            "--per-uuid-timeout-seconds",
            str(per_uuid_timeout_seconds),
            "--ground-truth-jsonl",
            str(ground_truth_jsonl),
            "--llm-provider",
            provider,
        ]
        if capture_output:
            cmd.append("--capture-output")
        else:
            cmd.append("--no-capture-output")

        print("[run]", " ".join(cmd), flush=True)
        procs.append(subprocess.Popen(cmd, env=base_env))

    return procs


def _wait_all(procs: List[subprocess.Popen]) -> None:
    failed: List[Tuple[int, int]] = []
    for i, p in enumerate(procs):
        rc = p.wait()
        if rc != 0:
            failed.append((i, rc))
    if failed:
        msg = ", ".join([f"shard{i}:rc={rc}" for i, rc in failed])
        raise SystemExit(f"Some shards failed: {msg}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--providers",
        nargs="+",
        default=["qwen", "deepseek"],
        choices=["qwen", "deepseek"],
        help="Which providers to run",
    )
    ap.add_argument("--shard-count", type=int, default=4)
    ap.add_argument("--per-uuid-timeout-seconds", type=int, default=900)
    ap.add_argument("--capture-output", action="store_true", default=True)
    ap.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable to use",
    )
    ap.add_argument(
        "--mabc-root",
        type=Path,
        default=Path("mABC"),
    )
    ap.add_argument(
        "--input-json",
        type=Path,
        default=Path("aiops2021/inputs/all_input_clean.json"),
    )
    ap.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("aiops2021/outputs"),
    )
    ap.add_argument(
        "--ground-truth-placeholder",
        type=Path,
        default=Path("aiops2021/outputs/__DO_NOT_USE_GROUND_TRUTH__.jsonl"),
        help="Nonexistent path to prevent GT vocab loading during inference",
    )
    ap.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete part*.jsonl after successful merge+score",
    )
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    env_from_file = _load_env_file(args.mabc_root / ".env")
    base_env = dict(os.environ)
    base_env.update(env_from_file)

    runner_py = Path("mABC") / "AIOpsChallenge_Adapt" / "run_mabc_and_build_submission.py"
    merge_py = Path("aiops2021") / "merge_and_score_mabc_all_llm.py"
    update_txt_py = Path("aiops2021") / "update_results_txt_minimal.py"

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    for provider in args.providers:
        _require_keys(provider, base_env)

        part_prefix = f"submission_mabc_all_llm_{provider}"
        out_prefix = args.outputs_dir / part_prefix

        print(f"\n===== mABC(all) provider={provider} =====", flush=True)
        procs = _spawn_shards(
            python_exe=args.python,
            runner_py=runner_py,
            mabc_root=args.mabc_root,
            input_json=args.input_json,
            out_prefix=str(out_prefix),
            provider=provider,
            shard_count=int(args.shard_count),
            per_uuid_timeout_seconds=int(args.per_uuid_timeout_seconds),
            capture_output=bool(args.capture_output),
            ground_truth_jsonl=args.ground_truth_placeholder,
            base_env=base_env,
        )
        _wait_all(procs)

        submission_out = args.outputs_dir / f"submission_mabc_all_allstats_clean_{provider}.jsonl"
        report_zh = args.outputs_dir / f"report_mabc_all_{provider}_zh.json"
        report_en = args.outputs_dir / f"report_mabc_all_{provider}_en.json"

        cmd = [
            str(args.python),
            str(merge_py),
            "--part-prefix",
            part_prefix,
            "--out-jsonl",
            str(submission_out),
            "--report-zh",
            str(report_zh),
            "--report-en",
            str(report_en),
        ]
        if args.cleanup_parts:
            cmd.append("--cleanup-parts")

        print("[merge+score]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=base_env)

    # Refresh the minimal score summary after reports are updated.
    if update_txt_py.exists():
        subprocess.run([str(args.python), str(update_txt_py)], check=True, env=base_env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
