"""Run OpenRCA (paper-style RCA-agent) on aiops2021 and score with AIOpsChallengeJudge.

This runs the OpenRCA RCA-agent baseline (controller+executor, code-interpreter style)
using AIOpsChallenge-style telemetry built from the already-prepared CausalRCA pkls.

Constraints:
- GT-free inference (ground truth is only used for scoring)
- Uses the no-leak input json (window only)

Providers:
- qwen: uses DashScope OpenAI-compatible endpoint
- deepseek: uses DeepSeek OpenAI-compatible endpoint

Required env vars (can also be present in mABC/.env):
- QWEN_API_KEY
- DEEPSEEK_API_KEY

Usage:
  .venv/bin/python aiops2021/run_openrca_agent_all_llm_qwen_vs_deepseek.py

Outputs (per provider):
- aiops2021/outputs/submission_openrca_agent_all_allstats_clean_{provider}.jsonl
- aiops2021/outputs/report_openrca_agent_all_{provider}_{zh,en}.json
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
    out_prefix: str,
    shard_count: int,
    per_uuid_timeout_seconds: int,
    phase: str,
    dataset_root: Path,
    base_env: Dict[str, str],
) -> List[subprocess.Popen]:
    procs: List[subprocess.Popen] = []

    for shard_index in range(shard_count):
        out_jsonl = Path(f"{out_prefix}.part{shard_index}.jsonl")
        cmd = [
            str(python_exe),
            str(runner_py),
            "--phase",
            str(phase),
            "--dataset-root",
            str(dataset_root),
            "--out-jsonl",
            str(out_jsonl),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
            "--per-uuid-timeout-seconds",
            str(per_uuid_timeout_seconds),
        ]

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
    )
    ap.add_argument("--shard-count", type=int, default=2)
    ap.add_argument("--per-uuid-timeout-seconds", type=int, default=900)
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
        help="Used only to load provider keys from mABC/.env",
    )
    ap.add_argument(
        "--phase",
        choices=["phase1", "phase2"],
        default="phase1",
        help="Which dataset phase to run",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root containing dataset/AIOpsChallenge/...",
    )
    ap.add_argument(
        "--pkls-dir",
        type=Path,
        default=Path("CausalRCA/data_aiops/aiops2021_all_allstats_clean/pkls"),
        help="CausalRCA-generated pkls (GT-free telemetry)",
    )
    ap.add_argument(
        "--input-json",
        type=Path,
        default=Path("aiops2021/inputs/all_input_clean.json"),
        help="No-leak input.json (used only for uuid/window instruction)",
    )
    ap.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("aiops2021/outputs"),
    )
    ap.add_argument(
        "--cleanup-parts",
        action="store_true",
        help="Delete part*.jsonl after successful merge+score",
    )
    ap.add_argument(
        "--skip-build-dataset",
        action="store_true",
        help="Skip rebuilding dataset/AIOpsChallenge/{phase}/ from pkls + clean input.json",
    )
    return ap.parse_args(argv)


def _set_openrca_llm_env(provider: str, base_env: Dict[str, str]) -> Dict[str, str]:
    env = dict(base_env)

    if provider == "qwen":
        key = (env.get("QWEN_API_KEY") or os.environ.get("QWEN_API_KEY") or "").strip()
        if not key:
            raise SystemExit("Missing QWEN_API_KEY")
        env["API_KEY"] = key
        env["OPENRCA_API_CONFIG_PATH"] = str(Path("OpenRCA/rca/api_config.yaml"))

    elif provider == "deepseek":
        key = (env.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        if not key:
            raise SystemExit("Missing DEEPSEEK_API_KEY")
        env["API_KEY"] = key
        env["OPENRCA_API_CONFIG_PATH"] = str(Path("OpenRCA/rca/api_config_deepseek.yaml"))

    else:
        raise SystemExit(f"Unknown provider: {provider}")

    return env


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    env_from_file = _load_env_file(args.mabc_root / ".env")
    base_env = dict(os.environ)
    base_env.update(env_from_file)

    # 1) Build OpenRCA dataset (query.csv + telemetry csvs). GT-free.
    build_py = Path("OpenRCA/AIOpsChallenge_Adapt/build_openrca_dataset_from_causalrca_pkls.py")
    if not args.skip_build_dataset:
        cmd = [
            str(args.python),
            str(build_py),
            "--phase",
            str(args.phase),
            "--pkls-dir",
            str(args.pkls_dir),
            "--input-json",
            str(args.input_json),
            "--dataset-root",
            str(args.dataset_root),
        ]
        print("[build-dataset]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=base_env)

    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    runner_py = Path("OpenRCA") / "AIOpsChallenge_Adapt" / "run_openrca_agent_and_build_submission.py"
    merge_py = Path("aiops2021") / "merge_and_score_mabc_all_llm.py"
    update_txt_py = Path("aiops2021") / "update_results_txt_minimal.py"

    for provider in args.providers:
        _require_keys(provider, base_env)
        provider_env = _set_openrca_llm_env(provider, base_env)

        part_prefix = f"submission_openrca_agent_all_llm_{provider}"
        out_prefix = args.outputs_dir / part_prefix

        print(f"\n===== OpenRCA(RCA-agent) provider={provider} =====", flush=True)
        procs = _spawn_shards(
            python_exe=args.python,
            runner_py=runner_py,
            out_prefix=str(out_prefix),
            shard_count=int(args.shard_count),
            per_uuid_timeout_seconds=int(args.per_uuid_timeout_seconds),
            phase=str(args.phase),
            dataset_root=args.dataset_root,
            base_env=provider_env,
        )
        _wait_all(procs)

        submission_out = args.outputs_dir / f"submission_openrca_agent_all_allstats_clean_{provider}.jsonl"
        report_zh = args.outputs_dir / f"report_openrca_agent_all_{provider}_zh.json"
        report_en = args.outputs_dir / f"report_openrca_agent_all_{provider}_en.json"

        cmd = [
            str(args.python),
            str(merge_py),
            "--part-prefix",
            part_prefix,
            "--shard-count",
            str(int(args.shard_count)),
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
        subprocess.run(cmd, check=True, env=provider_env)

    if update_txt_py.exists():
        subprocess.run([str(args.python), str(update_txt_py)], check=True, env=base_env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
