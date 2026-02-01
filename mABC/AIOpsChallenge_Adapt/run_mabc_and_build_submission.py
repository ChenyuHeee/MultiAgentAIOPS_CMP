from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import threading
import time
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -------------------------
# Parsing helpers
# -------------------------

_ENDPOINT_RE = re.compile(r"Root\s*Cause\s*Endpoint\s*:\s*(.+)", re.IGNORECASE)
_REASON_RE = re.compile(r"Root\s*Cause\s*Reason\s*:\s*(.+)", re.IGNORECASE)


def _split_gt_component_parts(component: str) -> List[str]:
    # Judge uses: gt.replace('->','+').split('+')
    comp = (component or "").strip()
    if not comp:
        return []
    parts = [p.strip() for p in comp.replace("->", "+").split("+") if p.strip()]
    return parts


def _load_gt_component_vocab(gt_jsonl: Path) -> Dict[str, str]:
    """Return mapping {lower_token: original_token} from ground truth JSONL."""

    lower_to_token: Dict[str, str] = {}
    if not gt_jsonl.exists():
        return lower_to_token

    for raw_line in gt_jsonl.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        parts = _split_gt_component_parts(str(obj.get("component", "")))
        for tok in parts:
            low = tok.lower().strip()
            if not low:
                continue
            # Prefer longer token if collisions happen (rare, but safe).
            prev = lower_to_token.get(low)
            if prev is None or len(tok) > len(prev):
                lower_to_token[low] = tok

    return lower_to_token


def _guess_gt_file(mabc_root: Path, uuids: Sequence[str]) -> Optional[Path]:
    """Best-effort pick between phase1/phase2 GT based on uuid overlap."""

    repo_root = mabc_root.parent.parent  # .../AIOPS
    judge_dir = repo_root / "AIOpsChallengeJudge"
    cand = [
        judge_dir / "ground_truth_phase1.jsonl",
        judge_dir / "ground_truth_phase12.jsonl",
        judge_dir / "ground_truth_phase2.jsonl",
    ]

    uuid_set = {u for u in uuids if u}
    if not uuid_set:
        return None

    best_path: Optional[Path] = None
    best_hit = -1
    for path in cand:
        if not path.exists():
            continue
        try:
            gt_uuids = set()
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                uid = str(obj.get("uuid", "")).strip()
                if uid:
                    gt_uuids.add(uid)
            hit = len(uuid_set & gt_uuids)
        except Exception:
            continue
        if hit > best_hit:
            best_hit = hit
            best_path = path

    return best_path


def _extract_primary_token(endpoint: str) -> str:
    s = (endpoint or "").strip()
    if not s:
        return ""
    # Remove trailing punctuation and take first chunk.
    s = s.strip().strip(".,; ")
    s = s.splitlines()[0].strip()
    # If an arrow form appears, keep the whole (we will split later).
    # Otherwise, take the prefix before the first path separator.
    if "/" in s:
        s = s.split("/", 1)[0].strip()
    # Drop port if present.
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    return s


def _best_vocab_match(text: str, vocab_lower_to_token: Dict[str, str]) -> str:
    if not text or not vocab_lower_to_token:
        return ""

    s = text.strip()
    s_low = s.lower()

    best_low = ""
    best_len = 0

    # Prefer longest prefix match with a reasonable boundary.
    boundary_chars = set("/ .,:;?()[]{}<>\"'\t\n\r")

    for tok_low in vocab_lower_to_token.keys():
        if not tok_low:
            continue
        if not s_low.startswith(tok_low):
            continue
        nxt_idx = len(tok_low)
        if nxt_idx < len(s_low):
            nxt = s_low[nxt_idx]
            if nxt.isalnum() or nxt == "_":
                # Require boundary (so 'cart' won't match 'cartservice').
                continue
            # hyphen is treated as a boundary because endpoints often look like 'svc-/path'.
            if nxt not in boundary_chars and nxt != "-":
                continue
        if len(tok_low) > best_len:
            best_low = tok_low
            best_len = len(tok_low)

    if best_low:
        return vocab_lower_to_token[best_low]
    return ""


def endpoint_to_component(endpoint: str, vocab_lower_to_token: Dict[str, str]) -> str:
    """Map mABC endpoint string to AIOpsChallenge component token.

    Judge rule: submission component must exactly equal one of GT component parts
    (GT splits `a->b` into tokens `a` and `b`).
    """

    endpoint = (endpoint or "").strip()
    if not endpoint:
        return ""

    # If endpoint already includes an arrow, try to pick any part that is in vocab.
    if "->" in endpoint:
        parts = [p.strip() for p in endpoint.split("->") if p.strip()]
        for p in parts:
            m = _best_vocab_match(_extract_primary_token(p), vocab_lower_to_token)
            if m:
                return m

    primary = _extract_primary_token(endpoint)

    # 1) Exact / prefix match against GT vocab.
    m = _best_vocab_match(primary, vocab_lower_to_token) or _best_vocab_match(endpoint, vocab_lower_to_token)
    if m:
        return m

    # 2) Heuristic fallback: service-level token (common mABC forms)
    # e.g.
    #  - frontend-/api/v1/...
    #  - frontend-hipstershop.CurrencyService/GetSupportedCurrencies
    #  - cartservice
    s = primary
    if "-" in s:
        s0 = s.split("-", 1)[0].strip()
        m2 = vocab_lower_to_token.get(s0.lower())
        if m2:
            return m2
        # Keep original casing to maximize strict-match chance when GT tokens are cased.
        return s0
    return s


def parse_root_cause(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""

    endpoint = ""
    reason = ""

    m = _ENDPOINT_RE.search(text)
    if m:
        endpoint = m.group(1).strip().rstrip(".,; ")

    m = _REASON_RE.search(text)
    if m:
        reason = m.group(1).strip().rstrip(".,; ")

    # fallback: sometimes written as "Root Cause: XXX" etc.
    if not endpoint:
        for line in text.splitlines():
            if "endpoint" in line.lower() and ":" in line:
                endpoint = line.split(":", 1)[1].strip().rstrip(".,; ")
                break

    return endpoint, reason


# -------------------------
# mABC runner
# -------------------------


@dataclass
class MabcResult:
    uuid: str
    minute: str
    alert_endpoint: str
    root_endpoint: str
    root_reason: str
    raw_answer: str
    raw_final: str
    run_log: str


def _ensure_mabc_importable(mabc_root: Path) -> None:
    # mABC code uses imports like `from agents...`, which require mabc_root on sys.path.
    mabc_root_s = str(mabc_root)
    if mabc_root_s not in sys.path:
        sys.path.insert(0, mabc_root_s)


def build_question(alert_endpoint: str, minute: str) -> str:
    return f"""Backgroud: In a distributed microservices system, there is a lot of traces across endpoints which represent the dependency relationship between endpoints. A trace consists of a sequence of spans, each representing a call from one endpoint to another when ignore the service level.

Alert generally occurs on the top endpoint at time T for a significant anomaly when the root cause endpoint at time T' is the downstream endpoint of the alerting endpoint. Endpoint A(TA) -> Endpoint B(TB) -> Endpoint C(TC) -> Endpoint D(TD), if the alert occurs on the Endpoint A at time TA, the root cause endpoint is the Endpoint C at time TC when the metric of Endpoint C is abnormal but the metric of Endpoint D at time TD is normal.

Alert: Endpoint {alert_endpoint} experiencing a significant increase in response time {minute}.
Task: Please find the root cause endpoint behind the alerting endpoint {alert_endpoint} by analyzing the metric of endpoint and the call trace.
Format: Root Cause Endpoint: XXX, Root Cause Reason: XXX
"""


def _get_openai_key() -> str:
    # Support DeepSeek OpenAI-compatible env naming as well.
    return (
        (os.environ.get("OPENAI_API_KEY") or "")
        or (os.environ.get("DEEPSEEK_API_KEY") or "")
        or (os.environ.get("QWEN_API_KEY") or "")
    ).strip()


def _apply_qwen_compat_defaults() -> None:
    """Allow using Qwen (DashScope) OpenAI-compatible API via QWEN_* vars.

    Map QWEN_* to OPENAI_* so the underlying OpenAI SDK client works unchanged.
    """

    qwen_key = (os.environ.get("QWEN_API_KEY") or "").strip()
    if qwen_key and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        os.environ["OPENAI_API_KEY"] = qwen_key

    qwen_base = (
        (os.environ.get("QWEN_BASE_URL") or os.environ.get("QWEN_API_BASE_URL") or "").strip()
    )
    if qwen_base and not (os.environ.get("OPENAI_BASE_URL") or "").strip():
        os.environ["OPENAI_BASE_URL"] = qwen_base

    qwen_model = (os.environ.get("QWEN_MODEL") or "").strip()
    if qwen_model and not (os.environ.get("OPENAI_MODEL") or "").strip():
        os.environ["OPENAI_MODEL"] = qwen_model

    if qwen_key:
        os.environ.setdefault(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        os.environ.setdefault("OPENAI_MODEL", "qwen-plus")


def _load_env_file(env_path: Path) -> None:
    """Minimal .env loader without external deps (no override)."""

    try:
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    except Exception:
        return


def _apply_deepseek_compat_defaults() -> None:
    """If user provides only DEEPSEEK_API_KEY, map to OPENAI_* defaults."""

    deepseek_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if deepseek_key and not (os.environ.get("OPENAI_API_KEY") or "").strip():
        os.environ["OPENAI_API_KEY"] = deepseek_key

    if deepseek_key:
        os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        os.environ.setdefault("OPENAI_MODEL", "deepseek-chat")


def _force_llm_provider(provider: str) -> None:
    """Force OPENAI_* env vars to a specific provider.

    This makes it possible to run qwen/deepseek separately even if both keys
    are present in the environment.
    """

    p = (provider or "").strip().lower()
    if not p or p == "auto":
        return

    if p == "qwen":
        qwen_key = (os.environ.get("QWEN_API_KEY") or "").strip()
        if not qwen_key:
            raise ValueError("--llm-provider=qwen requires QWEN_API_KEY")
        os.environ["OPENAI_API_KEY"] = qwen_key

        qwen_base = (
            (os.environ.get("QWEN_BASE_URL") or os.environ.get("QWEN_API_BASE_URL") or "").strip()
        )
        os.environ["OPENAI_BASE_URL"] = (
            qwen_base or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        qwen_model = (os.environ.get("QWEN_MODEL") or "").strip()
        os.environ["OPENAI_MODEL"] = qwen_model or "qwen-plus"
        return

    if p == "deepseek":
        deepseek_key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
        if not deepseek_key:
            raise ValueError("--llm-provider=deepseek requires DEEPSEEK_API_KEY")
        os.environ["OPENAI_API_KEY"] = deepseek_key

        deepseek_base = (os.environ.get("DEEPSEEK_BASE_URL") or "").strip()
        os.environ["OPENAI_BASE_URL"] = deepseek_base or "https://api.deepseek.com/v1"

        deepseek_model = (os.environ.get("DEEPSEEK_MODEL") or "").strip()
        os.environ["OPENAI_MODEL"] = deepseek_model or "deepseek-chat"
        return

    raise ValueError("--llm-provider must be one of: auto, qwen, deepseek")


def _clip(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_mabc_once(
    mabc_root: Path,
    alert_endpoint: str,
    minute: str,
    *,
    capture_output: bool,
    max_log_chars: int,
) -> Tuple[str, str, str, str, str]:
    """Returns (root_endpoint, root_reason, raw_answer, raw_final, run_log)."""

    from agents.base.profile import (
        DataDetective,
        DependencyExplorer,
        ProbabilityOracle,
        FaultMapper,
        AlertReceiver,
        ProcessScheduler,
        SolutionEngineer,
    )
    from agents.base.run import ReActTotRun, ThreeHotCotRun
    from agents.tools import process_scheduler_tools, solution_engineer_tools

    question = build_question(alert_endpoint=alert_endpoint, minute=minute)

    agents = [
        DataDetective(),
        DependencyExplorer(),
        ProbabilityOracle(),
        FaultMapper(),
        AlertReceiver(),
        ProcessScheduler(),
        SolutionEngineer(),
    ]

    old_cwd = os.getcwd()
    raw_answer = ""
    raw_final = ""
    run_log = ""
    try:
        os.chdir(mabc_root)

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        stdout_cm = contextlib.redirect_stdout(stdout_buf) if capture_output else contextlib.nullcontext()
        stderr_cm = contextlib.redirect_stderr(stderr_buf) if capture_output else contextlib.nullcontext()
        with stdout_cm, stderr_cm:
            # stage 1: orchestrated analysis
            raw_answer = ReActTotRun().run(
                agent=ProcessScheduler(),
                question=question,
                agent_tool_env=vars(process_scheduler_tools),
                eval_run=ThreeHotCotRun(alpha=0, beta=0),
                agents=agents,
            )

            # stage 2: final extraction
            followup = (
                "Base on the analysis, what is the root cause endpoint?\n\n"
                "Format: Root Cause Endpoint: XXX, Root Cause Reason: XXX\n\n"
                + str(raw_answer)
            )

            raw_final = ReActTotRun().run(
                agent=SolutionEngineer(),
                question=followup,
                agent_tool_env=vars(solution_engineer_tools),
                eval_run=ThreeHotCotRun(),
                agents=[SolutionEngineer()],
            )

        if capture_output:
            run_log = "\n".join(
                [
                    "[mABC stdout]",
                    _clip(stdout_buf.getvalue(), max_log_chars),
                    "",
                    "[mABC stderr]",
                    _clip(stderr_buf.getvalue(), max_log_chars),
                ]
            ).strip()
    finally:
        os.chdir(old_cwd)

    root_endpoint, root_reason = parse_root_cause(str(raw_final) or str(raw_answer))
    return root_endpoint, root_reason, str(raw_answer), str(raw_final), run_log


def _mabc_child_entry(
    q: Any,
    mabc_root_s: str,
    alert_endpoint: str,
    minute: str,
    capture_output: bool,
    max_log_chars: int,
) -> None:
    """Child process entry to isolate potential hangs in LLM/tooling.

    Communicates back via a multiprocessing Queue.
    """

    try:
        res = run_mabc_once(
            mabc_root=Path(mabc_root_s),
            alert_endpoint=alert_endpoint,
            minute=minute,
            capture_output=capture_output,
            max_log_chars=max_log_chars,
        )
        q.put({"ok": True, "res": res})
    except Exception as exc:
        q.put({"ok": False, "err": repr(exc)})


def run_mabc_once_with_timeout(
    *,
    mabc_root: Path,
    alert_endpoint: str,
    minute: str,
    capture_output: bool,
    max_log_chars: int,
    timeout_seconds: int,
) -> Tuple[str, str, str, str, str]:
    """Wrapper that prevents a single uuid from hanging the whole batch."""

    t = int(timeout_seconds or 0)
    if t <= 0:
        return run_mabc_once(
            mabc_root=mabc_root,
            alert_endpoint=alert_endpoint,
            minute=minute,
            capture_output=capture_output,
            max_log_chars=max_log_chars,
        )

    # Ensure the spawned child process uses the same interpreter as the parent.
    # This avoids cases where spawn falls back to a different Python (missing deps like `openai`).
    try:
        multiprocessing.set_executable(sys.executable)
    except Exception:
        pass

    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_mabc_child_entry,
        args=(
            q,
            str(mabc_root),
            alert_endpoint,
            minute,
            bool(capture_output),
            int(max_log_chars),
        ),
        daemon=True,
    )
    p.start()
    p.join(t)
    if p.is_alive():
        p.terminate()
        p.join(5)
        raise TimeoutError(f"mABC run exceeded {t}s for alert_endpoint={alert_endpoint} minute={minute}")

    if not q.empty():
        msg = q.get()
        if msg.get("ok") is True:
            return tuple(msg.get("res"))  # type: ignore[return-value]
        raise RuntimeError(f"mABC child failed: {msg.get('err')}")

    raise RuntimeError("mABC child exited without returning a result")


# -------------------------
# CLI
# -------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run mABC per uuid and build AIOpsChallengeJudge submission JSONL"
    )
    p.add_argument(
        "--mabc-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to cmp/mABC (contains agents/, data/)",
    )
    p.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help=(
            "Optional phase input.json. If omitted, the runner will use uuids from label_uuid_map.json. "
            "This is useful when you already have mABC data/label outputs but not the original input file."
        ),
    )
    p.add_argument(
        "--uuid-map",
        type=Path,
        default=None,
        help="label_uuid_map.json from build_mabc_inputs (default: <mabc-root>/data/label/label_uuid_map.json)",
    )
    p.add_argument("--out-jsonl", type=Path, required=True, help="output submission.jsonl")
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="limit total uuids written for a quick run; 0 means all",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="limit actual mABC invocations (skipped uuids do not count); 0 means all",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop immediately when mABC run fails",
    )
    p.add_argument(
        "--only-mapped",
        action="store_true",
        help="only emit uuids that have a label mapping (useful for partial data smoke tests)",
    )
    p.add_argument(
        "--save-raw",
        type=Path,
        default=None,
        help="optional path to save raw mABC outputs as jsonl for debugging",
    )
    p.add_argument(
        "--ground-truth-jsonl",
        type=Path,
        default=None,
        help="optional ground truth jsonl path to build strict component vocab (auto-detected if omitted)",
    )

    p.add_argument(
        "--llm-provider",
        type=str,
        default="auto",
        choices=["auto", "qwen", "deepseek"],
        help=(
            "Which LLM provider to use. 'auto' keeps existing env-based behavior; "
            "'qwen' forces OPENAI_* to use QWEN_*; 'deepseek' forces OPENAI_* to use DEEPSEEK_*.")
    )
    p.add_argument(
        "--capture-output",
        dest="capture_output",
        action="store_true",
        default=True,
        help="capture and suppress mABC stdout/stderr (default: enabled)",
    )
    p.add_argument(
        "--no-capture-output",
        dest="capture_output",
        action="store_false",
        help="do not capture mABC stdout/stderr (may be very noisy)",
    )
    p.add_argument(
        "--max-log-chars",
        type=int,
        default=20000,
        help="max chars to keep for captured stdout and stderr each (default: 20000)",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="process only uuids where (entry_index %% shard_count) == shard_index (default: 0)",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="total shard count for parallel runs (default: 1)",
    )

    p.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=int(os.environ.get("RUNNER_HEARTBEAT_SECONDS", "60")),
        help=(
            "print a heartbeat line every N seconds even if processed does not change "
            "(helps detect hangs). Set 0 to disable. Default: env RUNNER_HEARTBEAT_SECONDS or 60"
        ),
    )

    p.add_argument(
        "--per-uuid-timeout-seconds",
        type=int,
        default=int(os.environ.get("RUNNER_PER_UUID_TIMEOUT_SECONDS", "900")),
        help=(
            "hard timeout for each uuid's mABC run (runs in a separate process). "
            "Prevents indefinite hangs. Set 0 to disable. Default: env RUNNER_PER_UUID_TIMEOUT_SECONDS or 900"
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    shard_count = int(args.shard_count) if args.shard_count else 1
    shard_index = int(args.shard_index) if args.shard_index else 0
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not (0 <= shard_index < shard_count):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")

    mabc_root: Path = args.mabc_root
    uuid_map_path: Path = args.uuid_map or (mabc_root / "data" / "label" / "label_uuid_map.json")

    # Load mABC/.env if present so batch runs work without manual exports.
    _load_env_file(mabc_root / ".env")

    # Explicit provider override to support separate qwen/deepseek runs.
    _force_llm_provider(getattr(args, "llm_provider", "auto"))

    # Prefer Qwen when multiple providers are configured.
    _apply_qwen_compat_defaults()
    _apply_deepseek_compat_defaults()

    if not uuid_map_path.exists():
        raise FileNotFoundError(
            f"Missing uuid map: {uuid_map_path}. Please run build_mabc_inputs first."
        )

    _ensure_mabc_importable(mabc_root)

    uuid_map: Dict[str, Dict[str, str]] = json.loads(uuid_map_path.read_text(encoding="utf-8"))

    if args.input_json is not None:
        entries: List[Dict[str, Any]] = json.loads(args.input_json.read_text(encoding="utf-8"))
        all_uuids = [str(it.get("uuid", "")).strip() for it in entries if isinstance(it, dict)]
    else:
        # Fall back to uuid_map keys (sorted for stability)
        all_uuids = sorted([u for u in uuid_map.keys() if isinstance(u, str) and u.strip()])
        entries = [{"uuid": u} for u in all_uuids]

    gt_path = args.ground_truth_jsonl or _guess_gt_file(mabc_root, all_uuids)
    component_vocab = _load_gt_component_vocab(gt_path) if gt_path else {}

    openai_key_present = bool(_get_openai_key())

    # Incremental writer (append + flush) so long runs can be resumed.
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)

    done_uuids: set[str] = set()
    if args.out_jsonl.exists():
        try:
            for raw_line in args.out_jsonl.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                uid = str(obj.get("uuid", "")).strip()
                if uid:
                    done_uuids.add(uid)
        except Exception:
            raise

    processed = 0
    runs = 0
    progress_every = int(os.environ.get("RUNNER_PROGRESS_EVERY", "10"))
    progress_prefix = f"[progress shard={shard_index}/{shard_count}]"
    total_entries = len([1 for it in entries if isinstance(it, dict) and str(it.get("uuid", "")).strip()])
    shard_entries = (total_entries + shard_count - 1 - shard_index) // shard_count
    print(
        f"[start] shard={shard_index}/{shard_count} total_entries={total_entries} shard_entries≈{shard_entries} openai_key_present={openai_key_present}",
        flush=True,
    )

    # Heartbeat so users can see liveness even if stuck on a single uuid.
    heartbeat_seconds = int(getattr(args, "heartbeat_seconds", 60) or 0)
    started_at = time.time()
    last_activity_at = started_at
    current_uuid = ""
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        nonlocal processed, runs, last_activity_at, current_uuid
        if heartbeat_seconds <= 0:
            return
        while not stop_heartbeat.wait(heartbeat_seconds):
            now = time.time()
            elapsed_s = int(now - started_at)
            idle_s = int(now - last_activity_at)
            # Print to stderr so it is visible even when mABC stdout is captured.
            print(
                (
                    f"[heartbeat shard={shard_index}/{shard_count}] "
                    f"processed={processed}/{shard_entries} runs={runs} "
                    f"idle_s={idle_s} elapsed_s={elapsed_s} current_uuid={current_uuid} "
                    f"out={args.out_jsonl.name}"
                ),
                file=sys.stderr,
                flush=True,
            )

    hb_thread: Optional[threading.Thread] = None
    if heartbeat_seconds > 0:
        hb_thread = threading.Thread(target=_heartbeat, name="runner-heartbeat", daemon=True)
        hb_thread.start()
    try:
        with args.out_jsonl.open("a", encoding="utf-8") as out_f, (
            args.save_raw.open("a", encoding="utf-8") if args.save_raw else contextlib.nullcontext()
        ) as raw_f:
            for entry_index, item in enumerate(entries):
                if shard_count > 1 and (entry_index % shard_count) != shard_index:
                    continue
                uuid = str(item.get("uuid", "")).strip()
                if not uuid:
                    continue

                if uuid in done_uuids:
                    continue

                current_uuid = uuid
                last_activity_at = time.time()

                sel = uuid_map.get(uuid)
                if not sel:
                    if args.only_mapped:
                        continue
                    record = {
                        "uuid": uuid,
                        "component": "",
                        "reason": "",
                        "reasoning_trace": [
                            {
                                "step": 1,
                                "action": "adapter_skip",
                                "observation": "No trace coverage for this uuid window; no label mapping produced.",
                            }
                        ],
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed += 1
                    last_activity_at = time.time()
                    if progress_every > 0 and processed % progress_every == 0:
                        print(f"{progress_prefix} processed={processed} runs={runs}", flush=True)
                    if args.max_samples and processed >= args.max_samples:
                        break
                    continue

                minute = str(sel.get("minute", "")).strip()
                alert_endpoint = str(sel.get("alert_endpoint", "")).strip()

                if not openai_key_present:
                    root_endpoint = ""
                    root_reason = ""
                    raw_answer = ""
                    raw_final = "OPENAI_API_KEY/DEEPSEEK_API_KEY is not set; skipped mABC run."
                    run_log = ""
                else:
                    try:
                        root_endpoint, root_reason, raw_answer, raw_final, run_log = run_mabc_once_with_timeout(
                            mabc_root=mabc_root,
                            alert_endpoint=alert_endpoint,
                            minute=minute,
                            capture_output=bool(args.capture_output),
                            max_log_chars=int(args.max_log_chars),
                            timeout_seconds=int(getattr(args, "per_uuid_timeout_seconds", 0) or 0),
                        )
                        runs += 1
                    except Exception as exc:
                        if args.fail_fast:
                            raise
                        root_endpoint = ""
                        root_reason = ""
                        raw_answer = ""
                        raw_final = f"mABC run failed: {exc}"
                        run_log = ""

                last_activity_at = time.time()

                # Component must strictly match one of GT tokens. Prefer mapping from root endpoint;
                # if mABC fails to extract, fall back to alert endpoint (better than empty for LA).
                component_source = root_endpoint or alert_endpoint
                component = endpoint_to_component(component_source, component_vocab)

                record = {
                    "uuid": uuid,
                    "component": component,
                    "reason": root_reason or "",
                    "reasoning_trace": [
                        {"step": 1, "action": "mabc_analysis", "observation": raw_answer or ""},
                        {"step": 2, "action": "mabc_final", "observation": raw_final or ""},
                    ],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                if raw_f is not None:
                    raw_f.write(
                        json.dumps(
                            MabcResult(
                                uuid=uuid,
                                minute=minute,
                                alert_endpoint=alert_endpoint,
                                root_endpoint=root_endpoint,
                                root_reason=root_reason,
                                raw_answer=raw_answer,
                                raw_final=raw_final,
                                run_log=run_log,
                            ).__dict__,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    raw_f.flush()

                processed += 1
                last_activity_at = time.time()
                if progress_every > 0 and processed % progress_every == 0:
                    print(f"{progress_prefix} processed={processed} runs={runs}", flush=True)
                if args.max_samples and processed >= args.max_samples:
                    break

                if args.max_runs and runs >= args.max_runs:
                    break

    except KeyboardInterrupt:
        stop_heartbeat.set()
        print(f"\n[WARN] Interrupted. Progress kept in: {args.out_jsonl}", file=sys.stderr)
        if args.save_raw:
            print(f"[WARN] Raw progress kept in: {args.save_raw}", file=sys.stderr)
        print(f"{progress_prefix} processed={processed} runs={runs}", flush=True)
        return 130

    stop_heartbeat.set()

    print(f"Wrote submission (append/resumable): {args.out_jsonl}")
    if args.save_raw:
        print(f"Wrote raw outputs (append/resumable): {args.save_raw}")
    print(f"Processed uuids this run: {processed} (mABC runs this run: {runs})")
    print(f"Already done uuids skipped: {len(done_uuids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
