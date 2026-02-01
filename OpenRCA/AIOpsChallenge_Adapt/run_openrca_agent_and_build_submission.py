from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


_OPENRCA_ROOT = Path(__file__).resolve().parents[1]
if str(_OPENRCA_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENRCA_ROOT))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _load_existing_uuids(out_jsonl: Path) -> Set[str]:
    if not out_jsonl.exists():
        return set()
    done: Set[str] = set()
    for raw in out_jsonl.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        uuid = str(obj.get("uuid", "")).strip()
        if uuid:
            done.add(uuid)
    return done


def _write_jsonl_append(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_metric_cmdb_ids(metric_csv: Path) -> Iterable[str]:
    if not metric_csv.exists():
        return
    with metric_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cmdb_id = str(r.get("cmdb_id", "")).strip()
            if cmdb_id:
                yield cmdb_id


def _collect_component_candidates(telemetry_root: Path, uuids: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    for uuid in uuids:
        metric_csv = telemetry_root / uuid / "metric" / "metric_service.csv"
        for cmdb_id in _iter_metric_cmdb_ids(metric_csv):
            seen.add(cmdb_id)
    return sorted(seen)


_REASON_CANDIDATES: List[str] = [
    # English
    "network",
    "packet loss",
    "latency",
    "timeout",
    "error",
    "service degradation",
    "high cpu",
    "high memory",
    "high disk read io",
    "high disk write io",
    "high disk space",
    "jvm oom heap",
    "high jvm cpu",
    # Chinese (aiops2021 GT common phrases)
    "网络故障",
    "网络丢包",
    "网络延迟",
    "应用故障",
    "资源故障",
    "CPU使用率高",
    "内存使用率过高",
    "磁盘IO读使用率过高",
    "磁盘IO写使用率过高",
    "磁盘空间使用率过高",
    "JVM CPU负载高",
    "JVM OOM Heap",
]


def _build_prompt_schema_and_cand(*, phase: str, component_candidates: Sequence[str]) -> Tuple[str, str]:
    cand_components = "\n".join([f"- {c}" for c in component_candidates])
    cand_reasons = "\n".join([f"- {r}" for r in _REASON_CANDIDATES])
    cand = (
        "## POSSIBLE ROOT CAUSE REASONS:\n\n"
        + cand_reasons
        + "\n\n## POSSIBLE ROOT CAUSE COMPONENTS:\n\n"
        + cand_components
        + "\n"
    )

    schema = f"""## TELEMETRY DIRECTORY STRUCTURE:

- Telemetry root: `dataset/AIOpsChallenge/{phase}/telemetry/{{uuid}}/`
- Each case directory contains: `metric/`, `trace/`, `log/`
- CSV files:
  - `metric/metric_service.csv`
  - `trace/trace_span.csv` (may be empty)
  - `log/log_service.csv` (may be empty)

## DATA SCHEMA

1) Metric file `metric/metric_service.csv`:

    timestamp,cmdb_id,kpi_name,value

- `timestamp` is Unix seconds (UTC).
- `cmdb_id` is a component/service identifier (string).
- `kpi_name` is one of: calls, avg_duration, error_rate, timeout_rate.
- `value` is numeric.

2) Trace file `trace/trace_span.csv` (may be empty):

    timestamp,cmdb_id,parent_id,span_id,trace_id,duration

3) Log file `log/log_service.csv` (may be empty):

    log_id,timestamp,cmdb_id,log_name,value

{cand}

## TIME ZONE

- All issues use UTC+8 time in reasoning if needed; telemetry timestamps are UTC seconds.
"""
    return schema, cand


@dataclass
class _BasicPrompt:
    schema: str
    cand: str


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):  # pragma: no cover
    raise _Timeout("per-uuid timeout")


def _parse_agent_answer(answer_text: str) -> Tuple[str, str]:
    """Parse controller final JSON.

    Expected format (stringified JSON object):
      {"1": {"root cause component": "...", "root cause reason": "..."}, ...}
    """
    s = (answer_text or "").strip()
    if not s:
        return "", ""
    try:
        obj = json.loads(s)
    except Exception:
        return "", ""
    if not isinstance(obj, dict) or not obj:
        return "", ""

    first = obj.get("1") if "1" in obj else next(iter(obj.values()))
    if not isinstance(first, dict):
        return "", ""

    comp = ""
    reason = ""
    for k, v in first.items():
        key = str(k).lower()
        if not isinstance(v, str):
            continue
        if "component" in key:
            comp = v.strip()
        if "reason" in key:
            reason = v.strip()

    return comp, reason


def _fallback_from_metric(metric_csv: Path) -> Tuple[str, str]:
    """Fallback without GT: pick component with highest anomaly score from metrics."""
    if not metric_csv.exists():
        return "", "service degradation"

    scores: Dict[str, float] = {}
    max_er: Dict[str, float] = {}
    max_tr: Dict[str, float] = {}
    max_dur: Dict[str, float] = {}

    with metric_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cmdb_id = str(r.get("cmdb_id", "")).strip()
            kpi = str(r.get("kpi_name", "")).strip()
            try:
                val = float(r.get("value") or 0.0)
            except Exception:
                val = 0.0
            if not cmdb_id:
                continue

            if kpi == "error_rate":
                max_er[cmdb_id] = max(max_er.get(cmdb_id, 0.0), val)
            elif kpi == "timeout_rate":
                max_tr[cmdb_id] = max(max_tr.get(cmdb_id, 0.0), val)
            elif kpi == "avg_duration":
                max_dur[cmdb_id] = max(max_dur.get(cmdb_id, 0.0), val)

    for cmdb_id in set(max_er) | set(max_tr) | set(max_dur):
        er = max_er.get(cmdb_id, 0.0)
        tr = max_tr.get(cmdb_id, 0.0)
        dur = max_dur.get(cmdb_id, 0.0)
        scores[cmdb_id] = (2.6 * er) + (3.0 * tr) + (0.06 * dur)

    if not scores:
        return "", "service degradation"

    top = max(scores.items(), key=lambda kv: kv[1])[0]
    if max_tr.get(top, 0.0) > 0:
        reason = "timeout"
    elif max_er.get(top, 0.0) > 0:
        reason = "error"
    elif max_dur.get(top, 0.0) > 0:
        reason = "latency"
    else:
        reason = "service degradation"
    return top, reason


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OpenRCA RCA-agent on AIOpsChallenge-style dataset and build submission JSONL")
    p.add_argument("--phase", choices=["phase1", "phase2"], default="phase1")
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--per-uuid-timeout-seconds", type=int, default=900)
    p.add_argument("--controller-max-step", type=int, default=20)
    p.add_argument("--controller-max-turn", type=int, default=5)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    phase_dir = args.dataset_root / "AIOpsChallenge" / args.phase
    query_csv = phase_dir / "query.csv"
    telemetry_root = phase_dir / "telemetry"
    if not query_csv.exists():
        raise FileNotFoundError(f"Missing query.csv: {query_csv}")
    if not telemetry_root.exists():
        raise FileNotFoundError(f"Missing telemetry dir: {telemetry_root}")

    rows = _read_csv_rows(query_csv)
    uuids_all = [str(r.get("task_index", "")).strip() for r in rows]
    uuids_all = [u for u in uuids_all if u]

    if args.limit and args.limit > 0:
        uuids_all = uuids_all[: int(args.limit)]

    # shard filter
    shard_uuids: List[str] = []
    for idx, u in enumerate(uuids_all):
        if int(args.shard_count) <= 1 or (idx % int(args.shard_count)) == int(args.shard_index):
            shard_uuids.append(u)

    done = _load_existing_uuids(args.out_jsonl) if args.resume else set()

    # Build candidates on full set (not just shard) so the agent can select any component.
    component_candidates = _collect_component_candidates(telemetry_root, uuids_all)
    schema, cand = _build_prompt_schema_and_cand(phase=args.phase, component_candidates=component_candidates)
    bp = _BasicPrompt(schema=schema, cand=cand)

    import rca.baseline.rca_agent.prompt.agent_prompt as ap
    from rca.baseline.rca_agent.rca_agent import RCA_Agent

    try:
        from loguru import logger  # type: ignore

        if not args.verbose:
            logger.remove()
            logger.add(sys.stdout, level="INFO")
    except Exception:  # pragma: no cover
        import logging

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("openrca-agent")

    # per-uuid timeout via alarm (works on Unix)
    signal.signal(signal.SIGALRM, _timeout_handler)

    agent = RCA_Agent(ap, bp)

    total = len(shard_uuids)
    logger.info(
        f"[openrca-agent] phase={args.phase} shard={args.shard_index}/{args.shard_count} total={total}"
    )

    for i, uuid in enumerate(shard_uuids, start=1):
        if uuid in done:
            continue

        instruction_from_query = ""
        # preserve original ordering: find row by uuid
        for r in rows:
            if str(r.get("task_index", "")).strip() == uuid:
                instruction_from_query = str(r.get("instruction", "") or "").strip()
                break

        objective = (
            f"{instruction_from_query}\n"
            f"Task: Using ONLY telemetry under dataset/AIOpsChallenge/{args.phase}/telemetry/{uuid}/, "
            f"identify the MOST LIKELY root cause component and root cause reason for this anomaly. "
            f"Return exactly ONE root cause." 
        ).strip()

        metric_csv = telemetry_root / uuid / "metric" / "metric_service.csv"

        try:
            signal.alarm(int(args.per_uuid_timeout_seconds))
            answer_text, trajectory, prompt = agent.run(
                objective,
                logger,
                max_step=int(args.controller_max_step),
                max_turn=int(args.controller_max_turn),
            )
            signal.alarm(0)

            component, reason = _parse_agent_answer(answer_text)
            if not component:
                component, reason_fallback = _fallback_from_metric(metric_csv)
                reason = reason or reason_fallback

            observation = (answer_text or "").strip()
            if len(observation) > 800:
                observation = observation[:800] + "..."

            row = {
                "uuid": uuid,
                "component": component,
                "reason": reason or "service degradation",
                "reasoning_trace": [
                    {
                        "step": 1,
                        "action": "openrca-rca-agent",
                        "observation": observation or "(empty agent output)",
                    }
                ],
            }
            _write_jsonl_append(args.out_jsonl, row)
            logger.info(f"[{i}/{total}] uuid={uuid} component={component} reason={reason}")

        except _Timeout:
            signal.alarm(0)
            component, reason = _fallback_from_metric(metric_csv)
            row = {
                "uuid": uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": [
                    {
                        "step": 1,
                        "action": "openrca-rca-agent-timeout-fallback",
                        "observation": "per-uuid timeout; used metric fallback",
                    }
                ],
            }
            _write_jsonl_append(args.out_jsonl, row)
            logger.warning(f"[{i}/{total}] uuid={uuid} TIMEOUT -> fallback component={component} reason={reason}")

        except Exception as e:
            signal.alarm(0)
            component, reason = _fallback_from_metric(metric_csv)
            row = {
                "uuid": uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": [
                    {
                        "step": 1,
                        "action": "openrca-rca-agent-error-fallback",
                        "observation": f"{type(e).__name__}: {e}",
                    }
                ],
            }
            _write_jsonl_append(args.out_jsonl, row)
            logger.exception(f"[{i}/{total}] uuid={uuid} ERROR -> fallback")

    logger.info(f"[openrca-agent] wrote: {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
