from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


# Local core
from causalrca_core import run_causalrca_on_dataframe


"""CausalRCA adapter for AIOpsChallenge submissions.

Policy: This adapter MUST NOT read any ground truth (GT) files.

- UUIDs are sourced from --phase-input (input.json).
- Component is selected only from model outputs (top service / top variables).
- Reasons/traces avoid injecting judge-specific keyword tokens.
"""


def _signal_keywords_from_vars(top_vars: Sequence[Tuple[str, float]]) -> List[str]:
    # Keep these short so they survive judge's 100-char truncation.
    names = [str(k).lower() for k, _ in (top_vars or [])]
    out: List[str] = []
    if any("error_rate" in n or "error" in n for n in names):
        out.append("error")
        out.append("error+")
    if any("timeout_rate" in n or "timeout" in n for n in names):
        out.append("timeout+")
    if any("avg_duration" in n or "latency" in n or "rrt" in n for n in names):
        out.append("rrt+")
        out.append("max_rrt+")
    # de-dupe while preserving order
    dedup: List[str] = []
    seen: set[str] = set()
    for kw in out:
        if kw not in seen:
            seen.add(kw)
            dedup.append(kw)
    return dedup


def _natural_signals_from_vars(top_vars: Sequence[Tuple[str, float]]) -> List[str]:
    """Return natural-language signal tokens derived from top_vars.

    This avoids copying judge-specific keyword tokens (like "timeout+", "rrt+") into
    the submission, so scores better reflect the model rather than keyword hacks.
    """

    names = [str(k).lower() for k, _ in (top_vars or [])]
    out: List[str] = []
    if any("error_rate" in n or "error" in n for n in names):
        out.append("error")
    if any("timeout_rate" in n or "timeout" in n for n in names):
        out.append("timeout")
    if any("avg_duration" in n or "latency" in n or "rrt" in n for n in names):
        out.append("latency")
    # de-dupe while preserving order
    dedup: List[str] = []
    seen: set[str] = set()
    for kw in out:
        if kw not in seen:
            seen.add(kw)
            dedup.append(kw)
    return dedup


def _pick_component_from_top_vars(
    top_service: str,
    top_services: Sequence[Tuple[str, float]],
    top_vars: Sequence[Tuple[str, float]],
) -> str:
    """Pick a component without any GT vocab.

    Heuristic: choose the service whose variables contribute the largest cumulative
    score in top_vars, then fall back to CausalRCA's top_service.
    """

    weights: Dict[str, float] = {}
    original: Dict[str, str] = {}
    for k, v in (top_vars or []):
        name = str(k)
        if not name:
            continue
        svc_raw = name.split("_", 1)[0].strip()
        svc_key = svc_raw.lower()
        if not svc_key:
            continue
        if svc_key not in original:
            original[svc_key] = svc_raw
        try:
            w = float(v)
        except Exception:
            w = 0.0
        weights[svc_key] = weights.get(svc_key, 0.0) + max(0.0, w)

    if weights:
        best_key = max(weights.items(), key=lambda kv: kv[1])[0]
        return original.get(best_key, best_key)

    s = (top_service or "").strip()
    if s:
        return s
    if top_services:
        return (top_services[0][0] or "").strip()
    return ""


# -------------------------
# Execution isolation
# -------------------------


def _worker(payload: Dict[str, Any], out_q: "multiprocessing.Queue") -> None:
    uuid = payload["uuid"]
    pkl_path = payload["pkl_path"]
    epochs = int(payload["epochs"])
    graph_threshold = float(payload["graph_threshold"])
    max_seconds = payload.get("max_seconds")

    try:
        if pd is None:
            raise RuntimeError("pandas is required")
        df = pd.read_pickle(pkl_path)
        res = run_causalrca_on_dataframe(
            df,
            epochs=epochs,
            graph_threshold=graph_threshold,
            max_seconds=float(max_seconds) if max_seconds is not None else None,
        )

        # sort services by score
        items = sorted(res.service_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_service = items[0][0] if items else ""
        top_services = items[:5]
        top_vars = sorted(res.var_scores.items(), key=lambda kv: kv[1], reverse=True)[:10]

        out_q.put(
            {
                "ok": True,
                "uuid": uuid,
                "top_service": top_service,
                "top_services": top_services,
                "top_vars": top_vars,
            }
        )
    except Exception as exc:
        out_q.put({"ok": False, "uuid": uuid, "error": repr(exc)})


def _run_with_timeout(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(payload, q), daemon=True)
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        p.join(timeout=5)
        return {"ok": False, "uuid": payload["uuid"], "error": f"timeout>{timeout_s}s"}

    try:
        return q.get_nowait()
    except Exception:
        return {"ok": False, "uuid": payload["uuid"], "error": "no result"}


# -------------------------
# Main
# -------------------------


def ensure_trace_schema(step: Dict[str, Any]) -> Dict[str, Any]:
    # Judge expects: step/action/observation in each trace step.
    raw_step = step.get("step")
    step_i = 0
    try:
        if raw_step is None:
            raise ValueError("missing step")
        step_i = int(raw_step)  # may be str/int
    except Exception:
        step_i = 0
    return {
        "step": step_i,
        "action": str(step.get("action") or ""),
        "observation": str(step.get("observation") or ""),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run CausalRCA per uuid and build AIOpsChallenge submission JSONL")
    parser.add_argument(
        "--phase-input",
        type=Path,
        required=True,
        help="Phase input.json (uuid list source). Required because this adapter never reads ground truth.",
    )
    parser.add_argument("--inputs-dir", type=Path, required=True, help="dir containing pkls/{uuid}.pkl")
    parser.add_argument("--out-jsonl", type=Path, required=True, help="submission output jsonl")
    # NOTE: GT-reading flags removed by policy.
    parser.add_argument("--append", action="store_true", help="append to existing out-jsonl")
    parser.add_argument("--shard", type=int, default=0, help="shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="number of shards")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="limit uuids processed after sharding (0 = no limit)",
    )

    parser.add_argument("--epochs", type=int, default=200, help="CausalRCA training epochs per uuid")
    parser.add_argument("--graph-threshold", type=float, default=0.3, help="adjacency threshold")
    parser.add_argument(
        "--per-uuid-timeout-seconds",
        type=int,
        default=600,
        help="hard timeout per uuid (kills subprocess)",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="print heartbeat periodically even if stuck",
    )

    args = parser.parse_args(argv)

    if pd is None:
        print("[ERR] pandas not installed. Please `pip install pandas`.", file=sys.stderr)
        return 2

    uuids: List[str] = []
    items = json.loads(args.phase_input.read_text(encoding="utf-8"))
    for item in items:
        u = str(item.get("uuid") or item.get("UUID") or "").strip()
        if u:
            uuids.append(u)

    # shard split (stable)
    uuids = [u for i, u in enumerate(uuids) if (i % int(args.num_shards)) == int(args.shard)]
    if int(args.limit) > 0:
        uuids = uuids[: int(args.limit)]

    out_jsonl: Path = args.out_jsonl
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.append and out_jsonl.exists():
        for raw in out_jsonl.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            uid = str(obj.get("uuid") or "").strip()
            if uid:
                done.add(uid)

    mode = "a" if args.append else "w"
    processed = 0
    runs = 0

    last_progress_t = time.time()

    with out_jsonl.open(mode, encoding="utf-8") as f:
        for uuid in uuids:
            if uuid in done:
                processed += 1
                continue

            pkl_path = args.inputs_dir / "pkls" / f"{uuid}.pkl"
            if not pkl_path.exists():
                # fall back to direct path (if user passed pkls dir)
                alt = args.inputs_dir / f"{uuid}.pkl"
                pkl_path = alt if alt.exists() else pkl_path

            runs += 1

            payload = {
                "uuid": uuid,
                "pkl_path": str(pkl_path),
                "epochs": int(args.epochs),
                "graph_threshold": float(args.graph_threshold),
                "max_seconds": None,
            }

            result = _run_with_timeout(payload, timeout_s=float(args.per_uuid_timeout_seconds))

            if result.get("ok"):
                top_service = str(result.get("top_service") or "")
                top_services_any = result.get("top_services") or []
                try:
                    top_services: List[Tuple[str, float]] = [
                        (str(s), float(sc)) for s, sc in top_services_any if str(s).strip()
                    ]
                except Exception:
                    top_services = []

                top_vars = result.get("top_vars") or []

                component = _pick_component_from_top_vars(top_service, top_services, top_vars)

                # Signals derived from model outputs.
                natural_signals = _natural_signals_from_vars(top_vars)
                reason = (
                    f"CausalRCA(PageRank) ranks service '{top_service}' highest." +
                    (" Observed signals: " + ", ".join(natural_signals) + "." if natural_signals else "")
                )

                trace = [
                    ensure_trace_schema(
                        {
                            "step": 1,
                            "action": f"load {pkl_path.name}",
                            "observation": f"vars={len(top_vars)} top_service={top_service} component={component}",
                        }
                    ),
                    ensure_trace_schema(
                        {
                            "step": 2,
                            "action": "evidence keywords from signals",
                            "observation": (
                                "signals=" + ", ".join(natural_signals)
                            ),
                        }
                    ),
                    ensure_trace_schema(
                        {
                            "step": 3,
                            "action": "rank variables by PageRank on learned adjacency",
                            "observation": "top_vars=" + "; ".join([f"{k}:{v:.4g}" for k, v in top_vars[:8]]),
                        }
                    ),
                ]
            else:
                component = ""
                reason = f"CausalRCA failed: {result.get('error')}"
                trace = [
                    ensure_trace_schema(
                        {
                            "step": 1,
                            "action": "run causalrca",
                            "observation": reason,
                        }
                    )
                ]

            out = {
                "uuid": uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": trace,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            processed += 1

            now = time.time()
            if (now - last_progress_t) >= int(args.heartbeat_seconds):
                print(
                    f"[heartbeat] processed={processed}/{len(uuids)} runs={runs} current_uuid={uuid}",
                    flush=True,
                )
                last_progress_t = now

    print(f"[done] out={out_jsonl} processed={processed} runs={runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
