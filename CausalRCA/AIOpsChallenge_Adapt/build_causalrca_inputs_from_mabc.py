from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame


_ISO_Z_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def parse_iso_z(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def minute_bucket_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:00")


def extract_window_minutes(
    anomaly_desc: str,
    max_minutes: int = 120,
    pre_minutes: int = 0,
    post_minutes: int = 0,
) -> List[str]:
    matches = _ISO_Z_RE.findall(anomaly_desc or "")
    if len(matches) < 2:
        return []

    start_dt = parse_iso_z(matches[0])
    end_dt = parse_iso_z(matches[1])
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    start_dt = start_dt.replace(second=0, microsecond=0)
    end_dt = end_dt.replace(second=0, microsecond=0)

    pre = max(0, int(pre_minutes or 0))
    post = max(0, int(post_minutes or 0))
    if pre:
        start_dt = start_dt - timedelta(minutes=pre)
    if post:
        end_dt = end_dt + timedelta(minutes=post)

    minutes: List[str] = []
    cur = start_dt
    while cur <= end_dt and len(minutes) < max_minutes:
        minutes.append(minute_bucket_utc(cur))
        cur = cur + timedelta(minutes=1)
    return minutes


@dataclass
class Agg:
    calls: int = 0
    duration_sum: float = 0.0
    error_calls: float = 0.0
    timeout_calls: float = 0.0


def endpoint_to_service(endpoint: str) -> str:
    s = (endpoint or "").strip()
    if not s:
        return ""
    if "-" in s:
        return s.split("-", 1)[0].strip()
    return s


def _safe_col_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # keep it file/df friendly
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def build_uuid_dataframe(
    endpoint_stats: Mapping[str, Mapping[str, Mapping[str, Any]]],
    minutes: Sequence[str],
    entity_mode: str,
    max_entities: int,
) -> "DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required. Please `pip install pandas`.")

    # minute -> entity -> Agg
    by_minute: Dict[str, Dict[str, Agg]] = {m: {} for m in minutes}
    entity_calls_total: Dict[str, int] = {}
    entity_max_error_rate: Dict[str, float] = {}
    entity_max_timeout_rate: Dict[str, float] = {}
    entity_max_avg_duration: Dict[str, float] = {}

    for endpoint, minute_map in endpoint_stats.items():
        if not isinstance(minute_map, dict):
            continue

        if entity_mode == "service":
            entity = endpoint_to_service(endpoint)
        else:
            entity = endpoint

        entity = entity.strip()
        if not entity:
            continue

        for minute in minutes:
            row_any = minute_map.get(minute)
            if not isinstance(row_any, dict):
                continue
            row: Dict[str, Any] = row_any

            calls = int(row.get("calls") or 0)
            if calls <= 0:
                continue

            avg_dur = float(row.get("average_duration") or 0.0)
            err_rate = float(row.get("error_rate") or 0.0) / 100.0
            to_rate = float(row.get("timeout_rate") or 0.0) / 100.0

            prev = entity_max_error_rate.get(entity)
            if prev is None or err_rate > prev:
                entity_max_error_rate[entity] = err_rate
            prev = entity_max_timeout_rate.get(entity)
            if prev is None or to_rate > prev:
                entity_max_timeout_rate[entity] = to_rate
            prev = entity_max_avg_duration.get(entity)
            if prev is None or avg_dur > prev:
                entity_max_avg_duration[entity] = avg_dur

            m = by_minute[minute].get(entity)
            if m is None:
                m = Agg()
                by_minute[minute][entity] = m

            m.calls += calls
            m.duration_sum += avg_dur * calls
            m.error_calls += err_rate * calls
            m.timeout_calls += to_rate * calls

            entity_calls_total[entity] = entity_calls_total.get(entity, 0) + calls

    # Pick top entities to avoid exploding variable count.
    # Note: this is the only “lossy” step.
    #
    # Default policy keeps high-traffic entities, but also reserves slots for
    # high-signal entities (error/timeout/latency), which tends to help RCA.
    all_by_calls = [e for e, _ in sorted(entity_calls_total.items(), key=lambda kv: kv[1], reverse=True)]

    if max_entities <= 0:
        entities = all_by_calls
    else:
        k = int(max_entities)
        k_signal = max(3, min(10, k // 3))

        by_err = [e for e, _ in sorted(entity_max_error_rate.items(), key=lambda kv: kv[1], reverse=True)][:k_signal]
        by_to = [e for e, _ in sorted(entity_max_timeout_rate.items(), key=lambda kv: kv[1], reverse=True)][:k_signal]
        by_dur = [e for e, _ in sorted(entity_max_avg_duration.items(), key=lambda kv: kv[1], reverse=True)][:k_signal]

        entities = []
        seen: set[str] = set()
        for e in (by_err + by_to + by_dur + all_by_calls):
            if e in seen:
                continue
            seen.add(e)
            entities.append(e)
            if len(entities) >= k:
                break

    records: List[Dict[str, float]] = []
    for minute in minutes:
        row: Dict[str, Any] = {"minute": minute}
        agg_map = by_minute.get(minute, {})
        for entity in entities:
            a = agg_map.get(entity)
            if a is None or a.calls <= 0:
                # keep explicit zeros to avoid NaNs in downstream learning
                row[f"{_safe_col_name(entity)}_calls"] = 0.0
                row[f"{_safe_col_name(entity)}_avg_duration"] = 0.0
                row[f"{_safe_col_name(entity)}_error_rate"] = 0.0
                row[f"{_safe_col_name(entity)}_timeout_rate"] = 0.0
                continue

            row[f"{_safe_col_name(entity)}_calls"] = float(a.calls)
            row[f"{_safe_col_name(entity)}_avg_duration"] = float(a.duration_sum / a.calls)
            row[f"{_safe_col_name(entity)}_error_rate"] = float(a.error_calls / a.calls)
            row[f"{_safe_col_name(entity)}_timeout_rate"] = float(a.timeout_calls / a.calls)

        records.append(row)

    df = pd.DataFrame.from_records(records)
    df = df.set_index("minute")
    return df


def _shift_minute_buckets(minutes: Sequence[str], hours: int) -> List[str]:
    if not minutes:
        return []
    out: List[str] = []
    for m in minutes:
        try:
            dt = datetime.strptime(m, "%Y-%m-%d %H:%M:00")
            dt = dt.replace(tzinfo=timezone.utc) + timedelta(hours=int(hours))
            out.append(minute_bucket_utc(dt))
        except Exception:
            continue
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build CausalRCA inputs from mABC endpoint_stats.json")
    parser.add_argument("--phase", type=str, default="phase1", help="phase name (for bookkeeping only)")
    parser.add_argument("--phase-input", type=Path, required=True, help="AIOps Challenge input.json for the phase")
    parser.add_argument(
        "--mabc-endpoint-stats",
        type=Path,
        default=Path("cmp/mABC/data/metric/endpoint_stats.json"),
        help="mABC endpoint_stats.json (calls/avg_duration/error/timeout per minute)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output dir, will create pkls/ and uuid_to_pkl.json",
    )
    parser.add_argument(
        "--entity-mode",
        choices=["service", "endpoint"],
        default="service",
        help="use service prefix or full endpoint as entity",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=30,
        help="cap number of entities (services/endpoints) per uuid; prevents huge variable graphs",
    )
    parser.add_argument(
        "--max-window-minutes",
        type=int,
        default=120,
        help="cap anomaly window length (safety for malformed inputs)",
    )
    parser.add_argument(
        "--pre-minutes",
        type=int,
        default=0,
        help="extend window backward by N minutes as baseline context",
    )
    parser.add_argument(
        "--post-minutes",
        type=int,
        default=0,
        help="extend window forward by N minutes (optional)",
    )
    args = parser.parse_args(argv)

    if pd is None:
        print("[ERR] pandas not installed. Please `pip install pandas`.", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir
    pkls_dir = out_dir / "pkls"
    pkls_dir.mkdir(parents=True, exist_ok=True)

    endpoint_stats = json.loads(args.mabc_endpoint_stats.read_text(encoding="utf-8"))
    phase_items = json.loads(args.phase_input.read_text(encoding="utf-8"))

    uuid_to_pkl: Dict[str, str] = {}

    empty_feature_uuids: List[str] = []

    total = 0
    for item in phase_items:
        uuid = str(item.get("uuid") or item.get("UUID") or "").strip()
        if not uuid:
            continue
        desc = str(item.get("Anomaly Description") or item.get("anomaly_description") or "")
        minutes = extract_window_minutes(
            desc,
            max_minutes=int(args.max_window_minutes),
            pre_minutes=int(args.pre_minutes),
            post_minutes=int(args.post_minutes),
        )
        if not minutes:
            continue

        df = build_uuid_dataframe(
            endpoint_stats=endpoint_stats,
            minutes=minutes,
            entity_mode=args.entity_mode,
            max_entities=int(args.max_entities),
        )

        if getattr(df, "shape", (0, 0))[1] == 0:
            # Some AIOps2021 inputs have an 8-hour time-base mismatch (UTC vs UTC+8).
            # If we get an empty frame, try shifting the window before giving up.
            recovered = False
            for h in (-8, 8):
                shifted = _shift_minute_buckets(minutes, hours=h)
                if not shifted:
                    continue
                df2 = build_uuid_dataframe(
                    endpoint_stats=endpoint_stats,
                    minutes=shifted,
                    entity_mode=args.entity_mode,
                    max_entities=int(args.max_entities),
                )
                if getattr(df2, "shape", (0, 0))[1] > 0:
                    df = df2
                    recovered = True
                    print(f"[WARN] uuid={uuid} window shifted {h}h to recover non-empty features", flush=True)
                    break

            if not recovered:
                empty_feature_uuids.append(uuid)
                if len(empty_feature_uuids) <= 5:
                    print(
                        f"[WARN] uuid={uuid} has 0 feature columns for its window; "
                        "this usually means the endpoint_stats.json does not cover the phase's minutes.",
                        flush=True,
                    )

        pkl_path = pkls_dir / f"{uuid}.pkl"
        df.to_pickle(pkl_path)
        uuid_to_pkl[uuid] = str(pkl_path)
        total += 1

        if total % 50 == 0:
            print(f"[build] phase={args.phase} built={total}", flush=True)

    (out_dir / "uuid_to_pkl.json").write_text(
        json.dumps(uuid_to_pkl, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if empty_feature_uuids:
        (out_dir / "uuids_empty_features.json").write_text(
            json.dumps(empty_feature_uuids, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(
        f"[done] phase={args.phase} built_uuids={total} out={out_dir} "
        f"empty_feature_uuids={len(empty_feature_uuids)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
