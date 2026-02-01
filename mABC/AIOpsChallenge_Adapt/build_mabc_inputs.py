from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


def _pc_divide(a: Any, b: Any) -> Any:
    """Compat wrapper to avoid stub mismatches across pyarrow versions."""

    fn = getattr(pc, "divide", None) or getattr(pc, "divide_checked", None)
    if fn is None:
        raise AttributeError("pyarrow.compute.divide is not available")
    return fn(a, b)


def _pc_is_in(values: Any, value_set: Any) -> Any:
    """Compat wrapper to avoid stub mismatches across pyarrow versions."""

    fn = getattr(pc, "is_in", None)
    if fn is None:
        raise AttributeError("pyarrow.compute.is_in is not available")
    return fn(values, value_set=value_set)


# -------------------------
# Time helpers
# -------------------------

_ISO_Z_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def parse_iso_z(ts: str) -> datetime:
    # Example: 2025-06-05T16:10:02Z
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def minute_bucket_utc_from_ms(ms: int) -> str:
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:00")


def minute_bucket_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:00")


def extract_window_start_minute(anomaly_desc: str) -> Optional[str]:
    matches = _ISO_Z_RE.findall(anomaly_desc or "")
    if not matches:
        return None
    start_dt = parse_iso_z(matches[0])
    return minute_bucket_utc(start_dt)


def extract_window_minutes(anomaly_desc: str, max_minutes: int = 120) -> List[str]:
    """Return a list of UTC minute buckets within [start, end].

    We keep this bounded because some inputs may be malformed.
    """

    matches = _ISO_Z_RE.findall(anomaly_desc or "")
    if len(matches) < 2:
        m = extract_window_start_minute(anomaly_desc)
        return [m] if m else []

    start_dt = parse_iso_z(matches[0])
    end_dt = parse_iso_z(matches[1])
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    # align to minute buckets
    start_dt = start_dt.replace(second=0, microsecond=0)
    end_dt = end_dt.replace(second=0, microsecond=0)

    minutes: List[str] = []
    cur = start_dt
    while cur <= end_dt and len(minutes) < max_minutes:
        minutes.append(minute_bucket_utc(cur))
        cur = cur + timedelta(minutes=1)

    return minutes


def collect_allowed_minute_ids(
    input_json: Path,
    max_minutes: int = 120,
    pre_minutes: int = 0,
    post_minutes: int = 0,
) -> Set[int]:
    """Collect minute bucket ids (unix_ms // 60000) that appear in anomaly windows.

    We filter trace rows to these minute buckets to speed up parsing.
    """

    try:
        items = json.loads(input_json.read_text(encoding="utf-8"))
    except Exception:
        return set()

    allowed: Set[int] = set()
    for item in items:
        desc = str(item.get("Anomaly Description") or item.get("anomaly_description") or "")
        matches = _ISO_Z_RE.findall(desc)
        if len(matches) < 2:
            continue
        start_dt = parse_iso_z(matches[0]).replace(second=0, microsecond=0)
        end_dt = parse_iso_z(matches[1]).replace(second=0, microsecond=0)
        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt

        if int(pre_minutes) > 0:
            start_dt = start_dt - timedelta(minutes=int(pre_minutes))
        if int(post_minutes) > 0:
            end_dt = end_dt + timedelta(minutes=int(post_minutes))

        cur = start_dt
        n = 0
        cap = int(max_minutes) + max(0, int(pre_minutes)) + max(0, int(post_minutes))
        while cur <= end_dt and n < cap:
            # minute bucket id
            allowed.add(int(cur.timestamp() // 60))
            cur = cur + timedelta(minutes=1)
            n += 1
    return allowed


# -------------------------
# Aggregators
# -------------------------


@dataclass
class MetricAgg:
    calls: int = 0
    total_duration_ms: float = 0.0
    errors: int = 0
    timeouts: int = 0


MetricIndex = Mapping[str, Mapping[str, MetricAgg]]
TopologyIndex = Mapping[str, Mapping[str, Set[str]]]


def _iter_tag_items(tags: Any) -> Iterable[Dict[str, Any]]:
    if tags is None:
        return []
    if isinstance(tags, list):
        return [t for t in tags if isinstance(t, dict)]
    return []


def _tag_value(tags: Any, *keys: str) -> Optional[Any]:
    want = set(keys)
    for t in _iter_tag_items(tags):
        k = t.get("key")
        if k in want:
            return t.get("value")
    return None


def _tag_int(tags: Any, *keys: str) -> Optional[int]:
    v = _tag_value(tags, *keys)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None


def _tag_str(tags: Any, *keys: str) -> Optional[str]:
    v = _tag_value(tags, *keys)
    if v is None:
        return None
    return str(v)


def _tag_bool(tags: Any, *keys: str) -> Optional[bool]:
    v = _tag_value(tags, *keys)
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _infer_error_timeout(tags: Any) -> Tuple[bool, bool]:
    """Best-effort heuristics from OpenTelemetry/Jaeger span tags."""

    is_timeout = False

    # direct timeout flags
    if _tag_bool(tags, "timeout") is True:
        is_timeout = True

    status_code = _tag_int(tags, "http.status_code", "http.response.status_code", "http.request.status_code")
    if status_code is not None and status_code in {408, 504}:
        is_timeout = True

    grpc_code = _tag_int(tags, "grpc.status_code")
    if grpc_code is not None and grpc_code == 4:  # DEADLINE_EXCEEDED
        is_timeout = True

    status_msg = _tag_str(tags, "status.message")
    if status_msg and any(w in status_msg.lower() for w in ["deadline", "timeout", "timed out"]):
        is_timeout = True

    # error heuristics
    if _tag_bool(tags, "error") is True:
        return True, is_timeout

    if status_code is not None and status_code >= 500:
        return True, is_timeout

    if grpc_code is not None and grpc_code != 0:
        # for grpc, any non-OK often indicates error (including timeout already handled)
        return True, is_timeout

    # OpenTelemetry status.code: 0=UNSET,1=OK,2=ERROR
    otel_code = _tag_int(tags, "status.code")
    if otel_code is not None and otel_code == 2:
        return True, is_timeout

    otel_status = _tag_str(tags, "otel.status_code")
    if otel_status and otel_status.strip().upper() == "ERROR":
        return True, is_timeout

    return False, is_timeout


def _normalize_path(path: str) -> str:
    path = path.strip()
    if not path:
        return path
    # remove query string
    if "?" in path:
        path = path.split("?", 1)[0]
    return path


def infer_endpoint_name(service: str, operation_name: str, tags: Any, endpoint_mode: str) -> str:
    """Return an mABC-style endpoint string.

    mABC examples typically look like: <service>-<path_or_rpc>
    """

    service = (service or "").strip()
    operation_name = (operation_name or "").strip()

    if endpoint_mode == "service":
        return service

    if endpoint_mode == "service_operation":
        return f"{service}-{operation_name}" if operation_name else service

    # http/grpc preferred (default)
    # HTTP: prefer route -> target -> url.path
    route = _tag_str(tags, "http.route")
    target = _tag_str(tags, "http.target")
    url_path = _tag_str(tags, "url.path")
    path = route or target or url_path
    if path:
        return f"{service}-{_normalize_path(path)}" if service else _normalize_path(path)

    # gRPC / RPC
    rpc_service = _tag_str(tags, "rpc.service")
    rpc_method = _tag_str(tags, "rpc.method", "grpc.method")
    if rpc_service and rpc_method:
        return f"{service}-{rpc_service}/{rpc_method}" if service else f"{rpc_service}/{rpc_method}"
    if operation_name:
        return f"{service}-{operation_name}" if service else operation_name
    return service or "unknown"


def _get_service_name(process_struct: Any) -> str:
    # process is a struct with child 0 = serviceName
    # In Arrow, a struct becomes dict-like in to_pylist conversion, but we avoid full conversion.
    # We'll rely on Parquet reading returning Python-native objects in batch.to_pydict() for just this field.
    # As a fallback, stringify.
    if isinstance(process_struct, dict):
        val = process_struct.get("serviceName")
        return str(val) if val is not None else ""
    try:
        # pyarrow StructScalar
        return str(process_struct["serviceName"].as_py())
    except Exception:
        return str(process_struct)


def _iter_trace_parquet_files(day_root: Path) -> List[Path]:
    trace_dir = day_root / "trace-parquet"
    files = sorted(trace_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No trace parquet files found under {trace_dir}")
    return files


def build_from_traces(
    day_roots: Sequence[Path],
    batch_size: int = 50_000,
    endpoint_mode: str = "http",
    allowed_minute_ids: Optional[Set[int]] = None,
    skip_tags: bool = False,
    skip_topology: bool = False,
) -> Tuple[MetricIndex, TopologyIndex]:
    """Build endpoint_stats + endpoint_maps from trace parquet.

    endpoint_mode:
      - http: service-<http.route|http.target|url.path> (fallback operationName)
      - service_operation: service-operationName
      - service: serviceName (legacy coarse mode)
    """

    metric: DefaultDict[str, DefaultDict[str, MetricAgg]] = defaultdict(lambda: defaultdict(MetricAgg))
    topo: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    # Fast path: when we only need (calls, avg duration) and do not build topology.
    # This avoids per-row Arrow scalar parsing and is orders of magnitude faster.
    if skip_tags and skip_topology:
        if endpoint_mode not in {"service", "service_operation"}:
            raise ValueError(
                "fast build requires --endpoint-mode service or service_operation (http needs tags)"
            )
        if pd is None:
            raise RuntimeError(
                "pandas is required for fast build. Please install it or disable --skip-tags/--skip-topology."
            )

        allowed_arr = None
        if allowed_minute_ids:
            allowed_arr = pa.array(sorted(allowed_minute_ids), type=pa.int64())

        for day_root in day_roots:
            if not day_root.exists():
                print(f"[WARN] day folder not found, skipping: {day_root}", file=sys.stderr)
                continue

            try:
                parquet_files = _iter_trace_parquet_files(day_root)
            except FileNotFoundError as exc:
                print(f"[WARN] {exc}; skipping day: {day_root}", file=sys.stderr)
                continue

            for parquet_path in parquet_files:
                pf = pq.ParquetFile(parquet_path)
                schema_names = set(pf.schema.names)
                has_process = "process" in schema_names
                if has_process:
                    columns = ["startTimeMillis", "duration", "process", "operationName"]
                else:
                    # AIOps 2021 Jaeger export often flattens serviceName at top-level.
                    columns = ["startTimeMillis", "duration", "serviceName", "operationName"]

                for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                    if allowed_arr is not None:
                        start_ms_col = batch.column(batch.schema.get_field_index("startTimeMillis"))
                        # AIOps2021 trace export is inconsistent: some days store epoch seconds in startTimeMillis.
                        # Normalize to epoch milliseconds by magnitude.
                        try:
                            sec_mask = pc.less(start_ms_col, pa.scalar(10_000_000_000, type=pa.int64()))
                            start_ms_col = pc.if_else(sec_mask, pc.multiply(start_ms_col, 1000), start_ms_col)
                        except Exception:
                            pass
                        minute_ids = pc.cast(
                            _pc_divide(start_ms_col, pa.scalar(60000, type=pa.int64())),
                            pa.int64(),
                        )
                        mask = _pc_is_in(minute_ids, value_set=allowed_arr)
                        try:
                            batch = batch.filter(mask)
                        except Exception:
                            pass
                        if batch.num_rows == 0:
                            continue

                    start_ms_arr = batch.column(batch.schema.get_field_index("startTimeMillis"))
                    try:
                        sec_mask = pc.less(start_ms_arr, pa.scalar(10_000_000_000, type=pa.int64()))
                        start_ms_arr = pc.if_else(sec_mask, pc.multiply(start_ms_arr, 1000), start_ms_arr)
                    except Exception:
                        pass
                    dur_arr = batch.column(batch.schema.get_field_index("duration"))
                    proc_arr = None
                    if has_process:
                        proc_arr = batch.column(batch.schema.get_field_index("process"))
                    op_arr = batch.column(batch.schema.get_field_index("operationName"))

                    # Extract nested struct field in Arrow (no Python row loops).
                    svc_arr = None
                    if has_process:
                        try:
                            # Extract nested struct field in Arrow (no Python row loops).
                            svc_arr = proc_arr.field("serviceName") if proc_arr is not None else None
                        except Exception:
                            svc_arr = None
                    else:
                        svc_arr = batch.column(batch.schema.get_field_index("serviceName"))

                    if svc_arr is None:
                        # slow fallback for this batch
                        for i in range(batch.num_rows):
                            start_ms = start_ms_arr[i].as_py()
                            duration = dur_arr[i].as_py()
                            op_name = op_arr[i].as_py() if op_arr[i].is_valid else ""
                            if start_ms is None or duration is None:
                                continue

                            try:
                                if int(start_ms) < 10_000_000_000:
                                    start_ms = int(start_ms) * 1000
                            except Exception:
                                continue

                            if has_process and proc_arr is not None:
                                service = _get_service_name(proc_arr[i]).strip()
                            else:
                                try:
                                    service = str(batch.column(batch.schema.get_field_index("serviceName"))[i].as_py() or "").strip()
                                except Exception:
                                    service = ""

                            if not service:
                                continue
                            if endpoint_mode == "service":
                                endpoint = service
                            else:
                                endpoint = f"{service}-{op_name}" if op_name else service
                            minute = minute_bucket_utc_from_ms(int(start_ms))
                            m = metric[endpoint][minute]
                            m.calls += 1
                            m.total_duration_ms += float(duration) / 1000.0
                        continue

                    tbl = pa.table(
                        {
                            "service": svc_arr,
                            "operation": op_arr,
                            "start_ms": start_ms_arr,
                            "dur_us": dur_arr,
                        }
                    )
                    df = tbl.to_pandas(split_blocks=True, self_destruct=True)
                    if df.empty:
                        continue

                    # Basic cleaning.
                    df = df.dropna(subset=["service", "start_ms", "dur_us"])  # type: ignore[assignment]
                    if df.empty:
                        continue

                    if endpoint_mode == "service":
                        df["endpoint"] = df["service"].astype(str)
                    else:
                        # service_operation
                        op = df["operation"].fillna("").astype(str)
                        svc = df["service"].astype(str)
                        df["endpoint"] = svc.where(op.eq(""), svc + "-" + op)

                    minute_ms = (df["start_ms"].astype("int64") // 60000) * 60000
                    df["minute"] = pd.to_datetime(minute_ms, unit="ms", utc=True).dt.strftime(
                        "%Y-%m-%d %H:%M:00"
                    )
                    df["dur_ms"] = df["dur_us"].astype("float64") / 1000.0

                    g = (
                        df.groupby(["endpoint", "minute"], sort=False)["dur_ms"]
                        .agg(["size", "sum"])
                        .reset_index()
                    )
                    for row in g.itertuples(index=False):
                        endpoint = str(row.endpoint)
                        minute = str(row.minute)
                        calls = int(row.size)
                        total_dur = float(row.sum)
                        if not endpoint or not minute or calls <= 0:
                            continue
                        m = metric[endpoint][minute]
                        m.calls += calls
                        m.total_duration_ms += total_dur

        return metric, topo

    # (traceID, spanID) -> endpoint mapping; and pending children waiting for parent.
    span_endpoint: Dict[Tuple[str, str], str] = {}
    pending_children: DefaultDict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    def resolve_pending_for(parent_trace: str, parent_span: str, parent_endpoint: str) -> None:
        key = (parent_trace, parent_span)
        if key not in pending_children:
            return
        for minute, child_service in pending_children.pop(key):
            if parent_endpoint and child_service and parent_endpoint != child_service:
                topo[parent_endpoint][minute].add(child_service)

    def parent_ref(refs: Any) -> Optional[Tuple[str, str]]:
        # refs is a list of structs with fields refType/spanID/traceID
        if refs is None:
            return None
        try:
            # pyarrow returns list of dicts after to_pylist, but we avoid full conversion; here it's per-row.
            for ref in refs:
                if isinstance(ref, dict):
                    if ref.get("refType") == "CHILD_OF":
                        pt = ref.get("traceID")
                        ps = ref.get("spanID")
                        if pt and ps:
                            return str(pt), str(ps)
                else:
                    try:
                        if ref["refType"].as_py() == "CHILD_OF":
                            pt = ref["traceID"].as_py()
                            ps = ref["spanID"].as_py()
                            if pt and ps:
                                return str(pt), str(ps)
                    except Exception:
                        continue
        except TypeError:
            return None
        return None

    for day_root in day_roots:
        if not day_root.exists():
            print(f"[WARN] day folder not found, skipping: {day_root}", file=sys.stderr)
            continue

        try:
            parquet_files = list(_iter_trace_parquet_files(day_root))
        except FileNotFoundError as exc:
            print(f"[WARN] {exc}; skipping day: {day_root}", file=sys.stderr)
            continue

        print(
            f"[build] scanning day={day_root.name} parquet_files={len(parquet_files)}",
            flush=True,
        )

        for parquet_path in parquet_files:
            pf = pq.ParquetFile(parquet_path)
            try:
                approx_rows = int(pf.metadata.num_rows) if pf.metadata is not None else -1
            except Exception:
                approx_rows = -1
            print(
                f"[build] reading {day_root.name}/{parquet_path.name} rows≈{approx_rows}",
                flush=True,
            )
            columns = ["startTimeMillis", "duration", "process", "operationName"]
            if not skip_topology:
                columns = ["traceID", "spanID", "references"] + columns
            if not skip_tags:
                columns = columns + ["tags"]

            allowed_arr = None
            if allowed_minute_ids:
                # epoch minutes as int64
                allowed_arr = pa.array(sorted(allowed_minute_ids), type=pa.int64())

            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                # Fast pre-filter by minute buckets at Arrow level to reduce to_pydict overhead.
                if allowed_arr is not None:
                    start_ms_col = batch.column(batch.schema.get_field_index("startTimeMillis"))
                    # epoch minutes = floor(start_ms / 60000)
                    # Use divide+cast because some pyarrow versions may not expose floor_divide.
                    minute_ids = pc.cast(
                        _pc_divide(start_ms_col, pa.scalar(60000, type=pa.int64())),
                        pa.int64(),
                    )
                    mask = _pc_is_in(minute_ids, value_set=allowed_arr)
                    try:
                        batch = batch.filter(mask)
                    except Exception:
                        # Fallback: if filter not available, keep unfiltered batch.
                        pass
                    if batch.num_rows == 0:
                        continue

                # Iterate rows using Arrow scalars to avoid converting nested columns for the whole batch.
                start_ms_arr = batch.column(batch.schema.get_field_index("startTimeMillis"))
                dur_arr = batch.column(batch.schema.get_field_index("duration"))
                process_arr = batch.column(batch.schema.get_field_index("process"))
                op_arr = batch.column(batch.schema.get_field_index("operationName"))
                tags_arr = (
                    batch.column(batch.schema.get_field_index("tags")) if (not skip_tags) else None
                )

                trace_ids = (
                    batch.column(batch.schema.get_field_index("traceID")) if (not skip_topology) else None
                )
                span_ids = (
                    batch.column(batch.schema.get_field_index("spanID")) if (not skip_topology) else None
                )
                refs_arr = (
                    batch.column(batch.schema.get_field_index("references"))
                    if (not skip_topology)
                    else None
                )

                for i in range(batch.num_rows):
                    start_ms = start_ms_arr[i].as_py()
                    duration = dur_arr[i].as_py()

                    if start_ms is None or duration is None:
                        continue

                    trace_id = trace_ids[i].as_py() if trace_ids is not None else None
                    span_id = span_ids[i].as_py() if span_ids is not None else None
                    refs = refs_arr[i] if refs_arr is not None else None
                    process = process_arr[i]
                    op_name = op_arr[i].as_py() if op_arr[i].is_valid else None
                    tags = tags_arr[i] if tags_arr is not None else None

                    service = _get_service_name(process).strip()
                    if not service:
                        continue

                    endpoint = infer_endpoint_name(
                        service=service,
                        operation_name=str(op_name or ""),
                        tags=tags,
                        endpoint_mode=endpoint_mode,
                    )
                    if not endpoint:
                        continue

                    minute = minute_bucket_utc_from_ms(int(start_ms))

                    # metrics
                    m = metric[endpoint][minute]
                    m.calls += 1
                    # jaeger duration is typically microseconds; mABC expects ms-like units
                    m.total_duration_ms += float(duration) / 1000.0

                    if not skip_tags:
                        is_error, is_timeout = _infer_error_timeout(tags)
                        if is_error:
                            m.errors += 1
                        if is_timeout:
                            m.timeouts += 1

                    if not skip_topology:
                        if trace_id is None or span_id is None:
                            continue
                        trace_id_s = str(trace_id)
                        span_id_s = str(span_id)

                        # build mapping
                        span_endpoint[(trace_id_s, span_id_s)] = endpoint
                        resolve_pending_for(trace_id_s, span_id_s, endpoint)

                        pref = parent_ref(refs)
                        if pref is None:
                            continue
                        parent_trace, parent_span = pref
                        parent_endpoint = span_endpoint.get((parent_trace, parent_span))
                        if parent_endpoint is None:
                            pending_children[(parent_trace, parent_span)].append((minute, endpoint))
                        else:
                            if parent_endpoint and endpoint and parent_endpoint != endpoint:
                                topo[parent_endpoint][minute].add(endpoint)

            # avoid unbounded growth between files
            span_endpoint.clear()
            pending_children.clear()

    return metric, topo


def finalize_endpoint_stats(metric: MetricIndex) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for service, minute_map in metric.items():
        out[service] = {}
        for minute, agg in minute_map.items():
            calls = agg.calls
            avg = (agg.total_duration_ms / calls) if calls else 0.0
            error_rate = (agg.errors / calls * 100.0) if calls else 0.0
            timeout_rate = (agg.timeouts / calls * 100.0) if calls else 0.0
            success_rate = 100.0 - error_rate
            out[service][minute] = {
                "calls": int(calls),
                "success_rate": float(success_rate),
                "error_rate": float(error_rate),
                "average_duration": float(avg),
                "timeout_rate": float(timeout_rate),
            }
    return out


def finalize_endpoint_maps(topo: TopologyIndex) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for parent, minute_map in topo.items():
        out[parent] = {}
        for minute, children in minute_map.items():
            out[parent][minute] = sorted(children)
    return out


def anomaly_score(stats: Mapping[str, Any]) -> float:
    # Heuristic score for label selection: duration dominates, errors/timeouts amplify.
    dur = float(stats.get("average_duration", 0.0))
    er = float(stats.get("error_rate", 0.0)) / 100.0
    tr = float(stats.get("timeout_rate", 0.0)) / 100.0
    calls = float(stats.get("calls", 0.0))
    # Prefer endpoints with some traffic.
    traffic_boost = 1.0 if calls >= 3 else 0.8
    return dur * (1.0 + 2.0 * er + 3.0 * tr) * traffic_boost


def pick_alert_endpoint(endpoint_stats: Dict[str, Dict[str, Dict[str, float]]], minute: str) -> Optional[str]:
    best_endpoint: Optional[str] = None
    best_score = -1.0
    for endpoint, minute_map in endpoint_stats.items():
        stats = minute_map.get(minute)
        if not stats:
            continue
        score = anomaly_score(stats)
        if score > best_score:
            best_score = score
            best_endpoint = endpoint
    return best_endpoint


def pick_best_minute_and_endpoint(
    endpoint_stats: Dict[str, Dict[str, Dict[str, float]]],
    candidate_minutes: Sequence[str],
) -> Tuple[Optional[str], Optional[str]]:
    best_minute: Optional[str] = None
    best_endpoint: Optional[str] = None
    best_score = -1.0
    for minute in candidate_minutes:
        ep = pick_alert_endpoint(endpoint_stats, minute)
        if ep is None:
            continue
        score = anomaly_score(endpoint_stats[ep][minute])
        if score > best_score:
            best_score = score
            best_minute = minute
            best_endpoint = ep
    return best_minute, best_endpoint


def build_paths(endpoint_maps: Dict[str, Dict[str, List[str]]], minute: str, root: str, max_paths: int, max_depth: int) -> List[List[str]]:
    # Generate a few simple downstream paths using BFS on the minute-specific adjacency.
    if max_depth < 1:
        return []

    adj: Dict[str, List[str]] = {}
    for parent, minute_map in endpoint_maps.items():
        children = minute_map.get(minute)
        if children:
            adj[parent] = children

    paths: List[List[str]] = []
    queue: List[List[str]] = [[root]]
    seen_paths: Set[Tuple[str, ...]] = set()

    while queue and len(paths) < max_paths:
        path = queue.pop(0)
        if len(path) >= 2:
            key = tuple(path)
            if key not in seen_paths:
                seen_paths.add(key)
                paths.append(path)
                if len(paths) >= max_paths:
                    break
        if len(path) >= max_depth:
            continue
        last = path[-1]
        for child in adj.get(last, [])[:50]:
            if child in path:
                continue
            queue.append(path + [child])

    # if no downstream, provide a trivial self path to satisfy mABC shape
    if not paths:
        return [[root]]
    return paths


def build_label_json(
    input_json: Path,
    endpoint_stats: Dict[str, Dict[str, Dict[str, float]]],
    endpoint_maps: Dict[str, Dict[str, List[str]]],
    max_paths: int,
    max_depth: int,
) -> Tuple[Dict[str, Dict[str, List[List[str]]]], Dict[str, Dict[str, str]]]:

    def _minute_str_to_id(minute: str) -> Optional[int]:
        try:
            dt = datetime.strptime(minute, "%Y-%m-%d %H:%M:00").replace(tzinfo=timezone.utc)
        except Exception:
            return None
        return int(dt.timestamp() // 60)

    def _pick_nearest_available_minute(target_minute: str, max_delta_minutes: int = 60) -> Optional[str]:
        """Fallback when a uuid window has no trace coverage.

        We pick the nearest minute that exists in `available_minutes` so we can still
        generate a label_uuid_map entry and allow mABC to run.
        """

        target_id = _minute_str_to_id(target_minute)
        if target_id is None:
            return None
        # Linear scan is fine here (<= a few 10k minutes worst case).
        best_minute: Optional[str] = None
        best_delta = max_delta_minutes + 1
        for minute_id, minute_str in available_minutes_by_id:
            delta = abs(minute_id - target_id)
            if delta < best_delta:
                best_delta = delta
                best_minute = minute_str
                if best_delta == 0:
                    break
        if best_minute is None or best_delta > max_delta_minutes:
            return None
        return best_minute

    entries = json.loads(input_json.read_text(encoding="utf-8"))
    by_minute: DefaultDict[str, Dict[str, List[List[str]]]] = defaultdict(dict)
    uuid_map: Dict[str, Dict[str, str]] = {}

    available_minutes: Set[str] = set()
    for service_minutes in endpoint_stats.values():
        available_minutes.update(service_minutes.keys())

    available_minutes_by_id: List[Tuple[int, str]] = []
    for m in available_minutes:
        mid = _minute_str_to_id(m)
        if mid is not None:
            available_minutes_by_id.append((mid, m))

    for item in entries:
        desc = str(item.get("Anomaly Description", ""))
        uuid = str(item.get("uuid", "")).strip()
        window_minutes = extract_window_minutes(desc)
        if not window_minutes:
            continue

        # Only keep minutes we actually have traces for.
        in_window = [m for m in window_minutes if m in available_minutes]
        if not in_window:
            # Fallback: choose nearest trace-covered minute to the window start.
            start_minute = window_minutes[0]
            nearest = _pick_nearest_available_minute(start_minute, max_delta_minutes=720)
            if nearest is None:
                continue
            candidate_minutes = [nearest]
        else:
            candidate_minutes = in_window

        minute, alert = pick_best_minute_and_endpoint(endpoint_stats, candidate_minutes)
        if minute is None or alert is None:
            continue

        if uuid:
            uuid_map[uuid] = {"minute": minute, "alert_endpoint": alert}

        if alert in by_minute[minute]:
            continue

        paths = build_paths(endpoint_maps, minute, alert, max_paths=max_paths, max_depth=max_depth)
        by_minute[minute][alert] = paths

    # sort keys for stable output
    label = {k: by_minute[k] for k in sorted(by_minute.keys())}
    return label, uuid_map


# -------------------------
# CLI
# -------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mABC-readable inputs from AIOps Challenge parquet data")
    p.add_argument("--aiops-data-root", type=Path, required=True, help="Path to AIOPS /data directory")
    p.add_argument(
        "--day",
        type=str,
        action="append",
        required=True,
        help="Day folder under /data, e.g. 2025-06-07. Can be repeated.",
    )
    p.add_argument("--input-json", type=Path, required=True, help="Phase input.json (uuid + anomaly window)")
    p.add_argument(
        "--allowed-pre-minutes",
        type=int,
        default=0,
        help="Extend anomaly windows backward by N minutes when collecting allowed trace minutes (speed filter).",
    )
    p.add_argument(
        "--allowed-post-minutes",
        type=int,
        default=0,
        help="Extend anomaly windows forward by N minutes when collecting allowed trace minutes (speed filter).",
    )
    p.add_argument(
        "--no-minute-filter",
        action="store_true",
        help="Disable allowed-minute pre-filter and parse all trace minutes for the selected days (slower, but safest).",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="mABC data root to write into (will create metric/topology/label subdirs)",
    )
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument("--max-paths", type=int, default=5)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument(
        "--endpoint-mode",
        type=str,
        default="http",
        choices=["http", "service_operation", "service"],
        help="Endpoint granularity. http is highest fidelity (default).",
    )
    p.add_argument(
        "--skip-tags",
        action="store_true",
        help="Skip reading span tags; sets error/timeout rates to 0 (faster).",
    )
    p.add_argument(
        "--skip-topology",
        action="store_true",
        help="Skip building endpoint dependency graph (faster).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast build preset: equivalent to --skip-tags --skip-topology. (Recommended for full runs.)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    day_roots = [args.aiops_data_root / day for day in args.day]

    if args.fast:
        args.skip_tags = True
        args.skip_topology = True

    if args.skip_tags and args.endpoint_mode == "http":
        # http mode relies on tags for route/path extraction; fall back to a stable no-tag endpoint.
        print(
            "[WARN] --skip-tags is incompatible with --endpoint-mode http; falling back to service_operation",
            file=sys.stderr,
        )
        args.endpoint_mode = "service_operation"

    if args.no_minute_filter:
        allowed_minute_ids = set()
    else:
        allowed_minute_ids = collect_allowed_minute_ids(
            args.input_json,
            pre_minutes=int(args.allowed_pre_minutes),
            post_minutes=int(args.allowed_post_minutes),
        )
    metric_raw, topo_raw = build_from_traces(
        day_roots=day_roots,
        batch_size=args.batch_size,
        endpoint_mode=args.endpoint_mode,
        allowed_minute_ids=allowed_minute_ids,
        skip_tags=args.skip_tags,
        skip_topology=args.skip_topology,
    )
    endpoint_stats = finalize_endpoint_stats(metric_raw)
    endpoint_maps = finalize_endpoint_maps(topo_raw)

    (args.out_root / "metric").mkdir(parents=True, exist_ok=True)
    (args.out_root / "topology").mkdir(parents=True, exist_ok=True)
    (args.out_root / "label").mkdir(parents=True, exist_ok=True)

    (args.out_root / "metric" / "endpoint_stats.json").write_text(
        json.dumps(endpoint_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_root / "topology" / "endpoint_maps.json").write_text(
        json.dumps(endpoint_maps, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    label, uuid_map = build_label_json(
        input_json=args.input_json,
        endpoint_stats=endpoint_stats,
        endpoint_maps=endpoint_maps,
        max_paths=args.max_paths,
        max_depth=args.max_depth,
    )
    (args.out_root / "label" / "label.json").write_text(
        json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (args.out_root / "label" / "label_uuid_map.json").write_text(
        json.dumps(uuid_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote endpoint_stats.json: {args.out_root / 'metric' / 'endpoint_stats.json'}")
    print(f"Wrote endpoint_maps.json:  {args.out_root / 'topology' / 'endpoint_maps.json'}")
    print(f"Wrote label.json:          {args.out_root / 'label' / 'label.json'}")
    print(f"Wrote label_uuid_map.json: {args.out_root / 'label' / 'label_uuid_map.json'}")
    print(f"Label minutes: {len(label)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
