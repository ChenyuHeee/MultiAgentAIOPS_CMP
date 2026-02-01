"""Build aiops2021 input.json and ground_truth.jsonl from aiops21_groundtruth.csv.

Why this exists
- The three method adapters (mABC/CausalRCA/OpenRCA) expect an AIOpsChallenge-style input.json
  with a free-form "Anomaly Description" that contains ISO-8601 Z timestamps.
- The judge expects a ground_truth.jsonl with unique uuids.

The provided aiops2021/ground_truth.jsonl contains a duplicate uuid (aiops21-1475), which
breaks the judge. The CSV has enough fields (time + start/end) to disambiguate.

Outputs
- aiops2021/inputs/train_input.json
- aiops2021/inputs/test_input.json
- aiops2021/inputs/all_input.json
- aiops2021/outputs/ground_truth_train.jsonl
- aiops2021/outputs/ground_truth_test.jsonl
- aiops2021/outputs/ground_truth_all.jsonl
- aiops2021/outputs/days_train.txt / days_test.txt / days_all.txt

Notes
- We interpret st_time/ed_time strings as Asia/Shanghai local time and convert to UTC Z.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(frozen=True)
class Row:
    uuid: str
    service: str
    anomaly_type: str
    fault_cat: str
    fault_content: str
    start_iso_z: str
    end_iso_z: str
    day: str
    split: str  # train/test


def _parse_local_ts_to_iso_z(s: str) -> str:
    # Example: 2021-03-04 11:50:00.000000
    # Treat as Asia/Shanghai (UTC+8) then convert to UTC.
    dt_local = datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f")
    # Manual tz conversion: UTC = local - 8h
    dt_utc = dt_local.replace()  # naive
    dt_utc = dt_utc.replace()  # keep naive
    # subtract 8 hours
    from datetime import timedelta

    dt_utc = dt_utc - timedelta(hours=8)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_anomaly_type(s: str) -> List[str]:
    # Handles values like "JVM;\nCPU" and "JVM;MEMORY"
    raw = (s or "").replace("\n", ";")
    parts = [p.strip() for p in raw.split(";")]
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        out.append(p)
    return out


def _iter_rows(csv_path: Path) -> Iterable[Row]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # column names include Chinese; DictReader keeps them as-is.
            id_raw = (r.get("id") or "").strip()
            time_raw = (r.get("time") or "").strip()
            service = (r.get("service") or "").strip()
            anomaly_type = (r.get("anomaly_type") or "").strip()
            fault_cat = (r.get("故障类别") or "").strip()
            fault_content = (r.get("故障内容") or "").strip()
            st_time = (r.get("st_time") or "").strip()
            ed_time = (r.get("ed_time") or "").strip()
            split = (r.get("data_type") or "").strip().lower() or "train"

            if not id_raw or not time_raw:
                continue

            # Make uuid unique even when id repeats.
            # time is in ms.
            try:
                time_ms = int(float(time_raw))
            except Exception:
                time_ms = 0

            uuid = f"aiops21-{id_raw}-{time_ms}"

            if not st_time or not ed_time:
                # fall back: treat time_ms as point-in-time window of 5 minutes
                from datetime import timedelta

                dt_utc = datetime.utcfromtimestamp(time_ms / 1000.0)
                start_iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_iso = (dt_utc + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                start_iso = _parse_local_ts_to_iso_z(st_time)
                end_iso = _parse_local_ts_to_iso_z(ed_time)

            # Day folder in aiops2021 is local day; derive from st_time string.
            day = st_time.split(" ", 1)[0] if st_time else datetime.utcfromtimestamp(time_ms / 1000.0).strftime("%Y-%m-%d")

            yield Row(
                uuid=uuid,
                service=service,
                anomaly_type=anomaly_type,
                fault_cat=fault_cat,
                fault_content=fault_content,
                start_iso_z=start_iso,
                end_iso_z=end_iso,
                day=day,
                split=split,
            )


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_input(rows: Sequence[Row], *, include_label_fields: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if include_label_fields:
            anomaly_tokens = _normalize_anomaly_type(r.anomaly_type)
            anomaly_part = ",".join(anomaly_tokens) if anomaly_tokens else r.anomaly_type
            desc = (
                f"uuid={r.uuid} window=[{r.start_iso_z},{r.end_iso_z}] "
                f"service={r.service} anomaly_type={anomaly_part} fault={r.fault_cat}/{r.fault_content}"
            )
            out.append(
                {
                    "uuid": r.uuid,
                    "service": r.service,
                    "anomaly_type": r.anomaly_type,
                    "data_type": r.split,
                    "Anomaly Description": desc,
                }
            )
        else:
            # Clean evaluation mode: do NOT leak label-like fields into inputs.
            # Keep only the time window (needed by adapters to slice data).
            desc = f"uuid={r.uuid} window=[{r.start_iso_z},{r.end_iso_z}]"
            out.append(
                {
                    "uuid": r.uuid,
                    "data_type": r.split,
                    "Anomaly Description": desc,
                }
            )
    return out


def _build_gt(rows: Sequence[Row]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        anomaly_tokens = _normalize_anomaly_type(r.anomaly_type)
        reason = f"{r.fault_cat} / {r.fault_content}".strip(" /")
        # Keep reason_keywords similar to existing ground_truth.jsonl
        reason_keywords: List[str] = []
        if r.service:
            reason_keywords.append(r.service)
        reason_keywords += anomaly_tokens if anomaly_tokens else ([r.anomaly_type] if r.anomaly_type else [])
        if r.fault_cat:
            reason_keywords.append(r.fault_cat)
        if r.fault_content:
            reason_keywords.append(r.fault_content)
        if r.split:
            reason_keywords.append(r.split)

        out.append(
            {
                "uuid": r.uuid,
                "component": r.service,
                "reason": reason,
                "reason_keywords": reason_keywords,
                "evidence_points": [],
            }
        )
    return out


def _unique_days(rows: Sequence[Row]) -> List[str]:
    days: Set[str] = set(r.day for r in rows if r.day)
    return sorted(days)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("aiops2021/aiops21_groundtruth.csv"),
        help="Path to aiops21_groundtruth.csv",
    )
    args = p.parse_args()

    rows_all = list(_iter_rows(args.csv))
    rows_train = [r for r in rows_all if r.split == "train"]
    rows_test = [r for r in rows_all if r.split == "test"]

    # write inputs
    _write_json(Path("aiops2021/inputs/all_input.json"), _build_input(rows_all, include_label_fields=True))
    _write_json(Path("aiops2021/inputs/train_input.json"), _build_input(rows_train, include_label_fields=True))
    _write_json(Path("aiops2021/inputs/test_input.json"), _build_input(rows_test, include_label_fields=True))

    _write_json(Path("aiops2021/inputs/all_input_clean.json"), _build_input(rows_all, include_label_fields=False))
    _write_json(Path("aiops2021/inputs/train_input_clean.json"), _build_input(rows_train, include_label_fields=False))
    _write_json(Path("aiops2021/inputs/test_input_clean.json"), _build_input(rows_test, include_label_fields=False))

    # write ground truth
    _write_jsonl(Path("aiops2021/outputs/ground_truth_all.jsonl"), _build_gt(rows_all))
    _write_jsonl(Path("aiops2021/outputs/ground_truth_train.jsonl"), _build_gt(rows_train))
    _write_jsonl(Path("aiops2021/outputs/ground_truth_test.jsonl"), _build_gt(rows_test))

    # write days lists (used for mABC build)
    _write_lines(Path("aiops2021/outputs/days_all.txt"), _unique_days(rows_all))
    _write_lines(Path("aiops2021/outputs/days_train.txt"), _unique_days(rows_train))
    _write_lines(Path("aiops2021/outputs/days_test.txt"), _unique_days(rows_test))

    print(f"rows_all={len(rows_all)} train={len(rows_train)} test={len(rows_test)}")
    print("days_all:", ", ".join(_unique_days(rows_all)))


if __name__ == "__main__":
    main()
