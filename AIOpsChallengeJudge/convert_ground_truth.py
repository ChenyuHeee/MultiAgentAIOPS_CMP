#!/usr/bin/env python3
"""Convert raw AIOps phase ground truth files to evaluator schema."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:  # pragma: no cover - CLI helper
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


def flatten_keywords(value: Any) -> List[str]:
    items: List[str] = []
    if value is None:
        return items
    if isinstance(value, str):
        candidate = value.strip()
        if candidate:
            items.append(candidate)
        return items
    if isinstance(value, (list, tuple, set)):
        for element in value:
            items.extend(flatten_keywords(element))
        return items
    if isinstance(value, dict):
        for element in value.values():
            items.extend(flatten_keywords(element))
        return items
    candidate = str(value).strip()
    if candidate:
        items.append(candidate)
    return items


def unique_keywords(keywords: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for keyword in keywords:
        lowered = keyword.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(keyword)
    return ordered


def stringify_instance(record: Dict[str, Any]) -> str:
    instance = record.get("instance")
    if isinstance(instance, list):
        cleaned = [str(item).strip() for item in instance if str(item).strip()]
        # 若有源/目的，恢复使用 "source->destination"；否则用 "+" 连接实例
        if record.get("source") and record.get("destination"):
            return f"{record['source']}->{record['destination']}"
        return "+".join(cleaned)
    if isinstance(instance, str):
        return instance.strip()
    if instance is None or instance == "":
        return ""
    return str(instance).strip()


def build_component(record: Dict[str, Any]) -> str:
    primary = stringify_instance(record)
    if not primary:
        service = str(record.get("service", "")).strip()
        if service:
            primary = service
    if not primary:
        primary = str(record.get("instance_type", "")).strip()
    return primary or "unknown"


def build_reason(record: Dict[str, Any], component: str) -> str:
    category = str(record.get("fault_category", "")).strip()
    fault_type = str(record.get("fault_type", "")).strip()
    header_parts = [part for part in (category, fault_type) if part]

    details: List[str] = []
    for label, value in (
        ("service", record.get("service")),
        ("instance", stringify_instance(record)),
        ("component", component),
    ):
        if not value:
            continue
        details.append(f"{label}={value}")
    if record.get("source") and record.get("destination"):
        details.append(f"path={record['source']}->{record['destination']}")

    reason = " / ".join(header_parts) if header_parts else "unknown fault"
    if details:
        reason = f"{reason} ({', '.join(details)})"
    return reason


def build_reason_keywords(record: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for field in ("fault_category", "fault_type", "instance_type", "service", "source", "destination"):
        value = record.get(field)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                candidates.append(trimmed)
        elif isinstance(value, list):
            candidates.extend(str(item).strip() for item in value if str(item).strip())
    instance_value = stringify_instance(record)
    if instance_value:
        candidates.append(instance_value)
    candidates.extend(flatten_keywords(record.get("key_observation")))
    for observation in record.get("key_observations", []) or []:
        if isinstance(observation, dict):
            candidates.extend(flatten_keywords(observation.get("keyword")))
    return unique_keywords([kw for kw in candidates if kw])


def build_evidence_points(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    observations = record.get("key_observations") or []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        label_parts: List[str] = []
        for field in ("type", "category", "subtype"):
            value = observation.get(field)
            if isinstance(value, str) and value.strip():
                label_parts.append(value.strip())
        label = ":".join(label_parts) or "observation"
        keywords = unique_keywords(flatten_keywords(observation.get("keyword")))
        if keywords:
            evidence.append({"type": label, "keywords": keywords})
    single = unique_keywords(flatten_keywords(record.get("key_observation")))
    if single:
        evidence.append({"type": "summary", "keywords": single})
    return evidence


def convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
    uuid = record.get("uuid")
    if not isinstance(uuid, str) or not uuid.strip():
        raise ValueError("Every ground-truth record must include a non-empty uuid")
    component = build_component(record)
    converted = {
        "uuid": uuid,
        "component": component,
        "reason": build_reason(record, component),
        "reason_keywords": build_reason_keywords(record),
        "evidence_points": build_evidence_points(record),
    }
    return converted


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert phase ground truth to evaluator schema")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input phase ground-truth files (JSONL)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
        help="Destination JSONL file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    aggregated: List[Dict[str, Any]] = []
    for path in args.inputs:
        aggregated.extend(read_jsonl(path))

    converted = [convert_record(record) for record in aggregated]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in converted:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"Wrote {len(converted)} records to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main(sys.argv[1:]))
