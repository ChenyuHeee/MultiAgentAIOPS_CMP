from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


SUFFIXES: Tuple[str, ...] = ("calls", "avg_duration", "error_rate", "timeout_rate")


def _load_uuid_to_instruction(input_json: Path) -> Dict[str, str]:
	data = json.loads(input_json.read_text(encoding="utf-8"))
	out: Dict[str, str] = {}
	for row in data:
		uuid = str(row.get("uuid", "")).strip()
		if not uuid:
			continue
		instr = str(row.get("Anomaly Description", "")).strip()
		out[uuid] = instr
	return out


def _ensure_placeholder_csv(path: Path, header: List[str]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists() and path.stat().st_size > 0:
		return
	with path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(header)


def _iter_pkl_paths(pkls_dir: Path) -> Iterable[Path]:
	for p in sorted(pkls_dir.glob("*.pkl")):
		if p.is_file():
			yield p


def _pkl_to_metric_long(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])

	idx = pd.to_datetime(df.index)
	ts = (idx.view("int64") // 1_000_000_000).astype("int64")

	parts: List[pd.DataFrame] = []
	for suffix in SUFFIXES:
		suffix_cols = [c for c in df.columns if isinstance(c, str) and c.endswith("_" + suffix)]
		if not suffix_cols:
			continue
		sub = df[suffix_cols].copy()
		sub.columns = [c[: -(len(suffix) + 1)] for c in suffix_cols]
		long = sub.reset_index(drop=True)
		long.insert(0, "timestamp", ts)
		long = long.melt(id_vars=["timestamp"], var_name="cmdb_id", value_name="value")
		long.insert(2, "kpi_name", suffix)
		long = long[["timestamp", "cmdb_id", "kpi_name", "value"]]
		parts.append(long)

	if not parts:
		return pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"])

	out = pd.concat(parts, ignore_index=True)
	out["cmdb_id"] = out["cmdb_id"].astype(str)
	out["kpi_name"] = out["kpi_name"].astype(str)
	out["value"] = pd.to_numeric(out["value"], errors="coerce").fillna(0.0)
	out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(0).astype("int64")
	return out


def build_dataset(phase: str, pkls_dir: Path, input_json: Path, dataset_root: Path) -> None:
	uuid_to_instr = _load_uuid_to_instruction(input_json)

	system_dir = dataset_root / "AIOpsChallenge" / phase
	system_dir.mkdir(parents=True, exist_ok=True)

	query_csv = system_dir / "query.csv"
	query_csv.parent.mkdir(parents=True, exist_ok=True)
	with query_csv.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=["task_index", "instruction", "scoring_points"])
		writer.writeheader()
		for uuid, instr in sorted(uuid_to_instr.items()):
			writer.writerow({"task_index": uuid, "instruction": instr, "scoring_points": ""})

	for pkl_path in _iter_pkl_paths(pkls_dir):
		uuid = pkl_path.stem
		telemetry_dir = system_dir / "telemetry" / uuid

		metric_dir = telemetry_dir / "metric"
		trace_dir = telemetry_dir / "trace"
		log_dir = telemetry_dir / "log"
		metric_dir.mkdir(parents=True, exist_ok=True)
		trace_dir.mkdir(parents=True, exist_ok=True)
		log_dir.mkdir(parents=True, exist_ok=True)

		df = pd.read_pickle(pkl_path)
		metric_long = _pkl_to_metric_long(df)
		metric_csv = metric_dir / "metric_service.csv"
		metric_long.to_csv(metric_csv, index=False)

		_ensure_placeholder_csv(
			trace_dir / "trace_span.csv",
			header=["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"],
		)
		_ensure_placeholder_csv(
			log_dir / "log_service.csv",
			header=["log_id", "timestamp", "cmdb_id", "log_name", "value"],
		)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--phase", choices=["phase1", "phase2"], required=True)
	parser.add_argument("--pkls-dir", type=Path, required=True)
	parser.add_argument("--input-json", type=Path, required=True)
	parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
	args = parser.parse_args()

	if not args.pkls_dir.exists():
		raise FileNotFoundError(f"pkls-dir not found: {args.pkls_dir}")
	if not args.input_json.exists():
		raise FileNotFoundError(f"input-json not found: {args.input_json}")

	build_dataset(args.phase, args.pkls_dir, args.input_json, args.dataset_root)


if __name__ == "__main__":
	main()

