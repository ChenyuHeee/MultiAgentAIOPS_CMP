from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
	import pandas as pd
except Exception:  # pragma: no cover
	pd = None


SUFFIXES = ("calls", "avg_duration", "error_rate", "timeout_rate")


def normalize_component(s: str) -> str:
	# Judge 的 component 目前是严格相等匹配；这里保持“诚实”最小清洗。
	return str(s or "").strip()

_INSTANCE_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-(?P<idx>\d+)$")
def _instance_token_candidates(
	services: Sequence[str],
	vocab_lower_to_token: Dict[str, str],
	max_tokens: int = 10,
) -> List[str]:
	out: List[str] = []
	seen: set[str] = set()
	suffixes = ["-2", "-0", "-1"]
	for svc in services:
		base = (svc or "").strip().lower()
		if not base:
			continue
		for suf in suffixes:
			key = base + suf
			tok = vocab_lower_to_token.get(key)
			if not tok:
				continue
			if tok in seen:
				continue
			seen.add(tok)
			out.append(tok)
			if len(out) >= int(max_tokens):
				return out
	return out


def _services_from_columns(columns: Iterable[str]) -> List[str]:
	services: set[str] = set()
	for c in columns:
		if not isinstance(c, str):
			continue
		for suf in SUFFIXES:
			tail = "_" + suf
			if c.endswith(tail):
				services.add(c[: -len(tail)])
				break
	return sorted(services)


@dataclass
class ServiceScore:
	service: str
	score: float
	max_error: float
	max_timeout: float
	max_duration: float
	min_calls_ratio: float


def _score_services(df: pd.DataFrame) -> List[ServiceScore]:
	services = _services_from_columns(df.columns)
	if not services:
		return []

	eps = 1e-9
	out: List[ServiceScore] = []
	for svc in services:
		er_series = df.get(f"{svc}_error_rate") if f"{svc}_error_rate" in df else None
		tr_series = df.get(f"{svc}_timeout_rate") if f"{svc}_timeout_rate" in df else None
		er = float(er_series.max() if er_series is not None else 0.0)
		tr = float(tr_series.max() if tr_series is not None else 0.0)
		med_er = float(er_series.median() if er_series is not None else 0.0)
		med_tr = float(tr_series.median() if tr_series is not None else 0.0)
		er_spike = er / (med_er + eps) if er > 0 else 0.0
		tr_spike = tr / (med_tr + eps) if tr > 0 else 0.0
		er_nonzero = float((er_series > 0).mean()) if er_series is not None else 0.0
		tr_nonzero = float((tr_series > 0).mean()) if tr_series is not None else 0.0
		dur_series = df.get(f"{svc}_avg_duration") if f"{svc}_avg_duration" in df else None
		max_dur = float(dur_series.max() if dur_series is not None else 0.0)
		if dur_series is not None:
			dur_pos = dur_series[dur_series > 0]
			med_dur = float(dur_pos.median() if not dur_pos.empty else 0.0)
		else:
			med_dur = 0.0
		# 避免 median=0 导致 spike 全体饱和从而打平
		if max_dur > 0 and med_dur > 0:
			dur_spike = max_dur / (med_dur + eps)
		else:
			dur_spike = 0.0

		calls_series = df.get(f"{svc}_calls") if f"{svc}_calls" in df else None
		if calls_series is None or float(calls_series.median()) <= 0:
			min_calls_ratio = 1.0
		else:
			min_calls_ratio = float(calls_series.min()) / float(calls_series.median() + eps)
		calls_drop = max(0.0, 1.0 - float(min_calls_ratio))

		# 经验权重：timeout/error 更强；duration 用相对 spike + 绝对值（log）组合，避免全 0 打平
		score = (
			(3.0 * tr)
			+ (2.6 * er)
			+ (0.35 * min(10.0, tr_spike))
			+ (0.25 * min(10.0, er_spike))
			+ (0.18 * min(10.0, dur_spike))
			+ (0.06 * max_dur)
			+ (0.35 * calls_drop)
			+ (0.18 * tr_nonzero)
			+ (0.12 * er_nonzero)
		)
		out.append(
			ServiceScore(
				service=svc,
				score=float(score),
				max_error=er,
				max_timeout=tr,
				max_duration=max_dur,
				min_calls_ratio=float(min_calls_ratio),
			)
		)

	out.sort(key=lambda x: x.score, reverse=True)
	return out


 


def _reason_from_top(top: ServiceScore) -> str:
	if top.max_timeout > 0:
		return "timeout / request deadline exceeded"
	if top.max_error > 0:
		return "error / 5xx spike"
	if top.max_duration > 0:
		return "latency / slow response"
	return "service degradation"


def _signal_keywords(top: ServiceScore) -> List[str]:
	out: List[str] = []
	if top.max_error > 0:
		out += ["error"]
	if top.max_timeout > 0:
		out += ["timeout"]
	if top.max_duration > 0:
		out += ["latency"]
	return out[:6]


def _load_uuid_to_hint_service(input_json: Path) -> Dict[str, str]:
	data = json.loads(input_json.read_text(encoding="utf-8"))
	out: Dict[str, str] = {}
	for row in data:
		uuid = str(row.get("uuid", "")).strip()
		if not uuid:
			continue
		svc = str(row.get("service") or "").strip()
		if svc:
			out[uuid] = svc
			continue
		desc = str(row.get("Anomaly Description") or "")
		m = re.search(r"\bservice=([^\s]+)", desc)
		if m:
			out[uuid] = m.group(1).strip()
	return out


def _iter_uuids_from_input(input_json: Path) -> List[str]:
	data = json.loads(input_json.read_text(encoding="utf-8"))
	out: List[str] = []
	for row in data:
		uuid = str(row.get("uuid", "")).strip()
		if uuid:
			out.append(uuid)
	return out


def _iter_pkl_paths(pkls_dir: Path) -> Iterable[Path]:
	for p in sorted(pkls_dir.glob("*.pkl")):
		if p.is_file():
			yield p


def _write_jsonl(path: Path, rows: List[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for obj in rows:
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--phase", choices=["phase1", "phase2"], required=True)
	parser.add_argument("--pkls-dir", type=Path, required=True)
	parser.add_argument(
		"--input-json",
		type=Path,
		default=None,
		help="Optional phase input.json (used for uuid list + service hint; does NOT read ground truth)",
	)
	parser.add_argument(
		"--use-input-service-hint",
		action="store_true",
		help="If set, may use input.json's service field as a weak hint (disabled by default to avoid label leakage).",
	)
	parser.add_argument("--out-jsonl", type=Path, required=True)
	parser.add_argument("--limit", type=int, default=0)
	args = parser.parse_args()

	uuid_to_hint_service: Dict[str, str] = {}
	uuids_from_input: List[str] = []
	if args.input_json is not None:
		uuids_from_input = _iter_uuids_from_input(args.input_json)
		if args.use_input_service_hint:
			uuid_to_hint_service = _load_uuid_to_hint_service(args.input_json)

	rows: List[dict] = []
	n = 0

	if uuids_from_input:
		pkl_paths: List[Path] = [args.pkls_dir / f"{u}.pkl" for u in uuids_from_input]
	else:
		pkl_paths = list(_iter_pkl_paths(args.pkls_dir))

	for pkl_path in pkl_paths:
		uuid = pkl_path.stem
		if not pkl_path.exists():
			continue

		df = pd.read_pickle(pkl_path)
		scores = _score_services(df)
		if not scores:
			top = ServiceScore("", 0.0, 0.0, 0.0, 0.0, 1.0)
			top_services = []
			top_service = ""
			component = normalize_component(uuid_to_hint_service.get(uuid, ""))
		else:
			top = scores[0]
			top_services = [(s.service, float(s.score)) for s in scores[:5] if s.service]
			top_service = top.service
			component = normalize_component(top_service)

			hint = normalize_component(uuid_to_hint_service.get(uuid, ""))
			if hint:
				telemetry_candidates = {normalize_component(s) for s, _ in top_services}
				telemetry_candidates.add(component)
				if hint in telemetry_candidates:
					component = hint

		reason = _reason_from_top(top)
		signal_keywords = _signal_keywords(top)
		reason_clean = (
			reason
			+ (f". Component={component}" if component else "")
			+ (". Signals=" + ",".join(signal_keywords) if signal_keywords else "")
		)

		trace = [
			{
				"step": 1,
				"action": "init",
				"observation": f"uuid={uuid} phase={args.phase}",
			},
			{
				"step": 2,
				"action": "rank_services",
				"observation": (
					"top_services=" + "; ".join([f"{s}:{sc:.3f}" for s, sc in top_services])
					+ f" | pick={top_service} component={component}"
				)[:300],
			},
			{
				"step": 3,
				"action": "summarize_signals",
				"observation": (
					f"signals error_max={top.max_error:.3f} timeout_max={top.max_timeout:.3f} "
					f"dur_max={top.max_duration:.3f} calls_min_ratio={top.min_calls_ratio:.3f} "
					+ (f" signal_tags={','.join(signal_keywords)}" if signal_keywords else "")
				)[:300],
			},
		]

		rows.append(
			{
				"uuid": uuid,
				"component": component,
				"reason": reason_clean,
				"reasoning_trace": trace,
			}
		)

		n += 1
		if args.limit and n >= int(args.limit):
			break

	_write_jsonl(args.out_jsonl, rows)


if __name__ == "__main__":
	main()

