from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


_OPENRCA_ROOT = Path(__file__).resolve().parents[1]
if str(_OPENRCA_ROOT) not in sys.path:
	sys.path.insert(0, str(_OPENRCA_ROOT))

from rca.api_router import get_chat_completion


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


def _iter_pkl_paths(pkls_dir: Path) -> Iterable[Path]:
	for p in sorted(pkls_dir.glob("*.pkl")):
		if p.is_file():
			yield p


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
class ServiceStats:
	service: str
	max_error: float
	max_timeout: float
	max_duration: float
	duration_spike: float
	calls_drop: float
	score: float


def _safe_median_positive(series: pd.Series) -> float:
	if series is None:
		return 0.0
	pos = series[pd.to_numeric(series, errors="coerce").fillna(0.0) > 0]
	if pos.empty:
		return 0.0
	return float(pos.median())


def summarize_df(df: pd.DataFrame, top_k: int = 8) -> Tuple[List[ServiceStats], List[str]]:
	services = _services_from_columns(df.columns)
	allowed = set(services)
	# 常见 node 名（在 GT 里出现，但 pkls 未必有）；允许 LLM 输出
	for i in range(1, 9):
		allowed.add(f"aiops-k8s-{i:02d}")
	allowed_list = sorted(allowed)

	if df.empty or not services:
		return [], allowed_list

	eps = 1e-9
	rows: List[ServiceStats] = []
	for svc in services:
		er = float(df[f"{svc}_error_rate"].max() if f"{svc}_error_rate" in df else 0.0)
		tr = float(df[f"{svc}_timeout_rate"].max() if f"{svc}_timeout_rate" in df else 0.0)
		dur_series = df.get(f"{svc}_avg_duration") if f"{svc}_avg_duration" in df else None
		max_dur = float(dur_series.max() if dur_series is not None else 0.0)
		med_dur = _safe_median_positive(dur_series) if dur_series is not None else 0.0
		dur_spike = max_dur / (med_dur + eps) if (max_dur > 0 and med_dur > 0) else 0.0
		calls_series = df.get(f"{svc}_calls") if f"{svc}_calls" in df else None
		if calls_series is None:
			calls_drop = 0.0
		else:
			med_calls = float(pd.to_numeric(calls_series, errors="coerce").fillna(0.0).median())
			min_calls = float(pd.to_numeric(calls_series, errors="coerce").fillna(0.0).min())
			calls_drop = max(0.0, 1.0 - (min_calls / (med_calls + eps))) if med_calls > 0 else 0.0

		score = (3.0 * tr) + (2.6 * er) + (0.20 * min(10.0, dur_spike)) + (0.06 * max_dur) + (0.35 * calls_drop)
		rows.append(
			ServiceStats(
				service=svc,
				max_error=er,
				max_timeout=tr,
				max_duration=max_dur,
				duration_spike=float(dur_spike),
				calls_drop=float(calls_drop),
				score=float(score),
			)
		)

	rows.sort(key=lambda r: r.score, reverse=True)
	return rows[: max(1, int(top_k))], allowed_list


_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json_obj(text: str) -> Optional[dict]:
	if not text:
		return None
	m = _JSON_RE.search(text)
	if not m:
		return None
	blob = m.group(0)
	try:
		return json.loads(blob)
	except Exception:
		return None


def _best_token_match(text: str, allowed_tokens: Sequence[str]) -> str:
	if not text:
		return ""
	s = text.strip().lower()
	best = ""
	best_len = 0
	boundary = set("/ .,:;?()[]{}<>\"'\t\n\r")
	for tok in allowed_tokens:
		t = tok.lower()
		if not t:
			continue
		if not s.startswith(t):
			continue
		nxt = len(t)
		if nxt < len(s):
			ch = s[nxt]
			if ch.isalnum() or ch == "_":
				continue
			if ch not in boundary and ch != "-":
				continue
		if len(t) > best_len:
			best = tok
			best_len = len(t)
	return best


def normalize_component(raw: str, allowed_tokens: Sequence[str]) -> str:
	"""Normalize to one of allowed tokens, or keep a->b when both parts are allowed."""
	s = (raw or "").strip()
	if not s:
		return ""
	s_low = s.lower()
	if "->" in s_low:
		left, right = [p.strip() for p in s_low.split("->", 1)]
		l = _best_token_match(left, allowed_tokens) or left
		r = _best_token_match(right, allowed_tokens) or right
		return f"{str(l).lower()}->{str(r).lower()}"
	best = _best_token_match(s, allowed_tokens)
	return str(best).lower() if best else s_low


def llm_predict_component_reason(
	uuid: str,
	instruction: str,
	stats: Sequence[ServiceStats],
	allowed_tokens: Sequence[str],
	temperature: float,
	max_retries: int,
	sleep_s: float,
) -> Tuple[str, str, str]:
	# 构造一个尽量“真实诊断”的 prompt：只给指标摘要，不塞 GT 关键词，不用 Judge vocab。
	table_lines = [
		"service,max_error,max_timeout,max_duration,duration_spike,calls_drop,score",
	]
	for r in stats[:8]:
		table_lines.append(
			f"{r.service},{r.max_error:.4f},{r.max_timeout:.4f},{r.max_duration:.4f},{r.duration_spike:.2f},{r.calls_drop:.2f},{r.score:.4f}"
		)
	table = "\n".join(table_lines)

	allowed_hint = ", ".join(list(allowed_tokens)[:40])
	if len(allowed_tokens) > 40:
		allowed_hint += ", ..."

	system = (
		"You are an experienced SRE doing root cause analysis for a microservices system. "
		"You will be given a short incident description and aggregated per-service metrics around the incident. "
		"Infer the most likely root cause component and a concise reason. "
		"Return ONLY a JSON object with keys: component, reason."
	)
	user = (
		f"uuid: {uuid}\n"
		f"incident: {instruction}\n\n"
		"Aggregated metrics table (higher score means more anomalous):\n"
		f"{table}\n\n"
		"Constraints:\n"
		"- component should be a single service name (lowercase) or an edge 'serviceA->serviceB' when you strongly believe it's a network/path issue.\n"
		"- Keep reason short (<= 30 words).\n"
		"- Candidate component tokens include (not exhaustive): "
		f"{allowed_hint}\n"
	)

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": user},
	]

	last_text = ""
	for attempt in range(1, max_retries + 1):
		try:
			text = str(get_chat_completion(messages, temperature=temperature) or "").strip()
		except Exception as e:
			last_text = f"ERROR: {type(e).__name__}: {e}"
			if sleep_s > 0:
				time.sleep(float(sleep_s) * attempt)
			continue
		last_text = text
		obj = _extract_json_obj(text)
		if isinstance(obj, dict) and isinstance(obj.get("component"), str) and isinstance(obj.get("reason"), str):
			comp = normalize_component(obj["component"], allowed_tokens)
			reason = obj["reason"].strip()
			return comp, reason, text

		# repair prompt
		messages.append(
			{
				"role": "assistant",
				"content": text,
			}
		)
		messages.append(
			{
				"role": "user",
				"content": "Output ONLY valid JSON like: {\"component\":\"...\",\"reason\":\"...\"}. No extra text.",
			}
		)
		if sleep_s > 0:
			time.sleep(float(sleep_s) * attempt)

	# fallback: pick top anomalous service
	fallback_component = stats[0].service.lower() if stats else "checkoutservice"
	fallback_reason = "service anomaly detected in aggregated metrics"
	return fallback_component, fallback_reason, last_text


def _write_jsonl_append(path: Path, rows: List[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as f:
		for obj in rows:
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_done_uuids(path: Path) -> set[str]:
	if not path.exists():
		return set()
	done: set[str] = set()
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
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


def _load_jsonl_uuid_to_obj(path: Path) -> Dict[str, dict]:
	"""Load jsonl into a uuid->obj mapping.

	If there are duplicate uuids, the last one wins.
	"""
	out: Dict[str, dict] = {}
	if not path.exists():
		return out
	for line in path.read_text(encoding="utf-8").splitlines():
		line = line.strip()
		if not line:
			continue
		try:
			obj = json.loads(line)
		except Exception:
			continue
		uuid = str(obj.get("uuid", "")).strip()
		if not uuid:
			continue
		out[uuid] = obj
	return out


def _write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	tmp = path.with_suffix(path.suffix + ".tmp")
	with tmp.open("w", encoding="utf-8") as f:
		for obj in rows:
			f.write(json.dumps(obj, ensure_ascii=False) + "\n")
	tmp.replace(path)


def _is_llm_error(obj: dict) -> bool:
	raw = str(obj.get("_llm_raw", "") or "")
	return raw.startswith("ERROR:")


def rerun_error_rows(
	*,
	phase: str,
	pkls_dir: Path,
	input_json: Path,
	out_jsonl: Path,
	temperature: float,
	max_retries: int,
	sleep_s: float,
	limit: int,
) -> None:
	uuid_to_instr = _load_uuid_to_instruction(input_json)
	uuid_to_obj = _load_jsonl_uuid_to_obj(out_jsonl)
	if not uuid_to_obj:
		raise FileNotFoundError(f"out-jsonl not found or empty: {out_jsonl}")

	error_uuids = [u for u, obj in uuid_to_obj.items() if _is_llm_error(obj)]
	error_uuids.sort()
	if limit:
		error_uuids = error_uuids[: int(limit)]

	print(f"[{phase}] rerun-errors: {len(error_uuids)}", flush=True)
	updated = 0
	for uuid in error_uuids:
		instr = uuid_to_instr.get(uuid, "")
		pkl_path = pkls_dir / f"{uuid}.pkl"
		if not pkl_path.exists() or not instr:
			continue

		df = pd.read_pickle(pkl_path)
		stats, allowed = summarize_df(df, top_k=10)
		component, reason, raw_text = llm_predict_component_reason(
			uuid=uuid,
			instruction=instr,
			stats=stats,
			allowed_tokens=allowed,
			temperature=temperature,
			max_retries=max_retries,
			sleep_s=sleep_s,
		)

		trace = [
			{"step": 1, "action": "init", "observation": f"uuid={uuid} phase={phase}"},
			{
				"step": 2,
				"action": "metrics_summary",
				"observation": ("top_services=" + ";".join([s.service for s in stats[:5]]))[:120],
			},
			{"step": 3, "action": "llm_decision", "observation": f"component={component} reason={reason}"[:120]},
		]

		uuid_to_obj[uuid] = {
			"uuid": uuid,
			"component": component,
			"reason": reason,
			"reasoning_trace": trace,
			"_llm_raw": raw_text[:500],
		}
		updated += 1
		print(f"[{phase}] fixed {uuid} -> {component}", flush=True)

	# Write back in a stable order (by uuid) to avoid duplicates.
	_write_jsonl_atomic(out_jsonl, [uuid_to_obj[u] for u in sorted(uuid_to_obj.keys())])
	print(f"[{phase}] rerun-errors done; updated={updated}", flush=True)


def run_phase(
	phase: str,
	pkls_dir: Path,
	input_json: Path,
	out_jsonl: Path,
	limit: int,
	resume: bool,
	temperature: float,
	max_retries: int,
	sleep_s: float,
) -> None:
	uuid_to_instr = _load_uuid_to_instruction(input_json)
	done = _load_done_uuids(out_jsonl) if resume else set()

	written = 0
	for pkl_path in _iter_pkl_paths(pkls_dir):
		uuid = pkl_path.stem
		if uuid not in uuid_to_instr:
			continue
		if uuid in done:
			continue

		df = pd.read_pickle(pkl_path)
		stats, allowed = summarize_df(df, top_k=10)
		component, reason, raw_text = llm_predict_component_reason(
			uuid=uuid,
			instruction=uuid_to_instr.get(uuid, ""),
			stats=stats,
			allowed_tokens=allowed,
			temperature=temperature,
			max_retries=max_retries,
			sleep_s=sleep_s,
		)

		trace = [
			{"step": 1, "action": "init", "observation": f"uuid={uuid} phase={phase}"},
			{"step": 2, "action": "metrics_summary", "observation": ("top_services=" + ";".join([s.service for s in stats[:5]]) )[:120]},
			{"step": 3, "action": "llm_decision", "observation": f"component={component} reason={reason}"[:120]},
		]

		row = {
			"uuid": uuid,
			"component": component,
			"reason": reason,
			"reasoning_trace": trace,
			"_llm_raw": raw_text[:500],
		}
		_write_jsonl_append(out_jsonl, [row])
		print(f"[{phase}] wrote {uuid} -> {component}", flush=True)

		written += 1
		if limit and written >= int(limit):
			break


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--phase", choices=["phase1", "phase2"], required=True)
	parser.add_argument("--pkls-dir", type=Path, required=True)
	parser.add_argument("--input-json", type=Path, required=True)
	parser.add_argument("--out-jsonl", type=Path, required=True)
	parser.add_argument("--limit", type=int, default=0)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument(
		"--rerun-errors",
		action="store_true",
		help="Re-run only rows whose _llm_raw starts with 'ERROR:' and rewrite out-jsonl in-place.",
	)
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--max-retries", type=int, default=2)
	parser.add_argument("--sleep", type=float, default=0.0)
	args = parser.parse_args()

	if not args.pkls_dir.exists():
		raise FileNotFoundError(f"pkls-dir not found: {args.pkls_dir}")
	if not args.input_json.exists():
		raise FileNotFoundError(f"input-json not found: {args.input_json}")

	if args.rerun_errors:
		rerun_error_rows(
			phase=args.phase,
			pkls_dir=args.pkls_dir,
			input_json=args.input_json,
			out_jsonl=args.out_jsonl,
			temperature=float(args.temperature),
			max_retries=int(args.max_retries),
			sleep_s=float(args.sleep),
			limit=int(args.limit),
		)
		return

	run_phase(
		phase=args.phase,
		pkls_dir=args.pkls_dir,
		input_json=args.input_json,
		out_jsonl=args.out_jsonl,
		limit=args.limit,
		resume=bool(args.resume),
		temperature=float(args.temperature),
		max_retries=int(args.max_retries),
		sleep_s=float(args.sleep),
	)


if __name__ == "__main__":
	main()
