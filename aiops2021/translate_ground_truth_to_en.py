from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s or ""))


# Phrase-level translations for reasons/keywords seen in aiops2021 GT.
# This is intentionally small and deterministic (no external services).
_PHRASE_MAP: list[tuple[str, str]] = [
    ("网络故障", "network"),
    ("网络丢包", "packet loss"),
    ("网络延迟", "latency"),
    ("应用故障", "application"),
    ("资源故障", "resource"),
    ("CPU使用率高", "high cpu"),
    ("内存使用率过高", "high memory"),
    ("磁盘IO读使用率过高", "high disk read io"),
    ("磁盘IO写使用率过高", "high disk write io"),
    ("磁盘空间使用率过高", "high disk space"),
    ("JVM CPU负载高", "high jvm cpu"),
    ("JVM OOM Heap", "jvm oom heap"),
]


def translate_text_zh_to_en(text: str) -> str:
    s = str(text or "")
    for zh, en in _PHRASE_MAP:
        s = s.replace(zh, en)
    # keep separators readable
    s = s.replace("/", "/")
    return s


def _core_tokens_from_en_phrase(en: str) -> List[str]:
    # Add small tokens likely to match submission reasons.
    tokens: List[str] = []
    for t in re.split(r"[^a-z0-9]+", (en or "").lower()):
        t = t.strip()
        if not t:
            continue
        # keep short but meaningful tokens
        if t in {"high", "and", "or", "the", "a", "an"}:
            continue
        tokens.append(t)
    return tokens


def translate_keywords(keywords: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    def add(tok: str) -> None:
        tok = str(tok or "").strip()
        if not tok:
            return
        if tok in seen:
            return
        seen.add(tok)
        out.append(tok)

    for kw in keywords or []:
        kw_s = str(kw or "").strip()
        if not kw_s:
            continue

        if _has_cjk(kw_s):
            en = translate_text_zh_to_en(kw_s)
            add(en)
            for t in _core_tokens_from_en_phrase(en):
                add(t)
        else:
            add(kw_s)
            # also add lower-cased token for robustness
            low = kw_s.lower().strip()
            if low and low != kw_s:
                add(low)

    return out


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        yield json.loads(raw)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def translate_gt_file(in_path: Path, out_path: Path) -> None:
    translated: List[Dict[str, Any]] = []
    for rec in iter_jsonl(in_path):
        reason = str(rec.get("reason") or "")
        rec["reason"] = translate_text_zh_to_en(reason) if _has_cjk(reason) else reason

        kws = rec.get("reason_keywords")
        if isinstance(kws, list):
            rec["reason_keywords"] = translate_keywords(kws)

        # evidence都是空的
        translated.append(rec)

    write_jsonl(out_path, translated)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Translate aiops2021 ground_truth.jsonl (zh->en) deterministically")
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    args = p.parse_args(argv)

    translate_gt_file(args.in_path, args.out_path)
    print(f"Wrote: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
