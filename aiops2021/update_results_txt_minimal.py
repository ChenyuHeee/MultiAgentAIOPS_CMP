"""Update 测评结果.txt with minimal numeric scores from existing judge reports.

Reads:
  aiops2021/outputs/report_*_all_{zh,en}.json
and writes a compact summary to:
  测评结果.txt

Missing reports are kept as '-'.
"""

from __future__ import annotations

import json
from datetime import date
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Score:
    la: float
    ta: float
    eff: float
    exp: float
    final: float


def _read_score(path: Path) -> Optional[Score]:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    metrics = obj.get("metrics") if isinstance(obj.get("metrics"), dict) else obj

    def f(key: str) -> float:
        v = metrics.get(key)
        if v is None:
            raise KeyError(key)
        return float(v)

    return Score(
        la=f("component_accuracy"),
        ta=f("reason_accuracy"),
        eff=f("efficiency"),
        exp=f("explainability"),
        final=f("final_score"),
    )


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}"


def _fmt_line(name: str, lang: str, s: Optional[Score]) -> str:
    if s is None:
        return f"{name} {lang}: LA - | TA - | Eff - | Exp - | Final -"
    return (
        f"{name} {lang}: LA {_fmt_pct(s.la)} | TA {_fmt_pct(s.ta)} | "
        f"Eff {_fmt_pct(s.eff)} | Exp {_fmt_pct(s.exp)} | Final {s.final:.2f}"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "aiops2021" / "outputs"

    mapping: Dict[Tuple[str, str], Path] = {
        ("OpenRCA(heuristic)", "zh"): out_dir / "report_openrca_all_zh.json",
        ("OpenRCA(heuristic)", "en"): out_dir / "report_openrca_all_en.json",
        ("OpenRCA(agent,qwen)", "zh"): out_dir / "report_openrca_agent_all_qwen_zh.json",
        ("OpenRCA(agent,qwen)", "en"): out_dir / "report_openrca_agent_all_qwen_en.json",
        ("OpenRCA(agent,deepseek)", "zh"): out_dir / "report_openrca_agent_all_deepseek_zh.json",
        ("OpenRCA(agent,deepseek)", "en"): out_dir / "report_openrca_agent_all_deepseek_en.json",
        ("CausalRCA", "zh"): out_dir / "report_causalrca_all_zh.json",
        ("CausalRCA", "en"): out_dir / "report_causalrca_all_en.json",
        ("mABC(qwen)", "zh"): out_dir / "report_mabc_all_qwen_zh.json",
        ("mABC(qwen)", "en"): out_dir / "report_mabc_all_qwen_en.json",
        ("mABC(deepseek)", "zh"): out_dir / "report_mabc_all_deepseek_zh.json",
        ("mABC(deepseek)", "en"): out_dir / "report_mabc_all_deepseek_en.json",
    }

    lines = [
        "aiops2021 all（159） AIOpsChallengeJudge 跑分汇总（reason-threshold=0.65）",
        f"更新时间：{date.today().isoformat()}",
        _fmt_line("OpenRCA(heuristic)", "zh", _read_score(mapping[("OpenRCA(heuristic)", "zh")])),
        _fmt_line("OpenRCA(heuristic)", "en", _read_score(mapping[("OpenRCA(heuristic)", "en")])),
        "",
        "",
        _fmt_line("OpenRCA(agent,qwen)", "zh", _read_score(mapping[("OpenRCA(agent,qwen)", "zh")])),
        _fmt_line("OpenRCA(agent,qwen)", "en", _read_score(mapping[("OpenRCA(agent,qwen)", "en")])),
        "",
        _fmt_line(
            "OpenRCA(agent,deepseek)",
            "zh",
            _read_score(mapping[("OpenRCA(agent,deepseek)", "zh")]),
        ),
        _fmt_line(
            "OpenRCA(agent,deepseek)",
            "en",
            _read_score(mapping[("OpenRCA(agent,deepseek)", "en")]),
        ),
        "",
        _fmt_line("CausalRCA", "zh", _read_score(mapping[("CausalRCA", "zh")])),
        _fmt_line("CausalRCA", "en", _read_score(mapping[("CausalRCA", "en")])),
        "",
        _fmt_line("mABC(qwen)", "zh", _read_score(mapping[("mABC(qwen)", "zh")])),
        _fmt_line("mABC(qwen)", "en", _read_score(mapping[("mABC(qwen)", "en")])),
        "",
        _fmt_line("mABC(deepseek)", "zh", _read_score(mapping[("mABC(deepseek)", "zh")])),
        _fmt_line("mABC(deepseek)", "en", _read_score(mapping[("mABC(deepseek)", "en")])),
        "",
    ]

    (root / "测评结果.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
