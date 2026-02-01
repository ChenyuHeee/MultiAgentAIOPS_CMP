"""
评测提交结果
根据result.jsonl 文件进行评测

"""
from __future__ import annotations # 类型注解前向引用支持

import argparse # 命令行参数解析
import json #引用JSON处理模块
import math # 数学函数库
import sys # 系统模块
from dataclasses import dataclass # 数据类支持
from pathlib import Path # 路径处理模块
from typing import Any, Dict, Iterable, List, Sequence, Tuple # 类型声明

from difflib import SequenceMatcher # 序列匹配工具，用于计算字符串相似度

# ---------------------------------------------------------------------------
# 本程序的数据结构
# ---------------------------------------------------------------------------

@dataclass # 数据类
class SampleScore: # 单条样本的打分结果
    # 单条样本的打分结果，用于 --show-details 或报告输出
    uuid: str
    component_correct: bool
    reason_correct: bool
    step_count: int
    evidence_hit: int
    evidence_total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_SUBMISSION_FIELDS = ("uuid", "component", "reason", "reasoning_trace") # 提交文件中每条记录必须包含的字段
REQUIRED_TRACE_FIELDS = ("step", "action", "observation") # 推理链中每条记录必须包含的字段


def load_jsonl(path: Path, allow_empty: bool = False) -> List[Dict[str, Any]]: # 载入 JSONL 文件，返回记录列表，Path为文件路径，allow_empty为是否允许空文件，Dict为数据结构
    # 逐行读入JSONL
    if not path.exists(): #文件不存在则抛出异常
        raise FileNotFoundError(f"File not found: {path}")

    records: List[Dict[str, Any]] = [] #记录数据列表
    with path.open("r", encoding="utf-8") as f: # 打开文件
        for line_no, line in enumerate(f, start=1): #逐行读取文件，line_no为行号，line为行内容，enumerate为迭代器，返回索引和值，其中f表示文件句柄，start为行号，默认为1
            stripped = line.strip() #去除行首尾空白
            if not stripped: # 如果行内容为空，则跳过
                continue
            try: 
                records.append(json.loads(stripped)) #加载JSON数据并添加到记录列表中
            except json.JSONDecodeError as exc: # 捕获JSON解码错误，JSONDecodeError为JSON解码错误，ValueError为值错误，exc为异常对象，except表示捕获异常，except 后面可以跟一个异常类型，表示只捕获该异常
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc # 抛出值错误，附带行号和文件路径信息，raise表示抛出异常，ValueError为值错误，raise的用法为抛出异常，参数为异常对象
    if not records and not allow_empty: # 如果记录列表为空且不允许空文件，则抛出异常
        raise ValueError(f"File {path} is empty; expected at least one record") # 抛出值错误，附带文件路径信息，ValueError为值错误
    return records # 返回记录列表


def build_index(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]: # 构建 uuid 到记录的映射索引
    # 将 uuid 映射到对应记录，方便随机访问
    index: Dict[str, Dict[str, Any]] = {} # 索引数据结构，语法为 Dict[key_type, value_type]
    for record in records: # 遍历记录列表
        uuid = record.get("uuid") # 获取 uuid
        if not isinstance(uuid, str) or not uuid: # 如果 uuid 不是字符串或者为空，则抛出异常
            raise ValueError("Each record must contain a non-empty string uuid") # 抛出值错误，附带 uuid 信息，ValueError为值错误
        if uuid in index: # 如果 uuid 已经存在，则抛出异常
            raise ValueError(f"Duplicate uuid detected: {uuid}")
        index[uuid] = record # 添加 uuid 到记录的映射
    return index # 返回索引


def ensure_submission_schema(record: Dict[str, Any]) -> None: # 校验提交记录的格式
    # 严格检查提交格式，提前抛出可读错误
    for field in REQUIRED_SUBMISSION_FIELDS:
        if field not in record:
            raise ValueError(f"Submission entry {record.get('uuid')} missing field '{field}'")

    if not isinstance(record["component"], str):
        raise ValueError(f"component must be string in entry {record['uuid']}")
    if not isinstance(record["reason"], str):
        raise ValueError(f"reason must be string in entry {record['uuid']}")

    trace = record["reasoning_trace"]
    if not isinstance(trace, list):
        raise ValueError(f"reasoning_trace must be a list in entry {record['uuid']}")
    for idx, step in enumerate(trace, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"reasoning_trace[{idx}] must be an object in entry {record['uuid']}")
        for field in REQUIRED_TRACE_FIELDS:
            if field not in step:
                raise ValueError(
                    f"reasoning_trace[{idx}] in entry {record['uuid']} missing field '{field}'"
                )
        if not isinstance(step["step"], int):
            raise ValueError(f"reasoning_trace[{idx}].step must be int in entry {record['uuid']}")
        if not isinstance(step["action"], str):
            raise ValueError(
                f"reasoning_trace[{idx}].action must be string in entry {record['uuid']}"
            )
        if not isinstance(step["observation"], str):
            raise ValueError(
                f"reasoning_trace[{idx}].observation must be string in entry {record['uuid']}"
            )


def tokenize(text: str) -> List[str]: # List[str]表示字符串列表，str表示字符串
    return [token for token in text.lower().split() if token] # 将文本转换为小写，并使用空格进行分词，并过滤掉空字符串，返回结果列表


WORD_TRUNCATE_LIMIT = 100


def truncate_words(text: str, limit: int) -> str: # 截断文本到指定的词数限制
    tokens = text.split() # 将文本转换为列表
    if len(tokens) <= limit: # 如果词数小于等于限制，则返回原始文本
        return text.strip() # 去除首尾空格
    return " ".join(tokens[:limit]) # 将列表转换为字符串，并返回


def sequence_similarity(a: str, b: str) -> float: # 计算两个字符串的相似度
    return SequenceMatcher(a=a.lower(), b=b.lower()).ratio() # 计算两个字符串的相似度，SequenceMatcher为序列匹配工具，ratio为相似度比例


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float: # 计算两个字符串列表的 Jaccard 相似度，jaccard相似度为两个集合的交集除以两个集合的并集
    set_a = set(a) # 将字符串列表转换为集合，set()函数可以将列表转换为集合
    set_b = set(b)
    if not set_a or not set_b: # 如果任一集合为空，则返回0
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b) # Jaccard 相似度计算两个集合的交集和并集，并返回交集除以并集的比值


def reason_matches(gt_reason: str, submission_reason: str, keywords: Sequence[str], threshold: float) -> bool: # 判断提交理由是否与Ground Truth一致，gt_reason为Ground Truth的理由，submission_reason为提交理由，keywords为关键词列表，threshold为相似度阈值
    # 先用关键词判断，再回退到相似度，避免语义描述差异造成误伤
    trimmed_submission = truncate_words(submission_reason, WORD_TRUNCATE_LIMIT) # 将提交理由截断到设定的单词数限制
    trimmed_gt = truncate_words(gt_reason, WORD_TRUNCATE_LIMIT) # 将Ground Truth的理由截断到设定的单词数限制
    submission_lower = trimmed_submission.lower() # 将提交理由转换为小写

    for keyword in keywords: # 遍历关键词列表
        kw_lower = keyword.lower().strip() # 将关键词转换为小写
        if kw_lower and kw_lower in submission_lower: # 如果关键词不为空且出现在提交理由中，则返回True
            return True

    seq_score = sequence_similarity(trimmed_submission, trimmed_gt) # 计算两个字符串的相似度
    token_score = jaccard_similarity(tokenize(trimmed_submission), tokenize(trimmed_gt)) # 计算两个字符串列表的 Jaccard 相似度
    return max(seq_score, token_score) >= threshold # 返回相似度是否满足阈值要求


def normalize_component(value: str) -> str: # 规范化组件名称，去除多余空白
    return value.strip()


def normalize_observation(value: str, char_limit: int = 100, word_limit: int = 20) -> str: # 规范化观察结果，截断到指定字符数和单词数，并去除多余空白
    snippet = value.strip()[:char_limit] # 截断字符串
    tokens = snippet.split() # 分词
    if len(tokens) <= word_limit: # 如果单词数小于等于限制，则返回原始字符串
        return " ".join(tokens) #
    return " ".join(tokens[:word_limit]) # 将列表转换为字符串，并返回结果


def evidence_hits(evidence_points: Sequence[Dict[str, Any]], reasoning_trace: Sequence[Dict[str, Any]]) -> Tuple[int, int]: # 计算推理链中提到的关键词数量
    # 对每个 evidence 点检查推理链是否提到对应关键词
    if not evidence_points: # 如果没有 evidence 点，则返回0
        return 0, 0

    normalized_observations = [normalize_observation(step.get("observation", "")) for step in reasoning_trace] # 将推理链中的观察结果转换为小写
    hits = 0 # 命中数量
    total = 0 # 总数
    for point in evidence_points: # 遍历每个 evidence 点
        keywords = point.get("keywords") or [] # 获取关键词列表
        if not isinstance(keywords, list) or not keywords: # 如果关键词不是列表或者为空，则跳过
            continue
        total += 1 # 总数加1
        lowered_keywords = [kw.lower().strip() for kw in keywords if isinstance(kw, str) and kw.strip()] # 将关键词转换为小写
        if not lowered_keywords: # 如果关键词列表为空，则跳过
            total -= 1
            continue
        if any(kw in obs.lower() for kw in lowered_keywords for obs in normalized_observations): # 检查推理链中是否提到关键词
            hits += 1
    return hits, total # 返回命中数量和总数


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------


def score_submission( # 计算提交分数
    ground_truth: Dict[str, Dict[str, Any]], # 输入的Ground Truth数据
    submission: Dict[str, Dict[str, Any]], # 输入的提交数据
    reason_threshold: float = 0.65, # 理由相似度阈值
) -> Tuple[Dict[str, float], List[SampleScore]]: # 返回总分和样本得分列表
    # 主打分流程：遍历 uuid，计算四项指标并汇总加权
    total_samples = len(ground_truth) # 总样本数
    component_hits = 0 # 组件命中数量
    reason_hits_total = 0 # 理由命中数量
    path_lengths: List[int] = [] # 路径长度列表
    total_evidence_hits = 0 # 证据命中数量
    total_evidence_points = 0 # 证据点总数
    per_sample: List[SampleScore] = [] # 样本得分列表

    for uuid, gt_entry in ground_truth.items(): # 遍历每个 uuid 和对应的 Ground Truth 记录
        submission_entry = submission[uuid] # 获取对应 uuid 的提交记录
        ensure_submission_schema(submission_entry) # 检查提交记录是否符合要求

        gt_component = normalize_component(gt_entry.get("component", "")) # 获取并规范化 Ground Truth 组件名称
        submission_component = normalize_component(submission_entry["component"]) # 获取并规范化提交组件名称
        component_correct = gt_component == submission_component and bool(gt_component) # 检查组件名称是否一致
        if component_correct: # 如果组件名称一致，则组件命中数量加1
            component_hits += 1

        gt_reason = gt_entry.get("reason", "") # 获取并规范化 Ground Truth 理由
        gt_keywords = gt_entry.get("reason_keywords") or [] # 获取关键词列表
        if not isinstance(gt_keywords, list): # 如果关键词不是列表，则将其转换为列表
            gt_keywords = [] # 将关键词转换为列表
        reason_correct = False # 理由是否正确
        if isinstance(gt_reason, str) and gt_reason.strip(): # 如果理由是字符串且不为空，则进行匹配
            reason_correct = reason_matches(gt_reason, submission_entry["reason"], gt_keywords, reason_threshold) # 检查理由是否匹配
        if reason_correct: # 如果理由正确，则理由命中数量加1
            reason_hits_total += 1

        trace_steps = submission_entry.get("reasoning_trace", []) # 获取推理链步骤
        step_count = len(trace_steps) # 推理链步骤数
        if component_correct: # 如果组件名称一致，则计算推理链长度
            path_lengths.append(step_count)

        evidence = gt_entry.get("evidence_points") or [] # 获取证据点
        evidence_hit, evidence_total = evidence_hits(evidence, trace_steps) # 计算证据点命中数量和总数
        total_evidence_hits += evidence_hit
        total_evidence_points += evidence_total

        per_sample.append( # 创建样本得分对象
            SampleScore( # 创建样本得分对象
                uuid=uuid,
                component_correct=component_correct,
                reason_correct=reason_correct,
                step_count=step_count,
                evidence_hit=evidence_hit,
                evidence_total=evidence_total,
            )
        )

    la = component_hits / total_samples
    ta = reason_hits_total / total_samples
    efficiency = 0.0
    if path_lengths: # 如果有推理链长度，则计算平均推理链长度
        apl = sum(path_lengths) / len(path_lengths) #
        efficiency = math.exp(-(apl - 5.0) / 5.0) # 计算效率
        efficiency = min(max(efficiency, 0.0), 1.0) # 归一化效率到0-1范围内

    explainability = 0.0
    if total_evidence_points: # 如果有证据点，则计算解释性
        explainability = total_evidence_hits / total_evidence_points # 计算解释性

    final_score = 100.0 * (0.40 * la + 0.40 * ta + 0.10 * efficiency + 0.10 * explainability) # 计算总得分

    metrics = {
        "component_accuracy": la,
        "reason_accuracy": ta,
        "efficiency": efficiency,
        "explainability": explainability,
        "final_score": final_score,
    }
    return metrics, per_sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace: # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate AIOps root-cause predictions") # 创建参数解析器
    parser.add_argument("--ground-truth", "-g", type=Path, required=True, help="Ground-truth JSONL file") # 添加参数
    parser.add_argument("--submission", "-s", type=Path, required=True, help="Submission JSONL file")# 添加参数
    parser.add_argument( # 添加参数
        "--report",
        "-o",
        type=Path,
        default=None,
        help="Optional path to save a JSON report with aggregate and per-sample scores",
    )
    parser.add_argument( # 添加参数
        "--reason-threshold",
        type=float,
        default=0.65,
        help="Minimum similarity score (0-1) required to accept a reason when keywords miss",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print per-sample breakdown to stdout",
    )
    return parser.parse_args(argv)


def format_percentage(value: float) -> str: # 格式化百分比
    return f"{value * 100:.2f}%"


def main(argv: Sequence[str]) -> int: # 主函数
    args = parse_args(argv) # 解析命令行参数

    ground_truth_records = load_jsonl(args.ground_truth) # 加载Ground Truth记录
    submission_records = load_jsonl(args.submission, allow_empty=True) # 加载提交记录

    gt_index = build_index(ground_truth_records) # 构建索引
    submission_index = build_index(submission_records) # 构建索引

    gt_uuids = set(gt_index.keys()) # 获取Ground Truth的UUID
    submission_uuids = set(submission_index.keys()) # 获取提交记录的UUID
    extra = sorted(submission_uuids - gt_uuids) # 获取额外的UUID，额外的uuid来自于提交记录，但不在Ground Truth中
    missing = sorted(gt_uuids - submission_uuids) # 获取缺失的UUID，缺失的uuid来自于Ground Truth，但不在提交记录中
    if extra: # 如果有额外的UUID，则打印警告
        print(
            "[WARN] submission contains uuids not present in ground truth; they will be ignored:",
            ", ".join(extra),
        )
        for uuid in extra:
            submission_index.pop(uuid, None)
    if missing: # 如果有缺失的UUID，则打印警告
        print(
            "[WARN] submission missing uuids present in ground truth; blank predictions will be assumed:",
            ", ".join(missing),
        )
        for uuid in missing:
            submission_index[uuid] = {
                "uuid": uuid,
                "component": "",
                "reason": "",
                "reasoning_trace": [],
            }

    metrics, per_sample = score_submission(gt_index, submission_index, args.reason_threshold) # 计算得分
# 打印结果
    print("===== Overall Metrics =====")
    print(f"Component Accuracy (LA): {format_percentage(metrics['component_accuracy'])}")
    print(f"Reason Accuracy (TA):    {format_percentage(metrics['reason_accuracy'])}")
    print(f"Efficiency:              {format_percentage(metrics['efficiency'])}")
    print(f"Explainability:          {format_percentage(metrics['explainability'])}")
    print(f"Final Score:             {metrics['final_score']:.2f}")

    if args.show_details: # 如果显示详细信息，则打印
        print("\n===== Per-sample Breakdown =====")
        header = "uuid,component_ok,reason_ok,steps,evidence_hit,evidence_total"
        print(header)
        for sample in per_sample:
            print(
                f"{sample.uuid},{sample.component_correct},{sample.reason_correct},"
                f"{sample.step_count},{sample.evidence_hit},{sample.evidence_total}"
            )

    if args.report: # 如果有报告，则保存
        report_payload = {
            "metrics": metrics,
            "samples": [sample.__dict__ for sample in per_sample],
        }
        args.report.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        print(f"\nReport saved to {args.report}")

    return 0


if __name__ == "__main__": # 入口
    try:
        sys.exit(main(sys.argv[1:])) # 执行主函数
    except Exception as exc:  # pragma: no cover - CLI convenience # 捕获异常并打印错误信息
        print(f"[ERROR] {exc}", file=sys.stderr) # 打印错误信息
        sys.exit(1) # 退出

