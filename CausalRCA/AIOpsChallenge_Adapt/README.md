# CausalRCA × AIOpsChallenge 适配层

该目录用于把本仓库的 CausalRCA 方法接入当前工作区的 AIOps Challenge 数据与官方 Judge。

## 设计目标

- 尽量不改变数据“质量/语义”：优先复用 `cmp/mABC` 已构建的 `endpoint_stats.json`（它来自 trace-parquet 的 minute 聚合，包含 calls/avg_duration/error_rate/timeout_rate）。
- 最小必要格式转换：仅把「按 endpoint/分钟」的统计转换成「按 uuid/分钟」的多变量时间序列 DataFrame（保存为 `.pkl`），供 CausalRCA 训练/推断使用。
- 输出严格符合官方 Judge submission JSONL schema。

## 文件说明

- `build_causalrca_inputs_from_mabc.py`
  - 输入：`cmp/mABC/data/metric/endpoint_stats.json` + phase 的 `input.json`（uuid + anomaly window）。
  - 输出：每个 uuid 一个 `.pkl`（pandas DataFrame），以及 `uuid_to_pkl.json`。

- `causalrca_core.py`
  - 将原仓库训练脚本里“学习邻接矩阵 + PageRank 排名”的核心逻辑抽成可调用函数。

- `run_causalrca_and_build_submission.py`
  - 按 uuid 读取 `.pkl`，调用 `causalrca_core.py` 得到 root cause 排名，构建 submission JSONL。
  - 支持 shard、append/resume、心跳、每 uuid 超时（避免单样本挂死）。

## 环境注意

- 原仓库 `requirements.txt` 固定了 `torch==1.10.2`，这在 macOS/ARM 或较新 Python 上可能无法直接安装。
- 适配层代码保持“尽量少改原仓库”，但你可能需要用 conda/venv 配一个兼容环境（例如 Python 3.9/3.10 + 对应 torch 版本）。

## 快速使用（示例）

1) 生成 phase1 输入：

```bash
python cmp/CausalRCA/AIOpsChallenge_Adapt/build_causalrca_inputs_from_mabc.py \
  --phase phase1 \
  --phase-input data/phaseone-main/input.json \
  --mabc-endpoint-stats cmp/mABC/data/metric/endpoint_stats.json \
  --out-dir cmp/CausalRCA/data_aiops/phase1
```

2) 跑推断并生成 submission：

```bash
python cmp/CausalRCA/AIOpsChallenge_Adapt/run_causalrca_and_build_submission.py \
  --phase-input data/phaseone-main/input.json \
  --inputs-dir cmp/CausalRCA/data_aiops/phase1 \
  --out-jsonl outputs/submissions/causalrca_phase1.jsonl
```

3) 评分：

```bash
python AIOpsChallengeJudge/evaluate.py \
  --ground-truth AIOpsChallengeJudge/ground_truth_phase1.jsonl \
  --submission outputs/submissions/causalrca_phase1.jsonl \
  --report outputs/reports/causalrca_phase1.json

说明：submission 生成阶段不会读取任何 Ground Truth（GT）文件；GT 仅用于最后的 Judge 评分。
```
