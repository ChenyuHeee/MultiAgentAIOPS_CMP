# OpenRCA × AIOps Challenge 适配层

这套适配层做两件事：

1) **把 AIOps Challenge 的 per-uuid 时序输入（复用 CausalRCA 产出的 `.pkl`）转换成 OpenRCA 期望的 `dataset/{SYSTEM}/telemetry/{DATE}/.../*.csv` 目录结构**，便于后续（可选）用 OpenRCA 的 `RCA-agent` 来做分析。
2) **提供一个不依赖 LLM 的启发式 baseline**，直接从 `.pkl` 生成符合官方 Judge 的 submission JSONL（含 `reasoning_trace`）。

> 设计取舍：为了避免对原始数据做额外清洗/重采样，这里复用你已有的 minute 聚合产物（CausalRCA 的 pkls），并把它“平铺”为 OpenRCA 常见的 `timestamp, cmdb_id, kpi_name, value` 形式。

## 1. 构建 OpenRCA 数据集（CSV）

从 `cmp/CausalRCA/data_aiops/{phase}_pre30/pkls/*.pkl` 生成 OpenRCA 风格 telemetry：

```bash
cd cmp/OpenRCA
/Users/hechenyu/projects/AIOPS/.venv/bin/python AIOpsChallenge_Adapt/build_openrca_dataset_from_causalrca_pkls.py \
	--phase phase1 \
	--pkls-dir ../CausalRCA/data_aiops/phase1_pre30/pkls \
	--input-json ../../data/phaseone-main/input.json \
	--dataset-root ./dataset
```

会生成类似：

- `dataset/AIOpsChallenge/phase1/query.csv`
- `dataset/AIOpsChallenge/phase1/telemetry/{uuid}/metric/metric_service.csv`
- `dataset/AIOpsChallenge/phase1/telemetry/{uuid}/trace/trace_span.csv`（空表占位）
- `dataset/AIOpsChallenge/phase1/telemetry/{uuid}/log/log_service.csv`（空表占位）

## 2. 生成 Judge submission（LLM-free）

```bash
cd cmp/OpenRCA
/Users/hechenyu/projects/AIOPS/.venv/bin/python AIOpsChallenge_Adapt/run_openrca_heuristic_and_build_submission.py \
	--phase phase1 \
	--pkls-dir ../CausalRCA/data_aiops/phase1_pre30/pkls \
	--input-json ../../data/phaseone-main/input.json \
	--out-jsonl ./outputs/submissions/openrca_heuristic_phase1_pre30.jsonl \
	--limit 20

# 可选：如果你明确确认 input.json 的 service 字段不包含标签泄露，可开启弱提示
# 	--use-input-service-hint
```

接着用官方 Judge 评分：

```bash

# 推荐：离线稳定评测（指向本机缓存的 sentence-transformers 模型快照）
SNAP=~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/<your_snapshot_hash>
TRANSFORMERS_OFFLINE=1 AIOPS_SENTENCE_MODEL="$SNAP" \
	/Users/hechenyu/projects/AIOPS/.venv/bin/python ../../AIOpsChallengeJudge/evaluate.py \
	--ground-truth ../../AIOpsChallengeJudge/ground_truth_phase1.jsonl \
	--submission ./outputs/submissions/openrca_heuristic_phase1_pre30.jsonl
```

## 说明

- 本适配器不会读取任何 Ground Truth（GT）文件。
- `component`：仅根据 telemetry（pkls）里的 service 统计与 input.json 的 service 提示（若存在）进行选择。
- `reasoning_trace`：只记录模型/统计得到的信号，不再塞入大量“命中关键词”的提示词。

> Judge schema 提醒：`reasoning_trace` 的每个 step 需要包含 `step`(int) / `action`(str) / `observation`(str)。
