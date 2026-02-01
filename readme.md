# MultiAgentAIOPS_CMP

本仓库用于在 **AIOps 2021** 数据上，对三种 RCA 方法进行统一输入输出适配、严格的 **GT-free 推理**（推理阶段不读取/不泄露 ground truth），并使用官方风格的评测脚本进行打分。

包含方法：
- **CausalRCA**（基于统计/因果的 RCA）
- **mABC**（多智能体 LLM RCA）
- **OpenRCA**（heuristic / RCA-agent；其中 RCA-agent 为 controller + executor 多轮交互、带代码执行的原论文/原仓库式流程）

评测：使用 [AIOpsChallengeJudge](AIOpsChallengeJudge/README.md) 对提交结果进行打分，并将“all-only”的纯数字摘要写入 [测评结果.txt](测评结果.txt)。

## 目录结构（关键部分）

- [aiops2021/](aiops2021/)
	- 放置 AIOps2021 原始数据（本仓库 **不提交** 原始数据），并提供生成输入、跑分、汇总脚本
- [CausalRCA/](CausalRCA/)
	- 适配脚本在 [CausalRCA/AIOpsChallenge_Adapt/](CausalRCA/AIOpsChallenge_Adapt/)
- [mABC/](mABC/)
	- 适配脚本在 [mABC/AIOpsChallenge_Adapt/](mABC/AIOpsChallenge_Adapt/)
- [OpenRCA/](OpenRCA/)
	- 适配脚本在 [OpenRCA/AIOpsChallenge_Adapt/](OpenRCA/AIOpsChallenge_Adapt/)
- [AIOpsChallengeJudge/](AIOpsChallengeJudge/)
	- 评分脚本与说明

## 重要约束（GT-free 推理）

- **推理阶段禁止读取 ground-truth**。
- 推理输入统一使用 `*_clean.json`（已移除可能泄露标签/答案的信息）。
- ground-truth 仅允许用于：
	- 评测打分（judge）
	- 可选的离线分析/对齐（不参与推理输入）

仓库通过 [.gitignore](.gitignore) 强制忽略：
- `aiops2021/2021-*` 原始日目录（parquet）
- `aiops2021/inputs/`、`aiops2021/outputs/`
- `aiops2021/aiops21_groundtruth.csv`、`aiops2021/ground_truth.jsonl`
- 以及 `.env`、`.venv` 等

## 环境准备

建议使用 Python 3.11：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

按需安装依赖（按你要跑的方法选择）：

```bash
pip install -r AIOpsChallengeJudge/requirements.txt
pip install -r CausalRCA/requirements.txt
pip install -r mABC/requirements.txt
pip install -r OpenRCA/requirements.txt
```

## 数据准备（不随仓库提交）

你需要自行把 AIOps2021 原始数据放到 `aiops2021/` 下的对应日期目录，例如：

```
aiops2021/
	2021-03-04/
		log-parquet/
		metric-parquet/
		trace-parquet/
	2021-03-05/
	...
```

然后生成统一输入（包含 clean 版本）：

```bash
.venv/bin/python aiops2021/build_aiops21_inputs.py
```

生成结果通常会在 `aiops2021/inputs/` 下（该目录被 git 忽略）。

## 运行与评分（示例：all）

下面以 `all_input_clean.json` 为推理输入示例。注意：`aiops2021/outputs/` 默认也被 git 忽略。

### 1) CausalRCA

```bash
# 例：可分片运行（num-shards=4）
.venv/bin/python CausalRCA/AIOpsChallenge_Adapt/run_causalrca_and_build_submission.py \
	--phase-input aiops2021/inputs/all_input_clean.json \
	--inputs-dir CausalRCA/data_aiops/aiops2021_all_allstats_clean \
	--out-jsonl aiops2021/outputs/submission_causalrca_all_allstats_clean.part0.jsonl \
	--epochs 200 --graph-threshold 0.3 --per-uuid-timeout-seconds 300 \
	--shard 0 --num-shards 4
```

完成后合并并评分（具体合并脚本以你当前流水线为准）。

### 2) mABC（LLM，多 provider）

mABC 支持分片并行，并会生成 submission jsonl。

```bash
.venv/bin/python mABC/AIOpsChallenge_Adapt/run_mabc_and_build_submission.py \
	--mabc-root mABC \
	--input-json aiops2021/inputs/all_input_clean.json \
	--out-jsonl aiops2021/outputs/submission_mabc_all_llm.part0.jsonl \
	--shard-index 0 --shard-count 4 \
	--per-uuid-timeout-seconds 900 --capture-output
```

然后合并与评分：

```bash
.venv/bin/python aiops2021/merge_and_score_mabc_all_llm.py --cleanup-parts
```

### 3) OpenRCA（heuristic 与 RCA-agent）

#### heuristic

```bash
.venv/bin/python OpenRCA/AIOpsChallenge_Adapt/run_openrca_heuristic_and_build_submission.py \
	--phase phase1 \
	--pkls-dir CausalRCA/data_aiops/aiops2021_all_allstats_clean/pkls \
	--input-json aiops2021/inputs/all_input_clean.json \
	--out-jsonl aiops2021/outputs/submission_openrca_all_allstats_clean.jsonl
```

#### RCA-agent（controller + executor 多轮）

RCA-agent 需要可用的 OpenAI-compatible 接口配置与 API Key（**不要提交到 git**）。

典型做法：
- 在本机环境变量里注入 key（或放在本地 `.env` 并自行加载）
- 指定 OpenRCA 的路由配置，例如：

```bash
export OPENRCA_API_CONFIG_PATH=OpenRCA/rca/api_config_deepseek.yaml
export API_KEY=YOUR_KEY
```

然后运行本仓库提供的对比脚本（qwen/deepseek 等）：

```bash
.venv/bin/python aiops2021/run_openrca_agent_all_llm_qwen_vs_deepseek.py
```

## 评分与结果汇总

评分产物通常在 `aiops2021/outputs/report_*.json`（zh/en）。

将纯数字摘要写入 [测评结果.txt](测评结果.txt)：

```bash
.venv/bin/python aiops2021/update_results_txt_minimal.py
```

## 注意事项

- 本仓库不包含 AIOps2021 原始数据与 ground-truth；你需要自行准备。
- `.env`、原始数据、模型输出、parquet 等均已被忽略；如你本地需要这些文件，请自行保留在工作区（不会影响推送）。
- 子目录中的项目来自对应上游仓库（详见各自 README 与 LICENSE）；本仓库主要提供统一适配与跑分流水线。