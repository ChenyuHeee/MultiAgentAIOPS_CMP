# MultiAgentAIOPS_CMP

本仓库用于对 **任意“数据格式相同”的数据集** 跑三种 RCA 方法，并在统一输入/输出约定下进行评测与汇总。

这里的“格式相同”指：
- 推理输入为 AIOpsChallenge 风格的 `input.json`（包含每个 case/uuid 的时间窗、目标服务/组件等必要字段）
- 输出为 AIOpsChallengeJudge 可评分的 `submission.jsonl`
- （可选）若需要评分：提供 `ground_truth.jsonl`（仅用于 judge，禁止用于推理）

仓库里提供的 `aiops2021/` 只是一个**示例数据集目录**与脚本集合：你可以把其他数据集放到任意目录，并把命令里的路径替换掉即可。

包含方法：
- **CausalRCA**（基于统计/因果的 RCA）
- **mABC**（多智能体 LLM RCA）
- **OpenRCA**（heuristic / RCA-agent；其中 RCA-agent 为 controller + executor 多轮交互、带代码执行的原论文/原仓库式流程）

评测：使用 [AIOpsChallengeJudge](AIOpsChallengeJudge/README.md) 对提交结果进行打分，并将“all-only”的纯数字摘要写入 [测评结果.txt](测评结果.txt)。

## 目录结构（关键部分）

- [aiops2021/](aiops2021/)
	- 示例目录：用于演示“同格式数据集”的放置方式与一键脚本（你可以替换为你自己的数据集目录）
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
- `.env`、`.venv`、日志、parquet/ckpt 等大文件
- `aiops2021/` 示例数据集下的 inputs/outputs/原始日目录/ground-truth 等

对你自己的数据集，建议也放在**不提交**的目录（例如你自己建一个 `local_data/<dataset_name>/`），或者在 `.gitignore` 里按同样规则添加忽略项。

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

## 数据准备（适用于任意同格式数据集）

### 推荐的数据目录约定

你可以为任意数据集建立一个目录（下面用 `<DATASET_ROOT>` 表示），并保证至少包含：

```
<DATASET_ROOT>/
	inputs/
		all_input_clean.json        # 推理输入（GT-free）
	outputs/                      # 推理输出 + judge 报告
	ground_truth.jsonl            # 可选：仅用于评分
```

如果你的数据也采用 “按天分目录 + parquet” 的形态（类似 AIOps2021），可以额外放置：

```
<DATASET_ROOT>/
	2021-xx-xx/
		log-parquet/
		metric-parquet/
		trace-parquet/
```

### 输入文件要求（关键）

各方法适配脚本使用的核心是 `--input-json` / `--phase-input` 指向的 **clean 输入**。
只要你的 `*_clean.json` 字段与本仓库的适配脚本兼容（即“数据格式相同”），就可以在不改代码的情况下替换为任意数据集。

如果你需要从原始数据生成 input.json：
- 可以参考 [aiops2021/build_aiops21_inputs.py](aiops2021/build_aiops21_inputs.py) 的思路，写一个你自己的 `build_<dataset>_inputs.py`

> 注意：本仓库不会把 inputs/outputs/raw 数据提交到 GitHub。

## 运行与评分（通用模板）

下面给出**通用命令模板**。你只需要把路径替换成你的 `<DATASET_ROOT>`。

建议先在 shell 里设几个变量，避免命令太长：

```bash
DATASET_ROOT=/abs/path/to/your_dataset
INPUT_JSON=$DATASET_ROOT/inputs/all_input_clean.json
OUT_DIR=$DATASET_ROOT/outputs
mkdir -p "$OUT_DIR"
```

### 1) CausalRCA

```bash
# 例：可分片运行（num-shards=4）。inputs-dir 为 CausalRCA 的中间产物目录（你可自定义命名）。
CAUSAL_INPUTS_DIR=CausalRCA/data_aiops/<your_dataset_tag>

.venv/bin/python CausalRCA/AIOpsChallenge_Adapt/run_causalrca_and_build_submission.py \
  --phase-input "$INPUT_JSON" \
  --inputs-dir "$CAUSAL_INPUTS_DIR" \
  --out-jsonl "$OUT_DIR/submission_causalrca.part0.jsonl" \
  --epochs 200 --graph-threshold 0.3 --per-uuid-timeout-seconds 300 \
  --shard 0 --num-shards 4
```

完成后你需要自行把 `submission_causalrca.part*.jsonl` 合并成一个 submission，再调用 judge 评分。

### 2) mABC（LLM，多 provider）

mABC 支持分片并行，并会生成 submission jsonl。

```bash
.venv/bin/python mABC/AIOpsChallenge_Adapt/run_mabc_and_build_submission.py \
  --mabc-root mABC \
  --input-json "$INPUT_JSON" \
  --out-jsonl "$OUT_DIR/submission_mabc.part0.jsonl" \
  --shard-index 0 --shard-count 4 \
  --per-uuid-timeout-seconds 900 --capture-output
```

合并 part 文件后再评分（你也可以参考 [aiops2021/merge_and_score_mabc_all_llm.py](aiops2021/merge_and_score_mabc_all_llm.py) 的实现）。

### 3) OpenRCA（heuristic 与 RCA-agent）

#### heuristic

```bash
# OpenRCA heuristic 依赖 CausalRCA 产出的 pkls。
PKLS_DIR=$CAUSAL_INPUTS_DIR/pkls

.venv/bin/python OpenRCA/AIOpsChallenge_Adapt/run_openrca_heuristic_and_build_submission.py \
  --phase phase1 \
  --pkls-dir "$PKLS_DIR" \
  --input-json "$INPUT_JSON" \
  --out-jsonl "$OUT_DIR/submission_openrca_heuristic.jsonl"
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

然后运行 RCA-agent。

如果你要复用本仓库的“一键对比脚本”（qwen/deepseek），它目前位于示例目录：
- [aiops2021/run_openrca_agent_all_llm_qwen_vs_deepseek.py](aiops2021/run_openrca_agent_all_llm_qwen_vs_deepseek.py)

对其他数据集：
- 推荐直接调用 [OpenRCA/AIOpsChallenge_Adapt/run_openrca_agent_and_build_submission.py](OpenRCA/AIOpsChallenge_Adapt/run_openrca_agent_and_build_submission.py)
- 或者复制/改造 `aiops2021/` 下的一键脚本，把路径替换为你的 `<DATASET_ROOT>`

## 评分与结果汇总（通用）

### 评分（AIOpsChallengeJudge）

judge 只依赖 ground truth 与 submission（二者必须 uuid 对齐），示例：

```bash
.venv/bin/python AIOpsChallengeJudge/evaluate.py \
	--ground-truth "$DATASET_ROOT/ground_truth.jsonl" \
	--submission "$OUT_DIR/submission_openrca_heuristic.jsonl" \
	--reason-threshold 0.65
```

### 结果汇总（纯数字）

本仓库现有的摘要脚本是面向 `aiops2021/outputs/report_*.json` 的：
- [aiops2021/update_results_txt_minimal.py](aiops2021/update_results_txt_minimal.py)

如果你希望对“任意数据集目录”也一键汇总，做法有两种：
- 方案 A（零改代码）：把你数据集的 report 文件复制/软链到 `aiops2021/outputs/` 下的同名位置，再运行 `update_results_txt_minimal.py`
- 方案 B（更通用）：轻改 `update_results_txt_minimal.py`，让它支持 `--out-dir <DATASET_ROOT>/outputs`（如你需要我可以顺手改）

## 注意事项

- 本仓库不包含任何数据集的原始数据与 ground-truth；你需要自行准备。
- `.env`、原始数据、模型输出、parquet 等均已被忽略；如你本地需要这些文件，请自行保留在工作区（不会影响推送）。
- 子目录中的项目来自对应上游仓库（详见各自 README 与 LICENSE）；本仓库主要提供统一适配与跑分流水线。