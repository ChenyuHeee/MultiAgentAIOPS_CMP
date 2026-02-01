# AIOpsChallenge → mABC 数据适配器

把 AIOps Challenge 2025 的 parquet 数据（`/data/<day>/*-parquet`）预处理成 mABC 可读的三件套：

- `cmp/mABC/data/metric/endpoint_stats.json`
- `cmp/mABC/data/topology/endpoint_maps.json`
- `cmp/mABC/data/label/label.json`

同时会额外输出一个对评测很关键的映射文件：

- `cmp/mABC/data/label/label_uuid_map.json`（每个 `uuid` 对应的告警分钟与告警入口 endpoint）

## 设计取舍（重要）

mABC 原仓库把“endpoint”当作 **API/方法级** 调用端点；AIOps Challenge 的 trace 是 Jaeger/OTel span（`process.serviceName` + `operationName` + `tags` + `references`）。为了尽量提高准确度，本适配器默认使用更细粒度的 endpoint（优先从 span tags 提取 HTTP 路由，其次 gRPC/RPC 方法，最后回退到 `operationName`）：

- 默认（推荐）：`--endpoint-mode http`
  - HTTP：`<service>-<http.route|http.target|url.path>`
  - gRPC/RPC：`<service>-<rpc.service>/<rpc.method>`
  - 兜底：`<service>-<operationName>`

仍支持兼容/对照用的粗粒度模式：

- `--endpoint-mode service_operation`：`<service>-<operationName>`
- `--endpoint-mode service`：`<service>`（最粗粒度）

## 运行

在仓库根目录（`/Users/hechenyu/projects/AIOPS`）执行：

```bash
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  -m cmp.mABC.AIOpsChallenge_Adapt.build_mabc_inputs \
  --aiops-data-root /Users/hechenyu/projects/AIOPS/data \
  --day 2025-06-06 --day 2025-06-07 --day 2025-06-08 \
  --input-json /Users/hechenyu/projects/AIOPS/data/phaseone-main/input.json \
  --out-root /Users/hechenyu/projects/AIOPS/cmp/mABC/data \
  --endpoint-mode http
```

生成完成后，mABC 侧可直接运行：

```bash
cd /Users/hechenyu/projects/AIOPS/cmp/mABC
python main/main.py
```

## 生成 submission + 官方指标评测（综合分/组件准确率/原因准确率）

如果你的论文对比表需要官方评测器口径（推荐），需要把 mABC 的输出转成评测器要求的 submission JSONL。

### 1) 配置 mABC 的 LLM（必须）

mABC 默认通过 OpenAI SDK 调用模型。请在运行前设置环境变量（不要把 key 写进代码）：

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"   # 或你可用的模型名
# 如使用本地/代理网关：
# export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
```

### 2) 生成 mABC 输入（三件套 + uuid 映射）

以 Phase1 为例（需要覆盖 `input.json` 涉及的日期；Phase1 大约是 2025-06-06~2025-06-14，且你的 /data 里缺少 2025-06-05）：

```bash
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  -m cmp.mABC.AIOpsChallenge_Adapt.build_mabc_inputs \
  --aiops-data-root /Users/hechenyu/projects/AIOPS/data \
  --day 2025-06-06 --day 2025-06-07 --day 2025-06-08 --day 2025-06-09 --day 2025-06-10 \
  --day 2025-06-11 --day 2025-06-12 --day 2025-06-13 --day 2025-06-14 \
  --input-json /Users/hechenyu/projects/AIOPS/data/phaseone-main/input.json \
  --out-root /Users/hechenyu/projects/AIOPS/cmp/mABC/data \
  --endpoint-mode http
```

会生成：
- `cmp/mABC/data/metric/endpoint_stats.json`
- `cmp/mABC/data/topology/endpoint_maps.json`
- `cmp/mABC/data/label/label.json`
- `cmp/mABC/data/label/label_uuid_map.json`

### 3) 逐 uuid 跑 mABC 并生成 submission JSONL

```bash
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  -m cmp.mABC.AIOpsChallenge_Adapt.run_mabc_and_build_submission \
  --mabc-root /Users/hechenyu/projects/AIOPS/cmp/mABC \
  --input-json /Users/hechenyu/projects/AIOPS/data/phaseone-main/input.json \
  --out-jsonl /Users/hechenyu/projects/AIOPS/outputs/submissions/mabc_phase1.jsonl \
  --save-raw /Users/hechenyu/projects/AIOPS/outputs/reports/mabc_phase1_raw.jsonl
```

### 4) 用 AIOpsChallengeJudge 计算官方口径指标

```bash
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/AIOpsChallengeJudge/evaluate.py \
  -g /Users/hechenyu/projects/AIOPS/AIOpsChallengeJudge/ground_truth_phase1.jsonl \
  -s /Users/hechenyu/projects/AIOPS/outputs/submissions/mabc_phase1.jsonl \
  --show-details
```

## 输出说明

- `endpoint_stats.json`：由 trace 聚合得到（calls、average_duration、error_rate、timeout_rate）。
  - `duration` 以微秒为主，本适配器会换算为毫秒再输出到 `average_duration`。
  - `error/timeout` 由 span tags（如 `http.status_code/grpc.status_code/status.code/error/timeout`）做启发式推断。
- `endpoint_maps.json`：按分钟聚合的 endpoint 调用依赖图。
- `label.json`：从 `input.json` 的异常时间窗口（UTC）生成告警入口。
  - 会在“异常窗口内、且 trace 覆盖到”的分钟中，挑选“最异常”的一分钟，并选该分钟下分数最高的 endpoint 作为告警入口（同时考虑时延/错误/超时）。
  - 再从该分钟的拓扑图生成少量下游路径样例。
- 仅会为“你本次处理的 trace 覆盖到的分钟”生成 label，避免产生大量空路径。

- `label_uuid_map.json`：为每个 `uuid` 记录本次选择的 `{minute, alert_endpoint}`。
  - 原始 mABC `label.json` 结构是按分钟聚合的，无法区分“同一分钟内的多个 uuid”；这个文件用来在后续跑批时做到“按 uuid 输出一条预测”，从而能生成官方评测所需的 submission JSONL。
