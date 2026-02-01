# AIOPS 评测命令速查

以下命令假设你已经在项目根目录 `/Users/hechenyu/projects/AIOPS` 下，并且激活了虚拟环境：

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 1. 语义评测程序 (`evaluate.py`)

```bash
# Phase 1
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/mtr/judge/evaluate.py \
  -g /Users/hechenyu/projects/AIOPS/mtr/judge/ground_truth_phase1.jsonl \
  -s /Users/hechenyu/projects/AIOPS/mtr/12.1/results_list_phase1.json

# Phase 2
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/mtr/judge/evaluate.py \
  -g /Users/hechenyu/projects/AIOPS/mtr/judge/ground_truth_phase2.jsonl \
  -s /Users/hechenyu/projects/AIOPS/mtr/12.1/results_list_phase2.json

# Phase 1 + 2 (需要先合并 submission)
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/mtr/judge/evaluate.py \
  -g /Users/hechenyu/projects/AIOPS/mtr/judge/ground_truth_phase12.jsonl \
  -s /Users/hechenyu/projects/AIOPS/mtr/12.1/combined_results.jsonl
```

> 如需临时合并 submission，可使用：
>
> ```bash
> cat mtr/12.1/results_list_phase1.json mtr/12.1/results_list_phase2.json > mtr/12.1/combined_results.jsonl
> ```

## 2. SequenceMatcher 备份评测程序 (`backups/evaluate_seqmatcher_backup.py`)

命令与语义评测完全一致，只需替换脚本路径：

```bash
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/mtr/judge/backups/evaluate_seqmatcher_backup.py \
  -g /Users/hechenyu/projects/AIOPS/mtr/judge/ground_truth_phase1.jsonl \
  -s /Users/hechenyu/projects/AIOPS/mtr/12.1/results_list_phase1.json
```

> 若你已切换到 `mtr/` 目录，也可以使用相对路径版本：
>
> ```bash
> python judge/evaluate.py -g judge/ground_truth_phase1.jsonl -s 12.1/results_list_phase1.json
> python judge/backups/evaluate_seqmatcher_backup.py -g judge/ground_truth_phase1.jsonl -s 12.1/results_list_phase1.json
> ```

## 3. 官方评测程序 (`AIOpsChallenge2025Eval-main/server/eval.py`)

官方评测需要 label/answer.json。我们已经在以下目录准备好：

- `phase1_local/` 对应 `results_list_phase1.json`
- `phase2_local/` 对应 `results_list_phase2.json`
- `combined/`  对应合并后的 submission

执行命令（保持单行，记得替换 `phaseX_local` 为实际目录）：

```bash
CONFIG_PATH="/Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/config/config.yaml" \
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/eval.py \
  -d /Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/playground/mtr_phase_eval/phase1_local \
  -l label.json -a answer.json -r result.json
```

Phase 2 与合并数据的运行示例：

```bash
# Phase 2
d="phase2_local"
CONFIG_PATH="/Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/config/config.yaml" \
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/eval.py \
  -d /Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/playground/mtr_phase_eval/${d} \
  -l label.json -a answer.json -r result.json

# Phase 1 + 2（combined）
d="combined"
CONFIG_PATH="/Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/config/config.yaml" \
/Users/hechenyu/projects/AIOPS/.venv/bin/python \
  /Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/eval.py \
  -d /Users/hechenyu/projects/AIOPS/AIOpsChallenge2025Eval-main/server/playground/mtr_phase_eval/${d} \
  -l label.json -a answer.json -r result.json
```

运行后，`result.json` 会出现在对应目录（如 `phase1_local/result.json`）。

---
如需调整阈值或输出报告，可参考 `README_package.md` 里的额外参数说明。
