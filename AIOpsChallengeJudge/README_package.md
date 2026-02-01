# AIOPS Local Evaluation Toolkit

This folder contains the scripts I use to evaluate local AIOps Challenge submissions. It includes a semantic evaluator, a lightweight SequenceMatcher fallback, utilities for converting ground-truth files, and the phase ground-truth datasets bundled for convenience.

## Directory Layout

- `evaluate.py` – semantic evaluator using sentence-transformers
- `backups/evaluate_seqmatcher_backup.py` – fallback string-matching evaluator with identical CLI
- `convert_ground_truth.py` – helper to convert official ground-truth JSONL downloads into the evaluator schema
- `ground_truth/` – bundled ground-truth splits (`phase1`, `phase2`, `phase12`)
- `requirements.txt` – Python dependencies needed for the semantic evaluator

## Environment Setup

1. Install Python 3.9+.
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

If Hugging Face download access is restricted, set `AIOPS_SENTENCE_MODEL` to a local model path or run the fallback evaluator in `backups/` (no external downloads required).

## Usage

Both evaluators share the same CLI. Replace the ground-truth file and submission path with your own.

```bash
# Semantic evaluator (default)
python evaluate.py -g ground_truth/ground_truth_phase1.jsonl -s path/to/submission.jsonl

# SequenceMatcher fallback
python backups/evaluate_seqmatcher_backup.py -g ground_truth/ground_truth_phase1.jsonl -s path/to/submission.jsonl
```

Optional flags:
- `--reason-threshold 0.6` – adjust semantic similarity threshold
- `--show-details` – print per-sample component/reason hits
- `--report report.json` – save aggregate and per-sample details to JSON

## Preparing New Ground Truth Files

Use `convert_ground_truth.py` to convert official releases into the evaluator schema:

```bash
python convert_ground_truth.py phase1_raw.jsonl phase2_raw.jsonl -o ground_truth/ground_truth_phase12.jsonl
```

The script accepts one or more input files and writes a JSONL file with normalized fields (`component`, `reason`, `reason_keywords`, `evidence_points`).

## Contact

Let me know if you need a sample submission or additional automation scripts.
