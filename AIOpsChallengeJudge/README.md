# AIOpsChallengeJudge

Standalone copy of the AIOPS judge scripts (`mtr/judge`) for scoring submissions.

## Usage
- Convert ground truth (if needed): `python convert_ground_truth.py --output ground_truth.jsonl <phase_files...>`
- Evaluate: `python evaluate.py --ground-truth ground_truth.jsonl --submission submissions.jsonl`

## Notes
- Component matching uses exact tokens; path strings (`source->destination`) remain as in the original judge.
