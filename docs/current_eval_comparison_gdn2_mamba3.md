# Current Evaluation Comparison: GatedLinearAttention2 vs GDN2 and Mamba-3

Last updated: 2026-06-21 KST, during the first 10B-token evaluation run.

This document summarizes what is already measured, what is still running, and what can and cannot be claimed from the current numbers.

## Short Answer

The final 10B-token checkpoint does not currently beat Gated DeltaNet-2 on any completed GDN2 Table 2 standard benchmark task.

Against Mamba-3, the answer depends on which Mamba-3 baseline is used:

| Comparison target | Current 10B result |
|---|---|
| Gated DeltaNet-2 recurrent | 0 wins on completed Table 2 tasks |
| Mamba-3 recurrent SISO | 1 win: BoolQ |
| Mamba-3 recurrent MIMO | 1 very small win: BoolQ |
| Mamba-3 hybrid SISO | 0 wins |
| Mamba-3 hybrid MIMO | 0 wins |

The BoolQ win against recurrent Mamba-3 is real numerically, but it is not enough to claim that this model is better overall. The model loses clearly on perplexity, LAMBADA, PIQA, HellaSwag, WinoGrande, ARC, and OpenBookQA.

RULER is still pending. The RULER jobs are running on GPU 0-3 and write JSON only at the end, so no partial score is visible yet.

## What Is Being Compared

Our evaluated model is:

```text
gdn2_kla_1.3B
```

The main checkpoint discussed here is:

```text
runs/outputs/tsz128x4k_10B_gdn2_kla_1.3B_fineweb_edu_10bt/hf_checkpoints/checkpoint-10B
```

Training setup:

| Item | Value |
|---|---|
| Training tokens | 10B |
| Training context length | 4K |
| Data source | FineWeb-Edu sample/100BT |
| Architecture family | recurrent-only gated linear attention |
| Hybrid attention | not used |
| Tokenizer | TinyLlama/TinyLlama_v1.1 |

The paper baselines are copied in:

```text
eval/gdn2_paper_baselines.json
```

Current local result files are under:

```text
runs/eval/gdn2_paper/
```

The main generated result summary is:

```text
runs/eval/gdn2_paper/GDN2_PAPER_EVAL_RESULTS.md
```

## Completed Table 2 Standard Results

These are the currently completed standard language modeling and commonsense results. Social IQA is excluded from our local average because the installed `datasets` version blocks the legacy dataset script used by the `lm_eval` task.

| Checkpoint | Avg 8 acc | Wiki PPL | LAMBADA PPL | LAMBADA acc | PIQA | HellaSwag norm | WinoGrande | ARC-Easy | ARC-Challenge | OpenBookQA | BoolQ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01B | 39.88 | 45.14 | 117.08 | 20.67 | 62.24 | 31.96 | 51.70 | 50.29 | 21.84 | 18.20 | 62.11 |
| 02B | 42.00 | 35.86 | 66.48 | 26.02 | 64.25 | 35.80 | 50.12 | 53.37 | 23.29 | 21.40 | 61.71 |
| 03B | 42.60 | 33.12 | 55.95 | 27.69 | 64.64 | 37.11 | 51.14 | 55.13 | 24.06 | 21.00 | 60.00 |
| 04B | 43.22 | 31.51 | 47.36 | 30.10 | 63.93 | 37.81 | 53.51 | 55.13 | 24.23 | 20.80 | 60.21 |
| 05B | n/a | 29.96 | 46.01 | 28.76 | 65.56 | 39.67 | 52.49 | n/a | n/a | n/a | 60.95 |
| 10B | 45.20 | 24.52 | 29.12 | 34.76 | 67.03 | 41.46 | 51.70 | 60.14 | 26.11 | 22.60 | 57.80 |
| GDN2 recurrent | 53.11 | 15.90 | 11.41 | 48.09 | 72.80 | 56.84 | 57.85 | 72.43 | 38.23 | 31.60 | 59.54 |

`05B` was still being parsed when this snapshot was written, so its average should be regenerated from the JSON before publication. The raw file is:

```text
runs/eval/gdn2_paper/05B/standard_lm_eval.json
```

## Final 10B vs GDN2 Recurrent

For perplexity, lower is better. For all other listed metrics, higher is better.

| Metric | Ours 10B | GDN2 recurrent | Result |
|---|---:|---:|---|
| Wiki PPL | 24.52 | 15.90 | worse |
| LAMBADA PPL | 29.12 | 11.41 | worse |
| LAMBADA acc | 34.76 | 48.09 | worse |
| PIQA | 67.03 | 72.80 | worse |
| HellaSwag norm | 41.46 | 56.84 | worse |
| WinoGrande | 51.70 | 57.85 | worse |
| ARC-Easy | 60.14 | 72.43 | worse |
| ARC-Challenge | 26.11 | 38.23 | worse |
| OpenBookQA | 22.60 | 31.60 | worse |
| BoolQ | 57.80 | 59.54 | worse |

Current conclusion:

```text
Final 10B checkpoint vs GDN2 recurrent on completed Table 2 tasks: 0 wins.
```

## Final 10B vs Mamba-3

### Against Mamba-3 Recurrent SISO

| Metric | Ours 10B | Mamba-3 SISO recurrent | Result |
|---|---:|---:|---|
| Wiki PPL | 24.52 | 16.30 | worse |
| LAMBADA PPL | 29.12 | 12.99 | worse |
| LAMBADA acc | 34.76 | 45.06 | worse |
| PIQA | 67.03 | 72.31 | worse |
| HellaSwag norm | 41.46 | 55.58 | worse |
| WinoGrande | 51.70 | 56.20 | worse |
| ARC-Easy | 60.14 | 70.45 | worse |
| ARC-Challenge | 26.11 | 34.56 | worse |
| OpenBookQA | 22.60 | 31.00 | worse |
| BoolQ | 57.80 | 55.90 | better by 1.90 |

Result:

```text
1 win, 9 losses.
```

### Against Mamba-3 Recurrent MIMO

| Metric | Ours 10B | Mamba-3 MIMO recurrent | Result |
|---|---:|---:|---|
| Wiki PPL | 24.52 | 16.45 | worse |
| LAMBADA PPL | 29.12 | 11.66 | worse |
| LAMBADA acc | 34.76 | 47.82 | worse |
| PIQA | 67.03 | 72.36 | worse |
| HellaSwag norm | 41.46 | 56.49 | worse |
| WinoGrande | 51.70 | 55.78 | worse |
| ARC-Easy | 60.14 | 72.38 | worse |
| ARC-Challenge | 26.11 | 38.07 | worse |
| OpenBookQA | 22.60 | 30.00 | worse |
| BoolQ | 57.80 | 57.74 | better by 0.06 |

Result:

```text
1 tiny win, 9 losses.
```

### Against Mamba-3 Hybrid

| Target | Result |
|---|---|
| Mamba-3 SISO hybrid | 0 wins, 10 losses |
| Mamba-3 MIMO hybrid | 0 wins, 10 losses |

The closest metric is BoolQ:

| Metric | Ours 10B | Mamba-3 SISO hybrid | Mamba-3 MIMO hybrid |
|---|---:|---:|---:|
| BoolQ | 57.80 | 57.86 | 57.98 |

So even BoolQ does not beat the hybrid Mamba-3 numbers.

## Partial Real-World Retrieval Results

Completed 10B real-world retrieval splits:

| Task | Ours 10B | GDN2 recurrent | Result |
|---|---:|---:|---|
| FDA contains | 5.35 | 19.98 | worse |
| TriviaQA exact match | 1.71 | 61.37 | worse |
| NQ exact match | 1.02 | 19.64 | worse |
| DROP EM | 0.13 | 17.87 | worse |
| DROP F1 | 2.89 | 17.87 | worse if compared directly |

Still running:

```text
swde,squad_completion
```

The partial real-world numbers are very poor. This usually means the model is not yet instruction or QA robust, the evaluation format may be harsh for this checkpoint, or both. It should not be used to claim long-context strength.

## RULER Status

RULER is the most important pending result for the long-context question.

Running jobs:

| GPU | RULER task | Lengths |
|---:|---|---|
| 0 | niah_single_1 | 1K, 2K, 4K, 8K |
| 1 | niah_single_2 | 1K, 2K, 4K, 8K |
| 2 | niah_single_3 | 1K, 2K, 4K, 8K |
| 3 | niah_multikey_1 | 1K, 2K, 4K, 8K |

Expected output files:

```text
runs/eval/gdn2_paper/10B/splits/ruler_niah_single_1.json
runs/eval/gdn2_paper/10B/splits/ruler_niah_single_2.json
runs/eval/gdn2_paper/10B/splits/ruler_niah_single_3.json
runs/eval/gdn2_paper/10B/splits/ruler_niah_multikey_1.json
```

Important detail:

```text
The current RULER script writes the JSON only after a task finishes.
```

That means the absence of a RULER JSON file does not mean the process is dead. It means there is no completed task file yet. Progress must be checked with `nvidia-smi`, process runtime, and log tails.

## RULER Intermediate Save Fix

The slow 10B RULER jobs were launched before intermediate saving was added, so they still behave in the old way: no result JSON appears until the process finishes.

The script has now been changed for future runs:

```text
scripts/ruler_eval_gla2.py
```

New behavior:

| Moment | JSON behavior |
|---|---|
| Process starts | writes an initial `status: running` file |
| Task starts | records `current_task` |
| Each length finishes | writes the score for that length immediately |
| All tasks finish | writes `status: complete` |

The JSON is written atomically through a temporary file and replace operation, so partially written files should not be observed by readers.

This means future RULER runs can be monitored directly from the output JSON. Example:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("runs/eval/gdn2_paper/01B/ruler_table3.json")
data = json.loads(path.read_text())
print(data["status"], data["updated_at_kst"])
print(data["results"])
PY
```

The already-running 10B RULER jobs were not restarted because that would throw away more than one hour of GPU work.

## What Can Be Claimed Now

Safe claims:

- The 10B checkpoint improves over earlier 1B-5B checkpoints on many standard metrics.
- The final 10B checkpoint is still far below GDN2 100B on standard Table 2 tasks.
- The final 10B checkpoint beats recurrent Mamba-3 only on BoolQ, and only barely against Mamba-3 MIMO.
- The final 10B checkpoint does not beat hybrid Mamba-3 on completed Table 2 tasks.
- Real-world retrieval partial results are weak so far.
- RULER is still the open question for long-context retrieval behavior.

Unsafe claims:

- Do not claim this 10B checkpoint beats GDN2 overall.
- Do not claim it beats Mamba-3 overall.
- Do not claim long-context superiority until RULER finishes.
- Do not use intermediate BoolQ spikes as an overall architecture win.

## What To Do Next

1. Wait for RULER split JSON files.
2. Merge RULER splits into `runs/eval/gdn2_paper/10B/ruler_table3.json`.
3. Regenerate `runs/eval/gdn2_paper/GDN2_PAPER_EVAL_RESULTS.md`.
4. Finish SWDE/SQuAD real-world retrieval.
5. Complete 06B-10B learning-curve standard evaluations.
6. Re-check whether any metric beats GDN2, Mamba-3 recurrent, or Mamba-3 hybrid.

Until RULER finishes, the honest summary is:

```text
The model is learning, but the final 10B checkpoint has not yet shown a broad benchmark win.
```
