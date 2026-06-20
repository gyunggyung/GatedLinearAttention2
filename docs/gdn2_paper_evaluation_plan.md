# GDN-2 Paper Evaluation Plan

This document fixes the benchmark list and execution plan for evaluating
GatedLinearAttention2 checkpoints against the Gated DeltaNet-2 paper.

Source:

- Paper: `Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention`
- arXiv: `2605.22791v1`
- Official repo: `https://github.com/NVlabs/GatedDeltaNet-2`
- Local baseline table: `eval/gdn2_paper_baselines.json`

## What Will Be Measured

### Table 2: Language Modeling And Commonsense

Run with `lm-eval-harness` through `scripts/lm_eval_gla2.py`.

- WikiText perplexity: `wikitext`
- LAMBADA perplexity and accuracy: `lambada_openai`
- PIQA: `piqa`
- HellaSwag normalized accuracy: `hellaswag`
- WinoGrande: `winogrande`
- ARC-Easy: `arc_easy`
- ARC-Challenge: `arc_challenge`
- OpenBookQA: `openbookqa`
- Social IQA: `social_iqa`
- BoolQ: `boolq`

The paper reports Wiki/LAMBADA perplexity and accuracy values. Accuracy values
are percentages. The paper's recurrent GDN-2 average accuracy target is `53.11`.

### Table 3: RULER NIAH

Run with direct RULER generation in `scripts/ruler_eval_gla2.py` so that the
paper's 1K/2K/4K/8K lengths are measured explicitly.

- `niah_single_1`: 1K, 2K, 4K, 8K
- `niah_single_2`: 1K, 2K, 4K, 8K
- `niah_single_3`: 1K, 2K, 4K
- `niah_multikey_1`: 1K, 2K, 4K

The installed lm-eval RULER task is also available, but its yaml metric list is
not guaranteed to expose the paper's 1K/2K metrics. The direct evaluator uses
the same NVIDIA RULER data-generation functions and computes string-match
accuracy by length.

### Table 4: Real-world Retrieval

Run with `lm-eval-harness` tasks:

- SWDE: `swde`
- SQuAD completion: `squad_completion`
- FDA: `fda`
- TriviaQA: `triviaqa`
- Natural Questions open: `nq_open`
- DROP: `drop`

The installed tasks use the `hazyresearch/based-*` datasets for SWDE, FDA, and
SQuAD completion, matching the recall-heavy completion style used by recurrent
model papers. Inputs are evaluated with `max_length=2048` to match the paper's
2K truncation.

## Checkpoints

The runner evaluates every available milestone checkpoint under:

```text
runs/outputs/tsz128x4k_10B_gdn2_kla_1.3B_fineweb_edu_10bt/hf_checkpoints/checkpoint-??B/model-ckpt.pth
```

Expected checkpoints are `01B` through `10B`. This lets us measure whether
performance improves monotonically with trained tokens or whether some tasks
plateau/regress.

## Parallel Execution

The scheduler is:

```bash
python scripts/run_gdn2_paper_eval.py --gpus 0,1,2,3,4,5,6,7 --require_10b
```

It builds one queue containing:

- one Table 2 job per checkpoint;
- one Table 3 RULER job per checkpoint;
- one Table 4 real-world retrieval job per checkpoint.

Each running job gets one GPU through `CUDA_VISIBLE_DEVICES=<gpu>`. When a GPU
finishes its current job, the scheduler immediately launches the next pending
job on that GPU.

## Automatic Start After Training

The watcher is:

```bash
setsid bash scripts/watch_and_run_gdn2_eval.sh >> runs/eval/gdn2_paper/watch_eval.log 2>&1 < /dev/null &
```

It waits for the current training PID, then launches the full evaluation only
after `checkpoint-10B` exists.

## Outputs

- Raw job outputs:
  `runs/eval/gdn2_paper/<checkpoint>/`
- Logs:
  `runs/eval/gdn2_paper/logs/`
- Job manifest:
  `runs/eval/gdn2_paper/job_manifest.json`
- Final markdown summary:
  `runs/eval/gdn2_paper/GDN2_PAPER_EVAL_RESULTS.md`

The summary script compares our values with the GDN-2 paper baseline JSON. It
does not claim a win unless the metric is actually measured for the same task
and checkpoint.

## Current Readiness

- `lm_eval` is installed.
- Dataset prefetch script exists: `scripts/prefetch_gdn2_eval_data.py`.
- GatedLinearAttention2 custom lm-eval adapter exists:
  `scripts/lm_eval_gla2.py`.
- Direct RULER Table 3 evaluator exists:
  `scripts/ruler_eval_gla2.py`.
- 8-GPU queue scheduler exists:
  `scripts/run_gdn2_paper_eval.py`.
- Training-completion watcher exists:
  `scripts/watch_and_run_gdn2_eval.sh`.

