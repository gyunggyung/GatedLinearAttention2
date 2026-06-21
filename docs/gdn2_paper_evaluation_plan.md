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
- Social IQA: `social_iqa` in the paper list, currently marked `n/a` in the
  accelerated local run because `datasets==4.8.5` rejects the legacy
  `allenai/social_i_qa` dataset script used by the installed lm-eval task.
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

Expected checkpoints are `01B` through `10B`. The `10B` checkpoint is evaluated
first because it is the main model to compare with the GDN-2 paper baselines.
After the full `10B` Table 2/3/4 pass finishes, the runner evaluates `01B`,
`02B`, ..., `09B` to measure whether performance improves monotonically with
trained tokens or whether some tasks plateau/regress.

## Parallel Execution

The scheduler is:

```bash
python scripts/run_gdn2_paper_eval.py --gpus 0,1,2,3,4,5,6,7 --require_10b
```

It builds phased queues:

1. `10B_first`: one Table 2 job, one Table 3 RULER job, and one Table 4
   real-world retrieval job for `checkpoint-10B`.
2. `learning_curve`: the same three jobs for the earlier checkpoints.

Each running job gets one GPU through `CUDA_VISIBLE_DEVICES=<gpu>`. During the
`10B_first` phase, only the three `10B` jobs are launched, so the remaining GPUs
can sit idle briefly. This is intentional: the first useful answer is whether
the final 10B-token model is competitive with GDN-2 and the paper's other
baselines. When all `10B` jobs finish, the markdown summary is regenerated.
Only then does the runner start the `01B` through `09B` learning-curve jobs. In
the learning-curve phase, any free GPU immediately takes the next pending job.

The current accelerated 10B run splits the final checkpoint across all 8 GPUs:

```bash
bash scripts/run_10b_eval_accelerated.sh
```

RULER Table 3 is split across GPUs 0-3. Table 2 without Social IQA and Table 4
retrieval are split across GPUs 4-7 with:

```bash
bash scripts/run_10b_lm_eval_retries.sh
```

The retry script uses the patched lm-eval adapter, writes split JSON files under
`runs/eval/gdn2_paper/10B/splits/`, then merges whatever completed split files
exist into `standard_lm_eval.json`, `real_world_lm_eval.json`, and
`ruler_table3.json`. The markdown summary is regenerated from those merged JSON
files, so partial results appear as soon as valid JSON exists.

## Automatic Start After Training

The watcher is:

```bash
setsid bash scripts/watch_and_run_gdn2_eval.sh >> runs/eval/gdn2_paper/watch_eval.log 2>&1 < /dev/null &
```

It waits for the current training PID, then launches the evaluation only after
`checkpoint-10B` exists. The watcher passes `--primary_checkpoint 10B` and
`--summarize_after_phase`, so `GDN2_PAPER_EVAL_RESULTS.md` is written once after
the 10B-only phase and then refreshed again after the full learning-curve phase.

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
- Accelerated 10B split runner exists:
  `scripts/run_10b_eval_accelerated.sh`.
- Patched 10B lm-eval retry runner exists:
  `scripts/run_10b_lm_eval_retries.sh`.
