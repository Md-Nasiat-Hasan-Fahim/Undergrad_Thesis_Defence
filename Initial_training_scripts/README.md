# ProGen2 fine-tuning: LoRA-SFT, DPO, KTO

Three independent, runnable scripts. Each loads `train.csv`, applies a LoRA adapter
(rank `r=8` by default), fine-tunes ProGen2, early-stops on validation loss, and saves
the best adapter.

| Script | Method | Uses from `train.csv` |
|---|---|---|
| `01_lora_sft_train.py` | Plain causal-LM SFT + LoRA | `thermo_protein_sequence` only |
| `02_dpo_train.py` | DPO + LoRA | `thermo_protein_sequence` (chosen) vs `meso_protein_sequence` (rejected), paired |
| `03_kto_train.py` | KTO + LoRA | `thermo_protein_sequence` (label=True/+1), `meso_protein_sequence` (label=False/-1), unpaired |

## Setup

```bash
pip install -r requirements.txt
```

`train.csv` must contain at least these two columns:

```
meso_protein_sequence,thermo_protein_sequence
MKT...,MKV...
...
```

## Run

```bash
python 01_lora_sft_train.py --train_csv train.csv --output_dir ./out-lora-sft
python 02_dpo_train.py      --train_csv train.csv --output_dir ./out-dpo
python 03_kto_train.py      --train_csv train.csv --output_dir ./out-kto
```

Each script exposes `--help` for the full list of hyperparameters (learning rate, batch
size, epochs, LoRA rank/alpha/dropout/target modules, early-stopping patience, eval
frequency, etc.).

## Shared design choices

- **Model**: `hugohrban/progen2-small` by default (`--model_name` to swap in
  medium/base/large/xlarge ProGen2 checkpoints). Loaded with `trust_remote_code=True`
  since ProGen2 uses a custom architecture/tokenizer.
- **LoRA**: `r=8`, `alpha=16`, `dropout=0.05`, `target_modules="all-linear"` — this
  auto-targets every `nn.Linear` layer so it works regardless of ProGen2's internal
  module names; override with an explicit comma-separated list if you want to target
  only attention or only MLP projections.
- **Prompt for DPO/KTO**: both scripts use ProGen2's sequence-start context token
  (`"1"` by default, `--prompt_token` to change) as a shared prompt, since the task
  gives full sequences rather than conditioning tags.
- **Early stopping**: `EarlyStoppingCallback` on `eval_loss`, `load_best_model_at_end=True`,
  matching `eval_strategy`/`save_strategy="steps"`.
- **Logging**: Python `logging` module for run-level messages (data sizes, model
  loading, final metrics) + Hugging Face `Trainer`'s built-in `tqdm` progress bars and
  step-level loss/eval logging (`logging_steps`).
- **Reproducibility**: `set_seed(42)` (override with `--seed`), fixed `random_state` for
  the train/val split.

## Notes / things to double-check before a real run

- **KTO class balance**: the default script gives `meso` and `thermo` equal
  `desirable_weight`/`undesirable_weight` (1.0/1.0). If your dataset isn't ~50/50, tune
  these per the KTO paper's guidance (ratio ≈ `n_undesirable / n_desirable`, clipped to
  `[1, 4/3]`).
- **`ref_model=None`**: both DPO and KTO trainers are given `peft_config` with no
  explicit `ref_model` — TRL keeps an internal frozen copy of the base model (LoRA
  adapters disabled) as the reference policy. This is the standard/recommended pattern
  for combining LoRA with DPO/KTO and avoids loading two full copies of the model.
- **`processing_class=tokenizer`**: this is the TRL ≥0.12 argument name (older TRL
  versions use `tokenizer=`). If you're pinned to an older TRL, rename the kwarg.
- **Sequence length**: `--max_length` / `--max_prompt_length` are generous defaults;
  tune to your actual protein length distribution to avoid wasted compute/OOM.
