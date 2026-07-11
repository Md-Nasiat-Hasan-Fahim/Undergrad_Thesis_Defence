#!/usr/bin/env python
"""
03_kto_train.py
================
KTO (Kahneman-Tversky Optimization) fine-tuning of ProGen2 with a LoRA adapter (r=8).

Expected input: train.csv with columns:
    meso_protein_sequence    -> undesirable completion  (flag = -1 / label = False)
    thermo_protein_sequence  -> desirable   completion  (flag = +1 / label = True)

Unlike DPO, KTO does NOT need paired data — every sequence (meso AND thermo) becomes its
own independent example with a binary desirability label. Both share the same prompt:
ProGen2's sequence-start context token ("1" by default).

Example:
    python 03_kto_train.py \
        --train_csv train.csv \
        --model_name hugohrban/progen2-small \
        --output_dir ./out-kto \
        --epochs 3 --lr 5e-5 --batch_size 4 --beta 0.1

Notes:
- We pass `peft_config` directly to KTOTrainer with ref_model=None: TRL keeps an internal
  frozen copy of the base model (adapters disabled) as the reference policy.
- If your desirable/undesirable counts are imbalanced, tune --desirable_weight /
  --undesirable_weight per the KTO paper's guidance (n_undesirable/n_desirable, clipped
  to [1, 4/3]) instead of leaving both at 1.0.
- LoRA rank is fixed per the task at r=8 by default (override with --lora_r if needed).
"""
import argparse
import logging
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, set_seed
from peft import LoraConfig, TaskType
from trl import KTOConfig, KTOTrainer

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("kto")


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="KTO fine-tuning of ProGen2 (meso=-1, thermo=+1)")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--meso_col", type=str, default="meso_protein_sequence")
    p.add_argument("--thermo_col", type=str, default="thermo_protein_sequence")
    p.add_argument(
        "--prompt_token",
        type=str,
        default="1",
        help="Shared prompt/context prepended to every example (ProGen2 BOS/context token).",
    )
    p.add_argument("--model_name", type=str, default="hugohrban/progen2-small")
    p.add_argument("--output_dir", type=str, default="./out-kto")
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.1, help="KTO KL-penalty coefficient.")
    p.add_argument("--desirable_weight", type=float, default=1.0)
    p.add_argument("--undesirable_weight", type=float, default=1.0)

    # LoRA / PEFT
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="all-linear")

    # Training hygiene
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_and_split(csv_path, meso_col, thermo_col, prompt_token, val_size, seed):
    logger.info(f"Loading '{csv_path}' ...")
    df = pd.read_csv(csv_path)
    for col in (meso_col, thermo_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}. Found: {list(df.columns)}")

    meso = df[[meso_col]].dropna().rename(columns={meso_col: "completion"})
    meso["label"] = False  # flag = -1 (undesirable)

    thermo = df[[thermo_col]].dropna().rename(columns={thermo_col: "completion"})
    thermo["label"] = True  # flag = +1 (desirable)

    full_df = pd.concat([meso, thermo], ignore_index=True)
    full_df["prompt"] = prompt_token
    full_df = full_df[["prompt", "completion", "label"]]

    n_desirable = int(full_df["label"].sum())
    n_undesirable = int((~full_df["label"]).sum())
    logger.info(f"Built {len(full_df)} KTO examples ({n_desirable} desirable / {n_undesirable} undesirable)")

    train_df, val_df = train_test_split(
        full_df, test_size=val_size, random_state=seed, stratify=full_df["label"]
    )
    logger.info(f"Split -> train: {len(train_df)} | val: {len(val_df)}")

    return (
        Dataset.from_pandas(train_df.reset_index(drop=True)),
        Dataset.from_pandas(val_df.reset_index(drop=True)),
    )


def build_tokenizer(model_name):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Args: {vars(args)}")

    tokenizer = build_tokenizer(args.model_name)
    train_ds, val_ds = load_and_split(
        args.train_csv, args.meso_col, args.thermo_col, args.prompt_token, args.val_size, args.seed
    )

    logger.info(f"Loading policy model '{args.model_name}' ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    target_modules = (
        "all-linear"
        if args.lora_target_modules == "all-linear"
        else [m.strip() for m in args.lora_target_modules.split(",")]
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    kto_config = KTOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        report_to=["none"],
        disable_tqdm=False,
        seed=args.seed,
    )

    trainer = KTOTrainer(
        model=model,
        ref_model=None,  # TRL manages an internal frozen reference from the LoRA base
        args=kto_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    logger.info("Starting KTO training ...")
    train_result = trainer.train()
    logger.info(f"Training complete. Train metrics: {train_result.metrics}")

    logger.info(f"Saving best KTO LoRA adapter + tokenizer to '{args.output_dir}'")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
