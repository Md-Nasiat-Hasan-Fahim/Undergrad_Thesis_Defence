#!/usr/bin/env python
"""
01_lora_sft_train.py
=====================
LoRA / PEFT supervised fine-tuning (SFT) of ProGen2 using a plain causal-LM
(next-token-prediction) objective on the "good" (thermostable) sequences only.

Expected input: train.csv with a column `thermo_protein_sequence`.

Example:
    python 01_lora_sft_train.py \
        --train_csv train.csv \
        --model_name hugohrban/progen2-small \
        --output_dir ./out-lora-sft \
        --epochs 5 --lr 1e-4 --batch_size 4

Notes:
- LoRA rank is fixed per the task at r=8 by default (override with --lora_r if needed).
- `--lora_target_modules all-linear` (the default) auto-targets every nn.Linear layer,
  which is architecture-agnostic and works regardless of ProGen2's internal module
  naming. Pass a comma-separated list (e.g. "qkv_proj,out_proj,fc_in,fc_out") to target
  specific modules instead.
"""
import argparse
import logging
import os
import sys

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("lora_sft")


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT of ProGen2 on thermostable sequences")
    p.add_argument("--train_csv", type=str, default="train.csv")
    p.add_argument("--seq_col", type=str, default="thermo_protein_sequence")
    p.add_argument(
        "--model_name",
        type=str,
        default="hugohrban/progen2-small",
        help="Any ProGen2 checkpoint on the HF Hub (small/medium/base/large/xlarge).",
    )
    p.add_argument("--output_dir", type=str, default="./out-lora-sft")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)

    # LoRA / PEFT
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="all-linear",
        help='Comma-separated module names, or "all-linear" to auto-target every nn.Linear.',
    )

    # Training hygiene
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_and_split(csv_path, seq_col, val_size, seed):
    logger.info(f"Loading '{csv_path}' ...")
    df = pd.read_csv(csv_path)
    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found in {csv_path}. Found: {list(df.columns)}")

    df = df[[seq_col]].dropna().drop_duplicates().reset_index(drop=True)
    logger.info(f"Loaded {len(df)} unique thermostable ('good') sequences")

    train_df, val_df = train_test_split(df, test_size=val_size, random_state=seed)
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
    train_ds, val_ds = load_and_split(args.train_csv, args.seq_col, args.val_size, args.seed)

    def tokenize_fn(batch):
        return tokenizer(batch[args.seq_col], truncation=True, max_length=args.max_length, padding=False)

    logger.info("Tokenizing datasets ...")
    train_ds = train_ds.map(
        tokenize_fn, batched=True, remove_columns=train_ds.column_names, desc="Tokenizing train"
    )
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names, desc="Tokenizing val")

    lengths = [len(x) for x in tqdm(train_ds["input_ids"], desc="Scanning sequence lengths")]
    logger.info(f"Token length (train) — mean: {sum(lengths) / len(lengths):.1f}, max: {max(lengths)}")

    logger.info(f"Loading base model '{args.model_name}' ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.info("Resizing token embeddings for newly added pad token")
        model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    target_modules = (
        "all-linear"
        if args.lora_target_modules == "all-linear"
        else [m.strip() for m in args.lora_target_modules.split(",")]
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=20,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    logger.info("Starting LoRA SFT training ...")
    train_result = trainer.train()
    logger.info(f"Training complete. Train metrics: {train_result.metrics}")

    logger.info(f"Saving best LoRA adapter + tokenizer to '{args.output_dir}'")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    eval_metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {eval_metrics}")


if __name__ == "__main__":
    main()
