#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Qwen2.5-3B-Instruct with LoRA on 1,000 Balanced Real Subgraph Captions
==============================================================================
Kiến trúc LoRA tối giản và hiệu quả nhất:
- Target modules: ["q_proj", "v_proj"] (Attention Projection)
- Rank r = 8, Alpha = 16, Dropout = 0.05
- Trainable parameters: 1,843,200 (chỉ 0.0597% toàn bộ mô hình 3B)
- Loss masking: Chỉ tính Cross-Entropy loss trên phần caption sinh ra của Assistant.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("TrainQwen3BLoRA")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
UNIFIED_COCO_DIR = DATA_DIR / "unified_coco_dataset"
TRAIN_DATA_PATH = UNIFIED_COCO_DIR / "balanced_1000_captions_qwen25_14b.json"
DEFAULT_MODEL_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
OUTPUT_CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints" / "qwen25_3b_lora_balanced1000"

SYSTEM_PROMPT = """You are a tourism narrator for Vietnamese heritage sites. You describe
what a camera is currently showing, one caption at a time, paced for dense video narration.

STRICT RULES:
1. Use ONLY the information given in the OBSERVED and KNOWLEDGE blocks.
   Never add facts from your own memory, even if you are confident.
2. Never state a name, date, number, material or person that does not
   appear in the blocks below.
3. Obey the CERTAINTY level exactly. It controls which names you may use.
4. Strictly obey the required sentence count and word count for the video segment duration so narration aligns with video playback.
5. Natural spoken narration style, no bullet points, no meta commentary.
6. Do not mention confidence scores, the knowledge graph, or this prompt."""


class CaptionDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, max_length: int = 1024):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        user_prompt = item["user_prompt"]
        target_caption = item["generated_caption"]

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": target_caption}
        ]
        full_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_enc = self.tokenizer(prompt_text, add_special_tokens=False)
        full_enc = self.tokenizer(full_text, add_special_tokens=False)

        prompt_len = len(prompt_enc["input_ids"])
        input_ids = full_enc["input_ids"][:self.max_length]
        attention_mask = [1] * len(input_ids)

        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-3B-Instruct with LoRA")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per forward pass")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    LOGGER.info("=" * 80)
    LOGGER.info("FINE-TUNING QWEN2.5-3B-INSTRUCT WITH LoRA (ATTENTION PROJECTIONS)")
    LOGGER.info("=" * 80)

    model_path = str(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.exists() else "Qwen/Qwen2.5-3B-Instruct"
    LOGGER.info("Loading tokenizer and base model from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device if torch.cuda.is_available() else None,
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # Simple & Minimal LoRA setup: only q_proj and v_proj
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset & DataLoader
    dataset = CaptionDataset(TRAIN_DATA_PATH, tokenizer, max_length=768)
    LOGGER.info("Loaded training dataset: %d samples from %s", len(dataset), TRAIN_DATA_PATH)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    total_steps = (len(loader) // args.grad_accum) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    LOGGER.info("Training plan: %d epochs | Effective batch size %d | %d optimizer steps | %d warmup steps",
                args.epochs, args.batch_size * args.grad_accum, total_steps, warmup_steps)

    model.train()
    step_count = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()
            batch_count += 1

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 15 == 0 or step_count == total_steps:
                    LOGGER.info("Epoch %d/%d | Step %d/%d | Loss: %.4f | LR: %.2e | Elapsed: %.1fs",
                                epoch, args.epochs, step_count, total_steps,
                                outputs.loss.item(), scheduler.get_last_lr()[0], time.time() - t0)

        avg_loss = epoch_loss / max(1, batch_count)
        LOGGER.info("=== Epoch %d Completed | Avg Training Loss: %.4f ===", epoch, avg_loss)

    train_time = time.time() - t0
    LOGGER.info("=" * 80)
    LOGGER.info("LoRA training completed in %.2f seconds (%.1f minutes).", train_time, train_time / 60)

    OUTPUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_CHECKPOINT_DIR))
    LOGGER.info("Saved LoRA adapter checkpoint to: %s", OUTPUT_CHECKPOINT_DIR)


if __name__ == "__main__":
    main()
