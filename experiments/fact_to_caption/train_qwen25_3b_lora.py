#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_qwen25_3b_lora.py
=============================================================================
Fine-tunes Qwen2.5-3B-Instruct with LoRA on verified SFT dataset to learn:
  1. Epistemic Certainty Grounding: condition narration assertiveness on c = mu_F * mu_L.
  2. Hallucination Suppression: omit unverified / contradictory candidate facts.
  3. Direct Visual Assertion: assert scene graph visual observations without hedging.

Configuration:
  - Base Model: Qwen2.5-3B-Instruct (bfloat16)
  - LoRA: r=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  - Epochs: 3, Effective Batch Size: 8 (batch 2 x grad_accum 4), LR: 2e-4
  - Saves best checkpoint based on validation loss.
=============================================================================
"""

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("TrainQwen3BLoRA")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_VERIFIED_SFT = REPO_ROOT / "experiments" / "fact_to_caption" / "sft_dataset_verified_train.json"
CHECKPOINT_DIR = REPO_ROOT / "experiments" / "checkpoints_caption" / "qwen25_3b_certainty_lora"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
if not MODEL_PATH.exists():
    MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_INSTRUCTION = """You are an expert cultural heritage narrator for Vietnamese landmarks.
Narrate the video scene based on visual evidence from the scene graph and verified cultural knowledge facts.
Directly assert visual scene graph observations. Express confirmed knowledge facts with appropriate certainty (c = mu_F * mu_L), and strictly ignore unverified or dubious claims."""


class SFTDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items = []
        
        for s in samples:
            lm = s["landmark_name"]
            vis = ", ".join(s.get("scene_elements", []))
            
            facts_lines = []
            for f in s.get("scene_graph_facts", []):
                facts_lines.append(f"- [VISUAL OBSERVATION] {f['fact_text']}")
            for f in s.get("top5_knowledge_facts", []):
                c = f.get("certainty_c", 0.5)
                policy = f.get("policy_label", "ASSERT")
                if policy == "OMIT":
                    facts_lines.append(f"- [KNOWLEDGE FACT - OMIT] {f['fact_text']} (Certainty: {c:.2f})")
                elif policy == "HEDGE":
                    facts_lines.append(f"- [KNOWLEDGE FACT - HEDGE] {f['fact_text']} (Certainty: {c:.2f})")
                else:
                    facts_lines.append(f"- [KNOWLEDGE FACT - ASSERT] {f['fact_text']} (Certainty: {c:.2f})")
                    
            facts_str = "\n".join(facts_lines)
            
            user_msg = f"[LANDMARK SCENE]\nSite: {lm}\nObserved Visual Elements: {vis}\n\n[CANDIDATE FACTS & OBSERVATIONS]\n{facts_str}\n\nNarrate the scene accurately."
            asst_msg = s["target_caption"]
            
            messages = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": asst_msg}
            ]
            
            prompt_only_msgs = [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_msg}
            ]
            
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(prompt_only_msgs, tokenize=False, add_generation_prompt=True)
            
            full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
            prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length)
            
            input_ids = full_enc["input_ids"]
            prompt_len = len(prompt_enc["input_ids"])
            
            # Label masking: set prompt tokens to -100 so loss is only calculated on assistant output
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[:max_length]
            
            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    
    for item in batch:
        cur_len = len(item["input_ids"])
        pad_len = max_len - cur_len
        input_ids.append(torch.cat([item["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(cur_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))
        
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("TRAINING QWEN2.5-3B CERTAINTY-CONTROLLED LORA MODEL")
    LOGGER.info("=" * 70)

    if not IN_VERIFIED_SFT.exists():
        LOGGER.error(f"Verified SFT dataset not found: {IN_VERIFIED_SFT}")
        sys.exit(1)

    with open(IN_VERIFIED_SFT, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data = data["train"]
    val_data = data.get("val", [])
    LOGGER.info(f"Loaded {len(train_data):,} Train and {len(val_data):,} Val samples.")

    # Load Tokenizer & Base Model
    LOGGER.info(f"Loading tokenizer & model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Create DataLoaders
    train_dataset = SFTDataset(train_data, tokenizer)
    val_dataset = SFTDataset(val_data, tokenizer) if val_data else train_dataset

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # Optimizer & Scheduler
    epochs = 3
    lr = 2e-4
    grad_accum_steps = 4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    LOGGER.info(f"Training for {epochs} epochs | Total optimization steps: {total_steps}")

    best_val_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        step_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")

            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            step_loss += loss.item() * grad_accum_steps
            train_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % (grad_accum_steps * 5) == 0:
                cur_step = (step + 1) // grad_accum_steps
                LOGGER.info(f"Epoch {epoch}/{epochs} | Step {cur_step} | Loss: {step_loss / (grad_accum_steps * 5):.4f}")
                step_loss = 0.0

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to("cuda")
                labels = batch["labels"].to("cuda")
                attention_mask = batch["attention_mask"].to("cuda")
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else avg_train_loss
        LOGGER.info(f"Epoch {epoch:02d} Completed | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            LOGGER.info(f"New best validation loss: {best_val_loss:.4f}! Saving LoRA adapter checkpoint...")
            model.save_pretrained(CHECKPOINT_DIR)
            tokenizer.save_pretrained(CHECKPOINT_DIR)

    elapsed = time.time() - t0
    LOGGER.info("=" * 70)
    LOGGER.info(f"FINE-TUNING COMPLETE in {elapsed:.1f}s ({elapsed/60.0:.2f} mins)!")
    LOGGER.info(f"Best Validation Loss: {best_val_loss:.4f}")
    LOGGER.info(f"Model saved to: {CHECKPOINT_DIR}")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
