#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_nli_inference.py
=============================================================================
High-Throughput Cross-Encoder NLI Inference on RTX 5090 GPU.

Evaluates (Premise, Hypothesis) pairs with DeBERTa-v3-large:
  MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

Protocol:
  1. Touch /tmp/release_vram to release VRAM from memory-holding background task.
  2. Load model in bfloat16 on CUDA.
  3. Batch inference (batch_size=64, max_length=256).
  4. Save nli_predictions.json.
  5. In finally block: clean up CUDA memory and remove /tmp/release_vram.
=============================================================================
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("BatchNLIInference")

VRAM_RELEASE_FLAG = Path("/tmp/release_vram")
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


class NLIPairDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        item = self.tasks[idx]
        return item["task_id"], item["premise"], item["hypothesis"]


def collate_fn_builder(tokenizer, max_length=256):
    def collate_fn(batch):
        task_ids = [b[0] for b in batch]
        premises = [b[1] for b in batch]
        hypotheses = [b[2] for b in batch]
        enc = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return task_ids, enc
    return collate_fn


def run_inference(tasks_path: str, output_path: str, batch_size: int = 64):
    LOGGER.info(f"Setting VRAM release flag at {VRAM_RELEASE_FLAG}...")
    VRAM_RELEASE_FLAG.touch()
    time.sleep(2)  # Wait for holder process to yield memory

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    try:
        LOGGER.info(f"Loading {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Check dtype support
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        LOGGER.info(f"Using dtype: {dtype}")

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype
        ).to(device)
        model.eval()

        id2label = {idx: label.lower() for idx, label in model.config.id2label.items()}
        LOGGER.info(f"Model label mapping: {id2label}")
        e_idx = [k for k, v in id2label.items() if "entail" in v][0]
        c_idx = [k for k, v in id2label.items() if "contra" in v][0]
        n_idx = [k for k, v in id2label.items() if "neut" in v][0]
        LOGGER.info(f"Indices: Entailment={e_idx}, Contradiction={c_idx}, Neutral={n_idx}")

        LOGGER.info(f"Loading tasks from {tasks_path}...")
        with open(tasks_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        LOGGER.info(f"Total tasks to score: {len(tasks):,}")

        dataset = NLIPairDataset(tasks)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn_builder(tokenizer)
        )

        predictions = {}
        total_batches = len(loader)
        start_t = time.time()
        last_log_t = start_t

        with torch.no_grad():
            for b_idx, (task_ids, enc) in enumerate(loader):
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                token_type_ids = enc.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    out = model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.softmax(out.logits.float(), dim=-1)

                pe_list = probs[:, e_idx].cpu().tolist()
                pc_list = probs[:, c_idx].cpu().tolist()
                pn_list = probs[:, n_idx].cpu().tolist()

                for tid, pe, pc, pn in zip(task_ids, pe_list, pc_list, pn_list):
                    predictions[tid] = {
                        "p_E": round(float(pe), 4),
                        "p_C": round(float(pc), 4),
                        "p_N": round(float(pn), 4),
                    }

                curr_t = time.time()
                if curr_t - last_log_t >= 10.0 or b_idx == total_batches - 1:
                    done_count = (b_idx + 1) * batch_size
                    elapsed = curr_t - start_t
                    speed = (b_idx + 1) * batch_size / elapsed
                    pct = (b_idx + 1) / total_batches * 100.0
                    LOGGER.info(f"Progress: [{b_idx+1}/{total_batches}] ({pct:.1f}%) | Speed: {speed:.1f} pairs/s | Elapsed: {elapsed:.1f}s")
                    last_log_t = curr_t

        total_elapsed = time.time() - start_t
        LOGGER.info(f"Completed inference on {len(predictions):,} pairs in {total_elapsed:.1f}s ({len(predictions)/total_elapsed:.1f} pairs/s).")

        LOGGER.info(f"Writing predictions to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f)
        LOGGER.info("Predictions saved successfully.")

    finally:
        LOGGER.info("Cleaning up GPU memory and restoring VRAM holding...")
        if "model" in locals():
            del model
        torch.cuda.empty_cache()
        if VRAM_RELEASE_FLAG.exists():
            VRAM_RELEASE_FLAG.unlink()
            LOGGER.info(f"Removed {VRAM_RELEASE_FLAG}. VRAM holding resumed.")


if __name__ == "__main__":
    t_path = sys.argv[1] if len(sys.argv) > 1 else "nli_inference_tasks.json"
    o_path = sys.argv[2] if len(sys.argv) > 2 else "nli_predictions.json"
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    run_inference(t_path, o_path, batch_size=bs)
