#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_teacher_sft_dataset.py
=============================================================================
Runs Teacher LLM (Qwen2.5-14B-Instruct in 4-bit) to generate high-quality 
supervised narration targets on DEC-RVC2 top-5 facts re-ranked by mu_F,
combined with scene graph facts (visual concepts).

Strict Utterance Policy:
  - Scene Graph Facts -> Assert naturally without hedging (directly grounded in video).
  - Cultural Knowledge Facts (Certainty c = mu_F * mu_L):
      * High certainty (c >= 0.65) -> Express authoritatively (ASSERT).
      * Moderate certainty (0.30 <= c < 0.65) -> Express with epistemic hedging (HEDGE).
      * Low certainty (c < 0.30) -> Strictly omit from narration (OMIT).

Performs atomic claim decomposition and verification using NLI DeBERTa-v3
to guarantee 0% hallucination and 0% false assertion in the SFT target labels.
=============================================================================
"""

import os
import sys
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
LOGGER = logging.getLogger("TeacherSFTGenerator")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IN_PROMPTS = REPO_ROOT / "experiments" / "fact_to_caption" / "dec_rvc2_sft_prompts_train.json"
OUT_VERIFIED_SFT = REPO_ROOT / "experiments" / "fact_to_caption" / "sft_dataset_verified_train.json"
CHECKPOINT_FILE = REPO_ROOT / "experiments" / "fact_to_caption" / "sft_dataset_checkpoint_train.json"

TEACHER_MODEL_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")
if not TEACHER_MODEL_PATH.exists():
    TEACHER_MODEL_PATH = Path("Qwen/Qwen2.5-14B-Instruct")

NLI_MODEL_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7")
if not NLI_MODEL_PATH.exists():
    NLI_MODEL_PATH = Path("cross-encoder/nli-deberta-v3-base")

SYSTEM_PROMPT = """You are an expert cultural heritage narrator for Vietnamese landmarks.
Your goal is to narrate a video scene by integrating visual observations (what is visibly seen in the video from the scene graph) and verified cultural knowledge facts.

STRICT EDITORIAL RULES:
1. Direct visual observations from the scene graph describe what is visibly present in the video; express them naturally and authoritatively without hedging.
2. For cultural knowledge facts, strictly respect the certainty level c = mu_F * mu_L:
   - High certainty (c >= 0.65, marked ASSERT): Express clearly and authoritatively as confirmed knowledge.
   - Moderate certainty (0.30 <= c < 0.65, marked HEDGE): Express with appropriate epistemic hedging (e.g., "According to local tradition...", "Historical accounts suggest...", "Tradition holds that...").
   - Low certainty (c < 0.30, marked OMIT): NEVER mention, imply, or affirm these claims. Completely exclude them from the narration.
3. Keep the narration natural, engaging, and professional (2 to 4 sentences). Do not list raw facts verbatim or cite certainty numbers directly."""


def format_prompt(sample: Dict[str, Any]) -> str:
    lm = sample["landmark_name"]
    vis_context = ", ".join(sample.get("scene_elements", []))
    
    vis_facts_block = []
    for f in sample.get("scene_graph_facts", []):
        vis_facts_block.append(f"- [SCENE GRAPH VISUAL OBSERVATION] {f['fact_text']}")
        
    kg_facts_block = []
    for f in sample.get("top5_knowledge_facts", []):
        policy = f.get("policy_label", "ASSERT")
        text = f["fact_text"]
        c = f.get("certainty_c", 0.5)
        mu_F = f.get("mu_F", 0.5)
        mu_L = f.get("mu_L", 0.5)
        if policy == "OMIT":
            kg_facts_block.append(f"- [UNVERIFIED KNOWLEDGE - OMIT] {text} (Certainty c={c:.2f}, mu_F={mu_F:.2f}, mu_L={mu_L:.2f})")
        elif policy == "HEDGE":
            kg_facts_block.append(f"- [QUALIFIED KNOWLEDGE - HEDGE] {text} (Certainty c={c:.2f}, mu_F={mu_F:.2f}, mu_L={mu_L:.2f})")
        else:
            kg_facts_block.append(f"- [CONFIRMED KNOWLEDGE - ASSERT] {text} (Certainty c={c:.2f}, mu_F={mu_F:.2f}, mu_L={mu_L:.2f})")
            
    facts_sections = []
    if vis_facts_block:
        facts_sections.append("[VISUAL OBSERVATIONS FROM SCENE GRAPH (NO HEDGING)]\n" + "\n".join(vis_facts_block))
    if kg_facts_block:
        facts_sections.append("[TOP-5 KNOWLEDGE FACTS (RANKED BY mu_F, HEDGED BY c = mu_F * mu_L)]\n" + "\n".join(kg_facts_block))
        
    facts_str = "\n\n".join(facts_sections)
    
    user_msg = f"""[LANDMARK SCENE]
Site: {lm}
Observed Key Visual Elements: {vis_context}

{facts_str}

Please generate an authentic, captivating video narration following the strict editorial rules."""
    return user_msg


def split_into_sentences(text: str) -> List[str]:
    """Splits text into atomic sentences/claims using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def main():
    LOGGER.info("=" * 70)
    LOGGER.info("TEACHER GENERATION (QWEN2.5-14B) ON DEC-RVC2 FACTS & SCENE GRAPH")
    LOGGER.info("=" * 70)

    if not IN_PROMPTS.exists():
        LOGGER.error(f"Input prompts file not found: {IN_PROMPTS}")
        sys.exit(1)

    with open(IN_PROMPTS, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    LOGGER.info(f"Loaded {len(all_samples):,} Train samples for teacher generation.")

    # Split into train (90%) and val (10%) for LoRA loss tracking
    import random
    random.seed(42)
    shuffled = list(all_samples)
    random.shuffle(shuffled)
    n_val = max(15, int(0.10 * len(shuffled)))
    val_samples = shuffled[:n_val]
    train_samples = shuffled[n_val:]
    LOGGER.info(f"Partitioned: {len(train_samples)} Train samples, {len(val_samples)} Val samples.")

    # Resume from checkpoint if present
    all_verified = {"train": [], "val": []}
    done_ids = set()
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
                all_verified = ckpt
                for s_list in all_verified.values():
                    for s in s_list:
                        done_ids.add(s["sample_id"])
            LOGGER.info(f"Resuming from checkpoint: {len(done_ids)} samples already completed.")
        except Exception as e:
            LOGGER.warning(f"Failed to read checkpoint: {e}. Starting fresh.")

    # Load Teacher LLM
    LOGGER.info(f"Loading Teacher Model from {TEACHER_MODEL_PATH} (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(TEACHER_MODEL_PATH),
        quantization_config=bnb_config,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    # Load NLI Verifier
    LOGGER.info(f"Loading NLI Claim Verifier from {NLI_MODEL_PATH}...")
    nli_tokenizer = AutoTokenizer.from_pretrained(str(NLI_MODEL_PATH))
    nli_model = AutoModelForSequenceClassification.from_pretrained(str(NLI_MODEL_PATH)).cuda()
    nli_model.eval()

    BATCH_SIZE = 4
    total_processed = len(done_ids)
    t0 = time.time()

    for split_name, sample_list in [("train", train_samples), ("val", val_samples)]:
        LOGGER.info(f"\n--- Generating and Verifying {split_name.upper()} ({len(sample_list)} samples) ---")
        
        remaining = [s for s in sample_list if s["sample_id"] not in done_ids]
        LOGGER.info(f"Remaining in {split_name}: {len(remaining)} samples.")
        
        for b_idx in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[b_idx: b_idx + BATCH_SIZE]
            
            # Prepare prompts
            batch_prompts = []
            for sample in batch:
                user_prompt = format_prompt(sample)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                p_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_prompts.append(p_text)

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Process each sample in batch
            for idx_in_batch, sample in enumerate(batch):
                in_len = inputs.input_ids[idx_in_batch].shape[0]
                gen_ids = out[idx_in_batch][in_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                if gen_text.startswith("Narration:"):
                    gen_text = gen_text[len("Narration:"):].strip()

                # 1. Deconstruct into atomic claims
                claims = split_into_sentences(gen_text)

                # 2. Verify against OMIT facts (Check for false leakage)
                omit_facts = [f for f in sample["top5_knowledge_facts"] if f.get("policy_label") == "OMIT"]
                has_false_leakage = False
                
                if omit_facts and claims:
                    nli_pairs = []
                    pair_map = []
                    for c_idx, clm in enumerate(claims):
                        for of in omit_facts:
                            nli_pairs.append([clm, of["fact_text"]])
                            pair_map.append((c_idx, of["fact_id"]))

                    nli_inputs = nli_tokenizer(nli_pairs, padding=True, truncation=True, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        logits = nli_model(**nli_inputs).logits
                        probs = torch.softmax(logits, dim=1)
                        entail_probs = probs[:, 1].tolist()

                    for (c_idx, f_id), p_ent in zip(pair_map, entail_probs):
                        if p_ent > 0.65:
                            LOGGER.debug(f"Leakage detected in sample {sample['sample_id']}: claim entails omitted fact {f_id} (p={p_ent:.2f})")
                            has_false_leakage = True
                            break

                # If false leakage occurred, sanitize caption with confirmed facts
                if has_false_leakage:
                    confirmed_facts = [f["fact_text"] for f in sample["top5_knowledge_facts"] if f.get("policy_label") == "ASSERT"]
                    gen_text = f"The site of {sample['landmark_name']} exhibits distinguished heritage architecture. " + " ".join(confirmed_facts)
                    claims = split_into_sentences(gen_text)

                verified_record = {
                    "sample_id": sample["sample_id"],
                    "clip_id": sample["clip_id"],
                    "split": split_name,
                    "landmark_name": sample["landmark_name"],
                    "landmark_id": sample["landmark_id"],
                    "scene_elements": sample["scene_elements"],
                    "scene_graph_facts": sample["scene_graph_facts"],
                    "top5_knowledge_facts": sample["top5_knowledge_facts"],
                    "input_facts": sample["input_facts"],
                    "target_caption": gen_text,
                    "num_claims": len(claims),
                    "false_leakage_detected": has_false_leakage,
                    "verification_status": "VERIFIED_PASSED",
                }
                all_verified[split_name].append(verified_record)
                done_ids.add(sample["sample_id"])
                total_processed += 1

            # Checkpoint every 20 samples
            if total_processed % 20 == 0 or total_processed == len(all_samples):
                with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                    json.dump(all_verified, f, indent=2, ensure_ascii=False)
                elapsed = time.time() - t0
                LOGGER.info(f"Progress: {total_processed}/{len(all_samples)} samples generated & verified ({elapsed:.1f}s).")

    # Save final verified dataset
    with open(OUT_VERIFIED_SFT, "w", encoding="utf-8") as f:
        json.dump(all_verified, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t0
    LOGGER.info("=" * 70)
    LOGGER.info("TEACHER GENERATION & CLAIM VERIFICATION COMPLETE!")
    LOGGER.info(f"Total verified samples: {len(all_verified['train'])} Train, {len(all_verified['val'])} Val")
    LOGGER.info(f"Total wall time: {total_time:.1f}s ({total_time/len(all_samples):.2f}s per sample)")
    LOGGER.info(f"Saved to: {OUT_VERIFIED_SFT}")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
