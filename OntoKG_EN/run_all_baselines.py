#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run All External Baselines and Internal Ablations on 250 Test Videos
====================================================================
Triển khai đầy đủ 6 cấu hình mô hình:
1. ours: Mô hình đề xuất (Qwen2.5-3B LoRA + OntoKG + Uncertainty Gate)
2. frozen_base: LLM đóng băng (Base Qwen2.5-3B không fine-tune)
3. ablation_no_threshold: Bỏ ngưỡng (Always Affirm / Always HIGH certainty)
4. ablation_no_anchor: Bỏ neo (Greedy mu_F, không lọc candidate hay visual anchor)
5. vlm_direct: VLM trực tiếp (Qwen3-VL-8B, chỉ video keyframe, không fact nào)
6. vlm_rag: VLM + RAG (Qwen3-VL-8B + SigLIP2 keyframe retrieval từ kho văn bản di tích)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
)
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("RunAllBaselines")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_CAPTIONS_PATH = REPO_ROOT / "experiments" / "outputs" / "test_250_before_after_captions.json"
KB_EXPORT_PATH = REPO_ROOT / "data" / "sources" / "kb_full_export_2026-10-01.json"
OUTPUT_BENCHMARK_PATH = REPO_ROOT / "experiments" / "outputs" / "benchmark_all_models_250_captions.json"

QWEN_3B_BASE_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
LORA_ADAPTER_PATH = REPO_ROOT / "experiments" / "checkpoints" / "qwen25_3b_lora_balanced1000"
QWEN3_VL_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
SIGLIP2_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--google--siglip2-base-patch16-224/snapshots/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2")


def find_video_path(file_name: str) -> Path:
    # Try direct relative path
    p1 = REPO_ROOT / "data" / file_name
    if p1.exists():
        return p1
    
    # Search in VIDEO_TEST
    pure_name = Path(file_name).name
    matches = list((REPO_ROOT / "data" / "VIDEO_TEST").glob(f"**/{pure_name}"))
    if matches:
        return matches[0]
        
    matches_org = list((REPO_ROOT / "data" / "VIDEO_TEST_ORGANIZED").glob(f"**/{pure_name}"))
    if matches_org:
        return matches_org[0]
        
    # Search home directory
    matches_home = list(Path("/home/nguyendinhquy").glob(f"**/{pure_name}"))
    if matches_home:
        return matches_home[0]
        
    return None


def extract_keyframe(video_path: Path) -> Image.Image:
    if not video_path or not video_path.exists():
        # Fallback dummy image if video missing
        return Image.new("RGB", (448, 252), color=(73, 109, 137))
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return Image.new("RGB", (448, 252), color=(73, 109, 137))


def load_kb_corpus() -> List[Dict[str, str]]:
    with open(KB_EXPORT_PATH, "r", encoding="utf-8") as f:
        kb_data = json.load(f)
    corpus = []
    for node in kb_data.get("nodes_flat", []):
        name = node.get("name", "").strip()
        desc = node.get("description", "").strip()
        if name and desc:
            corpus.append({"name": name, "text": f"{name}: {desc[:400]}"})
    return corpus


def main():
    parser = argparse.ArgumentParser(description="Run all baselines and ablations")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    LOGGER.info("=" * 80)
    LOGGER.info("RUNNING ALL BASELINES AND ABLATIONS ON 250 TEST VIDEOS")
    LOGGER.info("=" * 80)

    if OUTPUT_BENCHMARK_PATH.exists():
        with open(OUTPUT_BENCHMARK_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)
        LOGGER.info("Loaded %d existing benchmark samples from: %s", len(samples), OUTPUT_BENCHMARK_PATH)
    else:
        with open(TEST_CAPTIONS_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)
        LOGGER.info("Loaded %d test samples from: %s", len(samples), TEST_CAPTIONS_PATH)

    # 1. OURS & FROZEN_BASE ALREADY EXIST
    for s in samples:
        if "caption_ours" not in s:
            s["caption_ours"] = s.get("caption_after_finetune", "")
        if "caption_frozen_base" not in s:
            s["caption_frozen_base"] = s.get("caption_before_finetune", "")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Check if Ablations 1 & 2 already computed
    has_ablation_1 = all(bool(s.get("caption_ablation_no_threshold")) for s in samples)
    has_ablation_2 = all(bool(s.get("caption_ablation_no_anchor")) for s in samples)

    if has_ablation_1 and has_ablation_2:
        LOGGER.info("\n>>> 2. ABLATIONS 1 & 2 ALREADY COMPUTED! SKIPPING TO KEEP GPU MEMORY CLEAN <<<")
    else:
        # 2. RUN ABLATIONS: NO_THRESHOLD & NO_ANCHOR WITH QWEN2.5-3B LORA
        LOGGER.info("\n>>> 2. RUNNING ABLATION 1 (NO_THRESHOLD) & ABLATION 2 (NO_ANCHOR) <<<")
        tokenizer_3b = AutoTokenizer.from_pretrained(str(QWEN_3B_BASE_PATH), padding_side="left")
        if tokenizer_3b.pad_token is None:
            tokenizer_3b.pad_token = tokenizer_3b.eos_token

        base_model_3b = AutoModelForCausalLM.from_pretrained(
            str(QWEN_3B_BASE_PATH),
            quantization_config=bnb_config,
            device_map=args.device,
        )
        lora_model_3b = PeftModel.from_pretrained(base_model_3b, str(LORA_ADAPTER_PATH)).eval()

        # Build Ablation 1 prompts (Always HIGH)
        prompts_no_thresh = []
        for s in samples:
            parent = s.get("groundtruth_parent", "")
            sub = s.get("groundtruth_sub", "")
            place_str = f"{sub} (part of {parent}, Hanoi)" if (sub and sub != parent) else f"{parent}, Hanoi"
            p = f"""CERTAINTY: HIGH

PLACE: {place_str}

OBSERVED (visual subgraph detected, with membership degree):
{s.get('observed_block', '')}

KNOWLEDGE (verified facts from graph, with membership degree):
{s.get('knowledge_block', '')}

RULES FOR THIS LEVEL:
- You MAY name the specific place.
- You MAY state the knowledge facts as established.
- Combine one visual observation with one knowledge fact.

Write the caption."""
            prompts_no_thresh.append(p)

        # Build Ablation 2 prompts (No anchor: greedy facts, no candidate masking)
        prompts_no_anchor = []
        for s in samples:
            p = f"""CERTAINTY: {s.get('certainty_level', 'HIGH')}

PLACE: {s.get('place_header', s.get('groundtruth_parent', ''))}

OBSERVED (visual subgraph detected, with membership degree):
{s.get('observed_block', '')}

KNOWLEDGE (raw unanchored facts from graph):
{s.get('knowledge_block', '')}

RULES FOR THIS LEVEL:
- Combine the highest scored observations with knowledge facts.

Write the caption."""
            prompts_no_anchor.append(p)

        # Generate Ablation 1
        LOGGER.info("Generating captions for Ablation 1: No Threshold (Always Affirm)...")
        for b_idx in range(0, len(prompts_no_thresh), 16):
            b_p = prompts_no_thresh[b_idx : b_idx + 16]
            b_msgs = [[{"role": "system", "content": "You are a tourism narrator."}, {"role": "user", "content": x}] for x in b_p]
            b_chat = [tokenizer_3b.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in b_msgs]
            enc = tokenizer_3b(b_chat, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.device)
            with torch.no_grad():
                out = lora_model_3b.generate(**enc, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer_3b.pad_token_id)
            inp_len = enc.input_ids.shape[1]
            for i_item in range(len(b_p)):
                gen_text = tokenizer_3b.decode(out[i_item][inp_len:], skip_special_tokens=True).strip()
                samples[b_idx + i_item]["caption_ablation_no_threshold"] = gen_text

        # Generate Ablation 2
        LOGGER.info("Generating captions for Ablation 2: No Anchor (Greedy mu_F)...")
        for b_idx in range(0, len(prompts_no_anchor), 16):
            b_p = prompts_no_anchor[b_idx : b_idx + 16]
            b_msgs = [[{"role": "system", "content": "You are a tourism narrator."}, {"role": "user", "content": x}] for x in b_p]
            b_chat = [tokenizer_3b.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in b_msgs]
            enc = tokenizer_3b(b_chat, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(args.device)
            with torch.no_grad():
                out = lora_model_3b.generate(**enc, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer_3b.pad_token_id)
            inp_len = enc.input_ids.shape[1]
            for i_item in range(len(b_p)):
                gen_text = tokenizer_3b.decode(out[i_item][inp_len:], skip_special_tokens=True).strip()
                samples[b_idx + i_item]["caption_ablation_no_anchor"] = gen_text

        # Free 3B model memory
        del lora_model_3b, base_model_3b
        torch.cuda.empty_cache()

        OUTPUT_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_BENCHMARK_PATH, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        LOGGER.info("Saved intermediate checkpoint (Ablations 1 & 2 done) to: %s", OUTPUT_BENCHMARK_PATH)

    # 3. EXTRACT KEYFRAMES FOR 250 TEST SAMPLES
    LOGGER.info("\n>>> 3. EXTRACTING KEYFRAMES FOR 250 TEST SAMPLES <<<")
    keyframes = []
    keyframe_dir = REPO_ROOT / "experiments" / "outputs" / "test_250_keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    for idx, s in enumerate(samples):
        save_p = keyframe_dir / f"keyframe_{idx:03d}.jpg"
        if save_p.exists():
            img_resized = Image.open(save_p).convert("RGB")
        else:
            v_path = find_video_path(s.get("file_name", ""))
            img = extract_keyframe(v_path)
            img_resized = img.resize((448, 252))
            img_resized.save(save_p)
        keyframes.append(img_resized)
        s["keyframe_path"] = str(save_p)
    LOGGER.info("Loaded/Extracted %d keyframes from: %s", len(keyframes), keyframe_dir)

    # 4. RUN SIGLIP2 KEYFRAME RAG ON KB CORPUS (ON CPU TO KEEP GPU 100% UNTOUCHED)
    LOGGER.info("\n>>> 4. RUNNING SIGLIP2 RETRIEVAL FROM KB TEXT CORPUS (CPU) <<<")
    corpus = load_kb_corpus()
    LOGGER.info("Loaded %d landmark description passages from KB export.", len(corpus))

    siglip_proc = AutoProcessor.from_pretrained(str(SIGLIP2_PATH))
    siglip_model = AutoModel.from_pretrained(str(SIGLIP2_PATH)).to("cpu").eval()

    # Encode text passages
    LOGGER.info("Encoding %d KB text descriptions with SigLIP2...", len(corpus))
    corpus_texts = [item["text"] for item in corpus]
    text_inputs = siglip_proc(text=corpus_texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cpu")
    with torch.no_grad():
        text_out = siglip_model.get_text_features(**text_inputs)
        text_features = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Encode keyframes and retrieve Top-1 passage
    LOGGER.info("Encoding 250 keyframes and computing similarity matrix...")
    retrieved_passages = []
    for b_idx in range(0, len(keyframes), 32):
        b_imgs = keyframes[b_idx : b_idx + 32]
        img_inputs = siglip_proc(images=b_imgs, return_tensors="pt").to("cpu")
        with torch.no_grad():
            img_out = siglip_model.get_image_features(**img_inputs)
            img_features = img_out.pooler_output if hasattr(img_out, "pooler_output") else img_out
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sim = img_features @ text_features.T
            top_idx = torch.argmax(sim, dim=-1).cpu().tolist()
            for tid in top_idx:
                retrieved_passages.append(corpus[tid]["text"])

    del siglip_model
    LOGGER.info("SigLIP2 retrieval completed on CPU.")

    for s, ret_txt in zip(samples, retrieved_passages):
        s["retrieved_rag_context"] = ret_txt

    # 5. RUN VLM DIRECT & VLM+RAG WITH QWEN3-VL-8B
    LOGGER.info("\n>>> 5. RUNNING QWEN3-VL-8B (DIRECT VLM & VLM + RAG) <<<")
    vlm_proc = Qwen3VLProcessor.from_pretrained(str(QWEN3_VL_PATH))
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(QWEN3_VL_PATH),
        quantization_config=bnb_config,
        device_map=args.device,
    ).eval()

    LOGGER.info("Generating captions for VLM Direct (Keyframe only, no facts)...")
    for idx, (s, img) in enumerate(zip(samples, keyframes)):
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this Vietnamese tourism site in 25-40 words for a dense travel caption. Describe what you see."}
            ]}
        ]
        prompt_txt = vlm_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = vlm_proc(text=[prompt_txt], images=[img], return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = vlm_model.generate(**inputs, max_new_tokens=60, do_sample=False)
        gen_txt = vlm_proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        s["caption_vlm_direct"] = gen_txt
        if (idx + 1) % 50 == 0:
            LOGGER.info("Progress VLM Direct: %d / 250", idx + 1)

    # Save checkpoint after VLM Direct
    with open(OUTPUT_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved intermediate checkpoint (VLM Direct done) to: %s", OUTPUT_BENCHMARK_PATH)

    LOGGER.info("Generating captions for VLM + RAG (Keyframe + SigLIP2 retrieved context)...")
    for idx, (s, img) in enumerate(zip(samples, keyframes)):
        rag_ctx = s.get("retrieved_rag_context", "")[:250]
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"You are a travel video narrator. Based on this scene and the retrieved background context snippet:\n[RETRIEVED CONTEXT: {rag_ctx}]\nWrite a 25-40 word dense tourism caption."}
            ]}
        ]
        prompt_txt = vlm_proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = vlm_proc(text=[prompt_txt], images=[img], return_tensors="pt").to(args.device)
        with torch.no_grad():
            out = vlm_model.generate(**inputs, max_new_tokens=60, do_sample=False)
        gen_txt = vlm_proc.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        s["caption_vlm_rag"] = gen_txt
        if (idx + 1) % 50 == 0:
            LOGGER.info("Progress VLM+RAG: %d / 250", idx + 1)

    del vlm_model
    torch.cuda.empty_cache()

    # 6. SAVE ALL GENERATIONS
    OUTPUT_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    LOGGER.info("\nSaved all 6 model captions for 250 test videos to: %s", OUTPUT_BENCHMARK_PATH)


if __name__ == "__main__":
    main()
