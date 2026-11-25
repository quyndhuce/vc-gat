#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Blind Pairwise Head-to-Head Captions using Qwen2.5-14B Judge
====================================================================
Chấm đối đầu theo cặp (Pairwise Head-to-Head Evaluation) giữa mô hình đề xuất ('ours')
và 5 mô hình cơ sở / ablation:
1. vlm_direct: VLM trực tiếp (Qwen3-VL-8B)
2. vlm_rag: VLM + RAG (Qwen3-VL-8B + SigLIP2)
3. frozen_base: Base LLM đóng băng không fine-tune
4. ablation_no_threshold: Bỏ ngưỡng (Always Affirm)
5. ablation_no_anchor: Bỏ neo (Greedy mu_F)

3 câu hỏi thẩm định cho mỗi cặp:
- Q1: Caption nào đúng với cảnh hơn? (Scene Consistency)
- Q2: Caption nào đáng tin hơn? (Factual Reliability / No Hallucination)
- Q3: Caption nào muốn nghe hơn? (Audio/Narrative Engagement)

Báo cáo tỷ lệ Thắng / Hòa / Thua (Win / Tie / Loss %) cho từng cặp đối đầu.
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("PairwiseJudge")

REPO_ROOT = Path(__file__).resolve().parent.parent
BLIND_DATASET_PATH = REPO_ROOT / "experiments" / "outputs" / "pairwise_blind_for_evaluation.json"
KEYS_DATASET_PATH = REPO_ROOT / "experiments" / "outputs" / "pairwise_blind_keys.json"
OUTPUT_EVALUATED_PATH = REPO_ROOT / "experiments" / "outputs" / "pairwise_evaluated_qwen14b_full.json"
OUTPUT_REPORT_PATH = REPO_ROOT / "experiments" / "outputs" / "pairwise_head_to_head_report.json"

QWEN_7B_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
QWEN_14B_PATH = Path("/home/nguyendinhquy/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8")

JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous expert judge specializing in evaluating tourist video narrations and audio guide captions for cultural heritage sites in Hanoi, Vietnam.
You will be presented with a ground truth cultural heritage site and two blind candidate captions (Model A and Model B).
You must compare them objectively across 3 distinct questions without position bias.

Question 1: Which caption is more consistent with the scene and true to the actual heritage site? (q1_more_scene_consistent)
- Prefer captions that correctly identify the true place without misnaming landmarks or making up physical features not present at that heritage site.

Question 2: Which caption is more factual, reliable, and trustworthy? (q2_more_factual_reliable)
- Heavily penalize captions that hallucinate incorrect historical facts, cite nonexistent dynasties, or falsely name other unrelated tourist attractions.
- A restrained, cautious caption is strictly more reliable than an overconfident caption that states falsehoods.

Question 3: Which caption is more engaging, pleasant, and natural to listen to as a travel audio guide? (q3_more_engaging_to_listen)
- Evaluate fluency, narrative tone, flow, and travel atmosphere.

For each question, select strictly one of: "Model A", "Model B", or "Tie".
Output your decision strictly as a JSON object with this schema:
```json
{
  "q1_more_scene_consistent": "Model A" | "Model B" | "Tie",
  "q2_more_factual_reliable": "Model A" | "Model B" | "Tie",
  "q3_more_engaging_to_listen": "Model A" | "Model B" | "Tie",
  "brief_rationale": "short explanation of difference"
}
```"""


def build_judge_prompt(item: Dict) -> str:
    ctx = item.get("groundtruth_context", {})
    parent = ctx.get("parent_landmark", "Hanoi Cultural Heritage Site")
    sub = ctx.get("sub_landmark", "")
    certainty = ctx.get("certainty_level", "HIGH")
    
    place_desc = f"{sub} ({parent})" if (sub and sub != parent) else parent
    
    prompt = f"""[HERITAGE SITE INFORMATION]
True Landmark: {place_desc}
Visual Certainty Level: {certainty} (HIGH = clear distinctive landmark; MEDIUM = ambiguous / candidate structures; LOW = generic feature)

[CANDIDATE CAPTIONS]
Model A: "{item.get('caption_a', '')}"
Model B: "{item.get('caption_b', '')}"

Compare Model A and Model B on the 3 questions and return the JSON evaluation:"""
    return prompt


def parse_judge_output(raw_text: str) -> Dict[str, str]:
    # Default fallback
    res = {
        "q1_more_scene_consistent": "Tie",
        "q2_more_factual_reliable": "Tie",
        "q3_more_engaging_to_listen": "Tie",
        "brief_rationale": "Failed to parse judge output."
    }
    
    try:
        # Try extracting JSON block
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if brace_match:
                data = json.loads(brace_match.group(1))
            else:
                data = {}
                
        def clean_choice(val: str) -> str:
            val_str = str(val).lower()
            if "model a" in val_str or val_str == "a":
                return "Model A"
            elif "model b" in val_str or val_str == "b":
                return "Model B"
            else:
                return "Tie"
                
        if "q1_more_scene_consistent" in data:
            res["q1_more_scene_consistent"] = clean_choice(data["q1_more_scene_consistent"])
        if "q2_more_factual_reliable" in data:
            res["q2_more_factual_reliable"] = clean_choice(data["q2_more_factual_reliable"])
        if "q3_more_engaging_to_listen" in data:
            res["q3_more_engaging_to_listen"] = clean_choice(data["q3_more_engaging_to_listen"])
        if "brief_rationale" in data:
            res["brief_rationale"] = str(data["brief_rationale"])
            
    except Exception:
        # Regex heuristics fallback
        if re.search(r"q1.*?(model a|model b|tie)", raw_text, re.IGNORECASE):
            m = re.search(r"q1.*?(model a|model b|tie)", raw_text, re.IGNORECASE)
            g = m.group(1).lower()
            res["q1_more_scene_consistent"] = "Model A" if "a" in g else ("Model B" if "b" in g else "Tie")
        if re.search(r"q2.*?(model a|model b|tie)", raw_text, re.IGNORECASE):
            m = re.search(r"q2.*?(model a|model b|tie)", raw_text, re.IGNORECASE)
            g = m.group(1).lower()
            res["q2_more_factual_reliable"] = "Model A" if "a" in g else ("Model B" if "b" in g else "Tie")
        if re.search(r"q3.*?(model a|model b|tie)", raw_text, re.IGNORECASE):
            m = re.search(r"q3.*?(model a|model b|tie)", raw_text, re.IGNORECASE)
            g = m.group(1).lower()
            res["q3_more_engaging_to_listen"] = "Model A" if "a" in g else ("Model B" if "b" in g else "Tie")
            
    return res


def main():
    parser = argparse.ArgumentParser(description="Pairwise Head-to-Head Judge")
    parser.add_argument("--model-size", type=str, default="7b", choices=["7b", "14b"], help="Model size for judge (7b or 14b)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    args = parser.parse_args()

    judge_model_path = QWEN_14B_PATH if args.model_size == "14b" else QWEN_7B_PATH

    LOGGER.info("=" * 80)
    LOGGER.info("STARTING PAIRWISE HEAD-TO-HEAD EVALUATION (%s JUDGE)", args.model_size.upper())
    LOGGER.info("=" * 80)

    # Load items (resume if partial evaluations exist)
    if OUTPUT_EVALUATED_PATH.exists():
        with open(OUTPUT_EVALUATED_PATH, "r", encoding="utf-8") as f:
            blind_items = json.load(f)
        LOGGER.info("Resumed %d items from existing evaluated file.", len(blind_items))
    else:
        with open(BLIND_DATASET_PATH, "r", encoding="utf-8") as f:
            blind_items = json.load(f)
        LOGGER.info("Loaded %d blind pairs from dataset.", len(blind_items))

    with open(KEYS_DATASET_PATH, "r", encoding="utf-8") as f:
        keys_map = json.load(f)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    LOGGER.info("Loading Judge Model (%s) from: %s", args.model_size.upper(), judge_model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(judge_model_path), padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(judge_model_path),
        quantization_config=bnb_config,
        device_map=args.device,
    ).eval()

    # Identify pending indices
    pending_indices = [
        i for i, item in enumerate(blind_items)
        if not item.get("evaluation", {}).get("q1_more_scene_consistent")
    ]
    LOGGER.info("Pending items to evaluate: %d / %d", len(pending_indices), len(blind_items))

    total_pending = len(pending_indices)
    for b_start in range(0, total_pending, args.batch_size):
        batch_idx_subset = pending_indices[b_start : b_start + args.batch_size]
        batch_items = [blind_items[i] for i in batch_idx_subset]
        prompts = [build_judge_prompt(item) for item in batch_items]
        
        batch_msgs = [
            [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": p}]
            for p in prompts
        ]
        chat_texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_msgs]
        enc = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(args.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        inp_len = enc.input_ids.shape[1]
        for idx_in_batch, item in enumerate(batch_items):
            gen_text = tokenizer.decode(out[idx_in_batch][inp_len:], skip_special_tokens=True).strip()
            parsed = parse_judge_output(gen_text)
            item["evaluation"] = parsed

        if (b_start + len(batch_items)) % 80 == 0 or (b_start + len(batch_items)) == total_pending:
            # Save periodic progress
            with open(OUTPUT_EVALUATED_PATH, "w", encoding="utf-8") as f:
                json.dump(blind_items, f, indent=2, ensure_ascii=False)
            LOGGER.info("Evaluated and saved progress: %d / %d pending pairs (%.1f%%)...",
                        b_start + len(batch_items), total_pending, (b_start + len(batch_items))/total_pending*100)

    # Save final evaluated JSON
    with open(OUTPUT_EVALUATED_PATH, "w", encoding="utf-8") as f:
        json.dump(blind_items, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved full pairwise evaluations to: %s", OUTPUT_EVALUATED_PATH)

    # Free model memory
    del model, tokenizer
    torch.cuda.empty_cache()

    # Aggregate Statistics Win / Tie / Loss (%)
    LOGGER.info("\nAggregating Win / Tie / Loss statistics against each baseline...")

    baseline_stats = defaultdict(lambda: {
        "q1_more_scene_consistent": {"win": 0, "tie": 0, "loss": 0},
        "q2_more_factual_reliable": {"win": 0, "tie": 0, "loss": 0},
        "q3_more_engaging_to_listen": {"win": 0, "tie": 0, "loss": 0},
        "total": 0,
        "name": ""
    })

    for item in blind_items:
        pair_id = item["pair_id"]
        eval_dict = item.get("evaluation", {})
        key_info = keys_map.get(pair_id, {})

        b_key = key_info.get("baseline_comparison", "unknown")
        b_name = key_info.get("baseline_name", b_key)
        is_ours_a = key_info.get("is_ours_model_a", True)

        stats = baseline_stats[b_key]
        stats["name"] = b_name
        stats["total"] += 1

        for q_key in ["q1_more_scene_consistent", "q2_more_factual_reliable", "q3_more_engaging_to_listen"]:
            choice = eval_dict.get(q_key, "Tie")
            if choice == "Tie":
                stats[q_key]["tie"] += 1
            elif (choice == "Model A" and is_ours_a) or (choice == "Model B" and not is_ours_a):
                stats[q_key]["win"] += 1
            else:
                stats[q_key]["loss"] += 1

    summary_report = {}
    for b_key, data in baseline_stats.items():
        tot = data["total"]
        b_report = {
            "baseline_key": b_key,
            "baseline_name": data["name"],
            "total_pairs": tot,
            "questions": {}
        }
        for q_key in ["q1_more_scene_consistent", "q2_more_factual_reliable", "q3_more_engaging_to_listen"]:
            w = data[q_key]["win"]
            t = data[q_key]["tie"]
            l = data[q_key]["loss"]
            b_report["questions"][q_key] = {
                "ours_win": w,
                "ours_win_pct": round(w / tot * 100, 2),
                "tie": t,
                "tie_pct": round(t / tot * 100, 2),
                "ours_loss": l,
                "ours_loss_pct": round(l / tot * 100, 2),
            }
        
        # Overall across all questions
        all_wins = sum(data[q]["win"] for q in ["q1_more_scene_consistent", "q2_more_factual_reliable", "q3_more_engaging_to_listen"])
        all_ties = sum(data[q]["tie"] for q in ["q1_more_scene_consistent", "q2_more_factual_reliable", "q3_more_engaging_to_listen"])
        all_losses = sum(data[q]["loss"] for q in ["q1_more_scene_consistent", "q2_more_factual_reliable", "q3_more_engaging_to_listen"])
        all_tot = tot * 3
        b_report["overall"] = {
            "ours_win_pct": round(all_wins / all_tot * 100, 2),
            "tie_pct": round(all_ties / all_tot * 100, 2),
            "ours_loss_pct": round(all_losses / all_tot * 100, 2),
        }
        summary_report[b_key] = b_report

        LOGGER.info(
            "=== Ours vs %s ===", data["name"]
        )
        LOGGER.info(
            "  Q1 (Scene):    Win: %5.2f%% | Tie: %5.2f%% | Loss: %5.2f%%",
            b_report["questions"]["q1_more_scene_consistent"]["ours_win_pct"],
            b_report["questions"]["q1_more_scene_consistent"]["tie_pct"],
            b_report["questions"]["q1_more_scene_consistent"]["ours_loss_pct"],
        )
        LOGGER.info(
            "  Q2 (Factual):  Win: %5.2f%% | Tie: %5.2f%% | Loss: %5.2f%%",
            b_report["questions"]["q2_more_factual_reliable"]["ours_win_pct"],
            b_report["questions"]["q2_more_factual_reliable"]["tie_pct"],
            b_report["questions"]["q2_more_factual_reliable"]["ours_loss_pct"],
        )
        LOGGER.info(
            "  Q3 (Engaging): Win: %5.2f%% | Tie: %5.2f%% | Loss: %5.2f%%",
            b_report["questions"]["q3_more_engaging_to_listen"]["ours_win_pct"],
            b_report["questions"]["q3_more_engaging_to_listen"]["tie_pct"],
            b_report["questions"]["q3_more_engaging_to_listen"]["ours_loss_pct"],
        )
        LOGGER.info(
            "  OVERALL:       Win: %5.2f%% | Tie: %5.2f%% | Loss: %5.2f%%",
            b_report["overall"]["ours_win_pct"],
            b_report["overall"]["tie_pct"],
            b_report["overall"]["ours_loss_pct"],
        )

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    LOGGER.info("\nSaved Pairwise Head-to-Head Report to: %s", OUTPUT_REPORT_PATH)


if __name__ == "__main__":
    main()
