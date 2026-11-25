#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Parametric Bootstrap 95% Confidence Intervals for Captioning Generation Metrics
==================================================================================
Runs B = 10,000 bootstrap iterations over the held-out test set (n = 149)
to compute point estimates, standard errors, and empirical 95% percentile CIs [2.5%, 97.5%]
for:
  1. Claim Precision (%)
  2. Knowledge Recall (K-Recall, %)
  3. Over-Assertion Rate (OAR, %)
  4. Policy Compliance (%)
  5. True VCR (Visual Contradiction Rate, %)
  6. Visual Scene Precision (VSP = 100 - True VCR, %)
  7. BLEU-4 and ROUGE-L

Evaluates:
  - 3 Individually Retrained Certainty Representations (Adapters):
    * Binary gate
    * HIGH/MEDIUM/LOW only (Ternary)
    * Continuous c + regime (Full Model)
  - Baselines:
    * Prompt-only LLM
    * KG-conditioned LLM
    * Hard-threshold Captioning
"""

import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path("/home/nguyendinhquy/TransMTL_OntoKG-main")
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.train_and_benchmark_3representation_ablation import (
    PROMPT_BUILDERS, evaluate_caption, generate_test_captions,
    build_binary_prompt, build_ternary_prompt, build_continuous_prompt,
    format_claim_as_sentence, LOCAL_3B, CHECKPOINTS_BASE_DIR,
    TEST_DATA_PATH, SYSTEM_PROMPT
)

BASE_MODEL_PATH = LOCAL_3B

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("BootstrapCIEval")

DATA_DIR = REPO_ROOT / "data" / "caption_dataset"
OUTPUT_JSON_PATH = DATA_DIR / "bootstrap_generation_ci_report.json"
PER_SAMPLE_OUTPUT_PATH = DATA_DIR / "per_sample_generation_eval_test149.json"

# Baselines prompt constructors
def build_prompt_only(sample: dict) -> str:
    return f"""OBSERVED SCENE:
{sample['observed_scene_graph']}

INSTRUCTION:
Write a cohesive 3–4 sentence narration paragraph (approx. 40–70 words) describing the scene.
Do not hallucinate facts not visually visible.

CAPTION:"""

def build_kg_conditioned(sample: dict) -> str:
    facts = sample.get("all_candidate_facts", sample.get("eligible_facts", []))
    facts_lines = [f"- {format_claim_as_sentence(f.get('head', sample['landmark_gt']), f['relation'], f['tail'])}" for f in facts]
    facts_str = "\n".join(facts_lines) if facts_lines else "- None"
    return f"""LANDMARK: {sample['landmark_gt']}
FACTS:
{facts_str}

OBSERVED:
{sample['observed_scene_graph']}

INSTRUCTION:
Write a cohesive 3–4 sentence narration paragraph (approx. 40–70 words) incorporating the landmark facts and observed scene.

CAPTION:"""

def build_hard_threshold(sample: dict) -> str:
    mu_L = sample.get("mu_L", 0.5)
    if mu_L >= 0.50:
        facts = sample.get("eligible_facts", [])
        facts_lines = [f"- {format_claim_as_sentence(f.get('head', sample['landmark_gt']), f['relation'], f['tail'])}" for f in facts]
        facts_str = "\n".join(facts_lines) if facts_lines else "- None"
        return f"""LANDMARK: {sample['landmark_gt']}
FACTS:
{facts_str}

OBSERVED:
{sample['observed_scene_graph']}

INSTRUCTION:
Write a cohesive 3–4 sentence narration paragraph (approx. 40–70 words).

CAPTION:"""
    else:
        return f"""LANDMARK: Unknown
FACTS: None

OBSERVED:
{sample['observed_scene_graph']}

INSTRUCTION:
Write a cohesive 3–4 sentence narration paragraph (approx. 40–70 words) describing only what is visually observed. Do not state any landmark name or facts.

CAPTION:"""

def run_bootstrap_ci(values: np.ndarray, num_bootstrap: int = 10000, seed: int = 42) -> dict:
    """Non-parametric percentile bootstrap CI [2.5%, 97.5%]."""
    np.random.seed(seed)
    n = len(values)
    idx = np.random.randint(0, n, size=(num_bootstrap, n))
    boot_means = np.mean(values[idx], axis=1)

    point_est = float(np.mean(values))
    std_err = float(np.std(boot_means, ddof=1))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "point_estimate": round(point_est, 2),
        "std_error": round(std_err, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "ci_str": f"{point_est:.1f} [{ci_lower:.1f}, {ci_upper:.1f}]",
    }

def main():
    LOGGER.info("Starting Bootstrap 95% Confidence Interval Benchmark...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f if line.strip() and not line.startswith("[ERROR")]
    N = len(test_samples)
    LOGGER.info(f"Loaded {N} held-out test samples.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Define targets to benchmark
    targets = [
        # 1. 3 Trained Adapters
        ("lora_binary", "Binary gate (Adapter)", build_binary_prompt, CHECKPOINTS_BASE_DIR / "qwen25_3b_lora_binary"),
        ("lora_ternary", "HIGH/MEDIUM/LOW only (Adapter)", build_ternary_prompt, CHECKPOINTS_BASE_DIR / "qwen25_3b_lora_ternary"),
        ("lora_continuous", "Continuous $c$ + regime (Ours)", build_continuous_prompt, CHECKPOINTS_BASE_DIR / "qwen25_3b_lora_continuous"),
        # 2. Base model baselines (zero-shot prompt baselines)
        ("base_prompt_only", "Prompt-only LLM (No KG/Cert)", build_prompt_only, None),
        ("base_kg_conditioned", "KG-conditioned LLM (Unfiltered)", build_kg_conditioned, None),
        ("base_hard_threshold", "Hard-threshold ($\\theta=0.50$)", build_hard_threshold, None),
    ]

    all_per_sample = {}
    all_bootstrap_summary = {}

    for key, name, prompt_fn, ckpt_path in targets:
        LOGGER.info("-" * 70)
        LOGGER.info(f"Evaluating target: {name} (Key: {key})")
        t0 = time.time()

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map=device if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        if ckpt_path is not None and ckpt_path.exists():
            LOGGER.info(f"Loading LoRA weights from {ckpt_path}")
            model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        else:
            LOGGER.info("Using raw Base Model without LoRA adapter")
            model = base_model

        prompts = [prompt_fn(s) for s in test_samples]
        captions = generate_test_captions(model, tokenizer, prompts, device)

        # Evaluate per sample
        sample_metrics_list = []
        for i, s in enumerate(test_samples):
            cap = captions[i]
            m = evaluate_caption(cap, s)
            # Convert [0, 1] ratios to percentage [0, 100]
            m_pct = {
                "sample_id": s.get("id", i),
                "scenario": s.get("scenario", ""),
                "certainty_level": s.get("certainty_level", ""),
                "caption": cap,
                "claim_precision": float(m["claim_precision"] * 100.0),
                "k_recall": float(m["k_recall"] * 100.0),
                "oar": float(m["oar"] * 100.0),
                "policy_compliant": float(m["policy_compliant"] * 100.0),
                "true_vcr": float(m["true_vcr"] * 100.0),
                "vsp": float((1.0 - m["true_vcr"]) * 100.0),
                "bleu4": float(m["bleu4"]),
                "rouge_l": float(m["rouge_l"]),
            }
            sample_metrics_list.append(m_pct)

        all_per_sample[key] = sample_metrics_list

        # Extract metric vectors
        metric_keys = ["claim_precision", "k_recall", "oar", "policy_compliant", "true_vcr", "vsp", "bleu4", "rouge_l"]
        bootstrap_target_summary = {
            "name": name,
            "n_samples": N,
            "eval_time_sec": round(time.time() - t0, 1),
            "metrics": {}
        }

        for mk in metric_keys:
            arr = np.array([m[mk] for m in sample_metrics_list], dtype=float)
            boot_res = run_bootstrap_ci(arr, num_bootstrap=10000, seed=42)
            bootstrap_target_summary["metrics"][mk] = boot_res

        all_bootstrap_summary[key] = bootstrap_target_summary
        LOGGER.info(f"✓ [{key}] ClaimPrec: {bootstrap_target_summary['metrics']['claim_precision']['ci_str']} | "
                    f"K-Recall: {bootstrap_target_summary['metrics']['k_recall']['ci_str']} | "
                    f"OAR: {bootstrap_target_summary['metrics']['oar']['ci_str']} | "
                    f"Policy: {bootstrap_target_summary['metrics']['policy_compliant']['ci_str']} | "
                    f"VSP: {bootstrap_target_summary['metrics']['vsp']['ci_str']}")

        # Cleanup GPU memory
        del model
        del base_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Format LaTeX Table
    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Bootstrap 95\\% confidence intervals ($B=10{,}000$ iterations) for generation metrics",
        "on the held-out test set ($n=149$). We report empirical point estimates and 95\\% percentile",
        "intervals $[2.5\\%, 97.5\\%]$ for each certainty representation and baseline architecture.}",
        "\\label{tab:bootstrap_ci_caption_generation}",
        "\\small",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Method / Representation} & \\textbf{Claim Prec. (\\%)} $\\uparrow$ & \\textbf{K-Recall (\\%)} $\\uparrow$ & \\textbf{OAR (\\%)} $\\downarrow$ & \\textbf{Policy (\\%)} $\\uparrow$ & \\textbf{VSP (\\%)} $\\uparrow$ \\\\",
        "\\midrule",
    ]

    for key, name, _, _ in targets:
        m = all_bootstrap_summary[key]["metrics"]
        cp = f"{m['claim_precision']['point_estimate']:.1f} [{m['claim_precision']['ci_lower']:.1f}, {m['claim_precision']['ci_upper']:.1f}]"
        kr = f"{m['k_recall']['point_estimate']:.1f} [{m['k_recall']['ci_lower']:.1f}, {m['k_recall']['ci_upper']:.1f}]"
        oar = f"{m['oar']['point_estimate']:.1f} [{m['oar']['ci_lower']:.1f}, {m['oar']['ci_upper']:.1f}]"
        pol = f"{m['policy_compliant']['point_estimate']:.1f} [{m['policy_compliant']['ci_lower']:.1f}, {m['policy_compliant']['ci_upper']:.1f}]"
        vsp = f"{m['vsp']['point_estimate']:.1f} [{m['vsp']['ci_lower']:.1f}, {m['vsp']['ci_upper']:.1f}]"
        if "lora_continuous" in key:
            line = f"\\textbf{{{name}}} & \\textbf{{{cp}}} & \\textbf{{{kr}}} & \\textbf{{{oar}}} & \\textbf{{{pol}}} & \\textbf{{{vsp}}} \\\\"
        else:
            line = f"{name} & {cp} & {kr} & {oar} & {pol} & {vsp} \\\\"
        latex_lines.append(line)

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    bootstrap_latex_table = "\n".join(latex_lines)

    print("\n" + "=" * 110)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (B = 10,000):")
    print("=" * 110)
    print(bootstrap_latex_table)

    # Save to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark": "Bootstrap 95% Confidence Intervals for Captioning Generation (B=10000)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "targets": all_bootstrap_summary,
            "bootstrap_latex_table": bootstrap_latex_table,
        }, f, indent=2, ensure_ascii=False)

    with open(PER_SAMPLE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_per_sample, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"✓ Saved Bootstrap summary to {OUTPUT_JSON_PATH}")
    LOGGER.info(f"✓ Saved per-sample evaluations to {PER_SAMPLE_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
