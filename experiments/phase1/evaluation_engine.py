#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_engine.py
=============================================================================
Core Evaluation Engine for Phase 1: End-to-End Captioning Evaluation.
Covers:
1. Evidence Packet Construction (E1 to E6) with strict isolation & budget rules.
2. Atomic Claim Extraction & Classification (HISTORICAL vs VISUAL, ASSERT vs HEDGE).
3. Ground-Truth Claim Verification (SUPPORTED, CONTRADICTED, UNKNOWN).
4. Mathematical Metric Computation (FP, KR, BKR@5, VSP, VFAR, Coverage, etc.).
5. Cluster Bootstrap Confidence Intervals (B = 10,000).
=============================================================================
"""

import os
import re
import json
import math
import logging
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

LOGGER = logging.getLogger("Phase1EvalEngine")

# Linguistic markers
ASSERTIVE_MARKERS = [
    "is known for", "was built", "was constructed", "was founded",
    "dates back", "was established", "is located", "has been", "features",
    "is a", "are a", "stands as", "serves as", "showcases", "houses",
    "honoring", "commemorates", "contains", "comprises", "consists of",
    "exhibits", "possesses", "surrounded by", "bordered by"
]

HEDGING_MARKERS = [
    "likely", "appears to", "possibly", "might be", "could be",
    "seems to", "suggests", "may have", "if this is", "believed to",
    "probably", "potentially", "perhaps", "may include", "typical of",
    "depending on", "indicates", "resembles", "evokes", "reportedly",
    "is said to", "is traditionally associated with", "thought to be"
]

VISUAL_KEYWORDS = [
    "see", "visible", "clearly", "captures", "shows", "features", "wall", "gate",
    "courtyard", "roof", "tower", "bridge", "lake", "tree", "exhibit", "statue",
    "temple", "pagoda", "building", "hall", "plaza", "square", "color", "red",
    "yellow", "brick", "stone", "water", "stalls", "walkway", "arch", "display",
    "ceramic", "silk", "lantern", "flag", "structure", "facade", "column", "pavilion"
]

HISTORICAL_KEYWORDS = [
    "built in", "constructed in", "century", "dynasty", "emperor", "king",
    "french rule", "colonial", "founded in", "war", "relic", "ancient",
    "history", "historical", "revolution", "monument", "heritage", "designated",
    "ho chi minh", "nguyen dynasty", "ly dynasty", "tran dynasty", "le dynasty"
]


def split_sentences(text: str) -> List[str]:
    """Splits caption into clean, coherent sentences."""
    cleaned = text.strip().replace("\n", " ")
    raw_sents = re.split(r'(?<=[.!?])\s+', cleaned)
    return [s.strip() for s in raw_sents if len(s.strip()) > 8]


def is_hedged(sentence: str) -> bool:
    s_lower = sentence.lower()
    return any(m in s_lower for m in HEDGING_MARKERS)


def classify_claim_type(claim: str) -> str:
    c_lower = claim.lower()
    has_hist = any(k in c_lower for k in HISTORICAL_KEYWORDS)
    has_vis = any(k in c_lower for k in VISUAL_KEYWORDS)
    if has_hist and not has_vis:
        return "HISTORICAL"
    elif has_vis and not has_hist:
        return "VISUAL"
    elif has_hist and has_vis:
        return "MIXED"
    else:
        # Default to historical if factual relation
        return "HISTORICAL" if any(w in c_lower for w in ["hanoi", "vietnam", "century", "capital"]) else "VISUAL"


def match_fact_tail(tail: str, text_lower: str) -> bool:
    tail_clean = tail.lower().strip()
    if not tail_clean or len(tail_clean) < 3:
        return False
    words = [w for w in tail_clean.split() if len(w) > 3]
    if tail_clean in text_lower:
        return True
    if len(words) >= 2 and sum(1 for w in words if w in text_lower) >= 2:
        return True
    return False


def verify_historical_claim(
    claim: str,
    gold_facts: List[Dict[str, Any]],
    negative_facts: List[Dict[str, Any]]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Verifies an atomic claim against independent reference:
    - SUPPORTED if entails / matches a gold fact (review == 'đúng').
    - CONTRADICTED if contradicts / matches a negative fact (review == 'sai').
    - UNKNOWN if insufficient evidence.
    """
    c_lower = claim.lower()
    # Check gold facts
    for gf in gold_facts:
        tail = gf.get("object") or gf.get("tail") or ""
        subj = gf.get("subject") or gf.get("head") or ""
        pred = gf.get("predicate") or gf.get("relation") or ""
        fact_text = gf.get("fact_text", f"{subj} {pred} {tail}").lower()

        if match_fact_tail(tail, c_lower) or fact_text in c_lower:
            return "SUPPORTED", gf

    # Check negative facts
    for nf in negative_facts:
        tail = nf.get("object") or nf.get("tail") or ""
        subj = nf.get("subject") or nf.get("head") or ""
        pred = nf.get("predicate") or nf.get("relation") or ""
        fact_text = nf.get("fact_text", f"{subj} {pred} {tail}").lower()

        if (tail and tail.lower() in c_lower and len(tail) > 3) or fact_text in c_lower:
            return "CONTRADICTED", nf

    return "UNKNOWN", None


def verify_visual_claim(
    claim: str,
    visual_concepts: List[str]
) -> str:
    """Verifies a visual claim against verified video concepts."""
    c_lower = claim.lower()
    for vc in visual_concepts:
        vc_clean = re.sub(r"\(\d+\.\d+\)", "", vc).lower().strip()
        if len(vc_clean) > 3 and vc_clean in c_lower:
            return "SUPPORTED"
        words = [w for w in vc_clean.split() if len(w) > 3]
        if len(words) >= 2 and sum(1 for w in words if w in c_lower) >= 2:
            return "SUPPORTED"
    return "UNKNOWN"


def compute_clip_metrics(
    claims: List[Dict[str, Any]],
    gold_facts: List[Dict[str, Any]],
    assigned_regimes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Computes all required metrics for a single clip:
    - FP, VerificationCoverage, FP_conservative
    - KR, BKR@5
    - VSP
    - VFAR, AssertedUnknownRate
    - EAP, OVR, OARgen (if assigned_regimes present)
    """
    hist_claims = [c for c in claims if c.get("type", c.get("claim_type")) in ("HISTORICAL", "MIXED")]
    vis_claims = [c for c in claims if c.get("type", c.get("claim_type")) in ("VISUAL", "MIXED")]

    # 1. Historical metrics
    n_s = sum(1 for c in hist_claims if c["verification"] == "SUPPORTED")
    n_c = sum(1 for c in hist_claims if c["verification"] == "CONTRADICTED")
    n_u = sum(1 for c in hist_claims if c["verification"] == "UNKNOWN")
    total_hist = len(hist_claims)

    fp = (n_s / (n_s + n_c)) if (n_s + n_c) > 0 else None
    coverage = ((n_s + n_c) / (n_s + n_c + n_u)) if (n_s + n_c + n_u) > 0 else None
    fp_conservative = (n_s / (n_s + n_c + n_u)) if (n_s + n_c + n_u) > 0 else None

    # 2. Knowledge Recall
    matched_gold_ids = set()
    for c in hist_claims:
        if c["verification"] == "SUPPORTED" and c.get("matched_fact_id"):
            matched_gold_ids.add(c["matched_fact_id"])
    
    n_gold = len(gold_facts)
    kr = (len(matched_gold_ids) / n_gold) if n_gold > 0 else None

    # 3. Budgeted Knowledge Recall (BKR@5)
    first_5_hist = hist_claims[:5]
    matched_5_ids = set()
    for c in first_5_hist:
        if c["verification"] == "SUPPORTED" and c.get("matched_fact_id"):
            matched_5_ids.add(c["matched_fact_id"])
    denom_bkr = min(5, n_gold)
    bkr5 = (len(matched_5_ids) / denom_bkr) if denom_bkr > 0 else None

    # 4. Visual-Semantic Precision (VSP)
    vis_s = sum(1 for c in vis_claims if c["verification"] == "SUPPORTED")
    total_vis = len(vis_claims)
    vsp = (vis_s / total_vis) if total_vis > 0 else None

    # 5. Verified False Assertion Rate (VFAR)
    asserted_claims = [c for c in hist_claims if c["style"] == "ASSERTED"]
    a_s = sum(1 for c in asserted_claims if c["verification"] == "SUPPORTED")
    a_c = sum(1 for c in asserted_claims if c["verification"] == "CONTRADICTED")
    a_u = sum(1 for c in asserted_claims if c["verification"] == "UNKNOWN")
    
    vfar = (a_c / (a_s + a_c)) if (a_s + a_c) > 0 else None
    aur = (a_u / (a_s + a_c + a_u)) if (a_s + a_c + a_u) > 0 else None

    res = {
        "n_hist_claims": total_hist,
        "n_vis_claims": total_vis,
        "n_s": n_s,
        "n_c": n_c,
        "n_u": n_u,
        "fp": fp,
        "coverage": coverage,
        "fp_conservative": fp_conservative,
        "kr": kr,
        "bkr5": bkr5,
        "vsp": vsp,
        "a_s": a_s,
        "a_c": a_c,
        "a_u": a_u,
        "vfar": vfar,
        "aur": aur
    }

    # Additional metrics for E5/E6
    if assigned_regimes is not None:
        omit_facts = [f for f in assigned_regimes if f.get("policy") == "OMIT"]
        hedge_facts = [f for f in assigned_regimes if f.get("policy") == "HEDGE"]
        assert_facts = [f for f in assigned_regimes if f.get("policy") == "ASSERT"]
        
        # OVR: OMIT facts stated
        omit_stated = 0
        for of in omit_facts:
            tail = of.get("tail") or of.get("object") or ""
            if any(tail.lower() in c.get("claim", c.get("sentence", "")).lower() for c in hist_claims if len(tail) > 3):
                omit_stated += 1
        ovr = (omit_stated / len(omit_facts)) if omit_facts else None

        # OARgen: Asserting HEDGE or OMIT facts
        over_assertions = 0
        for f in (omit_facts + hedge_facts):
            tail = f.get("tail") or f.get("object") or ""
            if any(tail.lower() in c.get("claim", c.get("sentence", "")).lower() and c.get("style") == "ASSERTED" for c in hist_claims if len(tail) > 3):
                over_assertions += 1
        denom_oar = len(omit_facts) + len(hedge_facts)
        oargen = (over_assertions / denom_oar) if denom_oar > 0 else None

        # EAP: Claims supported by evidence packet
        eap_supported = 0
        for c in hist_claims:
            if any(match_fact_tail(f.get("tail", "") or f.get("object", ""), c.get("claim", c.get("sentence", "")).lower()) for f in assigned_regimes):
                eap_supported += 1
        eap = (eap_supported / total_hist) if total_hist > 0 else None

        res.update({
            "ovr": ovr,
            "oargen": oargen,
            "eap": eap
        })

    return res


def cluster_bootstrap_ci(
    values_by_source: Dict[str, List[Optional[float]]],
    b_resamples: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes 95% Confidence Interval using Hierarchical Cluster Bootstrap
    grouped by source_video_id.
    """
    rng = np.random.RandomState(seed)
    source_keys = list(values_by_source.keys())
    n_sources = len(source_keys)
    if n_sources == 0:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(b_resamples):
        sampled_srcs = rng.choice(source_keys, size=n_sources, replace=True)
        sampled_vals = []
        for src in sampled_srcs:
            for v in values_by_source[src]:
                if v is not None and not math.isnan(v):
                    sampled_vals.append(v)
        if sampled_vals:
            boot_means.append(float(np.mean(sampled_vals)))

    if not boot_means:
        return 0.0, 0.0, 0.0

    boot_means.sort()
    lower = float(np.percentile(boot_means, 2.5))
    upper = float(np.percentile(boot_means, 97.5))
    mean = float(np.mean(boot_means))
    return mean, lower, upper
