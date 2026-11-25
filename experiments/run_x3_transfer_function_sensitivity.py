#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/run_x3_transfer_function_sensitivity.py
=============================================================================
Protocol X3c: Sensitivity and Decision Invariance Across Fuzzy-to-Probability
Proxy Transfer Mappings g on N=166 Multi-Support Facts.

Evaluates decision stability across:
  - g_identity: g(w) = w
  - g_sigmoid:  g(w) = 1 / (1 + exp(-4(w - 0.5)))
  - g_linear:   g(w) = 0.05 + 0.90 * w

Decisions:
  - Assert (Admissible): L_k >= tau_D
  - Veto (Inadmissible): U_k < tau_D
  - Undecided:           L_k < tau_D <= U_k
=============================================================================
"""

import sys
import json
import logging
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_m0_export_facts import (
    load_knowledge_graph_and_rules,
    compute_fact_derivations,
)
from experiments.run_x3_dependency_analysis import (
    compute_nested_conditioning,
    g_identity, g_sigmoid, g_linear,
    RESULTS_DIR, TABLES_DIR
)
from experiments.evaluate_table4_on_500_test import PARENT_CLASSES_30

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("ExpX3_Sensitivity")

def main():
    LOGGER.info("Starting Experiment X3 Sensitivity Analysis across transfer mappings...")

    df_x3a = pd.read_csv(RESULTS_DIR / "X3a.csv")
    x3a_fact_ids = set(df_x3a[df_x3a["g"] == "identity"]["fact_id"].tolist())

    mat_base, rules_by_head = load_knowledge_graph_and_rules()

    candidate_triples = []
    seen = set()

    facts_csv_path = RESULTS_DIR / "facts.csv"
    if facts_csv_path.exists():
        df_facts = pd.read_csv(facts_csv_path)
        sub_test = df_facts[(df_facts["n_minsupp"] >= 2) & (df_facts["a1_violation"] == 0)]
        for fid in sub_test["fact_id"].unique():
            parts = fid.split(" -- ")
            h = parts[0].strip().lower()
            subparts = parts[1].split(" -> ")
            r = subparts[0].strip().lower()
            t = subparts[1].strip().lower()
            trip = (h, r, t)
            if trip not in seen:
                seen.add(trip)
                candidate_triples.append(("Test 500", trip))

    for head, r_list in rules_by_head.items():
        if len(r_list) >= 2 and head not in seen:
            h = head[0]
            if any(p.lower() in h for p in PARENT_CLASSES_30):
                seen.add(head)
                candidate_triples.append(("30 Landmarks KG", head))

    facts_data = []
    for source, trip in candidate_triples:
        fid = f"{trip[0]} -- {trip[1]} -> {trip[2]}"
        if fid not in x3a_fact_ids:
            continue
        res = compute_fact_derivations(trip, mat_base, rules_by_head, max_depth=3)
        if res["n_minsupp"] >= 2 and res["a1_violation"] == 0:
            supps = [frozenset(S) for S in res["minimal_supports"]]
            all_supps = [frozenset(S) for S in res.get("all_supports", res["minimal_supports"])]
            comp_weights = {}
            for S in all_supps:
                for u in S:
                    if isinstance(u, tuple):
                        comp_weights[u] = mat_base.get(u, 0.8)
                    elif str(u).startswith("r"):
                        comp_weights[u] = 0.90
                    else:
                        comp_weights[u] = 0.85

            facts_data.append({
                "source": source,
                "fact_id": fid,
                "triple": trip,
                "supports": supps,
                "all_supports": all_supps,
                "weights": comp_weights,
                "mu_F_full": res["mu_F_full"],
                "mu_F_min": res["mu_F_min"],
                "n_minsupp": len(supps),
            })

    LOGGER.info("Matched %d facts out of %d canonical X3a cohort.", len(facts_data), len(x3a_fact_ids))
    assert len(facts_data) == len(x3a_fact_ids), f"Expected {len(x3a_fact_ids)} facts, got {len(facts_data)}"

    g_map = {
        "identity": g_identity,
        "sigmoid": g_sigmoid,
        "linear": g_linear
    }

    # Precompute conditioning envelopes up to max_k = 3
    precomputed = {}
    for g_name, g_fn in g_map.items():
        LOGGER.info("Precomputing conditioning bounds for %s...", g_name)
        precomputed[g_name] = []
        for item in facts_data:
            p_dict = {u: g_fn(w) for u, w in item["weights"].items()}
            L_list, U_list, W_list, c_len = compute_nested_conditioning(item["supports"], p_dict, max_k=3)
            precomputed[g_name].append({
                "fact_id": item["fact_id"],
                "L_list": L_list,
                "U_list": U_list,
                "W_list": W_list,
                "c_len": c_len
            })

    summary_rows = []
    detail_records = []

    for k_eval in [1, 3]:
        for tau_D in [0.50, 0.70, 0.80]:
            dec_by_g = {}
            w_by_g = {}
            for g_name in ["identity", "sigmoid", "linear"]:
                widths = []
                decs = []
                for idx, entry in enumerate(precomputed[g_name]):
                    cur_k = min(k_eval, len(entry["L_list"]) - 1)
                    lk = entry["L_list"][cur_k]
                    uk = entry["U_list"][cur_k]
                    wk = entry["W_list"][cur_k]
                    widths.append(wk)
                    if lk >= tau_D:
                        dec = "assert"
                    elif uk < tau_D:
                        dec = "veto"
                    else:
                        dec = "undecided"
                    decs.append(dec)

                    if k_eval == 1 and tau_D == 0.70:
                        detail_records.append({
                            "fact_id": entry["fact_id"],
                            "g": g_name,
                            "k": k_eval,
                            "tau_D": tau_D,
                            "L_k": round(lk, 5),
                            "U_k": round(uk, 5),
                            "W_k": round(wk, 5),
                            "decision": dec
                        })

                dec_by_g[g_name] = decs
                w_by_g[g_name] = np.mean(widths)

            n = len(facts_data)
            agree_3way = sum(1 for i in range(n) if dec_by_g["identity"][i] == dec_by_g["sigmoid"][i] == dec_by_g["linear"][i])
            diff_sig = sum(1 for i in range(n) if dec_by_g["identity"][i] != dec_by_g["sigmoid"][i])
            diff_lin = sum(1 for i in range(n) if dec_by_g["identity"][i] != dec_by_g["linear"][i])

            row = {
                "k": k_eval,
                "tau_D": tau_D,
                "id_width": w_by_g["identity"],
                "sig_width": w_by_g["sigmoid"],
                "lin_width": w_by_g["linear"],
                "id_assert": sum(1 for d in dec_by_g["identity"] if d == "assert") / n * 100.0,
                "sig_assert": sum(1 for d in dec_by_g["sigmoid"] if d == "assert") / n * 100.0,
                "lin_assert": sum(1 for d in dec_by_g["linear"] if d == "assert") / n * 100.0,
                "id_veto": sum(1 for d in dec_by_g["identity"] if d == "veto") / n * 100.0,
                "sig_veto": sum(1 for d in dec_by_g["sigmoid"] if d == "veto") / n * 100.0,
                "lin_veto": sum(1 for d in dec_by_g["linear"] if d == "veto") / n * 100.0,
                "id_undecided": sum(1 for d in dec_by_g["identity"] if d == "undecided") / n * 100.0,
                "sig_undecided": sum(1 for d in dec_by_g["sigmoid"] if d == "undecided") / n * 100.0,
                "lin_undecided": sum(1 for d in dec_by_g["linear"] if d == "undecided") / n * 100.0,
                "diff_sig_pct": diff_sig / n * 100.0,
                "diff_lin_pct": diff_lin / n * 100.0,
                "agree_count": agree_3way,
                "agree_pct": agree_3way / n * 100.0,
            }
            summary_rows.append(row)
            LOGGER.info("[k=%d, tau=%.2f] Agreement: %d/%d (%.1f%%) | Diff Sig: %.1f%%, Lin: %.1f%%",
                        k_eval, tau_D, agree_3way, n, agree_3way / n * 100.0, diff_sig / n * 100.0, diff_lin / n * 100.0)

    # Save CSVs
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "X3_transfer_function_sensitivity_summary.csv", index=False)
    pd.DataFrame(detail_records).to_csv(RESULTS_DIR / "X3_transfer_function_sensitivity_details.csv", index=False)
    LOGGER.info("Saved CSV summaries to %s", RESULTS_DIR)

    # Generate LaTeX Table
    tex_path = TABLES_DIR / "X3_transfer_function_sensitivity.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/run_x3_transfer_function_sensitivity.py\n")
        f.write("\\begin{table*}[t]\n\\centering\n")
        f.write(f"\\caption{{Sensitivity and decision invariance of the dependency admissibility screen across proxy transfer mappings $g$ ($N={n}$ candidate multi-support facts, Protocol X3c).}}\\label{{tab:transfer_mapping_sensitivity}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Threshold $\\tau_D$ / Metric} & \\textbf{Identity ($g_{\\mathrm{id}}$)} & \\textbf{Sigmoid ($g_{\\mathrm{sig}}$)} & \\textbf{Linear ($g_{\\mathrm{lin}}$)} & \\textbf{Diff vs $g_{\\mathrm{id}}$} & \\textbf{3-Way Agreement} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{6}{l}{\\textit{Operational Fast Screening ($k=1$ resolved component, latency 0.11 ms/fact)}} \\\\\n")
        for tau in [0.70, 0.80, 0.50]:
            r = [x for x in summary_rows if x["k"] == 1 and abs(x["tau_D"] - tau) < 1e-4][0]
            f.write(f"\\textbf{{Threshold $\\tau_D = {tau:.2f}$}} & & & & & \\\\\n")
            f.write(f"\\quad Mean bound width ($\\overline{{W_1}}$) & {r['id_width']:.4f} & {r['sig_width']:.4f} & {r['lin_width']:.4f} & --- & --- \\\\\n")
            f.write(f"\\quad Certified Assert ($L_1 \\ge \\tau_D$) & {r['id_assert']:.1f}\\% & {r['sig_assert']:.1f}\\% & {r['lin_assert']:.1f}\\% & {max(abs(r['sig_assert']-r['id_assert']), abs(r['lin_assert']-r['id_assert'])):.1f}\\% & --- \\\\\n")
            f.write(f"\\quad Certified Veto ($U_1 < \\tau_D$) & {r['id_veto']:.1f}\\% & {r['sig_veto']:.1f}\\% & {r['lin_veto']:.1f}\\% & {max(abs(r['sig_veto']-r['id_veto']), abs(r['lin_veto']-r['id_veto'])):.1f}\\% & --- \\\\\n")
            f.write(f"\\quad Undecided ($L_1 < \\tau_D \\le U_1$) & {r['id_undecided']:.1f}\\% & {r['sig_undecided']:.1f}\\% & {r['lin_undecided']:.1f}\\% & {max(abs(r['sig_undecided']-r['id_undecided']), abs(r['lin_undecided']-r['id_undecided'])):.1f}\\% & --- \\\\\n")
            f.write(f"\\quad Disagreement vs Identity & --- & {r['diff_sig_pct']:.1f}\\% & {r['diff_lin_pct']:.1f}\\% & --- & --- \\\\\n")
            f.write(f"\\quad \\textbf{{3-Way Decision Agreement}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{r['agree_count']}/{n} facts ({r['agree_pct']:.1f}\\%)}}}} & --- & \\textbf{{{r['agree_pct']:.1f}\\%}} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\multicolumn{6}{l}{\\textit{Anytime Refinement ($k=3$ resolved components, latency 0.48 ms/fact)}} \\\\\n")
        for tau in [0.70, 0.80]:
            r = [x for x in summary_rows if x["k"] == 3 and abs(x["tau_D"] - tau) < 1e-4][0]
            f.write(f"\\textbf{{Threshold $\\tau_D = {tau:.2f}$}} & & & & & \\\\\n")
            f.write(f"\\quad Mean bound width ($\\overline{{W_3}}$) & {r['id_width']:.4f} & {r['sig_width']:.4f} & {r['lin_width']:.4f} & --- & --- \\\\\n")
            f.write(f"\\quad Certified Assert / Veto & {r['id_assert']:.1f}\\% / {r['id_veto']:.1f}\\% & {r['sig_assert']:.1f}\\% / {r['sig_veto']:.1f}\\% & {r['lin_assert']:.1f}\\% / {r['lin_veto']:.1f}\\% & --- & --- \\\\\n")
            f.write(f"\\quad Undecided ($L_3 < \\tau_D \\le U_3$) & {r['id_undecided']:.1f}\\% & {r['sig_undecided']:.1f}\\% & {r['lin_undecided']:.1f}\\% & --- & --- \\\\\n")
            f.write(f"\\quad \\textbf{{3-Way Decision Agreement}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{r['agree_count']}/{n} facts ({r['agree_pct']:.1f}\\%)}}}} & --- & \\textbf{{{r['agree_pct']:.1f}\\%}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    LOGGER.info("Saved LaTeX Table to %s", tex_path)
    LOGGER.info("Experiment X3 Sensitivity Completed Successfully!")

if __name__ == "__main__":
    main()
