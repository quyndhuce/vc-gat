#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dempster-Shafer Evidential Fusion & Fuzzy Graph Alignment Framework.
Aligns Fuzzy Knowledge Graphs (F-KG) with Visual Scene Graphs (SG).
"""

import numpy as np

class DempsterShaferGraphMatcher:
    """
    Implements Dempster-Shafer Theory (DST) for evidential fusion 
    of visual, semantic, topological, and temporal evidence 
    to match Scene Graph nodes with Fuzzy Knowledge Graph nodes.
    """
    def __init__(self, conflict_threshold=0.8):
        self.conflict_threshold = conflict_threshold

    def construct_mass_function(self, scores, uncertainty_factor=0.2):
        """
        Convert raw similarity/confidence scores into a valid Basic Probability Assignment (BPA / Mass).
        scores: dict {candidate_v: raw_score}
        Returns: dict {candidate_v: mass, 'THETA': mass_theta}
        """
        mass = {}
        total = sum(scores.values())
        if total <= 1e-8:
            # Complete ignorance
            return {'THETA': 1.0}

        # Normalize valid scores to sum to (1 - uncertainty_factor)
        factor = (1.0 - uncertainty_factor) / total
        for cand, val in scores.items():
            mass[cand] = val * factor

        mass['THETA'] = uncertainty_factor
        return mass

    def dempster_combine_two(self, m1, m2):
        """
        Dempster's Rule of Combination for two mass functions m1 and m2.
        m_combined(A) = (1 / (1 - K)) * sum_{B \cap C = A} m1(B) * m2(C)
        K = sum_{B \cap C = \emptyset} m1(B) * m2(C)
        """
        combined = {}
        conflict_K = 0.0

        all_keys1 = list(m1.keys())
        all_keys2 = list(m2.keys())

        for k1 in all_keys1:
            for k2 in all_keys2:
                prob = m1[k1] * m2[k2]

                # Determine intersection set
                if k1 == 'THETA':
                    inter = k2
                elif k2 == 'THETA':
                    inter = k1
                elif k1 == k2:
                    inter = k1
                else:
                    inter = None # Empty set (Conflict)

                if inter is None:
                    conflict_K += prob
                else:
                    combined[inter] = combined.get(inter, 0.0) + prob

        # Check for extreme conflict (Dempster's Paradox)
        if conflict_K >= 1.0 - 1e-6:
            # Fallback to Murphy's Average Combination Rule under extreme conflict
            avg_mass = {}
            keys = set(m1.keys()).union(set(m2.keys()))
            for k in keys:
                avg_mass[k] = (m1.get(k, 0.0) + m2.get(k, 0.0)) / 2.0
            return avg_mass

        # Normalize by (1 - K)
        norm = 1.0 / (1.0 - conflict_K)
        for k in combined:
            combined[k] *= norm

        return combined

    def fuse_evidences(self, mass_functions_list):
        """Fuse an arbitrary list of mass functions sequentially."""
        if not mass_functions_list:
            return {'THETA': 1.0}

        current_m = mass_functions_list[0]
        for next_m in mass_functions_list[1:]:
            current_m = self.dempster_combine_two(current_m, next_m)
        return current_m

    def compute_belief_plausibility(self, fused_mass, candidates):
        """
        Compute Belief (Bel), Plausibility (Pl), and Pignistic Probability (BetP).
        """
        mass_theta = fused_mass.get('THETA', 0.0)
        num_cand = max(1, len(candidates))

        results = {}
        for c in candidates:
            bel = fused_mass.get(c, 0.0)
            pl = bel + mass_theta
            bet_p = bel + (mass_theta / num_cand)
            results[c] = {
                'Belief': bel,
                'Plausibility': pl,
                'PignisticProb': bet_p
            }
        return results

    def align_fuzzy_scene_graph(self, sg_nodes, fkg_nodes, sg_edges, fkg_edges,
                                vis_scores_mat, sem_scores_mat, topo_scores_mat, temp_scores_mat):
        """
        Full Evidential Fusion + Fuzzy Alignment Pipeline.
        """
        alignments = []
        n_sg = len(sg_nodes)
        n_fkg = len(fkg_nodes)

        for i in range(n_sg):
            sg_u = sg_nodes[i]
            candidates = [fkg_nodes[j]['id'] for j in range(n_fkg)]

            # 1. Build Mass Functions for 4 Evidence Sources
            vis_dict = {fkg_nodes[j]['id']: vis_scores_mat[i, j] for j in range(n_fkg)}
            sem_dict = {fkg_nodes[j]['id']: sem_scores_mat[i, j] for j in range(n_fkg)}
            topo_dict = {fkg_nodes[j]['id']: topo_scores_mat[i, j] for j in range(n_fkg)}
            temp_dict = {fkg_nodes[j]['id']: temp_scores_mat[i, j] for j in range(n_fkg)}

            m_vis = self.construct_mass_function(vis_dict, uncertainty_factor=0.15)
            m_sem = self.construct_mass_function(sem_dict, uncertainty_factor=0.20)
            m_topo = self.construct_mass_function(topo_dict, uncertainty_factor=0.25)
            m_temp = self.construct_mass_function(temp_dict, uncertainty_factor=0.10)

            # 2. Dempster Fusion
            fused_m = self.fuse_evidences([m_vis, m_sem, m_topo, m_temp])

            # 3. Compute Bel & Pl
            eval_res = self.compute_belief_plausibility(fused_m, candidates)

            # Select candidate with highest Pignistic Probability
            best_cand = max(candidates, key=lambda c: eval_res[c]['PignisticProb'])
            best_stats = eval_res[best_cand]

            alignments.append({
                'sg_node': sg_u['label'],
                'matched_fkg_node': best_cand,
                'Belief': best_stats['Belief'],
                'Plausibility': best_stats['Plausibility'],
                'PignisticProb': best_stats['PignisticProb'],
                'Fused_Mass': fused_m
            })

        return alignments

if __name__ == "__main__":
    print("Testing Dempster-Shafer Evidential Fusion for Fuzzy Graph Matching...")
    matcher = DempsterShaferGraphMatcher()

    # Toy Example: Scene Graph Node "bridge#0"
    sg_nodes = [{'label': 'bridge#0'}]
    fkg_nodes = [{'id': 'Thê Húc Bridge'}, {'id': 'Tràng Tiền Bridge'}, {'id': 'Long Biên Bridge'}]

    # Simulated Evidence Matrices
    vis_scores = np.array([[0.85, 0.10, 0.05]])  # Visual crop matches Thê Húc
    sem_scores = np.array([[0.90, 0.20, 0.15]])  # Text embedding matches Thê Húc
    topo_scores = np.array([[0.75, 0.30, 0.10]]) # Neighborhood matches Hoàn Kiếm Lake
    temp_scores = np.array([[0.95, 0.05, 0.00]]) # High intra-shot tracking stability

    alignments = matcher.align_fuzzy_scene_graph(
        sg_nodes, fkg_nodes, [], [],
        vis_scores, sem_scores, topo_scores, temp_scores
    )

    for a in alignments:
        print(f"\nSG Node: {a['sg_node']} ===> Matched F-KG: '{a['matched_fkg_node']}'")
        print(f"  • Belief (Bel):           {a['Belief']:.4f}")
        print(f"  • Plausibility (Pl):       {a['Plausibility']:.4f}")
        print(f"  • Pignistic Prob (BetP):   {a['PignisticProb']:.4f}")
        print(f"  • Complete Fused Mass m:   {a['Fused_Mass']}")
