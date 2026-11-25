"""
Build Global Merged Fuzzy Knowledge Graph (FKG).

Aggregates all 233 per-KB knowledge graphs into a single unified global graph:
  1. data/fkg_per_kb/fkg_global_merged.json (Unified SV & MV facts with Dempster/T-Conorm)
  2. data/fkg_per_kb/fkg_global_merged.ttl  (W3C RDF Turtle format for SPARQL/GraphDB/Neo4j)
  3. data/fkg_per_kb/fkg_global_networkx.pkl (NetworkX MultiDiGraph for GNN/TransMTL)
  4. data/fkg_per_kb/fkg_global_summary.json (Global statistics and distribution)
"""

import json
import logging
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("GlobalFKGMerger")

SINGLE_VALUED_RELATIONS = {
    "capital_of", "built_in", "founded_in", "inaugurated_in",
    "gifted_in", "has_height", "has_width", "has_length",
    "has_area", "recognized_in", "symbol_of", "former_name_of",
    "dated_to", "demolished_in", "renovated_in",
    # Fine-grained architectural attributes (single-valued)
    "has_number_of_levels", "has_number_of_columns", "has_shape",
    "has_roof_type", "has_roof_shape",
}


def t_conorm_algebraic(values: List[float]) -> float:
    """Probabilistic sum: 1 - ∏(1 - v)"""
    prod = 1.0
    for v in values:
        prod *= (1.0 - max(0.0, min(1.0, v)))
    return round(1.0 - prod, 5)


def fuse_global_fkg(
    base_dir: Path = Path("data/fkg_per_kb"),
    output_dir: Path = Path("data/fkg_per_kb"),
) -> Dict[str, Any]:
    index_path = base_dir / "kb_index.json"
    registry_path = base_dir / "shared_node_registry.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing {registry_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        kb_index = json.load(f)

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    uri_info = registry.get("uri_info", {})
    kbs = kb_index.get("kbs", {})

    LOGGER.info("Starting Global FKG Merge across %d KBs...", len(kbs))

    # Data structures for global fusion
    # (head_uri, relation) -> { tail_uri: [evidences] }
    global_sv_groups: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    # (head_uri, relation, tail_uri) -> [evidences]
    global_mv_facts: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    
    # Track head and tail surfaces
    surface_map: Dict[str, str] = {}
    all_nodes: Set[str] = set()

    kbs_processed = 0
    facts_loaded = 0

    for kb_id, info in kbs.items():
        kb_facts_path = base_dir / kb_id / "fkg_facts.json"
        if not kb_facts_path.exists():
            continue

        with open(kb_facts_path, "r", encoding="utf-8") as f:
            kb_data = json.load(f)

        # Collect nodes
        for node in kb_data.get("nodes", []):
            uri = node.get("uri")
            if uri:
                all_nodes.add(uri)
                if uri not in surface_map and node.get("canonical_surface"):
                    surface_map[uri] = node["canonical_surface"]

        # Collect SV groups
        for g in kb_data.get("single_valued_groups", []):
            h_uri = g.get("head_uri")
            rel = g.get("relation")
            if not h_uri or not rel:
                continue
            all_nodes.add(h_uri)
            if g.get("head_surface"):
                surface_map[h_uri] = g["head_surface"]

            key = (h_uri, rel)
            if key not in global_sv_groups:
                global_sv_groups[key] = {}

            for cand in g.get("candidates", []):
                t_uri = cand.get("tail_uri")
                if not t_uri:
                    continue
                all_nodes.add(t_uri)
                if cand.get("tail_surface"):
                    surface_map[t_uri] = cand["tail_surface"]

                if t_uri not in global_sv_groups[key]:
                    global_sv_groups[key][t_uri] = []
                
                # Append evidence
                for ev in cand.get("evidence", []):
                    global_sv_groups[key][t_uri].append(ev)
                    facts_loaded += 1

        # Collect MV facts
        for m in kb_data.get("multi_valued_facts", []):
            h_uri = m.get("head_uri")
            rel = m.get("relation")
            t_uri = m.get("tail_uri")
            if not h_uri or not rel or not t_uri:
                continue
            all_nodes.add(h_uri)
            all_nodes.add(t_uri)
            if m.get("head_surface"):
                surface_map[h_uri] = m["head_surface"]
            if m.get("tail_surface"):
                surface_map[t_uri] = m["tail_surface"]

            key = (h_uri, rel, t_uri)
            if key not in global_mv_facts:
                global_mv_facts[key] = []

            for ev in m.get("evidence", []):
                global_mv_facts[key].append(ev)
                facts_loaded += 1

        kbs_processed += 1

    LOGGER.info(
        "Collected raw evidence: %d SV group slots, %d MV fact slots from %d KBs",
        len(global_sv_groups), len(global_mv_facts), kbs_processed,
    )

    # 1. Global Dempster-Shafer fusion for SV groups
    fused_sv_groups = []
    conflict_count = 0

    for (h_uri, rel), cand_dict in global_sv_groups.items():
        candidates = []
        mass_assignments = []

        for t_uri, ev_list in cand_dict.items():
            # Combine evidence mass for this tail candidate
            conf_values = [
                ev.get("confidence", 0.5) * ev.get("authority", 0.8)
                for ev in ev_list
            ]
            m_val = t_conorm_algebraic(conf_values)
            m_val = min(0.99, max(0.01, m_val))
            mass_assignments.append((t_uri, m_val))

        # Dempster's rule across multiple candidates
        # For orthogonal hypothesis space:
        # K = sum_{i != j} m1(i) * m2(j)
        # Here we normalize masses if multiple candidates exist
        total_m = sum(m for _, m in mass_assignments)
        K = 0.0
        if len(mass_assignments) > 1:
            # Pairwise conflict estimation
            for i in range(len(mass_assignments)):
                for j in range(i + 1, len(mass_assignments)):
                    K += 2.0 * mass_assignments[i][1] * mass_assignments[j][1]
            K = min(1.0, K)

        if K >= 0.50:
            conflict_count += 1

        # Determine winner
        best_cand = max(mass_assignments, key=lambda x: x[1]) if mass_assignments else (None, 0.0)
        winner_uri, raw_bel = best_cand
        winner_bel = round(raw_bel * (1.0 - K * 0.5), 5)

        cand_output = []
        for t_uri, m_val in mass_assignments:
            cand_output.append({
                "tail_uri": t_uri,
                "tail_surface": surface_map.get(t_uri, t_uri),
                "bel": round(m_val, 5),
                "n_evidence": len(cand_dict[t_uri]),
                "methods": list(set(ev.get("method", "gliner_relex") for ev in cand_dict[t_uri])),
            })

        fused_sv_groups.append({
            "type": "single_valued",
            "head_uri": h_uri,
            "head_surface": surface_map.get(h_uri, h_uri),
            "relation": rel,
            "winner_uri": winner_uri,
            "winner_surface": surface_map.get(winner_uri, winner_uri) if winner_uri else None,
            "winner_bel": winner_bel,
            "conflict_K": round(K, 4),
            "needs_review": K >= 0.50,
            "candidates": cand_output,
        })

    # 2. Global T-Conorm fusion for MV facts
    fused_mv_facts = []
    for (h_uri, rel, t_uri), ev_list in global_mv_facts.items():
        masses = [
            ev.get("confidence", 0.5) * ev.get("authority", 0.8)
            for ev in ev_list
        ]
        mu_F = t_conorm_algebraic(masses)
        methods = list(set(ev.get("method", "gliner_relex") for ev in ev_list))
        max_conf = max((ev.get("confidence", 0.0) for ev in ev_list), default=0.0)

        fused_mv_facts.append({
            "type": "multi_valued",
            "head_uri": h_uri,
            "head_surface": surface_map.get(h_uri, h_uri),
            "relation": rel,
            "tail_uri": t_uri,
            "tail_surface": surface_map.get(t_uri, t_uri),
            "mu_F": mu_F,
            "max_confidence": round(max_conf, 4),
            "n_evidence": len(ev_list),
            "methods": methods,
        })

    # Sort MV facts by mu_F desc
    fused_mv_facts.sort(key=lambda x: x["mu_F"], reverse=True)

    LOGGER.info(
        "Global Fusion Complete: %d SV groups (%d conflicts K>=0.5), %d MV facts, %d unique nodes",
        len(fused_sv_groups), conflict_count, len(fused_mv_facts), len(all_nodes),
    )

    # 3. Export global JSON
    global_fkg_data = {
        "version": "1.0",
        "total_kbs": len(kbs),
        "total_nodes": len(all_nodes),
        "total_sv_groups": len(fused_sv_groups),
        "total_mv_facts": len(fused_mv_facts),
        "conflict_groups": conflict_count,
        "single_valued_groups": fused_sv_groups,
        "multi_valued_facts": fused_mv_facts,
    }
    json_path = output_dir / "fkg_global_merged.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(global_fkg_data, f, ensure_ascii=False, indent=2)
    LOGGER.info("Exported Global FKG JSON → %s", json_path)

    # 4. Export global NetworkX graph
    if HAS_NX:
        G = nx.MultiDiGraph()
        for uri in all_nodes:
            surface = surface_map.get(uri, uri)
            u_info = uri_info.get(uri, {})
            G.add_node(
                uri,
                label=surface,
                wikidata_id=u_info.get("wikidata_id", ""),
                mu_node=0.95 if u_info.get("wikidata_id") else 0.85,
            )
        for g in fused_sv_groups:
            if g.get("winner_uri"):
                G.add_edge(
                    g["head_uri"], g["winner_uri"],
                    relation=g["relation"],
                    mu=g["winner_bel"],
                    fusion="dempster",
                )
        for m in fused_mv_facts:
            G.add_edge(
                m["head_uri"], m["tail_uri"],
                relation=m["relation"],
                mu=m["mu_F"],
                fusion="t_conorm",
            )
        nx_path = output_dir / "fkg_global_networkx.pkl"
        with open(nx_path, "wb") as f:
            pickle.dump(G, f)
        LOGGER.info(
            "Exported Global NetworkX graph → %s (%d nodes, %d edges)",
            nx_path, G.number_of_nodes(), G.number_of_edges(),
        )

    # 5. Export W3C RDF Turtle (.ttl)
    ttl_path = output_dir / "fkg_global_merged.ttl"
    with open(ttl_path, "w", encoding="utf-8") as f:
        f.write("@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n")
        f.write("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n")
        f.write("@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n")
        f.write("@prefix fkg:  <http://transmtl.vn/ontology/> .\n")
        f.write("@prefix ent:  <http://transmtl.vn/ent/> .\n")
        f.write("@prefix kb:   <http://transmtl.vn/kb/> .\n\n")

        # Nodes
        for uri in sorted(all_nodes):
            surface = surface_map.get(uri, "")
            safe_uri = f"<{uri}>"
            f.write(f"{safe_uri} a fkg:Entity ;\n")
            if surface:
                safe_label = surface.replace('"', '\\"')
                f.write(f'    rdfs:label "{safe_label}"@en ;\n')
            u_info = uri_info.get(uri, {})
            if u_info.get("wikidata_id"):
                f.write(f'    fkg:wikidataId "{u_info["wikidata_id"]}" ;\n')
            f.write("    fkg:inFKG true .\n\n")

        # SV Facts (winners)
        for g in fused_sv_groups:
            if g.get("winner_uri"):
                h = f"<{g['head_uri']}>"
                t = f"<{g['winner_uri']}>"
                rel = f"fkg:{g['relation']}"
                f.write(f"{h} {rel} {t} .\n")

        # MV Facts
        for m in fused_mv_facts:
            h = f"<{m['head_uri']}>"
            t = f"<{m['tail_uri']}>"
            rel = f"fkg:{m['relation']}"
            f.write(f"{h} {rel} {t} .\n")

    LOGGER.info("Exported Global RDF Turtle (.ttl) → %s", ttl_path)

    # 6. Global Summary
    summary = {
        "total_kbs_merged": len(kbs),
        "total_unique_nodes": len(all_nodes),
        "total_single_valued_winners": len([g for g in fused_sv_groups if g.get("winner_uri")]),
        "total_multi_valued_facts": len(fused_mv_facts),
        "total_facts": len(fused_sv_groups) + len(fused_mv_facts),
        "conflict_count": conflict_count,
        "files_generated": [
            str(json_path),
            str(output_dir / "fkg_global_networkx.pkl"),
            str(ttl_path),
        ],
    }
    with open(output_dir / "fkg_global_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    LOGGER.info("Global FKG Merge completed successfully!")
    return summary


if __name__ == "__main__":
    fuse_global_fkg()
