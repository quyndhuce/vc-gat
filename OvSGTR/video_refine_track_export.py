#!/usr/bin/env python3
"""
Stage 2 of Video SGG Pipeline:
1. Load intermediate detections from detector_detections.json.
2. Run selective SigLIP 2 refinement on architectural & cultural boxes using vn_tourism_visual_universe.json.
3. Compute SigLIP 2 visual embeddings for each detection.
4. Perform cross-frame Object Tracking within each scene to unify entity IDs across keyframes.
5. Update relations with persistent track IDs and refined concept labels.
6. Export structured Scene Graph JSON for each scene (scene_XX_scene_graph.json) and a master video summary.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Domain Triage Policy
# ---------------------------------------------------------------------------
NATURE_PRIMITIVES = {
    'water', 'lake', 'river', 'sea', 'ocean', 'stream', 'pond', 'lagoon', 'waterfall',
    'tree', 'trees', 'plant', 'plants', 'leaf', 'leaves', 'forest', 'grass', 'bush', 'branch',
    'sky', 'cloud', 'clouds', 'mountain', 'hill', 'ground', 'dirt', 'sand', 'cliff', 'rock',
    'person', 'people', 'man', 'woman'
}

ARCHITECTURAL_AND_CULTURAL_KEYWORDS = {
    'bridge', 'temple', 'pagoda', 'building', 'house', 'palace', 'citadel',
    'shrine', 'tower', 'stupa', 'church', 'gate', 'pavilion',
    'roof', 'wall', 'pillar', 'column', 'door', 'window', 'balcony', 'railing',
    'stairs', 'tile', 'arch', 'courtyard', 'statue', 'stele', 'altar', 'bell', 'drum'
}


def should_refine_object(label: str) -> bool:
    label_clean = label.lower().strip()
    words = label_clean.replace('_', ' ').replace('-', ' ').split()
    for w in words:
        if w in NATURE_PRIMITIVES:
            return False
    for w in words:
        if w in ARCHITECTURAL_AND_CULTURAL_KEYWORDS:
            return True
    return False


# ---------------------------------------------------------------------------
# Visual Universe Loader
# ---------------------------------------------------------------------------
def load_visual_universe(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get("meta", {})
    templates = meta.get("prompt_templates", ["a photo of {label}"])

    concepts = []
    for bg in data.get("background_class", []):
        concepts.append({
            "id": bg["id"],
            "label_en": bg["label_en"],
            "description": bg["description"],
            "group": "background"
        })

    for group_name, items in data.get("object_concepts", {}).items():
        for item in items:
            concepts.append({
                "id": item["id"],
                "label_en": item["label_en"],
                "description": item["description"],
                "group": group_name
            })

    materials = []
    for attr in data.get("attribute_variables", []):
        if attr["id"] == "material":
            materials = attr["values"]
            break

    colours = []
    for attr in data.get("attribute_variables", []):
        if attr["id"] == "dominant_colour":
            colours = attr["values"]
            break

    return concepts, materials, colours, templates, data


# ---------------------------------------------------------------------------
# SigLIP 2 Scorer & Feature Extractor
# ---------------------------------------------------------------------------
class Siglip2Engine:
    def __init__(self, model_id="google/siglip2-base-patch16-224", device=None):
        from transformers import AutoProcessor, AutoModel

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Initializing SigLIP 2 ({model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

        self.concepts = None
        self.concept_embeds = None
        self.materials = None
        self.material_embeds = None
        self.colours = None
        self.colour_embeds = None

    def _encode_text_batch(self, texts):
        inputs = self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            embeds = out.pooler_output if hasattr(out, "pooler_output") else out
            return embeds / embeds.norm(dim=-1, keepdim=True)

    def index_universe(self, concepts, materials, colours, templates):
        self.concepts = concepts
        self.materials = materials
        self.colours = colours

        print(f"[INFO] Indexing {len(concepts)} concepts with SigLIP 2...")
        t_embeds = []
        for tpl in templates:
            texts = [tpl.replace("{label}", c["label_en"]).replace("{description}", c["description"]) for c in concepts]
            t_embeds.append(self._encode_text_batch(texts))
        self.concept_embeds = torch.stack(t_embeds, dim=0).mean(dim=0)
        self.concept_embeds = self.concept_embeds / self.concept_embeds.norm(dim=-1, keepdim=True)

        if materials:
            m_texts = [f"made of {m['label_en']}" for m in materials]
            self.material_embeds = self._encode_text_batch(m_texts)

        if colours:
            c_texts = [f"a {c['label_en']} object" for c in colours]
            self.colour_embeds = self._encode_text_batch(c_texts)

        print("[INFO] SigLIP 2 universe indexing complete!")

    @torch.no_grad()
    def get_image_embedding_and_scores(self, img_pil):
        inputs = self.processor(images=[img_pil], return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        img_feat = out.pooler_output if hasattr(out, "pooler_output") else out
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Concept probabilities
        sims = (img_feat @ self.concept_embeds.T).squeeze(0)
        scale = getattr(self.model, "logit_scale", torch.tensor(10.0)).exp().clamp(max=100.0)
        c_probs = torch.softmax(sims * scale, dim=-1)

        # Material
        mat_res = None
        if self.material_embeds is not None:
            m_sims = (img_feat @ self.material_embeds.T).squeeze(0)
            m_probs = torch.softmax(m_sims * scale, dim=-1)
            top_m = torch.argmax(m_probs).item()
            mat_res = {
                "id": self.materials[top_m]["id"],
                "label_en": self.materials[top_m]["label_en"],
                "confidence": round(float(m_probs[top_m].item()), 4)
            }

        # Colour
        col_res = None
        if self.colour_embeds is not None:
            c_sims = (img_feat @ self.colour_embeds.T).squeeze(0)
            c_probs_col = torch.softmax(c_sims * scale, dim=-1)
            top_c = torch.argmax(c_probs_col).item()
            col_res = {
                "id": self.colours[top_c]["id"],
                "label_en": self.colours[top_c]["label_en"],
                "confidence": round(float(c_probs_col[top_c].item()), 4)
            }

        feat_np = img_feat.squeeze(0).cpu().numpy()
        return feat_np, c_probs.cpu(), mat_res, col_res


# ---------------------------------------------------------------------------
# Cross-Frame Object Tracker within a Scene
# ---------------------------------------------------------------------------
def compute_iou(box1, box2):
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])

    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def track_objects_in_scene(keyframes_data, iou_weight=0.4, vis_weight=0.4, class_weight=0.2, match_threshold=0.35):
    """
    Assign persistent track_ids across keyframes within a single scene.
    Each detection receives 'track_id' (e.g. 'T01', 'T02'...).
    """
    track_counter = 1
    active_tracks = {}  # track_id -> dict(last_box, last_feat, label, history)

    for kf in keyframes_data:
        detections = kf["nodes"]
        if not detections:
            continue

        if not active_tracks:
            # First keyframe with detections: initialize all tracks
            for det in detections:
                tid = f"T{track_counter:02d}"
                track_counter += 1
                det["track_id"] = tid
                active_tracks[tid] = {
                    "track_id": tid,
                    "last_box": det["box_2d"],
                    "last_feat": det.get("visual_feature", None),
                    "label": det["detector_label"],
                    "siglip2_label": det.get("siglip2_label", det["detector_label"]),
                    "history": [(kf["frame_idx"], det["box_2d"])]
                }
            continue

        # Match current detections with active tracks
        track_ids = list(active_tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t_box = active_tracks[tid]["last_box"]
            t_feat = active_tracks[tid]["last_feat"]
            t_lbl = active_tracks[tid]["label"]

            for j, det in enumerate(detections):
                d_box = det["box_2d"]
                d_feat = det.get("visual_feature", None)
                d_lbl = det["detector_label"]

                iou = compute_iou(t_box, d_box)

                if t_feat is not None and d_feat is not None:
                    vis_sim = float(np.dot(t_feat, d_feat))
                    vis_sim = max(0.0, min(1.0, vis_sim))
                else:
                    vis_sim = 0.5

                cls_sim = 1.0 if t_lbl == d_lbl else 0.0

                affinity = iou_weight * iou + vis_weight * vis_sim + class_weight * cls_sim
                cost_matrix[i, j] = affinity

        # Hungarian assignment (maximize affinity)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            aff = cost_matrix[r, c]
            tid = track_ids[r]
            det = detections[c]

            # Condition to accept match: sufficient affinity AND same label (or very high visual similarity)
            if aff >= match_threshold and (det["detector_label"] == active_tracks[tid]["label"] or aff >= 0.65):
                det["track_id"] = tid
                matched_tracks.add(tid)
                matched_dets.add(c)
                # Update track state
                active_tracks[tid]["last_box"] = det["box_2d"]
                if det.get("visual_feature") is not None:
                    active_tracks[tid]["last_feat"] = det["visual_feature"]
                active_tracks[tid]["history"].append((kf["frame_idx"], det["box_2d"]))

        # Unmatched detections: start new tracks
        for j, det in enumerate(detections):
            if j not in matched_dets:
                tid = f"T{track_counter:02d}"
                track_counter += 1
                det["track_id"] = tid
                active_tracks[tid] = {
                    "track_id": tid,
                    "last_box": det["box_2d"],
                    "last_feat": det.get("visual_feature", None),
                    "label": det["detector_label"],
                    "siglip2_label": det.get("siglip2_label", det["detector_label"]),
                    "history": [(kf["frame_idx"], det["box_2d"])]
                }

    return active_tracks


# ---------------------------------------------------------------------------
# Main Stage 2 Execution
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Stage 2: SigLIP 2 Refinement, Object Tracking & Per-Scene JSON Export")
    p.add_argument('--input_json', type=str, default='video_output/detector_detections.json', help="Output from Stage 1")
    p.add_argument('--universe', type=str, default='vn_tourism_visual_universe.json', help="Visual universe taxonomy JSON")
    p.add_argument('--output_dir', type=str, default='video_output', help="Directory to save final scene graph JSONs")
    p.add_argument('--model_id', type=str, default='google/siglip2-base-patch16-224')
    p.add_argument('--edge_conf_threshold', type=float, default=0.050, help="Confidence threshold for relation edges")
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    assert os.path.exists(args.input_json), f"Input JSON not found: {args.input_json}"
    assert os.path.exists(args.universe), f"Visual universe not found: {args.universe}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)

    # 1. Load Visual Universe & Initialize SigLIP 2
    concepts, materials, colours, templates, raw_universe = load_visual_universe(args.universe)
    engine = Siglip2Engine(model_id=args.model_id, device=args.device)
    engine.index_universe(concepts, materials, colours, templates)

    scenes = stage1_data["scenes"]
    print(f"\n[INFO] Starting refinement & tracking on {len(scenes)} scenes...")

    all_scenes_summary = []

    for scene in scenes:
        s_id = scene["scene_id"]
        print(f"\n{'='*75}\n  Processing {s_id} ({scene['start_seconds']}s - {scene['end_seconds']}s, {len(scene['keyframes'])} keyframes)\n{'='*75}")

        # Step A: Refine detections on keyframes with SigLIP 2
        for kf in scene["keyframes"]:
            img_path = kf["image_path"]
            if not os.path.exists(img_path):
                continue
            img_pil = Image.open(img_path).convert("RGB")
            w, h = img_pil.size

            for node in kf["nodes"]:
                lbl = node["detector_label"]
                b = node["box_2d"]
                x0, y0, x1, y1 = b
                bw, bh = x1 - x0, y1 - y0
                mx, my = bw * 0.05, bh * 0.05
                cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
                cx1, cy1 = min(w, int(x1 + mx)), min(h, int(y1 + my))

                crop = img_pil.crop((cx0, cy0, cx1, cy1))

                needs_refine = should_refine_object(lbl)
                feat, c_probs, mat_res, col_res = engine.get_image_embedding_and_scores(crop)
                node["visual_feature"] = feat

                if needs_refine:
                    topk = torch.topk(c_probs, 2)
                    top1_c = concepts[topk.indices[0].item()]
                    top1_sc = float(topk.values[0].item())
                    top2_c = concepts[topk.indices[1].item()]
                    top2_sc = float(topk.values[1].item())

                    node["is_refined_by_siglip2"] = True
                    node["triage_type"] = "architectural_or_cultural"
                    node["siglip2_label"] = top1_c["label_en"]
                    node["siglip2_concept_id"] = top1_c["id"]
                    node["refined_concept"] = {
                        "id": top1_c["id"],
                        "label_en": top1_c["label_en"],
                        "description": top1_c["description"],
                        "group": top1_c["group"],
                        "siglip2_confidence": round(top1_sc, 4),
                        "runner_up": {
                            "id": top2_c["id"],
                            "label_en": top2_c["label_en"],
                            "siglip2_confidence": round(top2_sc, 4)
                        }
                    }
                    node["attributes"] = {
                        "material": mat_res,
                        "dominant_colour": col_res
                    }
                else:
                    node["is_refined_by_siglip2"] = False
                    node["triage_type"] = "natural_primitive_or_person"
                    node["siglip2_label"] = lbl
                    node["siglip2_concept_id"] = lbl
                    node["refined_concept"] = {
                        "id": lbl,
                        "label_en": lbl,
                        "group": "natural_primitive",
                        "siglip2_confidence": None,
                        "note": "Bypassed SigLIP 2 to preserve natural scene primitive"
                    }
                    node["attributes"] = {
                        "material": None,
                        "dominant_colour": None
                    }

        # Step B: Perform Object Tracking across keyframes in this scene
        active_tracks = track_objects_in_scene(scene["keyframes"])
        print(f"  ✓ Tracking complete: {len(active_tracks)} unique entities tracked across keyframes")

        # Step C: Build entity directory for this scene
        tracked_entities = []
        for tid, tinfo in sorted(active_tracks.items()):
            # Aggregate entity attributes from frames
            observed_nodes = []
            for kf in scene["keyframes"]:
                for node in kf["nodes"]:
                    if node.get("track_id") == tid:
                        observed_nodes.append((kf["frame_idx"], node))

            if not observed_nodes:
                continue

            # Pick highest score node for canonical info
            best_frame, best_node = max(observed_nodes, key=lambda x: x[1]["detector_score"])

            entity_record = {
                "track_id": tid,
                "detector_label": best_node["detector_label"],
                "detector_score": best_node["detector_score"],
                "siglip2_label": best_node["siglip2_label"],
                "siglip2_concept_id": best_node["siglip2_concept_id"],
                "is_refined_by_siglip2": best_node["is_refined_by_siglip2"],
                "attributes": best_node["attributes"],
                "frames_observed": [f for f, n in observed_nodes],
                "trajectory": [
                    {"frame_idx": f, "box_2d": n["box_2d"], "score": n["detector_score"]}
                    for f, n in observed_nodes
                ]
            }
            tracked_entities.append(entity_record)

        # Step D: Update Edges with Track IDs and Refined Labels
        scene_relations_summary = {}

        for kf in scene["keyframes"]:
            node_map = {n["node_idx"]: n for n in kf["nodes"]}
            updated_edges = []

            for e in kf.get("edges", []):
                s_node = node_map.get(e["subject_node_idx"])
                o_node = node_map.get(e["object_node_idx"])
                if not s_node or not o_node:
                    continue

                s_tid = s_node.get("track_id", f"node_{s_node['node_idx']}")
                o_tid = o_node.get("track_id", f"node_{o_node['node_idx']}")

                if s_tid == o_tid:
                    continue  # suppress self-relation

                conf = e["confidence"]
                if conf < args.edge_conf_threshold:
                    continue

                s_det_lbl = s_node["detector_label"]
                s_sig_lbl = s_node["siglip2_label"]
                o_det_lbl = o_node["detector_label"]
                o_sig_lbl = o_node["siglip2_label"]
                pred = e["predicate"]

                edge_dict = {
                    "subject_track_id": s_tid,
                    "subject_detector_label": s_det_lbl,
                    "subject_siglip2_label": s_sig_lbl,
                    "predicate": pred,
                    "object_track_id": o_tid,
                    "object_detector_label": o_det_lbl,
                    "object_siglip2_label": o_sig_lbl,
                    "confidence": conf,
                    "rel_score": e["rel_score"],
                    "detector_triple": f"{s_tid} ({s_det_lbl}) -> {pred} -> {o_tid} ({o_det_lbl})",
                    "siglip2_refined_triple": f"{s_tid} ({s_sig_lbl}) -> {pred} -> {o_tid} ({o_sig_lbl})"
                }
                updated_edges.append(edge_dict)

                # Aggregate to scene level
                triple_key = (s_tid, pred, o_tid)
                if triple_key not in scene_relations_summary:
                    scene_relations_summary[triple_key] = {
                        "subject_track_id": s_tid,
                        "subject_siglip2_label": s_sig_lbl,
                        "predicate": pred,
                        "object_track_id": o_tid,
                        "object_siglip2_label": o_sig_lbl,
                        "refined_triple": f"{s_tid} ({s_sig_lbl}) -> {pred} -> {o_tid} ({o_sig_lbl})",
                        "max_confidence": conf,
                        "frames_observed": [kf["frame_idx"]]
                    }
                else:
                    scene_relations_summary[triple_key]["max_confidence"] = max(
                        scene_relations_summary[triple_key]["max_confidence"], conf
                    )
                    scene_relations_summary[triple_key]["frames_observed"].append(kf["frame_idx"])

            kf["edges"] = updated_edges

            # Clean visual_feature array for JSON serialization
            for node in kf["nodes"]:
                node.pop("visual_feature", None)

        # Step E: Export Scene Graph JSON for this scene
        scene_json_path = out_dir / f"{s_id}_scene_graph.json"
        scene_export_data = {
            "scene_metadata": {
                "scene_id": s_id,
                "scene_index": scene["scene_index"],
                "video_path": stage1_data["video_path"],
                "start_frame": scene["start_frame"],
                "end_frame": scene["end_frame"],
                "start_seconds": scene["start_seconds"],
                "end_seconds": scene["end_seconds"],
                "duration_seconds": scene["duration_seconds"],
                "num_keyframes": scene["num_keyframes"],
                "fps": stage1_data["fps"]
            },
            "tracked_entities": tracked_entities,
            "scene_level_relations": sorted(list(scene_relations_summary.values()), key=lambda x: -x["max_confidence"]),
            "keyframes": scene["keyframes"]
        }

        with open(scene_json_path, "w", encoding="utf-8") as f:
            json.dump(scene_export_data, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Exported: {scene_json_path}")

        all_scenes_summary.append({
            "scene_id": s_id,
            "time_range": f"{scene['start_seconds']}s - {scene['end_seconds']}s",
            "duration": f"{scene['duration_seconds']}s",
            "json_path": str(scene_json_path),
            "num_entities": len(tracked_entities),
            "num_relations": len(scene_relations_summary),
            "top_entities": [f"{e['track_id']}: {e['siglip2_label']}" for e in tracked_entities[:5]],
            "top_triplets": [r["refined_triple"] for r in sorted(list(scene_relations_summary.values()), key=lambda x: -x["max_confidence"])[:4]]
        })

    # Step F: Export Master Video Summary JSON
    master_summary_path = out_dir / "video_scene_graphs_summary.json"
    master_data = {
        "video_path": stage1_data["video_path"],
        "fps": stage1_data["fps"],
        "num_scenes": len(all_scenes_summary),
        "scenes": all_scenes_summary
    }
    with open(master_summary_path, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*75}")
    print(f"[INFO] Pipeline Complete! Master summary saved to: {master_summary_path}")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
