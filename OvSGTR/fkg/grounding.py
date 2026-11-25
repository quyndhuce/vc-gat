import os
import json
import torch
import numpy as np
from PIL import Image

# Building-level concept IDs in vn_tourism_visual_universe.json
BUILDING_LEVEL_CONCEPTS = {
    'pagoda', 'temple_den', 'communal_house', 'imperial_palace', 'citadel_wall',
    'church', 'cathedral_gothic', 'monastery_hall', 'mosque', 'royal_tomb',
    'ancient_house', 'opera_house', 'museum_building', 'french_colonial_building',
    'railway_station', 'modern_skyscraper', 'presidential_palace', 'shrine_small',
    'city_gate', 'stupa', 'ancient_brick_sanctuary_tower'
}

# Component coarse labels from detector
COMPONENT_COARSE_LABELS = {
    'roof', 'window', 'pillar', 'column', 'door', 'gate'
}


class SemanticGroundingEngine:
    def __init__(self, universe_path, model_id="google/siglip2-base-patch16-224", cache_path="siglip2_prototypes_207x768.pt", device='cuda'):
        from transformers import AutoProcessor, AutoModel

        self.device = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
        print(f"[INFO] Initializing SigLIP 2 Semantic Grounding on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

        self.universe_path = universe_path
        self.cache_path = cache_path
        self.concepts, self.context_vars, self.templates = self._load_universe(universe_path)

        # Build building-level mask indices
        self.building_indices = [
            i for i, c in enumerate(self.concepts) if c["id"] in BUILDING_LEVEL_CONCEPTS
        ]

        # Step 5: Pre-encode prototype matrix
        self.prototype_matrix = self._load_or_compute_prototypes()

        # Pre-encode context attribute variables
        self.context_embeds = self._encode_context_variables()

    def _load_universe(self, universe_path):
        with open(universe_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("meta", {})
        templates = meta.get("prompt_templates", [
            "a photo of {label}",
            "a close-up photo of {label}",
            "a photo of {description}"
        ])

        concepts = []
        for bg in data.get("background_class", []):
            concepts.append({
                "id": bg["id"],
                "label_en": bg["label_en"],
                "description": bg["description"],
                "group": "background"
            })

        for grp, items in data.get("object_concepts", {}).items():
            for item in items:
                concepts.append({
                    "id": item["id"],
                    "label_en": item["label_en"],
                    "description": item["description"],
                    "group": grp
                })

        context_vars = []
        for attr in data.get("attribute_variables", []):
            if attr["id"] in ["time_of_day", "setting", "viewpoint", "crowd_level"]:
                context_vars.append(attr)

        return concepts, context_vars, templates

    def _encode_texts(self, text_list):
        inputs = self.processor(text=text_list, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            embeds = out.pooler_output if hasattr(out, "pooler_output") else out
            return embeds / embeds.norm(dim=-1, keepdim=True)

    def _load_or_compute_prototypes(self):
        """
        Bước 5: Mã hóa trước toàn bộ prompt văn bản (làm một lần)
        Kết quả là ma trận nguyên mẫu kích thước 207 × 768.
        """
        if os.path.exists(self.cache_path):
            print(f"[INFO] Loading cached SigLIP 2 prototype matrix from {self.cache_path}...")
            P = torch.load(self.cache_path, map_location=self.device)
            if P.shape == (len(self.concepts), 768):
                return P

        print(f"[INFO] Computing prototype matrix for {len(self.concepts)} concepts with 3 templates...")
        t_embeds = []
        for tpl in self.templates:
            texts = [
                tpl.replace("{label}", c["label_en"]).replace("{description}", c["description"])
                for c in self.concepts
            ]
            t_embeds.append(self._encode_texts(texts))

        P = torch.stack(t_embeds, dim=0).mean(dim=0)
        P = P / P.norm(dim=-1, keepdim=True)

        torch.save(P, self.cache_path)
        print(f"[INFO] Saved prototype matrix: {P.shape} to {self.cache_path}")
        return P

    def _encode_context_variables(self):
        context_dict = {}
        for attr in self.context_vars:
            aid = attr["id"]
            texts = [f"a photo with {val['label_en']} {aid.replace('_', ' ')}" for val in attr["values"]]
            context_dict[aid] = {
                "values": attr["values"],
                "embeds": self._encode_texts(texts)
            }
        return context_dict

    @torch.no_grad()
    def encode_crops_and_score_track(self, track, segment_frames, orig_w, orig_h):
        """
        Bước 6: Mã hóa vùng và chấm điểm khái niệm
        - Cắt vùng từ mỗi frame (nới bbox 15%)
        - SigLIP 2 image encoder -> lấy trung bình các frame được vector của track (768-d)
        - Nhân với ma trận nguyên mẫu P^T -> 207 similarity scores
        - Ràng buộc cấp bậc: che khái niệm cấp công trình nếu coarse_label là bộ phận hoặc diện tích nhỏ
        - Softmax với logit_scale
        - Giữ cả vector 207 chiều, không lấy argmax.
        """
        crop_embeds = []
        frame_dict = {f["frame_idx"]: f["image"] for f in segment_frames}

        for f_idx, box, _ in track["detections"]:
            if f_idx not in frame_dict:
                continue
            img_np = frame_dict[f_idx]
            x0, y0, x1, y1 = box
            bw = x1 - x0
            bh = y1 - y0

            # Expand bbox 15% margin
            mx = bw * 0.15
            my = bh * 0.15
            cx0 = max(0, int(x0 - mx))
            cy0 = max(0, int(y0 - my))
            cx1 = min(orig_w, int(x1 + mx))
            cy1 = min(orig_h, int(y1 + my))

            if cx1 <= cx0 or cy1 <= cy0:
                continue

            crop = img_np[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue

            crop_pil = Image.fromarray(crop)
            inputs = self.processor(images=[crop_pil], return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = feat / feat.norm(dim=-1, keepdim=True)
            crop_embeds.append(feat)

        if not crop_embeds:
            # Fallback uniform distribution
            return np.ones(len(self.concepts), dtype=np.float32) / len(self.concepts)

        # Average feature across frames
        track_feat = torch.stack(crop_embeds, dim=0).mean(dim=0)
        track_feat = track_feat / track_feat.norm(dim=-1, keepdim=True)

        # Cosine similarity with prototype matrix
        sims = (track_feat @ self.prototype_matrix.T).squeeze(0)  # (207,)
        scale = getattr(self.model, "logit_scale", torch.tensor(10.0)).exp().clamp(max=100.0)
        logits = sims * scale

        # Hierarchical constraints
        rep_box = track["rep_box"]
        rep_area = (rep_box[2] - rep_box[0]) * (rep_box[3] - rep_box[1])
        area_ratio = rep_area / (orig_w * orig_h)

        is_component = track["coarse_label"] in COMPONENT_COARSE_LABELS
        is_small = area_ratio < 0.05

        if is_component or is_small:
            logits[self.building_indices] = -1e4

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    @torch.no_grad()
    def score_all_tracks_in_segment(self, tracks, segment_frames, orig_w, orig_h, batch_size=64):
        """
        Batched version of track crop encoding for high-throughput GPU inference.
        """
        if not tracks:
            return []

        frame_dict = {f["frame_idx"]: f["image"] for f in segment_frames}
        crop_items = []

        for t_idx, track in enumerate(tracks):
            for f_idx, box, _ in track["detections"]:
                if f_idx not in frame_dict:
                    continue
                img_np = frame_dict[f_idx]
                x0, y0, x1, y1 = box
                bw = x1 - x0
                bh = y1 - y0

                mx = bw * 0.15
                my = bh * 0.15
                cx0 = max(0, int(x0 - mx))
                cy0 = max(0, int(y0 - my))
                cx1 = min(orig_w, int(x1 + mx))
                cy1 = min(orig_h, int(y1 + my))

                if cx1 <= cx0 or cy1 <= cy0:
                    continue

                crop = img_np[cy0:cy1, cx0:cx1]
                if crop.size == 0:
                    continue

                crop_pil = Image.fromarray(crop)
                crop_items.append((t_idx, crop_pil))

        if not crop_items:
            return [np.ones(len(self.concepts), dtype=np.float32) / len(self.concepts) for _ in tracks]

        # Batch compute image features
        all_feats = []
        for b_start in range(0, len(crop_items), batch_size):
            b_crops = [item[1] for item in crop_items[b_start : b_start + batch_size]]
            inputs = self.processor(images=b_crops, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_feats.append(feat)

        all_feats = torch.cat(all_feats, dim=0)

        # Group features by track index
        track_feats_map = {}
        for idx, (t_idx, _) in enumerate(crop_items):
            if t_idx not in track_feats_map:
                track_feats_map[t_idx] = []
            track_feats_map[t_idx].append(all_feats[idx])

        scale = getattr(self.model, "logit_scale", torch.tensor(10.0)).exp().clamp(max=100.0)

        track_probs_list = []
        for t_idx, track in enumerate(tracks):
            c_feats = track_feats_map.get(t_idx, [])
            if not c_feats:
                track_probs_list.append(np.ones(len(self.concepts), dtype=np.float32) / len(self.concepts))
                continue

            track_feat = torch.stack(c_feats, dim=0).mean(dim=0)
            track_feat = track_feat / track_feat.norm(dim=-1, keepdim=True)

            sims = (track_feat @ self.prototype_matrix.T).squeeze(0)
            logits = sims * scale

            rep_box = track["rep_box"]
            rep_area = (rep_box[2] - rep_box[0]) * (rep_box[3] - rep_box[1])
            area_ratio = rep_area / (orig_w * orig_h)

            is_component = track["coarse_label"] in COMPONENT_COARSE_LABELS
            is_small = area_ratio < 0.05

            if is_component or is_small:
                logits[self.building_indices] = -1e4

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            track_probs_list.append(probs)

        return track_probs_list

    @torch.no_grad()
    def score_context_on_full_frames(self, segment_frames):
        """
        Chấm điểm bối cảnh trên frame nguyên (không cắt vùng)
        """
        all_ctx_vals = []
        for f in segment_frames:
            img_pil = Image.fromarray(f["image"])
            inputs = self.processor(images=[img_pil], return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = feat / feat.norm(dim=-1, keepdim=True)

            scale = getattr(self.model, "logit_scale", torch.tensor(10.0)).exp().clamp(max=100.0)
            f_ctx = []
            for aid, cinfo in self.context_embeds.items():
                sims = (feat @ cinfo["embeds"].T).squeeze(0)
                probs = torch.softmax(sims * scale, dim=-1).cpu().numpy()
                f_ctx.extend(probs.tolist())
            all_ctx_vals.append(f_ctx)

        if not all_ctx_vals:
            return np.zeros(17, dtype=np.float32)

        return np.mean(all_ctx_vals, axis=0).astype(np.float32)

    def pool_segment_representation(self, track_memberships, context_vector):
        """
        Bước 7. Gộp track thành đoạn:
        Với mỗi khái niệm c: độ thuộc của đoạn bằng 1 trừ đi tích của (1 - độ thuộc của từng track vào c) [Noisy-OR]
        Ghép với vector bối cảnh được vector khoảng 207 + 17 chiều.
        """
        num_c = len(self.concepts)
        if not track_memberships:
            seg_concept_mem = np.zeros(num_c, dtype=np.float32)
        else:
            # track_memberships: (N_tracks, 207)
            # Noisy-OR: 1 - prod(1 - p_i)
            tm = np.clip(np.array(track_memberships), 0.0, 0.9999)
            seg_concept_mem = 1.0 - np.prod(1.0 - tm, axis=0)

        full_seg_vector = np.concatenate([seg_concept_mem, context_vector], axis=0)
        return seg_concept_mem, full_seg_vector
