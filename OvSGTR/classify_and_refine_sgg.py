#!/usr/bin/env python3
"""
Selective Knowledge Graph Refinement with SigLIP 2
Based on Vietnamese Tourism Visual Universe (vn_tourism_visual_universe.json)

Core Strategy (Tư duy phân loại & lọc chọn lọc):
1. NATURE_PRIMITIVES:
   - Elements of nature & background (tree, water, lake, sky, mountain...)
   - RATIONALE: Natural primitives are already well-localized by the detector and
     do NOT have cultural/architectural sub-classes. Subjecting them to cultural
     taxonomy causes false positive noise (e.g. water reflecting bridge misclassified as bridge).
   - ACTION: BYPASS SigLIP 2, retain detector label & confidence directly.

2. ARCHITECTURAL & CULTURAL OBJECTS:
   - Architecture: bridge, temple, pagoda, church, building, gate, tower...
   - Components: roof, wall, column, pillar, railing, balcony, stairs...
   - Artifacts & Relics: statue, stele, altar, bell, drum, plaque, board...
   - Cultural Transport: boat, sampan, basket boat, cyclo...
   - ACTION: RE-SCORE with SigLIP 2 against the Vietnamese Tourism Visual Universe
     to obtain fine-grained cultural identity + MATERIAL (chất liệu) + COLOUR (màu sắc).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

# ---------------------------------------------------------------------------
# Domain Triage Policy: Define object categories and their processing rules
# ---------------------------------------------------------------------------
# Nhóm 1: Các đối tượng cảnh quan tự nhiên & môi trường -> KHÔNG chấm lại bằng SigLIP 2
NATURE_PRIMITIVES = {
    # Nước / Thủy văn
    'water', 'lake', 'river', 'sea', 'ocean', 'stream', 'pond', 'lagoon', 'waterfall',
    # Cây cối / Thực vật
    'tree', 'trees', 'plant', 'plants', 'leaf', 'leaves', 'forest', 'grass', 'bush', 'branch', 'jungle', 'mangrove',
    # Địa hình & Bầu trời
    'sky', 'cloud', 'clouds', 'mountain', 'hill', 'ground', 'dirt', 'sand', 'dune', 'cliff', 'rock',
}

# Nhóm 2: Đối tượng kiến trúc, cấu kiện, chất liệu & hiện vật văn hóa -> CẦN chấm lại bằng SigLIP 2
ARCHITECTURAL_AND_CULTURAL_KEYWORDS = {
    # Công trình kiến trúc
    'bridge', 'temple', 'pagoda', 'building', 'house', 'palace', 'citadel',
    'shrine', 'tower', 'stupa', 'church', 'cathedral', 'mosque', 'gate', 'pavilion',
    # Cấu kiện kiến trúc & bộ phận
    'roof', 'wall', 'pillar', 'column', 'door', 'window', 'balcony', 'railing',
    'fence', 'stairs', 'tile', 'tiles', 'eaves', 'bracket', 'arch', 'archway', 'courtyard',
    # Hiện vật văn hóa / Tôn giáo / Tín ngưỡng
    'statue', 'stele', 'altar', 'burner', 'incense', 'bell', 'drum', 'plaque', 'board',
    'tablet', 'banner', 'lantern', 'lamp', 'flag', 'monument', 'carving',
    # Phương tiện văn hóa / Du lịch sông nước
    'boat', 'ship', 'sampan', 'coracle', 'cyclo', 'rickshaw', 'cart',
    # Trang phục / Hoạt động văn hóa
    'costume', 'dress', 'hat', 'aodai', 'puppet', 'textile', 'vase', 'pottery'
}


def should_refine_object(label: str) -> bool:
    """
    Decide whether an object should be refined by SigLIP 2.
    Returns True for architecture/materials/artifacts, False for natural primitives.
    """
    label_clean = label.lower().strip()
    words = label_clean.replace('_', ' ').replace('-', ' ').split()

    # If explicitly in nature primitives -> Bypass
    for w in words:
        if w in NATURE_PRIMITIVES:
            return False

    # If any word matches architectural or cultural keywords -> Refine
    for w in words:
        if w in ARCHITECTURAL_AND_CULTURAL_KEYWORDS:
            return True

    # Default for unidentified object: if not nature, allow refinement
    return False


# ---------------------------------------------------------------------------
# Visual Universe Loader
# ---------------------------------------------------------------------------
def load_visual_universe(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get("meta", {})
    templates = meta.get("prompt_templates", [
        "a photo of {label}",
        "a close-up photo of {label}",
        "a photo of {description}"
    ])

    concepts = []
    # Background classes
    for bg in data.get("background_class", []):
        concepts.append({
            "id": bg["id"],
            "label_en": bg["label_en"],
            "description": bg["description"],
            "group": "background"
        })

    # Object concepts
    for group_name, items in data.get("object_concepts", {}).items():
        for item in items:
            concepts.append({
                "id": item["id"],
                "label_en": item["label_en"],
                "description": item["description"],
                "group": group_name
            })

    # Material attributes
    materials = []
    for attr in data.get("attribute_variables", []):
        if attr["id"] == "material":
            materials = attr["values"]
            break

    # Dominant colour attributes
    colours = []
    for attr in data.get("attribute_variables", []):
        if attr["id"] == "dominant_colour":
            colours = attr["values"]
            break

    return concepts, materials, colours, templates, data


# ---------------------------------------------------------------------------
# SigLIP 2 Scorer
# ---------------------------------------------------------------------------
class Siglip2TaxonomyScorer:
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
        self.mat_embeds = None
        self.colours = None
        self.col_embeds = None

    def encode_text_list(self, text_list):
        inputs = self.processor(text=text_list, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            embeds = out.pooler_output if hasattr(out, 'pooler_output') else out
            return embeds / embeds.norm(dim=-1, keepdim=True)

    def index_universe(self, concepts, materials, colours, templates):
        self.concepts = concepts
        self.materials = materials
        self.colours = colours

        print(f"[INFO] Encoding {len(concepts)} concepts + materials + colours...")
        # 1. Concept embeddings (ensemble of prompt templates)
        ensemble = []
        for tpl in templates:
            texts = [tpl.replace("{label}", c["label_en"]).replace("{description}", c["description"])
                     for c in concepts]
            ensemble.append(self.encode_text_list(texts))
        self.concept_embeds = torch.stack(ensemble, dim=0).mean(dim=0)
        self.concept_embeds = self.concept_embeds / self.concept_embeds.norm(dim=-1, keepdim=True)

        # 2. Material embeddings
        if materials:
            mat_texts = [f"an object made of {m['label_en']}, {m['description']}" for m in materials]
            self.mat_embeds = self.encode_text_list(mat_texts)

        # 3. Colour embeddings
        if colours:
            col_texts = [f"a {c['label_en']} coloured object, {c['description']}" for c in colours]
            self.col_embeds = self.encode_text_list(col_texts)

        print("[INFO] Embeddings indexed successfully!")

    @torch.no_grad()
    def score_single_crop(self, crop_image):
        inputs = self.processor(images=[crop_image], return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        img_embed = out.pooler_output if hasattr(out, 'pooler_output') else out
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

        scale = self.model.logit_scale.exp() if hasattr(self.model, "logit_scale") else 1.0
        bias = self.model.logit_bias if hasattr(self.model, "logit_bias") else 0.0

        # Concept scores
        c_logits = (img_embed @ self.concept_embeds.T) * scale + bias
        c_probs = torch.softmax(c_logits, dim=-1)[0].cpu()

        # Material scores
        mat_res = None
        if self.mat_embeds is not None:
            m_logits = (img_embed @ self.mat_embeds.T) * scale + bias
            m_probs = torch.softmax(m_logits, dim=-1)[0].cpu()
            top_m_idx = m_probs.argmax().item()
            mat_res = {
                "id": self.materials[top_m_idx]["id"],
                "label": self.materials[top_m_idx]["label_en"],
                "score": m_probs[top_m_idx].item()
            }

        # Colour scores
        col_res = None
        if self.col_embeds is not None:
            col_logits = (img_embed @ self.col_embeds.T) * scale + bias
            col_probs = torch.softmax(col_logits, dim=-1)[0].cpu()
            top_col_idx = col_probs.argmax().item()
            col_res = {
                "id": self.colours[top_col_idx]["id"],
                "label": self.colours[top_col_idx]["label_en"],
                "score": col_probs[top_col_idx].item()
            }

        return c_probs, mat_res, col_res


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
_PALETTE = [
    (230, 57, 70), (244, 162, 97), (42, 157, 143), (38, 70, 83),
    (74, 144, 226), (155, 89, 182), (241, 196, 15), (46, 204, 113),
    (231, 76, 60), (52, 152, 219)
]

def render_selective_visualization(img_pil, records, output_path):
    img_ann = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img_ann, "RGBA")

    try:
        font_box = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_desc = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_box = font_h = font_sub = font_text = font_desc = ImageFont.load_default()

    # Draw bounding boxes
    for i, r in enumerate(records):
        color = _PALETTE[i % len(_PALETTE)]
        x0, y0, x1, y1 = r["box"]

        # Different box style: Solid for SigLIP 2 refined, Dashed outline or tag for nature
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)

        if r["is_refined"]:
            tag = f"#{i+1} [{r['siglip2_id']}] ({r['siglip2_score']*100:.0f}%)"
        else:
            tag = f"#{i+1} {r['ovsgtr_label']} [Nature: Giữ nguyên]"

        try:
            bb = font_box.getbbox(tag)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(tag) * 8, 14

        draw.rectangle([x0, max(0, y0 - th - 6), x0 + tw + 6, y0], fill=color + (230,))
        draw.text((x0 + 3, max(0, y0 - th - 4)), tag, fill=(255, 255, 255), font=font_box)

    # Side panel
    PANEL_W = 720
    ROW_H = 72
    PAD = 16
    H = max(img_ann.height, len(records) * ROW_H + PAD * 2 + 55)

    panel = Image.new("RGB", (PANEL_W, H), (18, 22, 34))
    pdraw = ImageDraw.Draw(panel)

    pdraw.text((PAD, PAD), "Chiến lược Tinh chỉnh Chọn lọc (Selective Knowledge Refinement)",
               fill=(255, 215, 64), font=font_h)
    pdraw.text((PAD, PAD + 24), "Kiến trúc/Chất liệu → Chấm SigLIP 2 | Cảnh quan tự nhiên → Giữ nguyên",
               fill=(140, 160, 190), font=font_sub)
    pdraw.line([(PAD, PAD + 42), (PANEL_W - PAD, PAD + 42)], fill=(60, 75, 105), width=1)

    for i, r in enumerate(records):
        y = PAD + 50 + i * ROW_H
        color = _PALETTE[i % len(_PALETTE)]

        # Badge index
        pdraw.rectangle([PAD, y, PAD + 32, y + 24], fill=color)
        pdraw.text((PAD + 5, y + 4), f"#{i+1:02d}", fill=(255, 255, 255), font=font_text)

        # OvSGTR Label
        pdraw.text((PAD + 42, y), f"Detector: {r['ovsgtr_label']}", fill=(100, 210, 255), font=font_text)
        pdraw.text((PAD + 42, y + 18), f"Score: {r['ovsgtr_score']:.2f}", fill=(150, 170, 195), font=font_desc)

        # Status arrow
        if r["is_refined"]:
            pdraw.text((PAD + 215, y + 8), "⟹", fill=(255, 215, 64), font=font_text)

            # Refined Label
            sig_txt = f"SigLIP 2: {r['siglip2_label']} [{r['siglip2_id']}]"
            pdraw.text((PAD + 250, y), sig_txt, fill=(140, 240, 160), font=font_text)

            mat_info = f"Chất liệu: {r.get('material', 'N/A')} | Màu: {r.get('colour', 'N/A')}"
            score_info = f"Độ tin cậy: {r['siglip2_score']*100:.1f}% | {mat_info}"
            pdraw.text((PAD + 250, y + 20), score_info, fill=(255, 200, 120), font=font_desc)
            pdraw.text((PAD + 250, y + 36), f"Nhóm: {r['siglip2_group']} | Kế tiếp: {r['runner_up']}",
                       fill=(150, 165, 190), font=font_desc)
        else:
            pdraw.text((PAD + 215, y + 8), "—", fill=(120, 130, 150), font=font_text)
            pdraw.text((PAD + 250, y), f"Tự nhiên ({r['ovsgtr_label']}): Giữ nguyên nhãn gốc",
                       fill=(170, 185, 210), font=font_text)
            pdraw.text((PAD + 250, y + 20), "Lý do: Đối tượng cảnh quan tự nhiên không cần định danh kiến trúc văn hóa.",
                       fill=(130, 145, 170), font=font_desc)
            pdraw.text((PAD + 250, y + 36), f"Phân loại: Tự nhiên / Môi trường (Bypass SigLIP 2)",
                       fill=(100, 120, 145), font=font_desc)

        pdraw.line([(PAD, y + ROW_H - 4), (PANEL_W - PAD, y + ROW_H - 4)], fill=(35, 45, 65), width=1)

    ratio = img_ann.height / panel.height
    panel_resized = panel.resize((int(panel.width * ratio), img_ann.height), Image.LANCZOS)

    combined = Image.new("RGB", (img_ann.width + panel_resized.width, img_ann.height), (18, 22, 34))
    combined.paste(img_ann, (0, 0))
    combined.paste(panel_resized, (img_ann.width, 0))
    combined.save(output_path, quality=95)
    print(f"\n[INFO] Saved selective refinement visualization → {output_path}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Selective Architecture & Material Refinement with SigLIP 2")
    parser.add_argument('--image', type=str, default='den-ngoc-son.jpg')
    parser.add_argument('--universe', type=str, default='vn_tourism_visual_universe.json')
    parser.add_argument('--output', type=str, default='selective_refinement_result.jpg')
    parser.add_argument('--model_id', type=str, default='google/siglip2-base-patch16-224')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    assert os.path.exists(args.image), f"Image not found: {args.image}"
    assert os.path.exists(args.universe), f"Universe not found: {args.universe}"

    # Load Universe
    concepts, materials, colours, templates, raw_universe = load_visual_universe(args.universe)

    # Detected bounding boxes on den-ngoc-son.jpg
    detected_boxes = [
        {"label": "bridge",       "score": 0.250, "box": [114.6, 220.2, 799.7, 428.9]},
        {"label": "temple",       "score": 0.264, "box": [0.8, 43.4, 210.6, 335.5]},
        {"label": "roof (gate)",  "score": 0.253, "box": [0.2, 115.4, 60.9, 190.1]},
        {"label": "roof (temple)","score": 0.334, "box": [26.4, 45.6, 203.5, 136.7]},
        {"label": "lake",         "score": 0.373, "box": [0.6, 326.3, 799.8, 531.5]},
        {"label": "water",        "score": 0.365, "box": [0.6, 326.3, 799.8, 531.5]},
        {"label": "tree",         "score": 0.258, "box": [0.7, 0.4, 799.6, 303.7]},
        {"label": "tree (left)",  "score": 0.210, "box": [0.6, 156.8, 124.3, 336.2]},
        {"label": "person",       "score": 0.403, "box": [395.0, 190.6, 444.1, 253.2]},
    ]

    # Initialize SigLIP 2
    scorer = Siglip2TaxonomyScorer(model_id=args.model_id, device=args.device)
    scorer.index_universe(concepts, materials, colours, templates)

    img = Image.open(args.image).convert("RGB")
    w, h = img.size

    records = []
    print("\n" + "=" * 125)
    print(f"{'#':2s} | {'Loại':10s} | {'Nhãn Detector':18s} | {'Kết quả SigLIP 2':30s} | {'Chất liệu':14s} | {'Màu sắc':12s} | {'Độ tin cậy'}")
    print("=" * 125)

    for i, d in enumerate(detected_boxes):
        needs_refine = should_refine_object(d["label"])

        if needs_refine:
            # Crop image patch
            x0, y0, x1, y1 = d["box"]
            bw, bh = x1 - x0, y1 - y0
            mx, my = bw * 0.05, bh * 0.05
            cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
            cx1, cy1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
            crop = img.crop((cx0, cy0, cx1, cy1))

            c_probs, mat_res, col_res = scorer.score_single_crop(crop)

            topk = torch.topk(c_probs, 2)
            top1_c = concepts[topk.indices[0].item()]
            top1_sc = topk.values[0].item()
            top2_c = concepts[topk.indices[1].item()]
            top2_sc = topk.values[1].item()

            mat_str = f"{mat_res['label']} ({mat_res['score']*100:.0f}%)" if mat_res else "N/A"
            col_str = f"{col_res['label']} ({col_res['score']*100:.0f}%)" if col_res else "N/A"

            rec = {
                "box": d["box"],
                "ovsgtr_label": d["label"],
                "ovsgtr_score": d["score"],
                "is_refined": True,
                "siglip2_id": top1_c["id"],
                "siglip2_label": top1_c["label_en"],
                "siglip2_desc": top1_c["description"][:50],
                "siglip2_group": top1_c["group"],
                "siglip2_score": top1_sc,
                "material": mat_str,
                "colour": col_str,
                "runner_up": f"{top2_c['id']} ({top2_sc*100:.1f}%)"
            }
            records.append(rec)

            print(f"{i+1:2d} | {'REFINE':10s} | {d['label']:18s} | {top1_c['label_en']:30s} | {mat_str:14s} | {col_str:12s} | {top1_sc*100:6.1f}%")
        else:
            # Bypass SigLIP 2
            rec = {
                "box": d["box"],
                "ovsgtr_label": d["label"],
                "ovsgtr_score": d["score"],
                "is_refined": False,
                "siglip2_id": d["label"],
                "siglip2_label": d["label"],
                "siglip2_desc": "Natural primitive retained from detector",
                "siglip2_group": "natural_primitive",
                "siglip2_score": d["score"],
                "material": "N/A (Nature)",
                "colour": "N/A",
                "runner_up": "None (Bypassed)"
            }
            records.append(rec)

            print(f"{i+1:2d} | {'BYPASS':10s} | {d['label']:18s} | {'[Giữ nguyên nhãn gốc]':30s} | {'N/A (Tự nhiên)':14s} | {'N/A':12s} | {d['score']*100:6.1f}% (Det)")

    print("=" * 125)

    # Render visualization
    render_selective_visualization(img, records, args.output)


if __name__ == '__main__':
    main()
