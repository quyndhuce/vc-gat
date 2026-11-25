#!/usr/bin/env python3
"""
Compare Object Labels: OvSGTR Detector vs SigLIP 2 Taxonomy Scorer
Using the Vietnamese Tourism Visual Universe (vn_tourism_visual_universe.json)

For each bounding box detected on the image:
1. OvSGTR identifies the bounding box and preliminary label.
2. SigLIP 2 scores the cropped patch against all concepts in vn_tourism_visual_universe.json.
3. Outputs a side-by-side comparison table and an annotated visual side panel.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

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
            "group": "background",
            "notes": bg.get("notes", "")
        })

    # Object concepts
    for group_name, items in data.get("object_concepts", {}).items():
        for item in items:
            concepts.append({
                "id": item["id"],
                "label_en": item["label_en"],
                "description": item["description"],
                "group": group_name,
                "notes": item.get("notes", "")
            })

    return concepts, templates, data


# ---------------------------------------------------------------------------
# SigLIP 2 Scorer
# ---------------------------------------------------------------------------
class Siglip2Scorer:
    def __init__(self, model_id="google/siglip2-base-patch16-224", device=None):
        from transformers import AutoProcessor, AutoModel

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[INFO] Loading SigLIP 2 ({model_id}) on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.text_embeds = None
        self.concepts = None

    def index_concepts(self, concepts, templates):
        self.concepts = concepts
        print(f"[INFO] Encoding {len(concepts)} visual universe concepts with SigLIP 2...")

        # Ensemble over templates
        all_template_embeds = []
        for tpl in templates:
            texts = []
            for c in concepts:
                txt = tpl.replace("{label}", c["label_en"]).replace("{description}", c["description"])
                texts.append(txt)

            inputs = self.processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
                embeds = out.pooler_output if hasattr(out, 'pooler_output') else out
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                all_template_embeds.append(embeds)

        # Average ensemble embeddings
        self.text_embeds = torch.stack(all_template_embeds, dim=0).mean(dim=0)
        self.text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)
        print("[INFO] Concept embeddings ready!")

    @torch.no_grad()
    def score_crops(self, crop_images):
        inputs = self.processor(images=crop_images, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        image_embeds = out.pooler_output if hasattr(out, 'pooler_output') else out
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp() if hasattr(self.model, "logit_scale") else 1.0
        logit_bias = self.model.logit_bias if hasattr(self.model, "logit_bias") else 0.0

        logits = (image_embeds @ self.text_embeds.T) * logit_scale + logit_bias
        softmax_probs = torch.softmax(logits, dim=-1)
        sigmoid_probs = torch.sigmoid(logits)

        return softmax_probs.cpu(), sigmoid_probs.cpu()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
_PALETTE = [
    (230, 57, 70), (244, 162, 97), (42, 157, 143), (38, 70, 83),
    (74, 144, 226), (155, 89, 182), (241, 196, 15), (46, 204, 113),
    (231, 76, 60), (52, 152, 219)
]

def render_comparison_visualization(img_pil, comparison_records, output_path):
    img_ann = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img_ann, "RGBA")

    try:
        font_box = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_panel_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_panel_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_desc = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_box = font_panel_h = font_panel_sub = font_text = font_desc = ImageFont.load_default()

    # Draw boxes on image
    for i, r in enumerate(comparison_records):
        color = _PALETTE[i % len(_PALETTE)]
        x0, y0, x1, y1 = r["box"]
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)

        label_txt = f"#{i+1} {r['ovsgtr_label']} → {r['siglip2_id']}"
        try:
            bb = font_box.getbbox(label_txt)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(label_txt) * 8, 14

        draw.rectangle([x0, max(0, y0 - th - 6), x0 + tw + 6, y0], fill=color + (230,))
        draw.text((x0 + 3, max(0, y0 - th - 4)), label_txt, fill=(255, 255, 255), font=font_box)

    # Side comparison panel
    PANEL_W = 680
    ROW_H = 68
    PAD = 16
    H = max(img_ann.height, len(comparison_records) * ROW_H + PAD * 2 + 55)

    panel = Image.new("RGB", (PANEL_W, H), (18, 22, 34))
    pdraw = ImageDraw.Draw(panel)

    pdraw.text((PAD, PAD), "Label Comparison: OvSGTR vs SigLIP 2 (VN Tourism Universe)",
               fill=(255, 215, 64), font=font_panel_h)
    pdraw.text((PAD, PAD + 24), "Bounding Boxes scored against vn_tourism_visual_universe.json",
               fill=(140, 160, 190), font=font_panel_sub)
    pdraw.line([(PAD, PAD + 42), (PANEL_W - PAD, PAD + 42)], fill=(60, 75, 105), width=1)

    for i, r in enumerate(comparison_records):
        y = PAD + 50 + i * ROW_H
        color = _PALETTE[i % len(_PALETTE)]

        # Badge index
        pdraw.rectangle([PAD, y, PAD + 32, y + 24], fill=color)
        pdraw.text((PAD + 5, y + 4), f"#{i+1:02d}", fill=(255, 255, 255), font=font_text)

        # OvSGTR Label
        pdraw.text((PAD + 42, y), f"OvSGTR: {r['ovsgtr_label']}", fill=(100, 210, 255), font=font_text)
        pdraw.text((PAD + 42, y + 18), f"Detector Score: {r['ovsgtr_score']:.2f}", fill=(150, 170, 195), font=font_desc)

        # Arrow
        pdraw.text((PAD + 230, y + 8), "⟹", fill=(255, 255, 255), font=font_text)

        # SigLIP 2 Universe Label
        siglip_title = f"SigLIP 2: {r['siglip2_label']} [{r['siglip2_id']}]"
        pdraw.text((PAD + 265, y), siglip_title, fill=(140, 240, 160), font=font_text)

        desc_str = f"Group: {r['siglip2_group']} | Prob: {r['siglip2_score']*100:.1f}% | Runner-up: {r['runner_up']}"
        pdraw.text((PAD + 265, y + 20), desc_str, fill=(200, 210, 230), font=font_desc)
        pdraw.text((PAD + 265, y + 36), f"Desc: {r['siglip2_desc']}", fill=(140, 150, 175), font=font_desc)

        pdraw.line([(PAD, y + ROW_H - 4), (PANEL_W - PAD, y + ROW_H - 4)], fill=(35, 45, 65), width=1)

    # Resize panel to match image height
    ratio = img_ann.height / panel.height
    panel_resized = panel.resize((int(panel.width * ratio), img_ann.height), Image.LANCZOS)

    combined = Image.new("RGB", (img_ann.width + panel_resized.width, img_ann.height), (18, 22, 34))
    combined.paste(img_ann, (0, 0))
    combined.paste(panel_resized, (img_ann.width, 0))
    combined.save(output_path, quality=95)
    print(f"\n[INFO] Saved visual comparison → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OvSGTR vs SigLIP 2 Visual Universe Label Comparison")
    parser.add_argument('--image', type=str, default='den-ngoc-son.jpg')
    parser.add_argument('--universe', type=str, default='vn_tourism_visual_universe.json')
    parser.add_argument('--output', type=str, default='comparison_ovsgtr_siglip2.jpg')
    parser.add_argument('--model_id', type=str, default='google/siglip2-base-patch16-224')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    assert os.path.exists(args.image), f"Image not found: {args.image}"
    assert os.path.exists(args.universe), f"Universe file not found: {args.universe}"

    # Load Visual Universe
    concepts, templates, universe_raw = load_visual_universe(args.universe)

    # Detected boxes by OvSGTR on den-ngoc-son.jpg
    # (Matches our earlier Swin-B OvD+R detections)
    detected_boxes = [
        {"label": "person",       "score": 0.403, "box": [395.0, 190.6, 444.1, 253.2]},
        {"label": "lake",         "score": 0.373, "box": [0.6, 326.3, 799.8, 531.5]},
        {"label": "water",        "score": 0.365, "box": [0.6, 326.3, 799.8, 531.5]},
        {"label": "roof",         "score": 0.334, "box": [26.4, 45.6, 203.5, 136.7]},
        {"label": "temple",       "score": 0.264, "box": [0.8, 43.4, 210.6, 335.5]},
        {"label": "tree",         "score": 0.258, "box": [0.7, 0.4, 799.6, 303.7]},
        {"label": "roof (gate)",  "score": 0.253, "box": [0.2, 115.4, 60.9, 190.1]},
        {"label": "bridge",       "score": 0.250, "box": [114.6, 220.2, 799.7, 428.9]},
        {"label": "tree (left)",  "score": 0.210, "box": [0.6, 156.8, 124.3, 336.2]},
        {"label": "person (deck)","score": 0.205, "box": [196.5, 186.5, 469.0, 268.9]},
    ]

    # Initialize SigLIP 2 and index concepts
    scorer = Siglip2Scorer(model_id=args.model_id, device=args.device)
    scorer.index_concepts(concepts, templates)

    # Crop patches with slight margin for context
    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    crops = []
    for d in detected_boxes:
        x0, y0, x1, y1 = d["box"]
        bw, bh = x1 - x0, y1 - y0
        mx, my = bw * 0.05, bh * 0.05
        cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
        cx1, cy1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
        crops.append(img.crop((cx0, cy0, cx1, cy1)))

    print(f"\n[INFO] Scoring {len(crops)} crops with SigLIP 2...")
    softmax_probs, sigmoid_probs = scorer.score_crops(crops)

    # Format comparison records
    records = []
    print("\n" + "=" * 125)
    print(f"{'#':2s} | {'Nhãn OvSGTR':20s} | {'Nhãn SigLIP 2 (VN Tourism Universe)':32s} | {'ID Universe':20s} | {'Softmax':7s} | {'Runner-up'}")
    print("=" * 125)

    for i, (d, s_probs) in enumerate(zip(detected_boxes, softmax_probs)):
        topk = torch.topk(s_probs, 3)
        top1_c = concepts[topk.indices[0].item()]
        top1_score = topk.values[0].item()

        top2_c = concepts[topk.indices[1].item()]
        top2_score = topk.values[1].item()

        rec = {
            "box": d["box"],
            "ovsgtr_label": d["label"],
            "ovsgtr_score": d["score"],
            "siglip2_id": top1_c["id"],
            "siglip2_label": top1_c["label_en"],
            "siglip2_desc": top1_c["description"][:45] + ("..." if len(top1_c["description"]) > 45 else ""),
            "siglip2_group": top1_c["group"],
            "siglip2_score": top1_score,
            "runner_up": f"{top2_c['id']} ({top2_score*100:.1f}%)"
        }
        records.append(rec)

        model_str = f"{d['label']} ({d['score']:.2f})"
        siglip_str = f"{top1_c['label_en']}"
        siglip_id = f"[{top1_c['id']}]"
        print(f"{i+1:2d} | {model_str:20s} | {siglip_str:32s} | {siglip_id:20s} | {top1_score*100:6.2f}% | {rec['runner_up']}")

    print("=" * 125)

    # Render comparison visualization
    render_comparison_visualization(img, records, args.output)


if __name__ == '__main__':
    main()
