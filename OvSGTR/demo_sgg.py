#!/usr/bin/env python3
"""
OvSGTR Demo: Scene Graph Generation on an Image
Supports both Fully Open-Vocabulary SGG (OvD+R) and Closed-set VG150 SGG.

Default model: vg-ovdr-swinb-mega-best.pth (Swin-B trained on MegaSG, ECCV 2024 Best Paper Candidate)
"""

import os
import sys
import argparse
import random
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import urllib.request
import torch
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'GroundingDINO'))

# PyTorch library path
try:
    torch_lib = Path(torch.__file__).parent / 'lib'
    cur_ld = os.environ.get('LD_LIBRARY_PATH', '')
    if str(torch_lib) not in cur_ld:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib}:{cur_ld}"
except Exception:
    pass

# Ensure GroundingDINO egg in sys.path
site_pkgs = os.path.dirname(os.path.dirname(torch.__file__))
for egg in glob.glob(os.path.join(site_pkgs, 'groundingdino-*.egg')):
    if egg not in sys.path:
        sys.path.insert(0, egg)
        break

# ---------------------------------------------------------------------------
# Default Open-Vocabulary Vocabularies
# ---------------------------------------------------------------------------
DEFAULT_OPEN_CATEGORIES = [
    'bridge', 'temple', 'lake', 'water', 'tree', 'railing', 'person', 'gate', 'roof', 'building'
]

DEFAULT_OPEN_PREDICATES = [
    '[UNK]', 'over', 'across', 'leading to', 'on', 'near', 'in',
    'part of', 'surrounded by', 'walking on', 'with', 'above'
]

# ---------------------------------------------------------------------------
# VG150 vocabulary (for closed-set mode)
# ---------------------------------------------------------------------------
VG150_OBJ_CATEGORIES = [
    '__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket',
    'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book',
    'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus',
    'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat',
    'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer',
    'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag',
    'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove',
    'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill',
    'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp',
    'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
    'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange',
    'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza',
    'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket',
    'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf',
    'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard',
    'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street',
    'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
    'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase',
    'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing',
    'wire', 'woman', 'zebra'
]

VG150_PREDICATES = [
    '[UNK]', 'above', 'across', 'against', 'along', 'and', 'at',
    'attached to', 'behind', 'belonging to', 'between', 'carrying',
    'covered in', 'covering', 'eating', 'flying in', 'for', 'from',
    'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of',
    'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near',
    'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of',
    'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under',
    'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears',
    'with'
]

# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
def make_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([T.Resize(800), normalize])


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(config_file, checkpoint_path, categories, predicates, device):
    from util.slconfig import SLConfig
    from models.registry import MODULE_BUILD_FUNCS

    print(f"[INFO] Config: {config_file}")
    cfg = SLConfig.fromfile(config_file)
    args = argparse.Namespace()
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)

    args.device = str(device)
    args.use_ema = False
    args.debug = False
    args.dataset_file = 'vg'

    # Handle rln_freq_bias if present in config
    if hasattr(args, 'rln_freq_bias') and args.rln_freq_bias:
        candidates = [
            REPO_ROOT / args.rln_freq_bias,
            REPO_ROOT / 'datasets' / 'vg_stats.pt',
            REPO_ROOT / 'data' / 'vg150' / 'vg_stats.pt',
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = str(c)
                break
        if found:
            args.rln_freq_bias = found
            print(f"[INFO] Freq bias: {found}")
        else:
            print("[WARN] Freq bias not found, disabling.")
            args.rln_freq_bias = None

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    clean_sd = {(k[7:] if k.startswith('module.') else k): v
                for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"[WARN] Missing ({len(missing)} keys): {missing[:3]}")
    if unexpected:
        print(f"[WARN] Unexpected ({len(unexpected)} keys): {unexpected[:3]}")

    model = model.to(device).eval()

    # Configure postprocessor
    pp = postprocessors['bbox']
    pp.rln_proj = getattr(model, "rln_proj", None)
    pp.rln_classifier = getattr(model, "rln_classifier", None)
    pp.rln_freq_bias = getattr(model, "rln_freq_bias", None)

    # Dictionary mappings
    pp.name2classes = {n: i for i, n in enumerate(categories)}
    pp.name2predicates = {p: i for i, p in enumerate(predicates)}
    pp.eval()
    pp.to(device)

    print("[INFO] Model ready!")
    return model, postprocessors


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def infer(model, postprocessors, image_path, categories, predicates, device):
    from groundingdino.util.misc import nested_tensor_from_tensor_list

    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    transform = make_transforms()
    img_t = transform(img_pil)
    samples = nested_tensor_from_tensor_list([img_t]).to(device)
    orig_size = torch.tensor([[orig_h, orig_w]], device=device)

    # Build prompts (omit background/[UNK] from text prompt)
    fg_cats = [c for c in categories if c != '__background__']
    fg_preds = [p for p in predicates if p != '[UNK]']

    caption = '. '.join(fg_cats).lower().strip() + '.'
    rel_caption = '. '.join(fg_preds).lower().strip() + '.'

    sgg_mode = getattr(model, "sgg_mode", "ovdr")
    if sgg_mode != "full":
        outputs = model(samples, captions=[caption], rel_captions=[rel_caption])
        results = postprocessors['bbox'](outputs, orig_size, cat_list=fg_cats)
    else:
        outputs = model(samples, captions=[caption])
        results = postprocessors['bbox'](outputs, orig_size)

    return results[0], img_pil


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
_COLOR_PALETTE = [
    (230, 57, 70),   # Red / Coral
    (244, 162, 97),  # Sandy Orange
    (42, 157, 143),  # Persian Green
    (38, 70, 83),    # Charcoal
    (74, 144, 226),  # Sky Blue
    (155, 89, 182),  # Amethyst Purple
    (241, 196, 15),  # Sunflower Yellow
    (46, 204, 113),  # Emerald Green
    (231, 76, 60),   # Alizarin Red
    (52, 152, 219),  # Dodger Blue
]

def _cls_color(cls_idx):
    return _COLOR_PALETTE[cls_idx % len(_COLOR_PALETTE)]


def draw_boxes(img_pil, result, categories, score_thr=0.2):
    """Draw bounding boxes and labels on image."""
    img = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    boxes  = result['boxes']
    labels = result['labels']
    scores = result['scores']
    keep   = scores > score_thr

    for box, lbl, sc in zip(boxes[keep], labels[keep], scores[keep]):
        cls_idx = int(lbl.item())
        name  = categories[cls_idx] if cls_idx < len(categories) else f"cls{cls_idx}"
        color = _cls_color(cls_idx)
        x0, y0, x1, y1 = box.tolist()

        # Box border
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)

        # Label badge
        txt = f"{name} {sc:.2f}"
        try:
            bb = font.getbbox(txt)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(txt) * 8, 14

        draw.rectangle([x0, max(0, y0 - th - 6), x0 + tw + 6, y0], fill=color + (220,))
        draw.text((x0 + 3, max(0, y0 - th - 4)), txt, fill=(255, 255, 255), font=font)

    return img, keep


def extract_triplets(result, categories, predicates, score_thr=0.2, topk=20, min_score=0.03):
    """Extract scene graph triplets from model predictions."""
    graph = result.get('graph', {})
    if not graph or 'all_node_pairs' not in graph:
        return []

    node_pairs       = graph['all_node_pairs']
    all_relation     = graph['all_relation']
    pred_boxes_class = graph.get('pred_boxes_class', None)
    pred_boxes_score = graph.get('pred_boxes_score', None)

    triplets = []
    seen = set()

    for pair_idx in range(len(node_pairs)):
        s_idx = int(node_pairs[pair_idx, 0].item())
        o_idx = int(node_pairs[pair_idx, 1].item())

        if pred_boxes_score is not None:
            s_sc = float(pred_boxes_score[s_idx].item()) if s_idx < len(pred_boxes_score) else 0.0
            o_sc = float(pred_boxes_score[o_idx].item()) if o_idx < len(pred_boxes_score) else 0.0
            if s_sc < score_thr or o_sc < score_thr:
                continue
        else:
            s_sc, o_sc = 1.0, 1.0

        rel_probs = all_relation[pair_idx]
        pred_idx  = int(rel_probs[1:].argmax().item()) + 1
        pred_score = float(rel_probs[pred_idx].item())

        if pred_score < min_score:
            continue

        sub_cls = int(pred_boxes_class[s_idx].item()) if pred_boxes_class is not None else 0
        obj_cls = int(pred_boxes_class[o_idx].item()) if pred_boxes_class is not None else 0

        sub_name  = categories[sub_cls]  if sub_cls  < len(categories)  else "obj"
        obj_name  = categories[obj_cls]  if obj_cls  < len(categories)  else "obj"
        pred_name = predicates[pred_idx] if pred_idx < len(predicates) else "rel"

        # Overall triplet confidence: rel_score * sqrt(s_score * o_score)
        triplet_conf = pred_score * ((s_sc * o_sc) ** 0.5)

        key = (sub_name, pred_name, obj_name)
        if key not in seen:
            seen.add(key)
            triplets.append(dict(
                subject=sub_name, predicate=pred_name, object=obj_name,
                rel_score=pred_score, triplet_score=triplet_conf,
                s_score=s_sc, o_score=o_sc
            ))

    triplets = sorted(triplets, key=lambda x: -x['triplet_score'])
    return triplets[:topk]


def make_triplet_panel(triplets, mode_name="OvSGTR", width=620):
    """Render dark UI panel listing detected Scene Graph relationships."""
    LINE_H = 26
    PAD = 16
    H = max(280, len(triplets) * LINE_H + PAD * 3 + 55)
    panel = Image.new("RGB", (width, H), (18, 22, 34))
    draw  = ImageDraw.Draw(panel)

    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_sub = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_body = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_num = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_title = font_sub = font_body = font_num = ImageFont.load_default()

    # Title
    draw.text((PAD, PAD), "Scene Graph — Top Relationships",
              fill=(255, 215, 64), font=font_title)
    draw.text((PAD, PAD + 24), f"Model: {mode_name} | Top Triplets (Subject → Predicate → Object)",
              fill=(140, 155, 185), font=font_sub)
    draw.line([(PAD, PAD + 42), (width - PAD, PAD + 42)], fill=(60, 75, 105), width=1)

    if not triplets:
        draw.text((PAD, PAD + 60), "No relationships meeting score threshold.",
                  fill=(180, 180, 180), font=font_body)
        return panel

    for i, t in enumerate(triplets):
        y = PAD + 52 + i * LINE_H

        # Number
        draw.text((PAD, y), f"#{i+1:02d}", fill=(110, 130, 160), font=font_num)

        # Subject
        sub_x = PAD + 38
        draw.text((sub_x, y), f"[{t['subject']}]", fill=(100, 210, 255), font=font_body)

        # Arrow + Predicate
        try:
            sw = font_body.getbbox(f"[{t['subject']}]")[2] - font_body.getbbox(f"[{t['subject']}]")[0]
        except Exception:
            sw = len(t['subject']) * 9 + 18
        pred_x = sub_x + sw + 6
        draw.text((pred_x, y), f"—{t['predicate']}→", fill=(255, 140, 100), font=font_body)

        # Object
        try:
            pw = font_body.getbbox(f"—{t['predicate']}→")[2] - font_body.getbbox(f"—{t['predicate']}→")[0]
        except Exception:
            pw = len(t['predicate']) * 9 + 24
        obj_x = pred_x + pw + 6
        draw.text((obj_x, y), f"[{t['object']}]", fill=(140, 240, 160), font=font_body)

        # Confidence
        conf_str = f"conf: {t['triplet_score']:.2f} (rel: {t['rel_score']:.2f})"
        try:
            cw = font_num.getbbox(conf_str)[2] - font_num.getbbox(conf_str)[0]
        except Exception:
            cw = len(conf_str) * 7
        draw.text((width - PAD - cw, y), conf_str, fill=(160, 170, 195), font=font_num)

    return panel


def save_result(img_annotated, triplets, output_path, mode_name="OvSGTR"):
    panel = make_triplet_panel(triplets, mode_name=mode_name, width=640)

    # Match panel height with annotated image
    tgt_h = img_annotated.height
    ratio = tgt_h / panel.height
    panel_w = max(480, int(panel.width * ratio))
    panel_resized = panel.resize((panel_w, tgt_h), Image.LANCZOS)

    combined = Image.new("RGB", (img_annotated.width + panel_resized.width, tgt_h), (18, 22, 34))
    combined.paste(img_annotated, (0, 0))
    combined.paste(panel_resized, (img_annotated.width, 0))
    combined.save(output_path, quality=95)
    print(f"\n[INFO] Saved composite result → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OvSGTR Scene Graph Generation Demo")
    p.add_argument('--image', type=str, default='den-ngoc-son.jpg',
                   help="Path to input image")
    p.add_argument('--mode', type=str, choices=['ovdr', 'closed'], default='ovdr',
                   help="Mode: 'ovdr' (Open-Vocabulary) or 'closed' (VG150)")
    p.add_argument('--config', type=str, default=None,
                   help="Override model config path")
    p.add_argument('--checkpoint', type=str, default=None,
                   help="Override model checkpoint path")
    p.add_argument('--categories', type=str, default=None,
                   help="Comma-separated custom object categories (e.g. 'bridge,temple,lake,tree,person')")
    p.add_argument('--predicates', type=str, default=None,
                   help="Comma-separated custom predicates (e.g. 'over,across,on,near,leading to')")
    p.add_argument('--output', type=str, default='result_den_ngoc_son.jpg',
                   help="Output path for annotated image")
    p.add_argument('--topk', type=int, default=15,
                   help="Number of top triplets to display")
    p.add_argument('--score_threshold', type=float, default=0.20,
                   help="Detection score threshold (default: 0.20)")
    p.add_argument('--device', type=str, default='cuda',
                   help="Device: 'cuda' or 'cpu'")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(1)

    # Determine config, checkpoint, and vocabulary based on mode
    if args.mode == 'ovdr':
        config_path = args.config or str(REPO_ROOT / 'config/GroundingDINO_SwinB_ovdr.py')
        ckpt_path = args.checkpoint or str(REPO_ROOT / 'checkpoints/vg-ovdr-swinb-mega-best.pth')

        if args.categories:
            categories = [c.strip().lower() for c in args.categories.split(',') if c.strip()]
        else:
            categories = DEFAULT_OPEN_CATEGORIES

        if args.predicates:
            raw_p = [p.strip().lower() for p in args.predicates.split(',') if p.strip()]
            predicates = ['[UNK]'] + [p for p in raw_p if p != '[UNK]']
        else:
            predicates = DEFAULT_OPEN_PREDICATES

        mode_name = "OvSGTR Swin-B (Open-Vocabulary)"
    else:
        config_path = args.config or str(REPO_ROOT / 'config/GroundingDINO_SwinB_full.py')
        ckpt_path = args.checkpoint or str(REPO_ROOT / 'checkpoints/vg-full-swinb-mega-best.pth')
        categories = VG150_OBJ_CATEGORIES
        predicates = VG150_PREDICATES
        mode_name = "OvSGTR Swin-B (Closed-set VG150)"

    print(f"[INFO] Mode: {args.mode.upper()}")
    print(f"[INFO] Categories ({len(categories)}): {categories}")
    print(f"[INFO] Predicates ({len(predicates)}): {predicates}")

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Build model
    model, postprocessors = build_model(config_path, ckpt_path, categories, predicates, device)

    # Run inference
    print(f"\n[INFO] Inferring on: {image_path} ...")
    result, img_pil = infer(model, postprocessors, image_path, categories, predicates, device)

    # Object summary
    scores = result['scores']
    labels = result['labels']
    boxes  = result['boxes']
    keep   = scores > args.score_threshold
    n_det  = keep.sum().item()

    print(f"\n{'='*65}")
    print(f"  Detected {n_det} objects (score > {args.score_threshold})")
    print(f"{'='*65}")
    for i, (lbl, sc) in enumerate(zip(labels[keep], scores[keep])):
        cls_idx = int(lbl.item())
        name = categories[cls_idx] if cls_idx < len(categories) else f"cls{cls_idx}"
        print(f"  {i+1:2d}. {name:16s}  score={sc:.3f}  "
              f"box={[round(v, 1) for v in boxes[keep][i].tolist()]}")

    # Draw boxes
    img_ann, keep_mask = draw_boxes(img_pil, result, categories, score_thr=args.score_threshold)

    # Extract Scene Graph Triplets
    triplets = extract_triplets(result, categories, predicates,
                                score_thr=args.score_threshold, topk=args.topk)

    print(f"\n{'='*65}")
    print(f"  Top-{len(triplets)} Scene Graph Triplets ({mode_name})")
    print(f"{'='*65}")
    for i, t in enumerate(triplets):
        print(f"  {i+1:2d}. [{t['subject']:12s}] "
              f"—{t['predicate']:14s}→ [{t['object']:12s}]  "
              f"conf={t['triplet_score']:.3f}  (rel_sc={t['rel_score']:.3f})")
    print(f"{'='*65}")

    # Save
    save_result(img_ann, triplets, args.output, mode_name=mode_name)


if __name__ == '__main__':
    main()
