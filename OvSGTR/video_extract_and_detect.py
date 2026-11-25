#!/usr/bin/env python3
"""
Stage 1 of Video SGG Pipeline:
1. Detect scene boundaries using PySceneDetect (ContentDetector).
2. For each scene, extract 3-5 representative keyframes.
3. Run OvSGTR Open-Vocabulary Scene Graph Generation on each keyframe.
4. Export intermediate scene & keyframe detections to detector_detections.json.
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
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

site_pkgs = os.path.dirname(os.path.dirname(torch.__file__))
for egg in glob.glob(os.path.join(site_pkgs, 'groundingdino-*.egg')):
    if egg not in sys.path:
        sys.path.insert(0, egg)
        break

from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS
from groundingdino.util.misc import nested_tensor_from_tensor_list

# Default open vocabulary
DEFAULT_CATEGORIES = [
    'bridge', 'temple', 'lake', 'water', 'tree', 'railing', 'person', 'gate', 'roof', 'building'
]
DEFAULT_PREDICATES = [
    '[UNK]', 'over', 'across', 'leading to', 'on', 'near', 'in',
    'part of', 'surrounded by', 'walking on', 'with', 'above'
]


def make_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([T.Resize(800), normalize])


def detect_scenes_and_keyframes(video_path, output_dir, num_keyframes=4, threshold=27.0, max_scenes=None):
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    print(f"\n[INFO] Opening video: {video_path}")
    video = open_video(video_path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=15))
    sm.detect_scenes(video)
    scene_list = sm.get_scene_list()

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / fps

    print(f"[INFO] Video duration: {duration:.2f}s ({total_frames} frames @ {fps:.1f} fps)")

    # Fallback if no cut detected: treat as single scene
    if not scene_list:
        print("[INFO] No scene cuts detected; treating entire video as 1 scene.")
        scene_ranges = [(0, total_frames)]
    else:
        scene_ranges = [(s.frame_num, e.frame_num) for s, e in scene_list]

    if max_scenes and len(scene_ranges) > max_scenes:
        print(f"[INFO] Limiting to first {max_scenes} scenes (out of {len(scene_ranges)} detected).")
        scene_ranges = scene_ranges[:max_scenes]

    print(f"[INFO] Processing {len(scene_ranges)} scene(s)...")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenes_metadata = []

    for s_idx, (start_f, end_f) in enumerate(scene_ranges, 1):
        scene_len = end_f - start_f
        s_dir = out_path / f"scene_{s_idx:02d}"
        s_dir.mkdir(parents=True, exist_ok=True)

        # Select num_keyframes evenly inside the scene (avoiding transition boundaries)
        k_count = min(num_keyframes, max(1, scene_len))
        if scene_len <= k_count:
            chosen_frames = list(range(start_f, end_f))
        else:
            step = scene_len / (k_count + 1)
            chosen_frames = [int(start_f + step * (i + 1)) for i in range(k_count)]

        # Unique & sorted frame indices
        chosen_frames = sorted(list(set(chosen_frames)))

        extracted_keyframes = []
        for f_idx in chosen_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_filename = f"frame_{f_idx:06d}.jpg"
            img_rel_path = str(s_dir / img_filename)
            Image.fromarray(frame_rgb).save(img_rel_path, quality=95)

            sec = f_idx / fps
            extracted_keyframes.append({
                "frame_idx": f_idx,
                "timestamp_sec": round(sec, 2),
                "image_path": img_rel_path,
                "width": frame_rgb.shape[1],
                "height": frame_rgb.shape[0]
            })

        scene_meta = {
            "scene_id": f"scene_{s_idx:02d}",
            "scene_index": s_idx,
            "start_frame": start_f,
            "end_frame": end_f,
            "start_seconds": round(start_f / fps, 2),
            "end_seconds": round(end_f / fps, 2),
            "duration_seconds": round((end_f - start_f) / fps, 2),
            "num_keyframes": len(extracted_keyframes),
            "keyframes": extracted_keyframes
        }
        scenes_metadata.append(scene_meta)
        print(f"  ✓ Scene {s_idx:02d}: {scene_meta['start_seconds']}s - {scene_meta['end_seconds']}s | {len(extracted_keyframes)} keyframes extracted")

    cap.release()
    return scenes_metadata, fps


def build_ovsgtr_model(config_path, checkpoint_path, categories, predicates, device):
    print(f"\n[INFO] Initializing OvSGTR Swin-B on {device}...")
    cfg = SLConfig.fromfile(config_path)
    args = argparse.Namespace()
    for k, v in cfg._cfg_dict.to_dict().items():
        setattr(args, k, v)
    args.device = str(device)
    args.use_ema = False
    args.debug = False
    args.dataset_file = 'vg'

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)

    print(f"[INFO] Loading weights: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    pp = postprocessors['bbox']
    pp.rln_proj = getattr(model, 'rln_proj', None)
    pp.rln_classifier = getattr(model, 'rln_classifier', None)
    pp.rln_freq_bias = getattr(model, 'rln_freq_bias', None)
    pp.name2classes = {n: i for i, n in enumerate(categories)}
    pp.name2predicates = {p: i for i, p in enumerate(predicates)}
    pp.eval().to(device)

    return model, pp


@torch.no_grad()
def infer_keyframe(model, postprocessor, image_path, categories, predicates, device, score_thr=0.20, min_rel_score=0.03):
    img_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img_pil.size

    transform = make_transforms()
    img_t = transform(img_pil)
    samples = nested_tensor_from_tensor_list([img_t]).to(device)
    orig_size = torch.tensor([[orig_h, orig_w]], device=device)

    caption = '. '.join([c for c in categories if c != '__background__']).lower().strip() + '.'
    rel_caption = '. '.join([p for p in predicates if p != '[UNK]']).lower().strip() + '.'

    outputs = model(samples, captions=[caption], rel_captions=[rel_caption])
    results = postprocessor(outputs, orig_size, cat_list=[c for c in categories if c != '__background__'])[0]

    scores = results['scores']
    labels = results['labels']
    boxes = results['boxes']
    keep = scores > score_thr

    nodes = []
    # Index mapping from old detection index to filtered node index
    keep_indices = torch.nonzero(keep).squeeze(-1).tolist()
    if isinstance(keep_indices, int):
        keep_indices = [keep_indices]

    for new_idx, orig_idx in enumerate(keep_indices):
        sc = float(scores[orig_idx].item())
        cls_idx = int(labels[orig_idx].item())
        lbl_name = categories[cls_idx] if cls_idx < len(categories) else f"cls_{cls_idx}"
        b = [round(float(v), 2) for v in boxes[orig_idx].tolist()]

        nodes.append({
            "node_idx": new_idx,
            "orig_idx": orig_idx,
            "box_2d": b,
            "detector_label": lbl_name,
            "detector_score": round(sc, 4)
        })

    # Extract relations
    edges = []
    graph = results.get('graph', {})
    if graph and 'all_node_pairs' in graph and len(nodes) > 1:
        node_pairs = graph['all_node_pairs']
        all_relation = graph['all_relation']
        pred_boxes_score = graph.get('pred_boxes_score', None)

        orig_to_new = {orig: new for new, orig in enumerate(keep_indices)}

        for pair_idx in range(len(node_pairs)):
            s_orig = int(node_pairs[pair_idx, 0].item())
            o_orig = int(node_pairs[pair_idx, 1].item())

            if s_orig not in orig_to_new or o_orig not in orig_to_new:
                continue

            s_new = orig_to_new[s_orig]
            o_new = orig_to_new[o_orig]
            if s_new == o_new:
                continue

            s_sc = float(pred_boxes_score[s_orig].item()) if pred_boxes_score is not None else 1.0
            o_sc = float(pred_boxes_score[o_orig].item()) if pred_boxes_score is not None else 1.0

            rel_probs = all_relation[pair_idx]
            pred_idx = int(rel_probs[1:].argmax().item()) + 1
            rel_score = float(rel_probs[pred_idx].item())

            if rel_score < min_rel_score:
                continue

            pred_name = predicates[pred_idx] if pred_idx < len(predicates) else f"pred_{pred_idx}"
            triplet_conf = rel_score * ((s_sc * o_sc) ** 0.5)

            edges.append({
                "subject_node_idx": s_new,
                "subject_label": nodes[s_new]["detector_label"],
                "predicate": pred_name,
                "object_node_idx": o_new,
                "object_label": nodes[o_new]["detector_label"],
                "rel_score": round(rel_score, 4),
                "confidence": round(triplet_conf, 4)
            })

    edges = sorted(edges, key=lambda x: -x["confidence"])

    return nodes, edges


def main():
    p = argparse.ArgumentParser(description="Stage 1: Scene Cut & OvSGTR Detection on Video")
    p.add_argument('--video', type=str, required=True, help="Path to video file")
    p.add_argument('--output_dir', type=str, default='video_output', help="Directory to save extracted keyframes & results")
    p.add_argument('--config', type=str, default='config/GroundingDINO_SwinB_ovdr.py')
    p.add_argument('--checkpoint', type=str, default='checkpoints/vg-ovdr-swinb-mega-best.pth')
    p.add_argument('--num_keyframes', type=int, default=4, help="Number of representative keyframes per scene (3-5)")
    p.add_argument('--max_scenes', type=int, default=None, help="Maximum number of scenes to process")
    p.add_argument('--scene_threshold', type=float, default=27.0, help="ContentDetector threshold for PySceneDetect")
    p.add_argument('--score_threshold', type=float, default=0.20, help="Object detection threshold")
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    assert os.path.exists(args.video), f"Video not found: {args.video}"
    device = torch.device(args.device if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    # Step 1: Detect scenes and extract keyframes
    scenes_meta, fps = detect_scenes_and_keyframes(
        video_path=args.video,
        output_dir=args.output_dir,
        num_keyframes=args.num_keyframes,
        threshold=args.scene_threshold,
        max_scenes=args.max_scenes
    )

    # Step 2: Build OvSGTR model
    categories = DEFAULT_CATEGORIES
    predicates = DEFAULT_PREDICATES
    model, postprocessor = build_ovsgtr_model(args.config, args.checkpoint, categories, predicates, device)

    # Step 3: Run SGG inference on each keyframe
    print("\n[INFO] Running OvSGTR Open-Vocabulary SGG on all keyframes...")
    total_kf = sum(len(s["keyframes"]) for s in scenes_meta)
    processed_count = 0

    for s in scenes_meta:
        for kf in s["keyframes"]:
            nodes, edges = infer_keyframe(
                model=model,
                postprocessor=postprocessor,
                image_path=kf["image_path"],
                categories=categories,
                predicates=predicates,
                device=device,
                score_thr=args.score_threshold
            )
            kf["nodes"] = nodes
            kf["edges"] = edges
            processed_count += 1
            print(f"  [{processed_count}/{total_kf}] {s['scene_id']} @ {kf['timestamp_sec']}s: {len(nodes)} objects, {len(edges)} relations")

    # Step 4: Export intermediate JSON
    out_json = Path(args.output_dir) / "detector_detections.json"
    export_payload = {
        "video_path": os.path.abspath(args.video),
        "fps": fps,
        "categories": categories,
        "predicates": predicates,
        "scenes": scenes_meta
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Stage 1 Completed! Output saved to: {out_json}")


if __name__ == '__main__':
    main()
