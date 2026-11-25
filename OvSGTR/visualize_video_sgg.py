#!/usr/bin/env python3
"""
Video Scene Graph Visualizer:
Renders an animated MP4 visualization of the video scene graph pipeline:
1. Annotated video frames showing bounding boxes with consistent track_id colors.
2. Directed relationship arrows connecting interacting tracked entities (Subject -> Predicate -> Object).
3. Dynamic side panel displaying:
   - Scene metadata & timestamps
   - Active tracked entities (with SigLIP 2 taxonomy & material/colour attributes)
   - Scene graph relation triplets with confidence scores
4. Encodes using ffmpeg (H.264 / yuv420p) for high compatibility.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Consistent color palette for track_ids
PALETTE = [
    (230, 57, 70),    # Red
    (244, 162, 97),   # Coral / Orange
    (42, 157, 143),   # Persian Green / Teal
    (233, 196, 106),  # Gold / Yellow
    (52, 152, 219),   # Dodger Blue
    (155, 89, 182),   # Amethyst Purple
    (46, 204, 113),   # Emerald Green
    (231, 76, 60),    # Alizarin Red
    (26, 188, 156),   # Turquoise
    (243, 156, 18),   # Orange Yellow
    (142, 68, 173),   # Violet
    (39, 174, 96),    # Nephritis Green
    (211, 84, 0),     # Pumpkin Orange
    (192, 57, 43),    # Crimson
    (127, 140, 141),  # Slate Grey
    (189, 195, 199)   # Silver
]

def get_track_color(track_id: str):
    try:
        idx = int(track_id.replace('T', '')) - 1
        return PALETTE[idx % len(PALETTE)]
    except Exception:
        return (100, 200, 255)


def draw_arrow(draw, pt1, pt2, color=(255, 215, 64), width=3, text=None, font=None):
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw main line
    draw.line([(x1, y1), (x2, y2)], fill=color + (240,), width=width)

    # Arrow head
    angle = np.arctan2(y2 - y1, x2 - x1)
    head_len = 16
    arrow_angle = np.pi / 6

    ax1 = x2 - head_len * np.cos(angle - arrow_angle)
    ay1 = y2 - head_len * np.sin(angle - arrow_angle)
    ax2 = x2 - head_len * np.cos(angle + arrow_angle)
    ay2 = y2 - head_len * np.sin(angle + arrow_angle)

    draw.polygon([(x2, y2), (ax1, ay1), (ax2, ay2)], fill=color + (255,))

    # Relationship label at midpoint
    if text and font:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        try:
            bb = font.getbbox(text)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(text) * 8, 14

        draw.rectangle([mx - tw/2 - 4, my - th/2 - 3, mx + tw/2 + 4, my + th/2 + 3], fill=(20, 24, 36, 230), outline=color + (255,), width=1)
        draw.text((mx - tw/2, my - th/2 - 1), text, fill=(255, 255, 255), font=font)


def render_side_panel(scene_info, keyframe_info, width=640, height=720):
    panel = Image.new("RGB", (width, height), (15, 19, 28))
    draw = ImageDraw.Draw(panel)

    try:
        font_h1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_h2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_h1 = font_h2 = font_sub = font_bold = font_text = ImageFont.load_default()

    PAD = 16
    y = PAD

    # --- Header Banner ---
    scene_id_str = f"{scene_info['scene_id'].upper()}: {scene_info['start_seconds']}s - {scene_info['end_seconds']}s (len {scene_info['duration_seconds']}s)"
    draw.text((PAD, y), scene_id_str, fill=(255, 215, 64), font=font_h1)
    y += 24

    kf_str = f"Keyframe #{keyframe_info['frame_idx']} @ t = {keyframe_info['timestamp_sec']}s | Pipeline: OvSGTR + SigLIP 2"
    draw.text((PAD, y), kf_str, fill=(140, 160, 190), font=font_sub)
    y += 20
    draw.line([(PAD, y), (width - PAD, y)], fill=(45, 58, 82), width=1)
    y += 12

    # --- Section 1: Tracked Entities in Frame ---
    nodes = keyframe_info.get("nodes", [])
    draw.text((PAD, y), f"ĐỐI TƯỢNG ĐỒNG NHẤT (Tracked Entities: {len(nodes)})", fill=(100, 210, 255), font=font_h2)
    y += 22

    # Render up to 6 entities to fit panel
    for n in nodes[:6]:
        tid = n.get("track_id", f"N{n['node_idx']}")
        col = get_track_color(tid)
        det_lbl = n["detector_label"]
        sig_lbl = n.get("siglip2_label", det_lbl)
        is_ref = n.get("is_refined_by_siglip2", False)

        # Track badge
        draw.rectangle([PAD, y, PAD + 38, y + 20], fill=col)
        draw.text((PAD + 5, y + 3), tid, fill=(255, 255, 255), font=font_bold)

        # Labels
        x_txt = PAD + 46
        if is_ref:
            main_label = f"{sig_lbl}"
            draw.text((x_txt, y), main_label, fill=(150, 250, 170), font=font_bold)
            attr = n.get("attributes", {})
            mat = attr.get("material")
            c_dom = attr.get("dominant_colour")
            mat_txt = f"Chất liệu: {mat['label_en']}" if mat else ""
            col_txt = f"Màu: {c_dom['label_en']}" if c_dom else ""
            details = " | ".join(filter(None, [mat_txt, col_txt]))
            draw.text((x_txt, y + 16), details or f"Detector: {det_lbl} ({n['detector_score']:.2f})", fill=(245, 190, 110), font=font_sub)
            y += 34
        else:
            main_label = f"{det_lbl} (Tự nhiên / Con người)"
            draw.text((x_txt, y), main_label, fill=(180, 200, 220), font=font_bold)
            draw.text((x_txt, y + 16), f"Score: {n['detector_score']:.2f} | Giữ nguyên nhãn gốc", fill=(130, 150, 175), font=font_sub)
            y += 34

    y += 6
    draw.line([(PAD, y), (width - PAD, y)], fill=(45, 58, 82), width=1)
    y += 12

    # --- Section 2: Active Scene Graph Triplets ---
    edges = keyframe_info.get("edges", [])
    draw.text((PAD, y), f"QUAN HỆ NGỮ CẢNH (Active Triplets: {len(edges)})", fill=(255, 170, 100), font=font_h2)
    y += 22

    # Show top triplets
    max_triplets = min(8, len(edges))
    for i, e in enumerate(edges[:max_triplets]):
        s_tid = e.get("subject_track_id", "")
        s_lbl = e.get("subject_siglip2_label", e.get("subject_label", ""))
        o_tid = e.get("object_track_id", "")
        o_lbl = e.get("object_siglip2_label", e.get("object_label", ""))
        pred = e.get("predicate", "")
        conf = e.get("confidence", 0.0)

        # Index #01
        draw.text((PAD, y), f"#{i+1:02d}", fill=(110, 130, 160), font=font_sub)

        # Subject
        sx = PAD + 28
        scol = get_track_color(s_tid)
        draw.text((sx, y), f"[{s_tid}] {s_lbl[:14]}", fill=scol, font=font_bold)

        # Arrow + Predicate
        try:
            sw = font_bold.getbbox(f"[{s_tid}] {s_lbl[:14]}")[2] - font_bold.getbbox(f"[{s_tid}] {s_lbl[:14]}")[0]
        except Exception:
            sw = len(f"[{s_tid}] {s_lbl[:14]}") * 8
        px = sx + sw + 6
        draw.text((px, y), f"—{pred}→", fill=(255, 140, 100), font=font_bold)

        # Object
        try:
            pw = font_bold.getbbox(f"—{pred}→")[2] - font_bold.getbbox(f"—{pred}→")[0]
        except Exception:
            pw = len(f"—{pred}→") * 8
        ox = px + pw + 6
        ocol = get_track_color(o_tid)
        draw.text((ox, y), f"[{o_tid}] {o_lbl[:14]}", fill=ocol, font=font_bold)

        # Score
        conf_str = f"conf: {conf:.2f}"
        try:
            cw = font_sub.getbbox(conf_str)[2] - font_sub.getbbox(conf_str)[0]
        except Exception:
            cw = 50
        draw.text((width - PAD - cw, y), conf_str, fill=(160, 175, 200), font=font_sub)

        y += 24

    return panel


def render_keyframe_composite(img_path, scene_info, keyframe_info, target_h=720):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    # Scale image to target height
    scale = target_h / orig_h
    new_w = int(orig_w * scale)
    img_resized = img.resize((new_w, target_h), Image.LANCZOS)
    draw = ImageDraw.Draw(img_resized, "RGBA")

    try:
        font_box = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_rel = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font_box = font_rel = ImageFont.load_default()

    nodes = keyframe_info.get("nodes", [])
    node_centers = {}

    # 1. Draw Bounding Boxes
    for n in nodes:
        tid = n.get("track_id", f"N{n['node_idx']}")
        col = get_track_color(tid)
        b = n["box_2d"]
        x0, y0, x1, y1 = [v * scale for v in b]

        # Save center for relations
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        node_centers[tid] = (cx, cy)

        # Box
        draw.rectangle([x0, y0, x1, y1], outline=col + (255,), width=3)

        # Label badge
        lbl = n.get("siglip2_label", n["detector_label"])
        txt = f"[{tid}] {lbl} ({n['detector_score']:.2f})"
        try:
            bb = font_box.getbbox(txt)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = len(txt) * 8, 14

        badge_w = tw + 8
        badge_h = th + 6
        if y0 >= badge_h:
            b_y0 = y0 - badge_h
            b_y1 = y0
        else:
            b_y0 = y0
            b_y1 = y0 + badge_h

        draw.rectangle([x0, b_y0, x0 + badge_w, b_y1], fill=col + (220,))
        draw.text((x0 + 4, b_y0 + 2), txt, fill=(255, 255, 255), font=font_box)

    # 2. Draw Top Relationship Arrows (limit to top 4 to avoid visual clutter)
    edges = keyframe_info.get("edges", [])
    for e in edges[:4]:
        s_tid = e.get("subject_track_id")
        o_tid = e.get("object_track_id")
        pred = e.get("predicate", "")

        if s_tid in node_centers and o_tid in node_centers:
            pt1 = node_centers[s_tid]
            pt2 = node_centers[o_tid]
            # Offset slightly if overlapping
            draw_arrow(draw, pt1, pt2, color=(255, 215, 64), width=3, text=f" {pred} ", font=font_rel)

    # 3. Render Right-side Information Panel
    panel = render_side_panel(scene_info, keyframe_info, width=640, height=target_h)

    # 4. Combine Left + Right
    total_w = new_w + panel.width
    composite = Image.new("RGB", (total_w, target_h), (15, 19, 28))
    composite.paste(img_resized, (0, 0))
    composite.paste(panel, (new_w, 0))

    return composite


def main():
    p = argparse.ArgumentParser(description="Render Video Scene Graph Visualization to MP4")
    p.add_argument('--video_output_dir', type=str, default='video_output', help="Directory containing scene_XX_scene_graph.json")
    p.add_argument('--output_video', type=str, default='video_output/visualized_scene_graph.mp4', help="Output MP4 path")
    p.add_argument('--fps', type=int, default=15, help="Video FPS")
    p.add_argument('--seconds_per_keyframe', type=float, default=2.5, help="Hold duration per keyframe in seconds")
    args = p.parse_args()

    out_dir = Path(args.video_output_dir)
    scene_files = sorted(list(out_dir.glob("scene_*_scene_graph.json")))

    assert len(scene_files) > 0, f"No scene graph JSON files found in: {args.video_output_dir}"

    print(f"\n[INFO] Found {len(scene_files)} scene JSON files. Starting visualization rendering...")

    frames_per_keyframe = int(args.fps * args.seconds_per_keyframe)
    all_composite_frames = []

    for s_file in scene_files:
        with open(s_file, "r", encoding="utf-8") as f:
            s_data = json.load(f)

        s_info = s_data["scene_metadata"]
        keyframes = s_data["keyframes"]

        print(f"  Rendering {s_info['scene_id']} ({len(keyframes)} keyframes)...")

        for kf in keyframes:
            img_path = kf["image_path"]
            if not os.path.exists(img_path):
                print(f"    [WARN] Image path not found: {img_path}")
                continue

            comp = render_keyframe_composite(img_path, s_info, kf, target_h=720)
            comp_np = cv2.cvtColor(np.array(comp), cv2.COLOR_RGB2BGR)

            # Repeat frame to give user time to read and observe
            for _ in range(frames_per_keyframe):
                all_composite_frames.append(comp_np)

    if not all_composite_frames:
        print("[ERROR] No frames rendered.")
        sys.exit(1)

    # Encode with cv2.VideoWriter + ffmpeg re-encode for standard H.264
    tmp_avi = out_dir / "temp_render.avi"
    h, w, _ = all_composite_frames[0].shape
    print(f"\n[INFO] Writing {len(all_composite_frames)} frames ({w}x{h} @ {args.fps} fps)...")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(str(tmp_avi), fourcc, args.fps, (w, h))
    for f in all_composite_frames:
        writer.write(f)
    writer.release()

    # Re-encode to H.264 MP4 with ffmpeg
    final_mp4 = Path(args.output_video)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_avi),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "22",
        str(final_mp4)
    ]
    print(f"[INFO] Running ffmpeg encoding → {final_mp4} ...")
    subprocess.run(cmd, check=True)

    if tmp_avi.exists():
        tmp_avi.unlink()

    print(f"\n{'='*70}")
    print(f"[SUCCESS] Visualized Scene Graph video created:")
    print(f"  Path: {final_mp4}")
    print(f"  Size: {final_mp4.stat().st_size / (1024*1024):.2f} MB")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
