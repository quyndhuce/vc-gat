import os
import json
import cv2
import numpy as np
from collections import defaultdict

TARGET_LANDMARKS = [
    "Hoan Kiem Lake",
    "Van Mieu Quoc Tu Giam",
    "HoangThanhThangLong",
    "West Lake",
    "One Pillar Pagoda",
    "Ho Chi Minh's Mausoleum",
    "The Old Quarter",
    "Vietnam National Museum of History",
    "Bat Trang Pottery Village",
    "Van Phuc Silk Village"
]


def load_dataset_split(metadata_path="/home/nguyendinhquy/graph-tourism/fuzzy_graph/output_segments/segments_metadata.json",
                       fuzzy_graph_root="/home/nguyendinhquy/graph-tourism/fuzzy_graph",
                       num_support=6,
                       num_test=3):
    """
    Parses segments_metadata.json, groups segments by video_id and landmark,
    and returns a deterministic split:
      - 10 landmarks
      - default 6 support + 3 test = 9 videos per landmark (90 videos total)
    """
    assert os.path.exists(metadata_path), f"Metadata not found: {metadata_path}"

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    landmark_videos = defaultdict(list)
    seen_vids = set()

    for r in records:
        src = r.get("source_video_path", "")
        for lm in TARGET_LANDMARKS:
            if f"/{lm}/" in src:
                vid = r.get("video_id")
                seg_path = os.path.join(fuzzy_graph_root, r.get("output_video_path", ""))
                if vid not in seen_vids and os.path.exists(seg_path):
                    cap = cv2.VideoCapture(seg_path)
                    dur = r.get("cut_duration", 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if fc > 25 and dur >= 5.0:
                        seen_vids.add(vid)
                        landmark_videos[lm].append({
                            "landmark": lm,
                            "video_id": vid,
                            "segment_id": r.get("segment_id"),
                            "path": seg_path,
                            "duration": dur,
                            "fps": fps,
                            "frame_count": fc
                        })
                break

    total_need = num_support + num_test
    dataset = {
        "landmarks": TARGET_LANDMARKS,
        "support": {},
        "test": {}
    }

    for lm in TARGET_LANDMARKS:
        vids = landmark_videos[lm][:total_need]
        assert len(vids) >= total_need, f"Landmark {lm} only has {len(vids)} valid videos, need {total_need}!"
        dataset["support"][lm] = vids[:num_support]
        dataset["test"][lm] = vids[num_support:total_need]

    return dataset


def get_all_90_videos(metadata_path="/home/nguyendinhquy/graph-tourism/fuzzy_graph/output_segments/segments_metadata.json",
                       fuzzy_graph_root="/home/nguyendinhquy/graph-tourism/fuzzy_graph",
                       num_per_lm=9):
    """
    Returns list of 90 videos (9 per landmark) and list of landmark names.
    """
    assert os.path.exists(metadata_path), f"Metadata not found: {metadata_path}"

    with open(metadata_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    landmark_videos = defaultdict(list)
    seen_vids = set()

    for r in records:
        src = r.get("source_video_path", "")
        for lm in TARGET_LANDMARKS:
            if f"/{lm}/" in src:
                vid = r.get("video_id")
                seg_path = os.path.join(fuzzy_graph_root, r.get("output_video_path", ""))
                if vid not in seen_vids and os.path.exists(seg_path):
                    cap = cv2.VideoCapture(seg_path)
                    dur = r.get("cut_duration", 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if fc > 25 and dur >= 5.0:
                        seen_vids.add(vid)
                        landmark_videos[lm].append({
                            "landmark": lm,
                            "video_id": vid,
                            "segment_id": r.get("segment_id"),
                            "path": seg_path,
                            "duration": dur,
                            "fps": fps,
                            "frame_count": fc
                        })
                break

    all_vids = []
    for lm in TARGET_LANDMARKS:
        vids = landmark_videos[lm][:num_per_lm]
        assert len(vids) >= num_per_lm, f"Landmark {lm} only has {len(vids)} valid videos, need {num_per_lm}!"
        all_vids.extend(vids)

    return all_vids, TARGET_LANDMARKS


def get_segment_frames(video_path, segment_duration=2.0, num_frames=4):
    """
    Bước 1: Cắt video thành các đoạn 2 giây. Mỗi đoạn lấy 3 đến 5 frame cách đều (mặc định 4 frame).
    Trả về list các segment, mỗi segment gồm:
      - segment_idx: int
      - start_sec: float
      - end_sec: float
      - frames: list of dict(frame_idx, timestamp_sec, image_rgb)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total_frames / fps

    frames_per_seg = int(segment_duration * fps)
    if frames_per_seg <= 0:
        frames_per_seg = int(2.0 * fps)

    num_segments = max(1, int(np.ceil(total_frames / frames_per_seg)))
    segments = []

    for s_idx in range(num_segments):
        start_f = s_idx * frames_per_seg
        end_f = min(total_frames, (s_idx + 1) * frames_per_seg)
        seg_len = end_f - start_f
        if seg_len <= 0:
            continue

        # Sample num_frames evenly
        k_count = min(num_frames, max(1, seg_len))
        step = seg_len / (k_count + 1)
        sampled_indices = [int(start_f + step * (i + 1)) for i in range(k_count)]
        sampled_indices = sorted(list(set(sampled_indices)))

        seg_frames = []
        for f_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sec = f_idx / fps
            seg_frames.append({
                "frame_idx": f_idx,
                "timestamp_sec": round(sec, 3),
                "image": frame_rgb
            })

        if seg_frames:
            segments.append({
                "segment_idx": s_idx,
                "start_sec": round(start_f / fps, 2),
                "end_sec": round(end_f / fps, 2),
                "frames": seg_frames
            })

    cap.release()
    return segments
