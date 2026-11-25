import os
import sys
import json
import time
import pickle
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))

from grounding import SemanticGroundingEngine
from relations import compute_segment_relations_and_triplets


def build_90_grounded_videos(
    stage1_90_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections_90vids.json",
    existing_grounded_cache="/home/nguyendinhquy/OvSGTR/fkg/output/grounded_segments_cache.pkl",
    existing_stage1_50="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json",
    output_cache_path="/home/nguyendinhquy/OvSGTR/fkg/output/grounded_videos_90.pkl",
    universe_path="/home/nguyendinhquy/OvSGTR/vn_tourism_visual_universe.json",
    prototype_cache="/home/nguyendinhquy/OvSGTR/fkg/output/siglip2_prototypes_207x768.pt",
    device="cuda"
):
    print("=" * 70)
    print("STAGE 2: Grounding and Relation Extraction for 90 Videos")
    print("=" * 70)

    # 1. Load Stage 1 90 videos
    with open(stage1_90_path, "r", encoding="utf-8") as f:
        s1_90 = json.load(f)
    videos_90 = s1_90.get("videos", [])
    print(f"[INFO] Loaded {len(videos_90)} videos from {stage1_90_path}")

    # 2. Check if final 90 cache already exists
    if os.path.exists(output_cache_path):
        print(f"[INFO] Loading cached 90 grounded videos from {output_cache_path}...")
        with open(output_cache_path, "rb") as f:
            data = pickle.load(f)
        return data["videos_by_landmark"], data["concepts"]

    # 3. Load existing 50 grounded videos if available
    cached_grounded_by_vid = {}
    concepts_meta = None
    if os.path.exists(existing_grounded_cache) and os.path.exists(existing_stage1_50):
        print(f"[INFO] Loading existing 50 grounded videos from {existing_grounded_cache}...")
        with open(existing_grounded_cache, "rb") as f:
            c50 = pickle.load(f)
        with open(existing_stage1_50, "r", encoding="utf-8") as f:
            s50 = json.load(f)

        s1_50_vids = s50["videos"]
        sup_cache = c50["support_segments"]
        test_cache = c50["test_videos"]
        concepts_meta = c50["engine_concepts"]
        landmarks = list(sup_cache.keys())

        for lm in landmarks:
            lm_s1_sup = [v for v in s1_50_vids if v["landmark"] == lm and v["split"] == "support"]
            sup_counts = [len(v["segments"]) for v in lm_s1_sup]
            offset = 0
            for v_rec, count in zip(lm_s1_sup, sup_counts):
                v_segs = sup_cache[lm][offset : offset + count]
                offset += count
                cached_grounded_by_vid[v_rec["video_id"]] = v_segs

            for v_rec in test_cache[lm]:
                cached_grounded_by_vid[v_rec["video_id"]] = v_rec["segments"]

        print(f"[INFO] Successfully reused {len(cached_grounded_by_vid)} already grounded videos!")

    # 4. Initialize Grounding Engine for new videos
    new_vids_to_process = [v for v in videos_90 if v["video_id"] not in cached_grounded_by_vid]
    print(f"[INFO] New videos needing SigLIP 2 grounding & relations: {len(new_vids_to_process)}")

    engine = None
    if new_vids_to_process:
        engine = SemanticGroundingEngine(
            universe_path=universe_path,
            cache_path=prototype_cache,
            device=device
        )
        concepts_meta = engine.concepts

    t0 = time.time()
    for idx, v_rec in enumerate(new_vids_to_process, 1):
        vid = v_rec["video_id"]
        lm = v_rec["landmark"]
        v_path = v_rec["video_path"]
        segments = v_rec.get("segments", [])

        print(f"[{idx}/{len(new_vids_to_process)}] Grounding {lm} (vid={vid}, {len(segments)} segs)...")
        cap = cv2.VideoCapture(v_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        processed_segments = []
        for seg in segments:
            s_idx = seg["segment_idx"]
            f_indices = seg.get("frame_indices", [])
            tracks = seg.get("tracks", [])

            seg_frames = []
            for local_idx, f_abs in enumerate(f_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_abs)
                ret, frame_bgr = cap.read()
                if ret and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    seg_frames.append({
                        "frame_idx": local_idx,
                        "abs_frame_idx": f_abs,
                        "image": frame_rgb
                    })

            # Fast batched track scoring
            track_memberships = engine.score_all_tracks_in_segment(tracks, seg_frames, orig_w, orig_h, batch_size=64)
            ctx_vec = engine.score_context_on_full_frames(seg_frames)
            seg_concept_mem, full_seg_vec = engine.pool_segment_representation(track_memberships, ctx_vec)

            sparse_triplets = compute_segment_relations_and_triplets(
                tracks, track_memberships, rel_threshold=0.20, top_concepts_per_track=5
            )

            processed_segments.append({
                "segment_idx": s_idx,
                "concept_membership": seg_concept_mem,
                "full_segment_vector": full_seg_vec,
                "sparse_triplets": sparse_triplets,
                "num_tracks": len(tracks),
                "num_triplets": len(sparse_triplets)
            })

        cap.release()
        cached_grounded_by_vid[vid] = processed_segments

    # 5. Assemble all 90 videos grouped by landmark
    videos_by_landmark = defaultdict(list)
    for v_rec in videos_90:
        vid = v_rec["video_id"]
        lm = v_rec["landmark"]
        if vid in cached_grounded_by_vid:
            videos_by_landmark[lm].append({
                "video_id": vid,
                "landmark": lm,
                "video_path": v_rec["video_path"],
                "duration": v_rec.get("duration", 0),
                "segments": cached_grounded_by_vid[vid]
            })

    total_assembled = sum(len(v) for v in videos_by_landmark.values())
    print(f"\n[INFO] Assembled {total_assembled} grounded videos across {len(videos_by_landmark)} landmarks.")
    for lm, vids in videos_by_landmark.items():
        print(f"  - {lm:35s}: {len(vids)} videos")

    # Save to disk
    with open(output_cache_path, "wb") as f:
        pickle.dump({
            "videos_by_landmark": dict(videos_by_landmark),
            "concepts": concepts_meta
        }, f)
    print(f"[INFO] Saved 90 grounded videos cache to {output_cache_path}")

    return videos_by_landmark, concepts_meta


if __name__ == "__main__":
    build_90_grounded_videos()
