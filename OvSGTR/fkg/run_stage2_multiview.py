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
from clustering import build_multiview_support_profiles, evaluate_multiview_test_videos


def load_or_compute_grounded_segments(
    detections_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json",
    cache_path="/home/nguyendinhquy/OvSGTR/fkg/output/grounded_segments_cache.pkl",
    universe_path="/home/nguyendinhquy/OvSGTR/vn_tourism_visual_universe.json",
    prototype_cache="/home/nguyendinhquy/OvSGTR/fkg/output/siglip2_prototypes_207x768.pt"
):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached grounded segments from {cache_path}...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["support_segments"], data["test_videos"], data["engine_concepts"]

    print("[INFO] Grounded segments cache not found. Computing with batched SigLIP 2 on GPU...")
    with open(detections_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)

    video_records = det_data.get("videos", [])
    engine = SemanticGroundingEngine(
        universe_path=universe_path,
        cache_path=prototype_cache,
        device="cuda"
    )

    support_segments = defaultdict(list)
    test_videos = defaultdict(list)
    total_videos = len(video_records)

    start_t = time.time()
    for v_idx, v_rec in enumerate(video_records, 1):
        vid = v_rec["video_id"]
        lm = v_rec["landmark"]
        split = v_rec["split"]
        v_path = v_rec["video_path"]
        segments = v_rec.get("segments", [])

        print(f"[{v_idx}/{total_videos}] Grounding {split.upper()} {lm} (vid={vid}, {len(segments)} segs)...")
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

        if split == "support":
            support_segments[lm].extend(processed_segments)
        else:
            test_videos[lm].append({
                "video_id": vid,
                "segments": processed_segments
            })

    print(f"[INFO] Grounding complete in {time.time() - start_t:.2f}s. Saving cache to {cache_path}...")
    with open(cache_path, "wb") as f:
        pickle.dump({
            "support_segments": dict(support_segments),
            "test_videos": dict(test_videos),
            "engine_concepts": engine.concepts
        }, f)

    return support_segments, test_videos, engine.concepts


def run_multiview_experiment(
    detections_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json",
    output_dir="/home/nguyendinhquy/OvSGTR/fkg/output",
    beta=1.0
):
    print("=" * 70)
    print("MULTI-VIEW CLUSTERING & RE-EVALUATION FOR 0% ACCURACY LANDMARKS")
    print("=" * 70)

    support_segs, test_vids, concepts_meta = load_or_compute_grounded_segments(detections_path=detections_path)
    target_landmarks = ["Van Mieu Quoc Tu Giam", "HoangThanhThangLong", "Van Phuc Silk Village"]

    for k_views in [3, 4]:
        print("\n" + "#" * 70)
        print(f"EXPERIMENT: CLUSTERING 3 WEAK LANDMARKS INTO {k_views} VIEWS")
        print("#" * 70)

        # 1. Build Multi-view Support Profiles
        clustered_cfg = {lm: k_views for lm in target_landmarks}
        profiles = build_multiview_support_profiles(
            support_segments_by_landmark=support_segs,
            clustered_landmarks=clustered_cfg,
            n_clusters=k_views,
            num_concepts=len(concepts_meta),
            beta=beta
        )

        # Print top distinctive concepts for each view of the 3 landmarks
        print(f"\n[DIAGNOSTIC] View Profiles for Clustered Landmarks (K={k_views}):")
        for lm in target_landmarks:
            print(f"\n  --- {lm} ({k_views} Views) ---")
            for v in profiles["landmark_views"][lm]:
                v_idx = v["view_idx"]
                n_segs = v["num_segments"]
                # Top distinctive concepts in this view
                top_c = np.argsort(-v["disc"])[:4]
                c_str = ", ".join([f"{concepts_meta[c]['label_en']} ({v['disc'][c]:.2f}, mem={v['node_mem'][c]:.2f})" for c in top_c])
                print(f"    View {v_idx} ({n_segs:2d} segs): {c_str}")

        # 2. Evaluate on 20 Test Videos
        results = evaluate_multiview_test_videos(
            test_video_segments=test_vids,
            multiview_profiles=profiles,
            concepts_meta=concepts_meta,
            beta=beta
        )

        # Save results
        out_file = os.path.join(output_dir, f"evaluation_multiview_{k_views}views.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 50)
        print(f"RESULTS FOR K={k_views} VIEWS:")
        print(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}% ({results['total_correct']}/{results['total_test_videos']})")
        print(f"Mean Margin (Correct):   {results['mean_margin_correct']:.4f}")
        print(f"Mean Margin (Incorrect): {results['mean_margin_incorrect']:.4f}")
        print("\nPer-Landmark Accuracy:")
        for lm, acc in results["per_landmark_accuracy"].items():
            mark = " [CLUSTERED]" if lm in target_landmarks else ""
            print(f"  {lm:35s}: {acc*100:5.1f}%{mark}")

        print("\nEarly Recognition Curve:")
        for prefix, acc in results["early_recognition_curve"].items():
            print(f"  {prefix:15s}: {acc*100:5.1f}%")

        print(f"\nDetailed results saved to: {out_file}")


if __name__ == "__main__":
    run_multiview_experiment()
