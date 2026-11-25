import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))

from grounding import SemanticGroundingEngine
from relations import compute_segment_relations_and_triplets
from statistics import aggregate_support_statistics
from reasoning import evaluate_test_videos


def run_stage2_reasoning(
    detections_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json",
    universe_path="/home/nguyendinhquy/OvSGTR/vn_tourism_visual_universe.json",
    prototype_cache="/home/nguyendinhquy/OvSGTR/fkg/output/siglip2_prototypes_207x768.pt",
    output_dir="/home/nguyendinhquy/OvSGTR/fkg/output",
    beta=1.0
):
    print("=" * 70)
    print("STAGE 2: Semantic Grounding, Fuzzy Relations, Statistics & Reasoning")
    print(f"Beta: {beta} | Universe: {universe_path}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Stage 1 Detections
    assert os.path.exists(detections_path), f"Detection file not found: {detections_path}"
    with open(detections_path, "r", encoding="utf-8") as f:
        det_data = json.load(f)

    video_records = det_data.get("videos", [])
    print(f"[INFO] Loaded Stage 1 Detections for {len(video_records)} videos.")

    # 2. Initialize SigLIP 2 Grounding Engine (Step 5)
    engine = SemanticGroundingEngine(
        universe_path=universe_path,
        cache_path=prototype_cache,
        device="cuda"
    )
    print(f"[INFO] Grounding Engine ready. Concepts: {len(engine.concepts)}")

    # 3. Process each video's segments: Grounding (Step 6-7) & Relations (Step 8)
    support_segments_by_landmark = defaultdict(list)
    test_videos_by_landmark = defaultdict(list)

    total_videos = len(video_records)
    start_time = time.time()

    for v_idx, v_rec in enumerate(video_records, 1):
        vid = v_rec["video_id"]
        lm = v_rec["landmark"]
        split = v_rec["split"]
        v_path = v_rec["video_path"]
        segments = v_rec.get("segments", [])

        print(f"[{v_idx}/{total_videos}] Grounding {split.upper()} {lm} (vid={vid}, {len(segments)} segs)...")

        # Open video capture once per video
        cap = cv2.VideoCapture(v_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        processed_segments = []

        for seg in segments:
            s_idx = seg["segment_idx"]
            f_indices = seg.get("frame_indices", [])
            tracks = seg.get("tracks", [])

            # Fetch frames from video
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

            # Step 6: Encode crops and score tracks
            track_memberships = []
            for t in tracks:
                probs = engine.encode_crops_and_score_track(t, seg_frames, orig_w, orig_h)
                track_memberships.append(probs)

            # Step 7: Pool segment representation via Noisy-OR & Context
            ctx_vec = engine.score_context_on_full_frames(seg_frames)
            seg_concept_mem, full_seg_vec = engine.pool_segment_representation(track_memberships, ctx_vec)

            # Step 8: Compute 7 fuzzy geometric relations & sparse triplets
            sparse_triplets = compute_segment_relations_and_triplets(
                tracks, track_memberships, rel_threshold=0.20, top_concepts_per_track=5
            )

            p_seg = {
                "segment_idx": s_idx,
                "concept_membership": seg_concept_mem,
                "full_segment_vector": full_seg_vec,
                "sparse_triplets": sparse_triplets,
                "num_tracks": len(tracks),
                "num_triplets": len(sparse_triplets)
            }
            processed_segments.append(p_seg)

        cap.release()

        if split == "support":
            support_segments_by_landmark[lm].extend(processed_segments)
        else:
            test_videos_by_landmark[lm].append({
                "video_id": vid,
                "segments": processed_segments
            })

    # Step 9: Aggregate Support Statistics & Profiles
    print("\n" + "=" * 70)
    print("STEP 9: AGGREGATING SUPPORT PROFILES FOR 10 LANDMARKS")
    print("=" * 70)
    profiles = aggregate_support_statistics(support_segments_by_landmark, num_concepts=len(engine.concepts))

    # Save profiles
    prof_out_path = os.path.join(output_dir, "support_profiles.json")
    saveable_profiles = {
        "landmarks": profiles["landmarks"],
        "support_counts": profiles["support_counts"],
        "norm_constants": profiles["norm_constants"].tolist(),
        "relation_membership_dicts": profiles["relation_membership_dicts"]
    }
    with open(prof_out_path, "w", encoding="utf-8") as f:
        json.dump(saveable_profiles, f, indent=2)
    np.save(os.path.join(output_dir, "node_membership_matrix.npy"), profiles["node_membership_matrix"])
    np.save(os.path.join(output_dir, "discriminativeness_matrix.npy"), profiles["discriminativeness_matrix"])
    print(f"[INFO] Support profiles saved to {prof_out_path}")

    # Steps 10-13: Evidence Scoring, Aggregation, Confusion Matrix & Evaluation
    print("\n" + "=" * 70)
    print("STEPS 10-13: REASONING & EVALUATING 20 TEST VIDEOS")
    print("=" * 70)
    eval_results = evaluate_test_videos(
        test_video_segments=test_videos_by_landmark,
        profiles=profiles,
        concepts_meta=engine.concepts,
        beta=beta
    )

    eval_out_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"[INFO] Evaluation results saved to {eval_out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {eval_results['overall_accuracy']*100:.2f}% ({eval_results['total_correct']}/{eval_results['total_test_videos']})")
    print(f"Mean Margin (Correct):   {eval_results['mean_margin_correct']:.4f}")
    print(f"Mean Margin (Incorrect): {eval_results['mean_margin_incorrect']:.4f}")

    print("\nPer-Landmark Accuracy:")
    for lm, acc in eval_results["per_landmark_accuracy"].items():
        print(f"  {lm:35s}: {acc*100:5.1f}%")

    print("\nEarly Recognition Curve:")
    for prefix, acc in eval_results["early_recognition_curve"].items():
        print(f"  {prefix:15s}: {acc*100:5.1f}%")

    print("\nTop-5 Contributing Concepts per Landmark (Diagnostic):")
    for lm, top_list in eval_results["top5_contributing_concepts"].items():
        c_strs = [f"{item['label_en']} ({item['score']:.3f})" for item in top_list]
        print(f"  {lm:35s}: {', '.join(c_strs)}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"STAGE 2 COMPLETE in {total_time:.2f}s.")
    print("=" * 70)


if __name__ == "__main__":
    run_stage2_reasoning()
