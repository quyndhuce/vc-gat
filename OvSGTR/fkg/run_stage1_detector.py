import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))

from dataset import load_dataset_split, get_segment_frames, TARGET_LANDMARKS
from detector import CoarseRegionDetector


def run_stage1_detection(
    metadata_path="/home/nguyendinhquy/graph-tourism/fuzzy_graph/output_segments/segments_metadata.json",
    output_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections_90vids.json",
    existing_cache="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json",
    num_support=6,
    num_test=3,
    segment_duration=2.0,
    num_frames=4,
    score_threshold=0.15
):
    print("=" * 70)
    print("STAGE 1: Frame Extraction, Coarse Detection, Filtering & Tracking")
    print(f"Dataset: {num_support} sup + {num_test} test per landmark = {(num_support+num_test)*len(TARGET_LANDMARKS)} videos total")
    print(f"Segment Duration: {segment_duration}s | Frames per segment: {num_frames}")
    print("=" * 70)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Load dataset split (e.g. 60 support + 30 test = 90 videos)
    dataset = load_dataset_split(metadata_path, num_support=num_support, num_test=num_test)
    print(f"[INFO] Loaded split: {len(TARGET_LANDMARKS)} landmarks.")
    for lm in TARGET_LANDMARKS:
        n_sup = len(dataset["support"][lm])
        n_test = len(dataset["test"][lm])
        print(f"  - {lm}: {n_sup} support, {n_test} test")

    # 2. Check existing progress for resuming
    processed_videos = {}
    source_cache = output_path if os.path.exists(output_path) else existing_cache
    if os.path.exists(source_cache):
        try:
            with open(source_cache, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for item in saved.get("videos", []):
                    processed_videos[item["video_id"]] = item
            print(f"[INFO] Resuming: found {len(processed_videos)} already processed videos from {source_cache}")
        except Exception as e:
            print(f"[WARN] Could not parse existing output file ({e}), starting fresh.")

    # 3. Initialize Coarse Region Detector (GroundingDINO Swin-B)
    detector = CoarseRegionDetector(device='cuda')

    total_videos = sum(len(dataset[s][lm]) for s in ["support", "test"] for lm in TARGET_LANDMARKS)
    video_idx = 0
    start_time = time.time()

    all_video_records = list(processed_videos.values())

    for split in ["support", "test"]:
        for lm in TARGET_LANDMARKS:
            vids = dataset[split][lm]
            for v_info in vids:
                video_idx += 1
                vid = v_info["video_id"]
                v_path = v_info["path"]

                if vid in processed_videos:
                    continue

                v_start = time.time()
                print(f"\n[{video_idx}/{total_videos}] [{split.upper()}] {lm} (vid={vid})")
                print(f"  Path: {v_path}")

                # Step 1: Slice video into 2-sec segments with 4 frames each
                segments = get_segment_frames(v_path, segment_duration=segment_duration, num_frames=num_frames)
                print(f"  Extracted {len(segments)} segments ({len(segments)*num_frames} frames)")

                seg_records = []
                total_tracks = 0

                for seg in segments:
                    s_idx = seg["segment_idx"]
                    frames = seg["frames"]

                    # Step 2: Detect coarse regions on each frame
                    frames_filtered = []
                    for f in frames:
                        raw_boxes, orig_w, orig_h = detector.detect_frame(f["image"], score_threshold=score_threshold)
                        # Step 3: Filter & merge regions
                        merged_boxes = detector.filter_and_merge_regions(raw_boxes, orig_w, orig_h)
                        frames_filtered.append(merged_boxes)

                    # Step 4: Track regions across frames within this segment
                    tracks = detector.track_regions_in_segment(frames_filtered, iou_threshold=0.5)
                    total_tracks += len(tracks)

                    seg_records.append({
                        "segment_idx": s_idx,
                        "start_sec": seg["start_sec"],
                        "end_sec": seg["end_sec"],
                        "frame_indices": [f["frame_idx"] for f in frames],
                        "tracks": tracks
                    })

                record = {
                    "video_id": vid,
                    "landmark": lm,
                    "split": split,
                    "video_path": v_path,
                    "duration": v_info["duration"],
                    "fps": v_info["fps"],
                    "total_segments": len(seg_records),
                    "total_tracks": total_tracks,
                    "segments": seg_records
                }

                all_video_records.append(record)
                processed_videos[vid] = record

                v_elapsed = time.time() - v_start
                print(f"  Done in {v_elapsed:.2f}s | Tracks: {total_tracks} ({total_tracks/max(1, len(seg_records)):.1f}/seg)")

                # Save atomically
                tmp_path = output_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "total_videos": len(all_video_records),
                        "videos": all_video_records
                    }, f, indent=2)
                os.replace(tmp_path, output_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"STAGE 1 COMPLETED in {total_time:.2f}s.")
    print(f"Total videos processed: {len(all_video_records)}")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_stage1_detection()
