import os
import sys
import json
import time
import pickle
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from transformers import AutoProcessor, AutoModel, AutoImageProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))

from dataset import TARGET_LANDMARKS
from baseline_siglip import LANDMARK_PROMPTS


def extract_and_cache_video_frame_features(
    stage1_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json",
    cache_siglip="/home/nguyendinhquy/OvSGTR/fkg/output/features_siglip2.pkl",
    cache_dino="/home/nguyendinhquy/OvSGTR/fkg/output/features_dinov2.pkl",
    device="cuda"
):
    """
    Trích xuất và lưu cache đặc trưng hình ảnh toàn khung hình (full-frame)
    của cả 50 video cho SigLIP 2 và DINOv2 trên GPU.
    """
    with open(stage1_path, "r", encoding="utf-8") as f:
        stage1_data = json.load(f)

    all_vids = stage1_data["videos"]
    print(f"[INFO] Total videos to extract features: {len(all_vids)}")

    # 1. SigLIP 2 Feature Extraction
    if os.path.exists(cache_siglip):
        print(f"[INFO] Found cached SigLIP 2 features at {cache_siglip}")
        with open(cache_siglip, "rb") as f:
            siglip_feats = pickle.load(f)
    else:
        print("[INFO] Extracting SigLIP 2 features for all 50 videos on GPU...")
        proc_siglip = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
        model_siglip = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device).eval()

        siglip_feats = {}
        t0 = time.time()
        for idx, v_rec in enumerate(all_vids, 1):
            vid = v_rec["video_id"]
            v_path = v_rec["video_path"]
            frame_indices = []
            for s in v_rec.get("segments", []):
                frame_indices.extend(s.get("frame_indices", []))

            cap = cv2.VideoCapture(v_path)
            frames = []
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()

            if not frames:
                siglip_feats[vid] = np.zeros((1, 768), dtype=np.float32)
                continue

            # Batch encode
            v_embeds = []
            for b in range(0, len(frames), 64):
                b_imgs = frames[b : b + 64]
                inputs = proc_siglip(images=b_imgs, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model_siglip.get_image_features(**inputs)
                    feat = out.pooler_output if hasattr(out, "pooler_output") else out
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    v_embeds.append(feat.cpu().numpy())

            siglip_feats[vid] = np.concatenate(v_embeds, axis=0)
            if idx % 10 == 0:
                print(f"  SigLIP 2: [{idx}/{len(all_vids)}] videos extracted ({time.time()-t0:.1f}s)")

        with open(cache_siglip, "wb") as f:
            pickle.dump(siglip_feats, f)
        print(f"[INFO] Saved SigLIP 2 features to {cache_siglip}")

    # 2. DINOv2 Feature Extraction
    if os.path.exists(cache_dino):
        print(f"[INFO] Found cached DINOv2 features at {cache_dino}")
        with open(cache_dino, "rb") as f:
            dino_feats = pickle.load(f)
    else:
        print("[INFO] Extracting DINOv2 features (facebook/dinov2-base) for all 50 videos on GPU...")
        proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

        dino_feats = {}
        t0 = time.time()
        for idx, v_rec in enumerate(all_vids, 1):
            vid = v_rec["video_id"]
            v_path = v_rec["video_path"]
            frame_indices = []
            for s in v_rec.get("segments", []):
                frame_indices.extend(s.get("frame_indices", []))

            cap = cv2.VideoCapture(v_path)
            frames = []
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            cap.release()

            if not frames:
                dino_feats[vid] = np.zeros((1, 768), dtype=np.float32)
                continue

            v_embeds = []
            for b in range(0, len(frames), 64):
                b_imgs = frames[b : b + 64]
                inputs = proc_dino(images=b_imgs, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model_dino(**inputs)
                    # CLS token as global frame embedding
                    feat = out.last_hidden_state[:, 0, :]
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                    v_embeds.append(feat.cpu().numpy())

            dino_feats[vid] = np.concatenate(v_embeds, axis=0)
            if idx % 10 == 0:
                print(f"  DINOv2: [{idx}/{len(all_vids)}] videos extracted ({time.time()-t0:.1f}s)")

        with open(cache_dino, "wb") as f:
            pickle.dump(dino_feats, f)
        print(f"[INFO] Saved DINOv2 features to {cache_dino}")

    return siglip_feats, dino_feats, all_vids


def compute_siglip_text_prototypes(device="cuda"):
    """
    Tạo 10 vector nguyên mẫu văn bản từ các prompt tiếng Anh cho SigLIP 2.
    """
    proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device).eval()

    prototypes = []
    for lm in TARGET_LANDMARKS:
        prompts = LANDMARK_PROMPTS[lm]
        inputs = proc(text=prompts, padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            embeds = out.pooler_output if hasattr(out, "pooler_output") else out
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            proto = embeds.mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True)
            prototypes.append(proto.cpu().numpy())

    return np.stack(prototypes, axis=0)  # (10, 768)


def evaluate_baseline_cross_validation(
    video_features_map,
    all_videos_list,
    method="image_prototype",  # 'image_prototype' or 'text_prototype'
    text_prototypes=None,
    scheme="4sup_1test"        # '4sup_1test' or '3sup_2test'
):
    """
    Chạy kiểm định chéo cho mô hình baseline:
    - method='image_prototype': Nguyên mẫu ảnh (Support videos cung cấp các ảnh mẫu để tính prototype)
    - method='text_prototype': Nguyên mẫu văn bản (Prompt tiếng Anh)
    """
    landmarks = TARGET_LANDMARKS
    num_landmarks = len(landmarks)
    lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}

    # Group videos by landmark
    vids_by_lm = defaultdict(list)
    for v in all_videos_list:
        vids_by_lm[v["landmark"]].append(v["video_id"])

    # Define folds
    if scheme == "4sup_1test":
        folds = [
            ([1, 2, 3, 4], [0]),
            ([0, 2, 3, 4], [1]),
            ([0, 1, 3, 4], [2]),
            ([0, 1, 2, 4], [3]),
            ([0, 1, 2, 3], [4]),
        ]
    else:
        folds = [
            ([0, 1, 2], [3, 4]),
            ([1, 2, 3], [4, 0]),
            ([2, 3, 4], [0, 1]),
            ([3, 4, 0], [1, 2]),
            ([4, 0, 1], [2, 3]),
        ]

    fold_accuracies = []
    confusion_matrix = np.zeros((num_landmarks, num_landmarks), dtype=np.int32)
    per_lm_correct = {lm: 0 for lm in landmarks}
    per_lm_total = {lm: 0 for lm in landmarks}

    for fold_idx, (sup_idxs, test_idxs) in enumerate(folds):
        # 1. Build prototypes for this fold
        if method == "text_prototype":
            # Fixed text prototypes across folds
            W_fold = text_prototypes  # (10, 768)
        else:
            # Build Image Prototype for each landmark from support videos
            proto_list = []
            for lm in landmarks:
                sup_vids = [vids_by_lm[lm][i] for i in sup_idxs]
                all_sup_frames = []
                for vid in sup_vids:
                    feats = video_features_map.get(vid)
                    if feats is not None and len(feats) > 0:
                        all_sup_frames.append(feats)

                if all_sup_frames:
                    cat_feats = np.concatenate(all_sup_frames, axis=0)
                    mean_f = np.mean(cat_feats, axis=0)
                    norm_f = mean_f / (np.linalg.norm(mean_f) + 1e-6)
                else:
                    norm_f = np.zeros(768, dtype=np.float32)
                proto_list.append(norm_f)

            W_fold = np.stack(proto_list, axis=0)  # (10, D)

        # 2. Test on query videos
        fold_corr = 0
        fold_tot = 0

        for gt_lm in landmarks:
            gt_idx = lm_to_idx[gt_lm]
            for t_idx in test_idxs:
                test_vid = vids_by_lm[gt_lm][t_idx]
                test_feats = video_features_map.get(test_vid)

                if test_feats is None or len(test_feats) == 0:
                    continue

                # Cosine similarities: (N_frames, 10)
                sims = test_feats @ W_fold.T
                # Mean similarity across video frames
                v_score = np.mean(sims, axis=0)
                pred_idx = int(np.argmax(v_score))

                is_c = (pred_idx == gt_idx)
                if is_c:
                    fold_corr += 1
                    per_lm_correct[gt_lm] += 1
                per_lm_total[gt_lm] += 1
                fold_tot += 1

                confusion_matrix[gt_idx, pred_idx] += 1

        fold_accuracies.append(fold_corr / max(1, fold_tot))

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    per_lm_acc = {lm: round(per_lm_correct[lm] / max(1, per_lm_total[lm]), 4) for lm in landmarks}

    return {
        "method": method,
        "scheme": scheme,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "total_evaluations": sum(per_lm_total.values()),
        "total_correct": sum(per_lm_correct.values()),
        "per_landmark_accuracy": per_lm_acc,
        "confusion_matrix": confusion_matrix.tolist()
    }


def main():
    print("=" * 70)
    print("BENCHMARK: ALL BASELINES ON 5-FOLD CROSS-VALIDATION (50 VIDEOS)")
    print("=" * 70)

    output_dir = "/home/nguyendinhquy/OvSGTR/fkg/output"
    os.makedirs(output_dir, exist_ok=True)

    siglip_feats, dino_feats, all_vids = extract_and_cache_video_frame_features()
    text_prototypes = compute_siglip_text_prototypes()

    schemes = ["4sup_1test", "3sup_2test"]

    all_results = {}

    for sch in schemes:
        print("\n" + "#" * 70)
        sch_name = "5-Fold (4 Support / 1 Test, 50 Videos)" if sch == "4sup_1test" else "5-Fold (3 Support / 2 Test, 100 Evals)"
        print(f"EVALUATING SCHEME: {sch_name}")
        print("#" * 70)

        # Baseline 1: SigLIP 2 Zero-Shot Text Prototypes
        print("\n>>> [1] Baseline: SigLIP 2 Zero-Shot Text Prototypes...")
        res_siglip_text = evaluate_baseline_cross_validation(
            siglip_feats, all_vids, method="text_prototype", text_prototypes=text_prototypes, scheme=sch
        )
        print(f"Accuracy: {res_siglip_text['mean_accuracy']*100:.2f}% ± {res_siglip_text['std_accuracy']*100:.2f}%")
        print(f"Folds: {[f'{a*100:.1f}%' for a in res_siglip_text['fold_accuracies']]}")

        # Baseline 2: SigLIP 2 Image Prototypes (Nguyên mẫu ảnh SigLIP)
        print("\n>>> [2] Baseline: SigLIP 2 Image Prototypes (Nguyên mẫu ảnh từ support videos)...")
        res_siglip_img = evaluate_baseline_cross_validation(
            siglip_feats, all_vids, method="image_prototype", scheme=sch
        )
        print(f"Accuracy: {res_siglip_img['mean_accuracy']*100:.2f}% ± {res_siglip_img['std_accuracy']*100:.2f}%")
        print(f"Folds: {[f'{a*100:.1f}%' for a in res_siglip_img['fold_accuracies']]}")

        # Baseline 3: DINOv2 Image Prototypes (Nguyên mẫu ảnh DINO)
        print("\n>>> [3] Baseline: DINOv2 Image Prototypes (Nguyên mẫu ảnh từ support videos)...")
        res_dino_img = evaluate_baseline_cross_validation(
            dino_feats, all_vids, method="image_prototype", scheme=sch
        )
        print(f"Accuracy: {res_dino_img['mean_accuracy']*100:.2f}% ± {res_dino_img['std_accuracy']*100:.2f}%")
        print(f"Folds: {[f'{a*100:.1f}%' for a in res_dino_img['fold_accuracies']]}")

        all_results[f"{sch}_siglip_text"] = res_siglip_text
        all_results[f"{sch}_siglip_img"] = res_siglip_img
        all_results[f"{sch}_dino_img"] = res_dino_img

    out_file = os.path.join(output_dir, "all_baselines_cross_validation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL BASELINES CROSS-VALIDATION COMPLETE!")
    print(f"Results saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
