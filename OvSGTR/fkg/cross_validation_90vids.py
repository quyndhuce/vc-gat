import os
import sys
import json
import time
import pickle
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))

from dataset import TARGET_LANDMARKS
from baseline_siglip import LANDMARK_PROMPTS


def extract_and_cache_90_frame_features(
    grounded_cache_path="/home/nguyendinhquy/OvSGTR/fkg/output/grounded_videos_90.pkl",
    cache_siglip_50="/home/nguyendinhquy/OvSGTR/fkg/output/features_siglip2.pkl",
    cache_dino_50="/home/nguyendinhquy/OvSGTR/fkg/output/features_dinov2.pkl",
    cache_siglip_90="/home/nguyendinhquy/OvSGTR/fkg/output/features_siglip2_90.pkl",
    cache_dino_90="/home/nguyendinhquy/OvSGTR/fkg/output/features_dinov2_90.pkl",
    device="cuda"
):
    """
    Extracts full-frame features for all 90 videos, reusing the 50 cached.
    """
    with open(grounded_cache_path, "rb") as f:
        grounded_data = pickle.load(f)
    vids_by_lm = grounded_data["videos_by_landmark"]

    all_vids = []
    for lm in TARGET_LANDMARKS:
        all_vids.extend(vids_by_lm[lm])

    # 1. SigLIP 2
    if os.path.exists(cache_siglip_90):
        print(f"[INFO] Found cached SigLIP 2 90-video features at {cache_siglip_90}")
        with open(cache_siglip_90, "rb") as f:
            siglip_feats = pickle.load(f)
    else:
        siglip_feats = {}
        if os.path.exists(cache_siglip_50):
            with open(cache_siglip_50, "rb") as f:
                siglip_feats = pickle.load(f)
            print(f"[INFO] Reused {len(siglip_feats)} existing SigLIP 2 video features")

        new_vids = [v for v in all_vids if v["video_id"] not in siglip_feats]
        if new_vids:
            print(f"[INFO] Extracting SigLIP 2 features for {len(new_vids)} new videos on GPU...")
            from transformers import AutoProcessor, AutoModel
            proc_siglip = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
            model_siglip = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device).eval()

            for idx, v_rec in enumerate(new_vids, 1):
                vid = v_rec["video_id"]
                v_path = v_rec["video_path"]
                frame_indices = []
                for s in v_rec.get("segments", []):
                    f_indices = [f.get("abs_frame_idx", 0) for f in s.get("frames", [])] if "frames" in s else []
                    frame_indices.extend(f_indices)

                cap = cv2.VideoCapture(v_path)
                frames = []
                if frame_indices:
                    for f_idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                else:
                    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    step = max(1, fc // 60)
                    for f_idx in range(0, fc, step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                cap.release()

                if not frames:
                    siglip_feats[vid] = np.zeros((1, 768), dtype=np.float32)
                    continue

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
                    print(f"  SigLIP 2: [{idx}/{len(new_vids)}] new videos extracted")

            with open(cache_siglip_90, "wb") as f:
                pickle.dump(siglip_feats, f)
            print(f"[INFO] Saved 90 SigLIP 2 features to {cache_siglip_90}")

    # 2. DINOv2
    if os.path.exists(cache_dino_90):
        print(f"[INFO] Found cached DINOv2 90-video features at {cache_dino_90}")
        with open(cache_dino_90, "rb") as f:
            dino_feats = pickle.load(f)
    else:
        dino_feats = {}
        if os.path.exists(cache_dino_50):
            with open(cache_dino_50, "rb") as f:
                dino_feats = pickle.load(f)
            print(f"[INFO] Reused {len(dino_feats)} existing DINOv2 video features")

        new_vids = [v for v in all_vids if v["video_id"] not in dino_feats]
        if new_vids:
            print(f"[INFO] Extracting DINOv2 features for {len(new_vids)} new videos on GPU...")
            from transformers import AutoImageProcessor, AutoModel
            proc_dino = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            model_dino = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()

            for idx, v_rec in enumerate(new_vids, 1):
                vid = v_rec["video_id"]
                v_path = v_rec["video_path"]
                cap = cv2.VideoCapture(v_path)
                fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, fc // 60)
                frames = []
                for f_idx in range(0, fc, step):
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
                        feat = out.last_hidden_state[:, 0, :]
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        v_embeds.append(feat.cpu().numpy())
                dino_feats[vid] = np.concatenate(v_embeds, axis=0)
                if idx % 10 == 0:
                    print(f"  DINOv2: [{idx}/{len(new_vids)}] new videos extracted")

            with open(cache_dino_90, "wb") as f:
                pickle.dump(dino_feats, f)
            print(f"[INFO] Saved 90 DINOv2 features to {cache_dino_90}")

    return siglip_feats, dino_feats


def evaluate_fkg_cv(vids_by_lm, landmarks, concepts, folds_indices, k_views=3, beta=1.0):
    num_landmarks = len(landmarks)
    lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}
    num_concepts = len(concepts)

    fold_accuracies = []
    overall_confusion_matrix = np.zeros((num_landmarks, num_landmarks), dtype=np.int32)
    per_lm_correct = {lm: 0 for lm in landmarks}
    per_lm_total = {lm: 0 for lm in landmarks}

    for fold_idx, (sup_idxs, test_idxs) in enumerate(folds_indices):
        landmark_views = {}
        for lm in landmarks:
            segs = []
            for i in sup_idxs:
                segs.extend(vids_by_lm[lm][i]["segments"])

            if k_views > 1 and len(segs) > k_views:
                X = np.stack([s["concept_membership"] for s in segs])
                km = KMeans(n_clusters=k_views, random_state=42, n_init=10).fit(X)
                clusters = [[] for _ in range(k_views)]
                for s, lbl in zip(segs, km.labels_):
                    clusters[lbl].append(s)
                clusters = [c for c in clusters if len(c) > 0]
            else:
                clusters = [segs]

            views = []
            for v_idx, c_segs in enumerate(clusters):
                Nv = len(c_segs)
                Sc = np.sum([s["concept_membership"] for s in c_segs], axis=0)
                node_m = Sc / max(1, Nv)

                PR_k = defaultdict(float)
                P_k = defaultdict(float)
                for s in c_segs:
                    c_mem = s["concept_membership"]
                    for trip_key, trip_score in s["sparse_triplets"].items():
                        PR_k[trip_key] += trip_score
                        parts = trip_key.split('_')
                        cA, cB = int(parts[0]), int(parts[2])
                        P_k[trip_key] += float(c_mem[cA] * c_mem[cB])

                rel_m = {}
                for trip_key, pr_val in PR_k.items():
                    p_val = P_k.get(trip_key, 0.0)
                    rm = min(1.0, pr_val / (p_val + 1e-4)) if p_val > 1e-6 else min(1.0, pr_val / Nv)
                    if rm >= 0.05:
                        rel_m[trip_key] = round(float(rm), 4)

                views.append({
                    "view_idx": v_idx,
                    "num_segments": Nv,
                    "node_mem": node_m,
                    "rel_mem": rel_m
                })
            landmark_views[lm] = views

        peak_nodes = np.zeros((num_landmarks, num_concepts), dtype=np.float32)
        for l_idx, lm in enumerate(landmarks):
            all_node_mems = [v["node_mem"] for v in landmark_views[lm]]
            peak_nodes[l_idx] = np.max(all_node_mems, axis=0)

        sum_across = np.sum(peak_nodes, axis=0, keepdims=True)

        for l_idx, lm in enumerate(landmarks):
            for v in landmark_views[lm]:
                node_m = v["node_mem"]
                disc_v = node_m / (sum_across.squeeze(0) + 1e-6)
                v["disc"] = disc_v
                term_nodes = float(np.sum(disc_v * node_m))
                term_rels = float(sum(v["rel_mem"].values()))
                v["norm_const"] = max(1e-4, term_nodes + beta * term_rels)

        fold_correct = 0
        fold_total = 0

        for gt_lm in landmarks:
            gt_idx = lm_to_idx[gt_lm]
            for t_idx in test_idxs:
                v_record = vids_by_lm[gt_lm][t_idx]
                segs = v_record["segments"]
                seg_scores = []

                for s in segs:
                    s_scores = []
                    for l_idx, lm in enumerate(landmarks):
                        views = landmark_views[lm]
                        v_scores = []
                        for vi in views:
                            tn = float(np.sum(vi["disc"] * s["concept_membership"] * vi["node_mem"]))
                            tr = sum(vi["rel_mem"].get(k, 0.0) * sc for k, sc in s["sparse_triplets"].items())
                            v_score = (tn + beta * tr) / vi["norm_const"]
                            v_scores.append(v_score)
                        s_scores.append(max(v_scores))
                    seg_scores.append(s_scores)

                v_scores = np.mean(seg_scores, axis=0)
                pred_idx = int(np.argmax(v_scores))

                is_corr = (pred_idx == gt_idx)
                if is_corr:
                    fold_correct += 1
                    per_lm_correct[gt_lm] += 1
                per_lm_total[gt_lm] += 1
                fold_total += 1
                overall_confusion_matrix[gt_idx, pred_idx] += 1

        fold_acc = fold_correct / fold_total
        fold_accuracies.append(fold_acc)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    per_lm_acc = {lm: round(per_lm_correct[lm] / max(1, per_lm_total[lm]), 4) for lm in landmarks}

    return {
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "total_evaluations": int(np.sum(overall_confusion_matrix)),
        "total_correct": int(np.trace(overall_confusion_matrix)),
        "per_landmark_accuracy": per_lm_acc,
        "confusion_matrix": overall_confusion_matrix.tolist()
    }


def evaluate_baseline_cv(vids_by_lm, landmarks, features_dict, folds_indices, method="image_prototype", text_prototypes=None):
    num_landmarks = len(landmarks)
    lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}

    fold_accuracies = []
    overall_confusion_matrix = np.zeros((num_landmarks, num_landmarks), dtype=np.int32)
    per_lm_correct = {lm: 0 for lm in landmarks}
    per_lm_total = {lm: 0 for lm in landmarks}

    for fold_idx, (sup_idxs, test_idxs) in enumerate(folds_indices):
        if method == "image_prototype":
            # Compute image prototype for each landmark
            landmark_prototypes = np.zeros((num_landmarks, 768), dtype=np.float32)
            for l_idx, lm in enumerate(landmarks):
                sup_embeds = []
                for i in sup_idxs:
                    vid = vids_by_lm[lm][i]["video_id"]
                    if vid in features_dict:
                        sup_embeds.append(features_dict[vid])
                if sup_embeds:
                    all_frames = np.concatenate(sup_embeds, axis=0)
                    proto = np.mean(all_frames, axis=0)
                    proto = proto / (np.linalg.norm(proto) + 1e-8)
                    landmark_prototypes[l_idx] = proto
        else:
            landmark_prototypes = text_prototypes

        fold_correct = 0
        fold_total = 0

        for gt_lm in landmarks:
            gt_idx = lm_to_idx[gt_lm]
            for t_idx in test_idxs:
                vid = vids_by_lm[gt_lm][t_idx]["video_id"]
                if vid in features_dict:
                    test_frames = features_dict[vid]
                    sims = np.dot(test_frames, landmark_prototypes.T)
                    v_scores = np.mean(sims, axis=0)
                    pred_idx = int(np.argmax(v_scores))
                else:
                    pred_idx = 0

                is_corr = (pred_idx == gt_idx)
                if is_corr:
                    fold_correct += 1
                    per_lm_correct[gt_lm] += 1
                per_lm_total[gt_lm] += 1
                fold_total += 1
                overall_confusion_matrix[gt_idx, pred_idx] += 1

        fold_acc = fold_correct / fold_total
        fold_accuracies.append(fold_acc)

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    per_lm_acc = {lm: round(per_lm_correct[lm] / max(1, per_lm_total[lm]), 4) for lm in landmarks}

    return {
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
        "total_evaluations": int(np.sum(overall_confusion_matrix)),
        "total_correct": int(np.trace(overall_confusion_matrix)),
        "per_landmark_accuracy": per_lm_acc,
        "confusion_matrix": overall_confusion_matrix.tolist()
    }


def run_full_90_videos_benchmark():
    grounded_cache = "/home/nguyendinhquy/OvSGTR/fkg/output/grounded_videos_90.pkl"
    assert os.path.exists(grounded_cache), f"Cache not found: {grounded_cache}"

    with open(grounded_cache, "rb") as f:
        data = pickle.load(f)

    vids_by_lm = data["videos_by_landmark"]
    concepts = data["concepts"]
    landmarks = TARGET_LANDMARKS

    print("=" * 70)
    print("BENCHMARKING ALL MODELS ON 90 VIDEOS (10 LANDMARKS x 9 VIDEOS)")
    print("=" * 70)

    # 1. Extract / Cache Full-frame features for SigLIP 2 and DINOv2
    siglip_feats, dino_feats = extract_and_cache_90_frame_features(grounded_cache)

    # Compute text prototypes for SigLIP 2
    from transformers import AutoProcessor, AutoModel
    proc_siglip = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    model_siglip = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to("cuda").eval()

    text_prototypes = np.zeros((len(landmarks), 768), dtype=np.float32)
    for l_idx, lm in enumerate(landmarks):
        prompts = LANDMARK_PROMPTS.get(lm, [f"a photo of {lm} in Hanoi"])
        inputs = proc_siglip(text=prompts, padding="max_length", return_tensors="pt").to("cuda")
        with torch.no_grad():
            t_out = model_siglip.get_text_features(**inputs)
            t_feat = t_out.pooler_output if hasattr(t_out, "pooler_output") else t_out
            t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
            proto = t_feat.mean(dim=0)
            proto = proto / proto.norm(dim=-1)
            text_prototypes[l_idx] = proto.cpu().numpy()

    # Define Cross-Validation Schemes
    # Scheme 1: 9-Fold Leave-One-Video-Out (8 support / 1 test, 90 evaluations)
    folds_9_scheme = [([j for j in range(9) if j != i], [i]) for i in range(9)]

    # Scheme 2: 3-Fold Cross-Validation (6 support / 3 test, 90 evaluations)
    folds_3_scheme = [
        ([3, 4, 5, 6, 7, 8], [0, 1, 2]),
        ([0, 1, 2, 6, 7, 8], [3, 4, 5]),
        ([0, 1, 2, 3, 4, 5], [6, 7, 8])
    ]

    all_results = {}

    for s_name, folds in [("9fold_8sup1test", folds_9_scheme), ("3fold_6sup3test", folds_3_scheme)]:
        print("\n" + "#" * 70)
        print(f"EVALUATING SCHEME: {s_name} ({len(folds)} folds, 90 evaluations)")
        print("#" * 70)

        # 1. FKG Multi-View (K=3)
        print("\n>>> [1] FKG Multi-View (K=3 Views)...")
        res_fkg_k3 = evaluate_fkg_cv(vids_by_lm, landmarks, concepts, folds, k_views=3)
        print(f"Accuracy: {res_fkg_k3['mean_accuracy']*100:.2f}% ± {res_fkg_k3['std_accuracy']*100:.2f}%")
        all_results[f"{s_name}_fkg_multiview_k3"] = res_fkg_k3

        # 2. FKG Baseline (K=1)
        print("\n>>> [2] FKG Baseline (K=1 View)...")
        res_fkg_k1 = evaluate_fkg_cv(vids_by_lm, landmarks, concepts, folds, k_views=1)
        print(f"Accuracy: {res_fkg_k1['mean_accuracy']*100:.2f}% ± {res_fkg_k1['std_accuracy']*100:.2f}%")
        all_results[f"{s_name}_fkg_baseline_k1"] = res_fkg_k1

        # 3. SigLIP 2 Zero-Shot Text
        print("\n>>> [3] Baseline: SigLIP 2 Zero-Shot Text Prototypes...")
        res_siglip_txt = evaluate_baseline_cv(vids_by_lm, landmarks, siglip_feats, folds, method="text_prototype", text_prototypes=text_prototypes)
        print(f"Accuracy: {res_siglip_txt['mean_accuracy']*100:.2f}% ± {res_siglip_txt['std_accuracy']*100:.2f}%")
        all_results[f"{s_name}_siglip_text"] = res_siglip_txt

        # 4. SigLIP 2 Image Prototypes
        print("\n>>> [4] Baseline: SigLIP 2 Image Prototypes...")
        res_siglip_img = evaluate_baseline_cv(vids_by_lm, landmarks, siglip_feats, folds, method="image_prototype")
        print(f"Accuracy: {res_siglip_img['mean_accuracy']*100:.2f}% ± {res_siglip_img['std_accuracy']*100:.2f}%")
        all_results[f"{s_name}_siglip_img"] = res_siglip_img

        # 5. DINOv2 Image Prototypes
        print("\n>>> [5] Baseline: DINOv2 Image Prototypes...")
        res_dino_img = evaluate_baseline_cv(vids_by_lm, landmarks, dino_feats, folds, method="image_prototype")
        print(f"Accuracy: {res_dino_img['mean_accuracy']*100:.2f}% ± {res_dino_img['std_accuracy']*100:.2f}%")
        all_results[f"{s_name}_dino_img"] = res_dino_img

    out_json = "/home/nguyendinhquy/OvSGTR/fkg/output/cv_results_90vids.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Saved complete 90-video cross-validation results to {out_json}")


if __name__ == "__main__":
    run_full_90_videos_benchmark()
