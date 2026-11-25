import os
import sys
import json
import time
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
from transformers import AutoProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'fkg'))

from dataset import TARGET_LANDMARKS

# English prompts capturing distinctive visual features for each landmark
LANDMARK_PROMPTS = {
    "Hoan Kiem Lake": [
        "a photo of Hoan Kiem Lake in Hanoi",
        "a photo of Turtle Tower in the middle of Hoan Kiem Lake",
        "a photo of The Huc red bridge crossing Hoan Kiem Lake",
        "a scenic view of Hoan Kiem Sword Lake with green water and willow trees",
        "a photo of Ngoc Son Temple at Hoan Kiem Lake in Hanoi"
    ],
    "Van Mieu Quoc Tu Giam": [
        "a photo of Van Mieu Temple of Literature in Hanoi",
        "a photo of Khue Van Cac pavilion at the Temple of Literature",
        "a photo of stone turtles carrying doctor stelae at the Temple of Literature",
        "a photo of the ancient red courtyards and pond of the Temple of Literature in Hanoi",
        "a photo of Dai Thanh Sanctuary at Van Mieu Temple of Literature"
    ],
    "HoangThanhThangLong": [
        "a photo of the Imperial Citadel of Thang Long in Hanoi",
        "a photo of Doan Mon ancient south gate at the Imperial Citadel of Thang Long",
        "a photo of the carved stone dragon steps of Kinh Thien Palace at Thang Long Citadel",
        "a photo of the ancient citadel wall and brick tower at the Imperial Citadel of Thang Long",
        "a photo of Hanoi Flag Tower Cot Co at Thang Long Imperial Citadel"
    ],
    "West Lake": [
        "a photo of West Lake Tay Ho in Hanoi",
        "a photo of Tran Quoc Pagoda on an islet in West Lake",
        "a scenic panoramic view of West Lake with expansive calm water in Hanoi",
        "a photo of Tay Ho Temple Phu Tay Ho next to the lake",
        "a sunset photo over West Lake in Hanoi"
    ],
    "One Pillar Pagoda": [
        "a photo of One Pillar Pagoda Chua Mot Cot in Hanoi",
        "a photo of the small Buddhist temple built on a single stone pillar over a lotus pond",
        "a close-up photo of the wooden One Pillar Pagoda resembling a lotus blossom",
        "a photo of One Pillar Pagoda surrounded by green trees and lotus flowers"
    ],
    "Ho Chi Minh's Mausoleum": [
        "a photo of Ho Chi Minh's Mausoleum in Ba Dinh Square Hanoi",
        "a photo of the grand gray stone monument of Ho Chi Minh Mausoleum",
        "a wide shot of Ba Dinh Square with green lawn in front of Ho Chi Minh's Mausoleum",
        "a photo of the monumental architecture of Ho Chi Minh Mausoleum"
    ],
    "The Old Quarter": [
        "a photo of Hanoi Old Quarter 36 streets",
        "a photo of narrow tube shophouses and crowded streets in Hanoi Old Quarter",
        "a busy street scene with motorbikes and vendors in the Old Quarter of Hanoi",
        "a photo of night market and street food stalls in Hanoi Old Quarter",
        "a photo of French colonial balconies on narrow streets of Hanoi Old Quarter"
    ],
    "Vietnam National Museum of History": [
        "a photo of the Vietnam National Museum of History in Hanoi",
        "a photo of the French colonial Indochina style building with yellow walls of the National Museum of History",
        "a photo of the exterior architecture of the Vietnam National Museum of History",
        "a photo of historic museum exhibition rooms at the National Museum of History in Hanoi"
    ],
    "Bat Trang Pottery Village": [
        "a photo of Bat Trang ceramic pottery village in Hanoi",
        "a photo of painted ceramic vases, porcelain bowls and pottery in Bat Trang",
        "a photo of a ceramic artisan shaping wet clay on a pottery wheel in Bat Trang",
        "a photo of traditional ceramic drying yards and pottery shops in Bat Trang village"
    ],
    "Van Phuc Silk Village": [
        "a photo of Van Phuc silk weaving village in Hanoi",
        "a photo of colourful umbrella canopy and silk lanterns hanging over the street in Van Phuc",
        "a photo of traditional wooden weaving looms and rolls of vibrant silk in Van Phuc",
        "a photo of silk clothing, scarves and textile shops in Van Phuc village"
    ]
}


class SigLIPZeroShotBaseline:
    def __init__(self, model_id="google/siglip2-base-patch16-224", device="cuda"):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        print(f"[INFO] Initializing SigLIP 2 Zero-Shot Baseline on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

        self.landmarks = TARGET_LANDMARKS
        self.prototype_matrix = self._encode_landmark_prototypes()

    def _encode_landmark_prototypes(self):
        """
        Với mỗi landmark: đưa các prompt tiếng Anh qua text encoder, lấy trung bình rồi chuẩn hóa.
        Kết quả: Ma trận nguyên mẫu 10 × 768.
        """
        print("[INFO] Encoding landmark text prompts into prototypes (10 x 768)...")
        prototypes = []

        for lm in self.landmarks:
            prompts = LANDMARK_PROMPTS[lm]
            inputs = self.processor(text=prompts, padding="max_length", return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)
                embeds = out.pooler_output if hasattr(out, "pooler_output") else out
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                proto = embeds.mean(dim=0)
                proto = proto / proto.norm(dim=-1, keepdim=True)
                prototypes.append(proto)

        W = torch.stack(prototypes, dim=0)  # (10, 768)
        print(f"[INFO] Prototype matrix ready: {W.shape}")
        return W

    @torch.no_grad()
    def evaluate_video(self, video_path, frame_indices=None, batch_size=64):
        """
        Đánh giá 1 video trực tiếp bằng SigLIP 2:
        - Đọc các frame (toàn cảnh nguyên vẹn, không crop, không detector, không scene graph)
        - Đưa qua SigLIP image encoder theo batch
        - Tính cosine similarity với 10 nguyên mẫu
        - Lấy trung bình điểm qua các frame của video, lấy argmax.
        """
        cap = cv2.VideoCapture(video_path)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_indices is None or len(frame_indices) == 0:
            # Sample 32 frames evenly
            step = max(1, total_f // 32)
            frame_indices = list(range(0, total_f, step))[:32]

        frames_pil = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_pil.append(Image.fromarray(rgb))

        cap.release()

        if not frames_pil:
            return np.zeros(len(self.landmarks)), []

        # Batch encode frames
        all_feats = []
        for b in range(0, len(frames_pil), batch_size):
            batch_imgs = frames_pil[b : b + batch_size]
            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = feat / feat.norm(dim=-1, keepdim=True)
            all_feats.append(feat)

        all_feats = torch.cat(all_feats, dim=0)  # (N_frames, 768)

        # Cosine similarity with 10 prototypes: (N_frames, 10)
        sims = (all_feats @ self.prototype_matrix.T).cpu().numpy()

        # Mean similarity across all frames: (10,)
        video_scores = np.mean(sims, axis=0)

        # Frame-by-frame similarities for early recognition
        return video_scores, sims


def run_baseline_evaluation(
    stage1_path="/home/nguyendinhquy/OvSGTR/fkg/output/stage1_detections.json",
    output_dir="/home/nguyendinhquy/OvSGTR/fkg/output"
):
    print("=" * 70)
    print("RUNNING ZERO-SHOT SigLIP 2 DIRECT BASELINE (NO SCENE GRAPH, NO FKG)")
    print("=" * 70)

    with open(stage1_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_vids = data["videos"]
    test_vids = [v for v in all_vids if v.get("split") == "test"]

    print(f"[INFO] Loaded {len(test_vids)} test videos (2 per landmark across 10 landmarks).")
    baseline = SigLIPZeroShotBaseline(device="cuda")

    landmarks = TARGET_LANDMARKS
    lm_to_idx = {lm: i for i, lm in enumerate(landmarks)}
    num_landmarks = len(landmarks)

    confusion_matrix = np.zeros((num_landmarks, num_landmarks), dtype=np.int32)
    video_predictions = []
    correct_margins = []
    incorrect_margins = []

    prefix_cutoffs = [4, 12, 20, 999]  # approx 1 seg (4f), 3 segs (12f), 5 segs (20f), all
    prefix_correct = {k: 0 for k in prefix_cutoffs}
    total_eval = 0

    start_time = time.time()

    for v_rec in test_vids:
        gt_lm = v_rec["landmark"]
        gt_idx = lm_to_idx[gt_lm]
        vid = v_rec["video_id"]
        v_path = v_rec["video_path"]

        # Collect all sampled frame indices from stage 1 segments
        frame_indices = []
        for s in v_rec.get("segments", []):
            frame_indices.extend(s.get("frame_indices", []))

        # Evaluate video directly with SigLIP 2
        v_scores, frame_sims = baseline.evaluate_video(v_path, frame_indices=frame_indices)
        total_eval += 1

        pred_idx = int(np.argmax(v_scores))
        sorted_scores = np.sort(v_scores)[::-1]
        margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else float(sorted_scores[0])

        is_corr = (pred_idx == gt_idx)
        confusion_matrix[gt_idx, pred_idx] += 1

        if is_corr:
            correct_margins.append(margin)
        else:
            incorrect_margins.append(margin)

        # Early recognition curve
        for cut in prefix_cutoffs:
            if len(frame_sims) > 0:
                p_scores = np.mean(frame_sims[: min(len(frame_sims), cut)], axis=0)
                if int(np.argmax(p_scores)) == gt_idx:
                    prefix_correct[cut] += 1

        video_predictions.append({
            "video_id": vid,
            "ground_truth": gt_lm,
            "predicted": landmarks[pred_idx],
            "is_correct": is_corr,
            "top1_score": round(float(sorted_scores[0]), 4),
            "top2_score": round(float(sorted_scores[1]), 4) if len(sorted_scores) > 1 else 0.0,
            "margin": round(margin, 4),
            "num_frames": len(frame_indices),
            "scores": {lm: round(float(v_scores[i]), 4) for i, lm in enumerate(landmarks)}
        })

    elapsed = time.time() - start_time
    total_corr = int(np.trace(confusion_matrix))
    overall_acc = total_corr / max(1, total_eval)

    per_lm_acc = {}
    for i, lm in enumerate(landmarks):
        row_sum = np.sum(confusion_matrix[i])
        per_lm_acc[lm] = float(confusion_matrix[i, i] / max(1, row_sum))

    early_curves = {}
    for cut, label in zip(prefix_cutoffs, ["first_1_seg", "first_3_segs", "first_5_segs", "all_segs"]):
        early_curves[label] = round(float(prefix_correct[cut] / max(1, total_eval)), 4)

    results = {
        "model": "google/siglip2-base-patch16-224 (Zero-Shot Direct Image Baseline)",
        "overall_accuracy": round(overall_acc, 4),
        "total_test_videos": total_eval,
        "total_correct": total_corr,
        "mean_margin_correct": round(float(np.mean(correct_margins)), 4) if correct_margins else 0.0,
        "mean_margin_incorrect": round(float(np.mean(incorrect_margins)), 4) if incorrect_margins else 0.0,
        "per_landmark_accuracy": per_lm_acc,
        "confusion_matrix": confusion_matrix.tolist(),
        "early_recognition_curve": early_curves,
        "predictions": video_predictions
    }

    out_file = os.path.join(output_dir, "baseline_siglip_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("ZERO-SHOT SigLIP 2 BASELINE RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {overall_acc*100:.2f}% ({total_corr}/{total_eval}) in {elapsed:.2f}s")
    print(f"Mean Margin (Correct):   {results['mean_margin_correct']:.4f}")
    print(f"Mean Margin (Incorrect): {results['mean_margin_incorrect']:.4f}")

    print("\nPer-Landmark Accuracy:")
    for lm, acc in per_lm_acc.items():
        print(f"  {lm:35s}: {acc*100:5.1f}%")

    print("\nEarly Recognition Curve:")
    for label, acc in early_curves.items():
        print(f"  {label:15s}: {acc*100:5.1f}%")

    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    run_baseline_evaluation()
