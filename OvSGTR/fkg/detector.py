import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'GroundingDINO'))

# Setup LD_LIBRARY_PATH for PyTorch
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

# Coarse prompt categories as specified in Bước 2
COARSE_CATEGORIES = [
    'building', 'tower', 'bridge', 'roof', 'statue', 'boat',
    'water', 'tree', 'person', 'bowl of food', 'gate', 'column', 'door', 'window'
]


def make_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([T.Resize(800), normalize])


def compute_iou(box1, box2):
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])

    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class CoarseRegionDetector:
    def __init__(self, config_path=None, checkpoint_path=None, device='cuda'):
        if config_path is None:
            config_path = str(REPO_ROOT / 'config/GroundingDINO_SwinB_ovdr.py')
        if checkpoint_path is None:
            checkpoint_path = str(REPO_ROOT / 'checkpoints/vg-ovdr-swinb-mega-best.pth')

        self.device = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
        print(f"[INFO] Initializing Coarse Detector on {self.device}...")

        cfg = SLConfig.fromfile(config_path)
        args = argparse.Namespace()
        for k, v in cfg._cfg_dict.to_dict().items():
            setattr(args, k, v)
        args.device = str(self.device)
        args.use_ema = False
        args.debug = False
        args.dataset_file = 'vg'

        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, criterion, postprocessors = build_func(args)

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model', ckpt)
        sd = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)
        self.model = model.to(self.device).eval()

        predicates = ['on', 'near', 'next to', 'has', 'in', 'behind', 'under', 'at']
        self.postprocessor = postprocessors['bbox']
        self.postprocessor.rln_proj = getattr(model, "rln_proj", None)
        self.postprocessor.rln_classifier = getattr(model, "rln_classifier", None)
        self.postprocessor.rln_freq_bias = getattr(model, "rln_freq_bias", None)
        self.postprocessor.name2classes = {n: i for i, n in enumerate(COARSE_CATEGORIES)}
        self.postprocessor.name2predicates = {p: i for i, p in enumerate(predicates)}
        self.postprocessor.eval().to(self.device)

        self.caption = '. '.join(COARSE_CATEGORIES).lower().strip() + '.'
        self.rel_caption = '. '.join(predicates).lower().strip() + '.'
        self.transform = make_transforms()

    @torch.no_grad()
    def detect_frame(self, frame_rgb, score_threshold=0.15):
        """
        Bước 2: Định vị vùng trên frame với prompt thô
        """
        img_pil = Image.fromarray(frame_rgb)
        orig_w, orig_h = img_pil.size

        img_t = self.transform(img_pil)
        samples = nested_tensor_from_tensor_list([img_t]).to(self.device)
        orig_size = torch.tensor([[orig_h, orig_w]], device=self.device)

        outputs = self.model(samples, captions=[self.caption], rel_captions=[self.rel_caption])
        results = self.postprocessor(outputs, orig_size, cat_list=COARSE_CATEGORIES)[0]

        scores = results['scores']
        labels = results['labels']
        boxes = results['boxes']
        keep = scores > score_threshold

        raw_boxes = []
        for b, lbl, sc in zip(boxes[keep], labels[keep], scores[keep]):
            cls_idx = int(lbl.item())
            name = COARSE_CATEGORIES[cls_idx] if cls_idx < len(COARSE_CATEGORIES) else f"cls_{cls_idx}"
            box_coords = [round(float(v), 2) for v in b.tolist()]
            raw_boxes.append({
                "box": box_coords,
                "label": name,
                "score": round(float(sc.item()), 4)
            })

        return raw_boxes, orig_w, orig_h

    @staticmethod
    def filter_and_merge_regions(raw_boxes, orig_w, orig_h):
        """
        Bước 3. Lọc và gộp vùng:
        Ba phép lọc, theo thứ tự:
        1. Bỏ vùng có cạnh ngắn dưới khoảng 40 pixel
        2. Bỏ vùng chiếm trên 50% diện tích khung hình khỏi danh sách tham gia quan hệ (vẫn giữ cho phần khái niệm)
        3. Gộp các vùng có IoU trên 0.85 thành một, giữ bbox có điểm cao nhất
        """
        frame_area = orig_w * orig_h

        # Lọc 1: Bỏ vùng có cạnh ngắn dưới khoảng 40 pixel
        pass1 = []
        for r in raw_boxes:
            b = r["box"]
            w = b[2] - b[0]
            h = b[3] - b[1]
            if min(w, h) >= 40.0:
                pass1.append(r)

        # Lọc 2: Đánh dấu vùng chiếm trên 50% diện tích khung hình
        pass2 = []
        for r in pass1:
            b = r["box"]
            w = b[2] - b[0]
            h = b[3] - b[1]
            area = w * h
            participate_in_relations = (area / frame_area <= 0.50)
            pass2.append({
                "box": b,
                "label": r["label"],
                "score": r["score"],
                "participate_in_relations": participate_in_relations,
                "area_ratio": area / frame_area
            })

        # Lọc 3: Gộp các vùng có IoU trên 0.85 thành một, giữ bbox có điểm cao nhất
        # Sắp xếp giảm dần theo điểm
        pass2 = sorted(pass2, key=lambda x: -x["score"])
        merged = []
        suppressed = set()

        for i in range(len(pass2)):
            if i in suppressed:
                continue
            merged.append(pass2[i])
            for j in range(i + 1, len(pass2)):
                if j in suppressed:
                    continue
                if compute_iou(pass2[i]["box"], pass2[j]["box"]) > 0.85:
                    suppressed.add(j)

        return merged

    @staticmethod
    def track_regions_in_segment(frames_filtered, iou_threshold=0.5):
        """
        Bước 4. Theo vết vùng qua các frame trong đoạn:
        Ghép các bbox thuộc cùng một vật thể thành track, bằng IoU giữa các frame liên tiếp (ngưỡng khoảng 0.5).
        Kết quả: mỗi đoạn có một danh sách track, mỗi track có bbox đại diện (trung bình) và danh sách frame.
        """
        tracks = []
        track_counter = 1

        for f_idx, frame_boxes in enumerate(frames_filtered):
            if f_idx == 0:
                for b in frame_boxes:
                    tracks.append({
                        "track_id": f"T{track_counter:02d}",
                        "coarse_label": b["label"],
                        "participate_in_relations": b["participate_in_relations"],
                        "detections": [(f_idx, b["box"], b["score"])],
                        "last_box": b["box"],
                        "last_frame": f_idx
                    })
                    track_counter += 1
                continue

            matched_det_indices = set()
            for t in tracks:
                best_iou = 0.0
                best_det_idx = -1

                for d_idx, b in enumerate(frame_boxes):
                    if d_idx in matched_det_indices:
                        continue
                    # Check same coarse label or IoU
                    iou = compute_iou(t["last_box"], b["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = d_idx

                if best_iou >= iou_threshold:
                    matched_det_indices.add(best_det_idx)
                    b = frame_boxes[best_det_idx]
                    t["detections"].append((f_idx, b["box"], b["score"]))
                    t["last_box"] = b["box"]
                    t["last_frame"] = f_idx

            # Unmatched detections become new tracks
            for d_idx, b in enumerate(frame_boxes):
                if d_idx not in matched_det_indices:
                    tracks.append({
                        "track_id": f"T{track_counter:02d}",
                        "coarse_label": b["label"],
                        "participate_in_relations": b["participate_in_relations"],
                        "detections": [(f_idx, b["box"], b["score"])],
                        "last_box": b["box"],
                        "last_frame": f_idx
                    })
                    track_counter += 1

        # Calculate representative bbox (average coordinates across frames)
        final_tracks = []
        for t in tracks:
            boxes = np.array([d[1] for d in t["detections"]])
            scores = [d[2] for d in t["detections"]]
            rep_box = [round(float(v), 2) for v in boxes.mean(axis=0)]
            final_tracks.append({
                "track_id": t["track_id"],
                "coarse_label": t["coarse_label"],
                "participate_in_relations": t["participate_in_relations"],
                "rep_box": rep_box,
                "max_score": round(max(scores), 4),
                "detections": t["detections"]  # list of (frame_idx_in_seg, box, score)
            })

        return final_tracks
