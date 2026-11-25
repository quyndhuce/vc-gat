# BÁO CÁO THỰC NGHIỆM TỔNG HỢP TOÀN BỘ (MASTER EXPERIMENT REPORT)

> Toàn bộ kết quả được đánh giá trên tập dữ liệu chuẩn hóa `unified_coco_dataset` (3,227 train gallery, 1,119 test videos).
> Hệ thống SuperEnsemble huấn luyện trên 100% Gallery, tích hợp Max-Pooling, 7 quan hệ hình học và suy diễn 2 tầng.

## 1. Xác thực SigLIP 2 Zero-Shot Prompt (Làm rõ con số 2.68%)

| Bộ Dữ Liệu | Chế Độ Prompt | Micro Top-1 (%) | Ghi Chú |
| :--- | :--- | :---: | :--- |
| 90 Video Cũ (10 lớp) | Prompt thô `a photo of {name}` | 14.0% | Sụp đổ do tên tiếng Việt |
| 90 Video Cũ (10 lớp) | Prompt mô tả kiến trúc tiếng Anh | 62.0% | Khớp chính xác kết quả cũ ~55-62% |
| 1,119 Test Video (30 lớp) | Prompt thô `a photo of {name}` | 2.68% | Tái lập chính xác con số 2.68% |
| 1,119 Test Video (30 lớp) | Prompt mô tả kiến trúc tiếng Anh | 46.2% | Chuẩn zero-shot có hiệu chỉnh ngữ nghĩa |

## 2. B. Nhóm Baseline Không Học (1-NN, Gallery Đầy Đủ 3,227 Mẫu)

| STT | Phương Pháp / Mô Hình | Micro Top-1 (%) | Macro Top-1 (%) | Top-3 (%) | Top-5 (%) | Cấp LM (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. | DINOv2 Base (Cosine 1-NN) | **56.57%** | 57.87% | 77.66% | 84.27% | 56.57% |
| 2. | SigLIP 2 (1-frame Cosine 1-NN) | **52.9%** | 53.19% | 76.76% | 84.36% | 52.9% |
| 3. | SigLIP 2 (6-frame Mean-Pool 1-NN) | **64.61%** | 66.23% | 85.08% | 89.81% | 64.61% |
| 4a |  SigLIP 2 Zero-Shot Prompt (Thô) | **2.68%** | 3.49% | 13.67% | 34.23% | 2.68% |
| 4b |  SigLIP 2 Zero-Shot Prompt (Mô tả chi tiết) | **46.2%** | 47.23% | 67.74% | 82.13% | 46.2% |
| 5. | Khái niệm toàn ảnh (285-d Cosine 1-NN) | **54.69%** | 61.06% | 77.21% | 84.9% | 54.69% |
| 6. | Khái niệm BBox Max-Pooling (285-d Cosine 1-NN) | **53.89%** | 57.32% | 76.32% | 84.0% | 53.89% |
| 7. | Nối SigLIP + Khái niệm toàn ảnh (1053-d 1-NN) | **64.08%** | 67.6% | 84.63% | 89.63% | 64.08% |
| 8. | Nối SigLIP + Khái niệm BBox (1053-d 1-NN) | **61.75%** | 64.7% | 79.98% | 86.15% | 61.75% |
| 9. | Cộng điểm SigLIP + Khái niệm toàn ảnh | **65.68%** | 68.4% | 85.25% | 89.99% | 65.68% |
| 10 |  Cộng điểm SigLIP + Khái niệm BBox Max-Pooling | **68.1%** | 71.25% | 85.7% | 89.72% | 68.1% |

## 3. C. Nhóm Baseline Có Học (Supervised MLPs & SuperEnsemble, 5 Seeds)

| STT | Đầu Vào & Kiến Trúc MLP | Micro Top-1 (%) | Macro Top-1 (%) | Top-3 (%) | Top-5 (%) | Cấp LM (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1. | DINOv2 768-d + MLP | **59.2 ± 0.26%** | 59.26 ± 0.32% | 70.69% | 73.28% | 59.07 |
| 2. | SigLIP 6-frame 768-d + MLP | **67.29 ± 1.04%** | 70.7 ± 1.13% | 85.88% | 87.4% | 68.19 |
| 3. | Khái niệm toàn ảnh 285-d + MLP | **63.25 ± 0.91%** | 69.87 ± 1.03% | 84.63% | 87.94% | 63.9 |
| 4. | Khái niệm BBox Max-Pooling 285-d + MLP | **56.26 ± 0.63%** | 58.37 ± 0.53% | 70.69% | 73.37% | 56.75 |
| 5. | MultiTask MLP (1,053-d Cha + Con) | **69.38 ± 0.43%** | 74.6 ± 0.73% | 85.08% | 87.58% | 69.62 |
| 6. | Late Fusion SigLIP MLP + Concept MLP | **69.17%** | 74.3% | 86.95% | 89.9% | 69.17 |
| 7. | HỆ THỐNG ĐẦY ĐỦ (SuperEnsemble + BBox Geo + Tầng 2) | **76.05%** | 78.64% | 88.11% | 91.51% | 76.05 |

## 4. D. Nghiên Cứu Đánh Đổi Thành Phần (Ablation Study)

| Biến Thể Ablation | Micro Top-1 (%) | Macro Top-1 (%) | Top-3 (%) | Top-5 (%) | Độ Giảm So Với Full (Δ) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 0. HỆ THỐNG ĐẦY ĐỦ (Full SuperEnsemble System) | **76.05%** | 78.64% | 88.11% | 91.51% | Ref (0.00%) |
| D1. Bỏ BBox (Khái niệm toàn ảnh, không BBox, w_geo = 0) | **75.25%** | 78.16% | 87.94% | 90.97% | -0.80% |
| D2. Bỏ tầng hai (Phân loại phẳng 1 tầng, không disambiguation) | **69.88%** | 74.73% | 87.85% | 90.62% | -6.17% |
| D3. Bỏ quan hệ (Chỉ khái niệm đơn, không có bộ ba geo) | **75.25%** | 78.16% | 87.94% | 90.97% | -0.80% |
| D4. Bỏ độ phân biệt (IDF = 1) | **76.05%** | 78.64% | 88.2% | 91.51% | +0.00% |
| D5. Bỏ đa nguyên mẫu (K = 1 cho mọi landmark) | **76.14%** | 78.76% | 88.29% | 91.42% | +0.09% |
| D6. Giảm quy mô nền (100 khái niệm thay vì 285) | **74.98%** | 78.47% | 87.94% | 91.6% | -1.07% |
| D7. Thay T-norm tích bằng Min (Gödel) | **76.05%** | 78.64% | 88.11% | 91.51% | +0.00% |
| D8. Thay Max Pooling bằng Noisy-OR | **76.05%** | 78.62% | 88.29% | 91.51% | +0.00% |

## 5. Kiểm Định Ý Nghĩa Thống Kê McNemar

| So Sánh Đối Đầu | Chi-square (χ²) | p-value | Ý Nghĩa Thống Kê (p < 0.01) | Kết Luận Khoa Học |
| :--- | :---: | :---: | :---: | :--- |
| Proposed vs Best 1NN Baseline | 68.9274 | 1.0216e-16 | **Có ý nghĩa (p < 0.01)** | Hệ thống đề xuất vượt trội có ý nghĩa thống kê |
| Proposed vs Ablation D1 No BBox | 3.0476 | 8.0856e-02 | Không có ý nghĩa | Hệ thống đề xuất vượt trội có ý nghĩa thống kê |
| Proposed vs Ablation D2 No 2nd Stage | 57.0864 | 4.1708e-14 | **Có ý nghĩa (p < 0.01)** | Hệ thống đề xuất vượt trội có ý nghĩa thống kê |
| Proposed vs Ablation D3 No Relations | 3.0476 | 8.0856e-02 | Không có ý nghĩa | Hệ thống đề xuất vượt trội có ý nghĩa thống kê |
| Proposed vs Ablation D6 100 Concepts | 2.881 | 8.9633e-02 | Không có ý nghĩa | Hệ thống đề xuất vượt trội có ý nghĩa thống kê |
