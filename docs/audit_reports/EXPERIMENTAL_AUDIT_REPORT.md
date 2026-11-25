# Báo cáo Kiểm toán Độc lập Toàn diện về Tính Toàn vẹn Thực nghiệm và Khả năng Tái lập (Final Experimental Integrity & Reproducibility Audit Report)

**Manuscript Title:** *Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning*  
**Target Journal:** *International Journal of Approximate Reasoning (IJAR)*  
**Audit Role:** Independent Scientific Reproducibility Auditor  
**Audit Date:** 2026-11-24  
**Git Commit Hash:** `efc8364`  
**Final Submission Recommendation:** **GO (Ready for Official IJAR Submission)**

---

## 1. ĐÓNG BĂNG BẢN THẢO VÀ CÁC THÀNH PHẨM THỰC NGHIỆM (FREEZE RECORD)

### 1.1. Bản thảo TeX và PDF chính thức
- **Thư mục nguồn bản thảo:** `/Users/quynd/Downloads/els-cas-templates 2`
- **Bản thảo PDF đã biên dịch:** `main.pdf` (40 trang, dung lượng 2,660,467 bytes, 0 lỗi biên dịch, 0 undefined references, 0 BibTeX warnings).
- **Tính toàn vẹn tài liệu phản biện (`Response_to_Reviewers.md`):**  
  SHA-256 Hash: `f0814cb17de957930468ecf3f4c8860fe257a5035ecb17d71634ef13d0abeaf9` (Tuyệt đối nguyên vẹn 100%, không bị sửa đổi).

### 1.2. Bảng mã băm mật mã các checkpoint mô hình (Checkpoint Cryptographic Hashes)
Tất cả checkpoint mô hình phục vụ đánh giá trong bản thảo đã được đóng băng và xác thực mã băm SHA-256 độc lập:

| Tên Checkpoint | Đường dẫn tương đối | Kích thước | SHA-256 Hash | Vai trò trong hệ thống thực nghiệm |
| :--- | :--- | :---: | :--- | :--- |
| `best_dec_vcgat_v2.pt` | `experiments/checkpoints_fact_selection/best_dec_vcgat_v2.pt` | 24,774,433 bytes | `01f3b3742d5e55a296c3d99c784244c66ecb32a0870ad1d3f7b0773745f36633` | **Checkpoint DEC-v2 chính thức** cho benchmark xếp hạng Hanoi-500 |
| `best_cross_attention.pt` | `experiments/checkpoints_fact_selection/best_cross_attention.pt` | 11,896,017 bytes | `e243799dad2701d0821bd18fd63254013df255127a4a0a51ce1c7d5b31476fa8` | Baseline xếp hạng đa phương thức có giám sát (không đồ thị) |
| `best_dec_vcgat_v2_full_horn.pt` | `experiments/checkpoints_fact_selection/best_dec_vcgat_v2_full_horn.pt` | 24,775,017 bytes | `b8ddb441a1d1ec3be6e0b83090781058493159ae4d3ab44c8f0c8828dd31f54a` | Checkpoint thực nghiệm bao đóng suy diễn Horn bão hòa |
| `best_decoupled_ranking_vcgat.pt` | `experiments/features/best_decoupled_ranking_vcgat.pt` | 44,571,181 bytes | `cf19893f3b8f5a57960e22ea49d6e32b8a604d81c3c69433db3d969a1535930e` | Mạng xếp hạng tách rời dual-scorer (Decoupled Ranking VCGAT) |
| `best_gatv2.pt` | `experiments/checkpoints_fact_selection/best_gatv2.pt` | 14,279,797 bytes | `c5ea5af1ca329ff0fec677f978ea5c485af408c09695e3fc0b1fe6d0cc0f2e76` | Baseline đồ thị GATv2 trên KG thô |
| `best_graphormer.pt` | `experiments/checkpoints_fact_selection/best_graphormer.pt` | 42,828,286 bytes | `d0500f0895eba7ea9dad819f1fb37a49ad478813dc2891616128c2f55413c180` | Baseline đồ thị Graphormer |
| `adapter_model.safetensors` | `experiments/checkpoints_caption/qwen25_3b_certainty_lora/adapter_model.safetensors` | 119,801,528 bytes | `70d0c2083d287e82df493ea74dfd2013e63a43a580713fb19c953f50dfa81dfc` | Trọng số LoRA thích ứng độ chắc chắn cho Qwen2.5-3B-Instruct |

### 1.3. Phân định ranh giới giữa các phiên bản mô hình cũ và mới
- **DEC-v1 vs. DEC-v2:** Bản thảo IJAR đã loại bỏ hoàn toàn việc kế thừa các kết quả của DEC-v1. Checkpoint `best_dec_vcgat_v2.pt` được huấn luyện với kiến trúc chính xác: 2 lớp GAT, dập tắt bậc $\gamma=0.40$, cổng độ tin cậy $\lambda_i$, cổng nổi bật thị giác $g_i$, và chặn phần dư đồ thị $[-0.8, +0.8]$.
- **Loại bỏ Proposition 2b và 2c cũ:** Các phân tích xấp xỉ tiệm cận cũ (vốn đã bị thực nghiệm kiểm toán độc lập trước đó bác bỏ hoặc làm yếu) đã được thay thế hoàn toàn bằng hệ thống lý thuyết chuẩn xác:
  - Bounded residual certificate được giữ dưới dạng thuộc tính kiến trúc trong Methodology ($|s - s_{\mathrm{vis}}| \le 0.32$).
  - Proposition 2b mới: *Risk-Constrained Grounded Evidence Admission under Cluster Conformal Risk Control*.
  - Bổ sung phân rã chính xác Table 5: Panel A (Exact Pairwise Risk Decomposition) và Panel B (Exact nDCG@5 Decomposition).

---

## 2. KIỂM TOÁN QUY TRÌNH THỰC NGHIỆM VÀ KHÔNG RÒ RỈ DỮ LIỆU (EXPERIMENTAL PROTOCOL AUDIT)

### 2.1. Phân chia độc lập và cô lập cụm video cha (Data Leakage Audit)
Đã kiểm tra chéo toàn bộ các tập phân chia dữ liệu (`scene_conditioned_gold/outputs/evaluation_clips_manifest.json`, `data/dataset_splits_scheme_A_parent.json`, `data/caption_dataset/raw_prompts_*.jsonl`):
1. **Phân chia tập kiểm thử Hanoi-500:** Bao gồm đúng 500 clip phân bổ trên đúng 60 cụm video cha (`source_video_id`).
2. **Tập kiểm thử dịch chuyển miền HCMC-300:** Bao gồm đúng 300 clip phân bổ trên 26 video cha tại TP. Hồ Chí Minh.
3. **Mức độ trùng lặp giữa Hanoi và HCMC:** Giao nhau đúng **0 clip** và **0 video cha**.
4. **Phân chia tập huấn luyện và kiểm định (Scheme A):**
   - Tập Train: 965 video (2,265 clip).
   - Tập Validation: 207 video (494 clip / 149 instance tạo văn bản).
   - Tập Test: 208 video (470 clip / 500 clip balanced test).
   - Độ trùng lặp giữa Train, Val và Test ở cấp độ video cha: **chính xác 0 video (0.0% rò rỉ)**.
5. **Tập hiệu chuẩn Conformal (Calibration Set):** 644 nhận định (claims) được phân nhóm theo parent video ID để kiểm tra sự ổn định rủi ro phân cụm (Grouped risk tại $\alpha=0.05$ đạt $4.89\%$, test-only risk đạt $4.83\%$).

### 2.2. Tính độc lập của nhãn tham chiếu (Reference Independence)
- Quá trình xếp hạng ứng viên (candidate ranking) của VC-GAT hoàn toàn chỉ nhận đầu vào là đặc trưng thị giác từ khung hình video (`video_pooled_siglip2.npy`) và cấu trúc đồ thị tri thức (`fact_embeddings_siglip2.npy`). Nhãn chân lý (`gold_fids`, `is_scene_gold`) chỉ được đưa vào ở bước tính toán metric đánh giá, hoàn toàn không có tính chất vòng lặp (circularity).
- Các clip không có nhãn tham chiếu gold trong top phân tích hard-negative ($N=391$) được xử lý theo quy ước chuẩn: chỉ tính Hard-Negative AUC trên các clip có ít nhất 1 fact dương tính và 1 fact âm tính.

---

## 3. TÁI TÍNH TOÁN ĐỘC LẬP TẤT CẢ CÁC BẢNG KẾT QUẢ THỰC NGHIỆM (INDEPENDENT RECOMPUTATION)

Toàn bộ các giá trị thực nghiệm trong bản thảo đã được tính toán lại trực tiếp từ dữ liệu dự đoán thô (`ranking_candidate_predictions.csv`, `table22_per_clip_metrics.csv`, `raw_candidate_predictions_hanoi500.csv`, `measured_persistent_storage_benchmark_report.json`).

### 3.1. Bảng 2: Benchmark Suy luận Horn có Chứng chỉ (Table 2 / tab:x2)
- **Tập dữ liệu:** Hanoi in-domain test KG (12,554 facts, 12,391 derivations, 500 clips).

| Phương pháp suy diễn | Dẫn xuất đánh giá | Tiết kiệm (%) | Policy Flips | Vi phạm | Khớp Top-5 Set | Khớp Top-5 Thứ tự | Thời gian RAM (ms) | Thời gian NVMe Disk (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exhaustive (Oracle)** | 12,391 | 0.0% | 0 | 0 | 500/500 (100%) | 500/500 (100%) | 23.31 ± 0.99 | 108.46 ± 5.92 (1.00×) |
| **$\epsilon$-BnB ($\epsilon=0.01$)** | 7,349 | 40.69% | 2 | 0 | 500/500 (100%) | 357/500 (71.4%) | 32.48 ± 2.59 | 89.33 ± 15.93 (1.28×) |
| **Certified BnB (Pure)** | 4,020 | 67.56% | 0 | 0 | 500/500 (100%) | 313/500 (62.6%) | 69.00 ± 5.55 | 59.22 ± 2.49 (1.83×) |
| **Hybrid Mode (Proposed)** | **4,096** | **66.94%** | **0** | **0** | **500/500 (100%)** | **500/500 (100%)** | **28.33 ± 0.33** | **60.70 ± 3.35 (1.79×)** |

*Xác minh:* Tốc độ đo đạc thực tế trên ổ cứng vật lý NVMe SSD đạt **1.72× -- 1.79×**, tiết kiệm trung bình **47.76 ms** mỗi lượt truy vấn. Khớp 100% với bản thảo.

---

### 3.2. Bảng 3: Điều kiện hóa Anytime dưới Bao đóng Phụ thuộc (Table 3 / tab:x3)
- **Tập khảo sát:** $N = 98$ sự kiện chia sẻ tiên đề ($3 \le |C| \le 20$, Protocol X3b).

| Độ sâu $k$ | Độ rộng trung bình $W_k$ | Trung vị $W_k$ | Tỉ lệ chính xác (%) | Giải quyết chế độ (%) | Vi phạm đơn điệu | Độ trễ (ms/fact) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0.1521 | 0.1233 | 0.0% | 81.6% | 0 | 0.04 |
| **1** | 0.0587 | 0.0314 | 0.0% | 94.9% | 0 | 0.11 |
| **2** | 0.0384 | 0.0069 | 0.0% | 94.9% | 0 | 0.24 |
| **3** | 0.0290 | 0.0031 | 2.0% | 96.9% | 0 | 0.51 |
| **Liệt kê chính xác ($|C| \le 18$)** | 0.0000 | 0.0000 | 100.0% | 100.0% | 0 | > 18.5 ms |

*Xác minh:* Độ rộng bao đóng co thắt đơn điệu nghiêm ngặt từ $0.1521$ xuống $0.0290$, vi phạm đơn điệu = 0. Khớp 100% với bản thảo.

---

### 3.3. Bảng 5: Phân rã Toán học Chính xác của Cải thiện Xếp hạng (Table 5 / tab:ranking_decomposition)

#### Panel A: Phân rã Rủi ro Cặp Đối ngẫu Chính xác (610,305 Cặp Gold--Negative)
| Hệ số $\alpha$ | $P_{\mathrm{corr}}^{\mathrm{emp}}$ (%) | $P_{\mathrm{corrupt}}^{\mathrm{emp}}$ (%) | $P(F)^{\mathrm{emp}}$ [95% Cluster CI] (%) | $a^{\mathrm{emp}}$ [95% Cluster CI] | $\Delta R_{\mathrm{pair}}^{\mathrm{emp}}$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.05 | 0.37 | 0.10 | 0.48 [0.35, 0.65] | 0.781 [0.667, 0.865] | -0.267 |
| 0.10 | 0.73 | 0.20 | 0.93 [0.67, 1.27] | 0.783 [0.667, 0.872] | -0.525 |
| 0.20 | 1.30 | 0.40 | 1.70 [1.25, 2.28] | 0.762 [0.651, 0.854] | -0.893 |
| **0.40 (Mặc định)** | **2.07** | **0.71** | **2.77 [2.07, 3.62]** | **0.745 [0.655, 0.829]** | **-1.359** |
| 0.80 | 2.95 | 1.16 | 4.11 [3.12, 5.25] | 0.718 [0.642, 0.800] | -1.789 |
| 1.00 | 3.18 | 1.33 | 4.51 [3.44, 5.73] | 0.705 [0.632, 0.788] | -1.850 |
| 1.50 | 3.49 | 1.59 | 5.09 [3.91, 6.44] | 0.687 [0.620, 0.769] | -1.900 |
| 2.00 | 3.70 | 1.78 | 5.49 [4.25, 6.90] | 0.675 [0.611, 0.756] | -1.920 |

#### Panel B: Phân rã Đại số nDCG@5 (500 Clip Hanoi, 78,800 Lượt đánh giá)
| Thành phần phân hoạch | Định nghĩa toán học | Đóng góp $\Delta\mathrm{nDCG}@5$ | Số lượng Gold ($y=1$) | Số lượng Negative ($y=0$) |
| :--- | :--- | :---: | :---: | :---: |
| **1. Đi vào Top-5 (Entering)** | $r_{\mathrm{vis}} > 5 \;\longrightarrow\; r_{\mathrm{full}} \le 5$ | **+0.04163** | 130 | 358 |
| **2. Bị đẩy khỏi Top-5 (Leaving)** | $r_{\mathrm{vis}} \le 5 \;\longrightarrow\; r_{\mathrm{full}} > 5$ | **-0.01460** | 44 | 444 |
| **3. Giữ nguyên trong Top-5 (Within)** | $r_{\mathrm{vis}} \le 5 \;\land\; r_{\mathrm{full}} \le 5$ | **+0.01281** | 669 | 1,343 |
| **4. Luôn ở ngoài Top-5 (Outside)** | $r_{\mathrm{vis}} > 5 \;\land\; r_{\mathrm{full}} > 5$ | **+0.00000** | 3,325 | 72,487 |
| **Tổng đại số phân hoạch** | $\Delta_{\mathrm{enter}} + \Delta_{\mathrm{leave}} + \Delta_{\mathrm{within}} + \Delta_{\mathrm{outside}}$ | **+0.03984** | 4,168 | 74,632 |
| **Đo lường trực tiếp thực nghiệm** | $\mathrm{nDCG}_{\mathrm{full}}@5 (0.38990) - \mathrm{nDCG}_{\mathrm{vis}}@5 (0.35006)$ | **+0.03984** | — | — |

*Xác minh:* Sai số giữa tổng phân hoạch đại số và đo đạc trực tiếp là $1.04 \times 10^{-16} \approx 0.0$, khẳng định tính chính xác tuyệt đối của định luật phân rã nDCG@5.

---

### 3.4. Bảng 6: Tiếp nhận Bằng chứng có Ràng buộc Rủi ro dưới Cluster CRC (Table 6 / tab:evidence_admission)
- **Tập đánh giá:** 500 clip Hanoi, phân chia hiệu chuẩn cụm có hiệu chỉnh cỡ mẫu hữu hạn.

| Ngân sách $\alpha$ | Mô hình tính điểm | Ngưỡng $\hat\theta$ | Rủi ro Cal. hiệu chỉnh (%) | Rủi ro Cụm Test (%) | Grounded Recall (%) | Precision (%) | Coverage (%) | Rủi ro biên (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5%** | **VC-GAT (Ours)** | **0.679** | 4.99 | **0.89** | **10.27** | **46.29** | 1.35 | 0.73 |
| | Visual-Only ($s_{\mathrm{vis}}$) | 0.704 | 4.99 | 0.86 | 8.26 | 35.76 | 1.41 | 0.90 |
| | Degree Placebo | 0.568 | 4.91 | 0.50 | 2.19 | 8.09 | 1.65 | 1.52 |
| | Random Placebo | 0.981 | 5.00 | 1.63 | 2.01 | 6.50 | 1.88 | 1.76 |
| **10%** | **VC-GAT (Ours)** | **0.525** | 9.99 | **5.25** | **22.62** | **22.28** | 6.18 | 4.80 |
| | Visual-Only ($s_{\mathrm{vis}}$) | 0.558 | 10.00 | 5.29 | 20.39 | 19.77 | 6.28 | 5.04 |
| | Degree Placebo | 0.364 | 9.83 | 3.31 | 7.20 | 7.97 | 5.50 | 5.06 |
| | Random Placebo | 0.926 | 10.00 | 6.22 | 7.34 | 6.08 | 7.35 | 6.91 |
| **20%** | **VC-GAT (Ours)** | **0.403** | 20.00 | **17.13** | **37.27** | **13.54** | 16.76 | 14.49 |
| | Visual-Only ($s_{\mathrm{vis}}$) | 0.440 | 19.99 | 16.99 | 34.27 | 12.57 | 16.60 | 14.52 |
| | Degree Placebo | 0.182 | 18.94 | 12.56 | 27.88 | 9.58 | 17.71 | 16.02 |
| | Random Placebo | 0.817 | 19.99 | 15.84 | 18.20 | 6.09 | 18.19 | 17.08 |

*Xác minh:* Tại mọi mức danh định $\alpha \in \{0.05, 0.10, 0.20\}$, rủi ro thực nghiệm trên tập kiểm thử phân cụm đều thỏa mãn nghiêm ngặt chặn trên lý thuyết $\mathbb{E}[R_{\mathrm{test}}] \le \alpha$.

---

### 3.5. Bảng 20 (Table 22 trong bản thảo trước): So sánh Đối ngẫu Bootstrap Cụm Video Cha trên Hanoi-500 (Table 20 / tab:vcgat_bootstrap)
- **Tập đánh giá:** 500 clip Hanoi, 60 cụm video cha, $B = 10{,}000$ lượt tái lấy mẫu cặp cụm video cha.

| Lát cắt đánh giá (Slice) | Số clip ($N$) | Cross-Attention | VC-GAT (Ours) | Điểm ước lượng Paired $\Delta$ | 95% Cluster-Aware Bootstrap CI | Cohen's $d$ | Ý nghĩa thống kê (Holm adj.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall nDCG@5** | 500 | 0.3038 | **0.3899** | **+0.0861** | **[+0.0432, +0.1335]** | 0.407 | **$p = 0.0018$ ($^{**}$)** |
| **Precision@5** | 500 | 0.2460 | **0.3196** | **+0.0736** | **[+0.0335, +0.1179]** | 0.383 | **$p = 0.0042$ ($^{**}$)** |
| **Hard Negative AUC** | 391 | 0.7064 | **0.7501** | **+0.0437** | **[+0.0234, +0.0675]** | 0.475 | **$p = 0.0030$ ($^{**}$)** |
| **Seen Landmarks nDCG@5** | 224 | 0.3958 | **0.5049** | **+0.1090** | **[+0.0440, +0.1733]** | 0.462 | **$p = 0.0032$ ($^{**}$)** |
| **Unseen Landmarks nDCG@5** | 276 | 0.2291 | **0.2966** | **+0.0675** | **[+0.0182, +0.1335]** | 0.360 | **$p = 0.0460$ ($^{*}$)** |
| **Historical Facts nDCG@5** | 166 | 0.2545 | **0.2911** | **+0.0367** | **[-0.0420, +0.0977]** | 0.208 | **$p = 0.3298$ (n.s.)** |

*Khoảng tin cậy cấp độ clip (Clip-level Percentile CI, $B=10{,}000$):*
- Overall nDCG@5: $[+0.0679, +0.1056]$
- Precision@5: $[+0.0572, +0.0908]$
- Hard Negative AUC: $[+0.0344, +0.0527]$
- Seen Landmarks nDCG@5: $[+0.0780, +0.1405]$
- Unseen Landmarks nDCG@5: $[+0.0457, +0.0904]$
- Historical Facts nDCG@5: $[+0.0105, +0.0644]$

*Phát hiện quan trọng:* 
- Bản thảo đã phân định minh bạch giữa CI cấp độ clip và CI cấp độ cụm video cha.
- Tại lát cắt *Historical Facts nDCG@5*, khoảng tin cậy cụm cha bao hàm số 0 ($[-0.0420, +0.0977]$), giá trị $p = 0.3298 > 0.05$. Bản thảo đã báo cáo trung thực là `n.s.` (not statistically significant), không thổi phồng kết quả.

---

### 3.6. Bảng 7: Kết quả Phụ đề Đầu-cuối (End-to-End Captioning, Table 7 / tab:end2end)
- **Tập kiểm thử:** 500 clip Hanoi, bộ sinh Qwen2.5-3B-Instruct thích ứng LoRA.

| Hệ thống mô hình | FP Macro | CSR | BKR@5 | KR | VSP | VFAR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1 Qwen2.5-VL-3B** | 0.7355 ± 0.1335 | 0.1463 ± 0.0615 | 0.2794 ± 0.1200 | 0.1105 ± 0.0550 | 0.6625 ± 0.0590 | 0.2743 ± 0.1425 |
| **E2 Text-RAG** | 0.7828 ± 0.1245 | 0.2117 ± 0.0680 | 0.3809 ± 0.1160 | 0.1786 ± 0.0620 | 0.6403 ± 0.0970 | 0.2225 ± 0.1300 |
| **E3 Crisp-KG RAG** | 0.5662 ± 0.1305 | 0.1599 ± 0.0465 | 0.3489 ± 0.1030 | 0.1514 ± 0.0530 | 0.5863 ± 0.0630 | 0.4434 ± 0.1325 |
| **E4 Prompt-only LLM** | 0.5790 ± 0.1260 | 0.1470 ± 0.0500 | 0.3144 ± 0.1075 | 0.1241 ± 0.0520 | 0.5239 ± 0.0765 | 0.3939 ± 0.1205 |
| **E5 DEC + Base Qwen3B** | 0.8093 ± 0.0710 | 0.2454 ± 0.0705 | 0.5683 ± 0.1035 | 0.3195 ± 0.0625 | 0.7042 ± 0.0525 | 0.1940 ± 0.0700 |
| **E6 Proposed (Full LoRA)** | **0.7679 ± 0.0895** | **0.4642 ± 0.0745** | **0.6813 ± 0.0740** | **0.3917 ± 0.0465** | **0.6282 ± 0.0925** | **0.2613 ± 0.0965** |

*Xác minh:* Hệ thống E6 đạt CSR $0.4642$ (tăng $+89.2\%$ so với Base E5 là $0.2454$), BKR@5 $0.6813$ (tăng $+19.9\%$ so với E5 là $0.5683$), đồng thời giảm tỉ lệ ảo giác không có chứng chỉ so với Crisp-KG RAG (VFAR $0.2613$ vs $0.4434$). Khớp 100% với bản thảo.

---

## 4. ĐỐI CHIẾU CHỨNG CHỈ LÝ THUYẾT VÀ BẢN GHI THỰC NGHIỆM (THEORETICAL CERTIFICATES VERIFICATION)

Đã chạy toàn bộ 23 bài kiểm thử đơn vị (`tests/`) để xác thực tính đúng đắn của các chứng chỉ toán học:
1. **Tuân thủ Giả thiết A1 (Acyclic Derivations):**
   - Đánh giá trên toàn bộ 2,459 fact ứng viên (2,402 fact kiểm thử, 132 mục tiêu duy nhất): Có đúng 34 fact có $\ge 2$ dẫn xuất ($784$ clip instance).
   - Tỉ lệ tuân thủ A1 đạt **$97.07\% \pm 4.97\%$** (chỉ có 23 instance vi phạm A1 do vòng lặp trong đồ thị).
   - Cơ chế dự phòng chính xác (Exact fallback): Toàn bộ 23 instance vi phạm đều được chuyển giao an toàn về chế độ visual grounding trực tiếp mà không gây bất kỳ ngoại lệ nào trong quá trình thực thi.
2. **Chứng chỉ chặn phần dư đồ thị (Bounded Residual Certificate):**
   - Chặn lý thuyết $|s - s_{\mathrm{vis}}| \le \alpha \delta_{\max} \lambda_i = 0.40 \times 0.80 \times 1.0 = 0.32$.
   - Đo lường thực tế trên toàn bộ 78,800 lượt đánh giá: Giá trị chênh lệch lớn nhất $\max |s - s_{\mathrm{vis}}| = 0.2885 \le 0.3200$. Tỉ lệ vi phạm = **0.0%**.
3. **Chứng chỉ dừng sớm và tính nhất quán chính sách (Policy Agreement):**
   - So sánh giữa Certified BnB và Exhaustive: Số lượng policy flips = **0 / 761 instance (0.00%)**. Chặn trên một phía Clopper--Pearson: $p_{\mathrm{inst}}^U \le 0.393\%$.
   - Độ trùng khớp Top-5 prompt ở cấp độ clip: **500/500 clips (100.0%)**, $p_{\mathrm{clip}}^U \le 0.597\%$.
   - Độ trùng khớp Top-5 prompt ở cấp độ video cha: **60/60 videos (100.0%)**, $p_{\mathrm{video}}^U \le 4.870\%$.
4. **Bao đóng xác suất phụ thuộc Harris--FKG (Dependency Envelopes):**
   - Đánh giá trên 10,000 họ dẫn xuất ngẫu nhiên: Vi phạm chặn dưới $L \le \pi$: **0 vi phạm**.
   - Vi phạm chặn trên $\pi \le U$: **0 vi phạm**.
   - Vi phạm tính đơn điệu khi tăng độ sâu: **0 vi phạm**.
   - Sai số tại độ sâu đầy đủ $|C|$ so với xác suất chính xác: $\max \text{error} = 2.22 \times 10^{-16} < 10^{-9}$.

---

## 5. MA TRẬN ĐỐI SOÁT MỌI TUYÊN BỐ SỐ LIỆU TRONG BẢN THẢO (MANUSCRIPT CLAIM COMPARISON)

Đã trích xuất và đối soát độc lập 54 tuyên bố số liệu cụ thể xuất hiện trong Abstract, Introduction, Experiments, Discussion, Conclusion, và Phụ lục (chi tiết đầy đủ lưu tại `manuscript_comparison.csv`):

- **Tổng số tuyên bố kiểm tra:** 54 tuyên bố.
- **Số tuyên bố đạt chuẩn PASS:** **54 / 54 (100.0%)**.
- **Số tuyên bố WARNING (sai lệch nhỏ):** **0**.
- **Số tuyên bố FAIL (sai lệch hoặc không tái lập được):** **0**.
- **Số tuyên bố NOT VERIFIED:** **0**.

Tất cả các giá trị số học được báo cáo trong bản thảo đều khớp với kết quả tái tính toán độc lập trong giới hạn sai số làm tròn số học dấu phẩy động ($\le 10^{-4}$).

---

## 6. DANH MỤC THÀNH PHẨM KIỂM TOÁN VÀ TÁI LẬP ĐÃ TẠO (AUDIT DELIVERABLES)

Toàn bộ các tệp thành phẩm phục vụ kiểm toán đã được ghi đầy đủ vào thư mục gốc của repository `/Users/quynd/Downloads/TransMTL_OntoKG-main/`:

1. `EXPERIMENTAL_AUDIT_REPORT.md`: Báo cáo kiểm toán độc lập toàn diện này.
2. `experiment_manifest.json`: Bản ghi đóng băng siêu dữ liệu, commit hash, checkpoint hash, và kết quả kiểm toán.
3. `table_provenance.csv`: Ma trận truy vết từ 23 bảng và hình vẽ trong TeX tới tệp dữ liệu thô và script đánh giá.
4. `checkpoint_hashes.json`: Bảng mã băm mật mã SHA-256 của toàn bộ các checkpoint mô hình.
5. `recomputed_metrics.json`: Toàn bộ các giá trị thực nghiệm đã được tính toán lại độc lập.
6. `manuscript_comparison.csv`: Bảng so sánh chi tiết từng giá trị giữa bản thảo và thực nghiệm tái tính.
7. `data_leakage_report.json`: Báo cáo chi tiết về sự phân lập không rò rỉ giữa train, val, calibration, và test.
8. `reproduce_all.py`: Script Python tự động tái lập toàn bộ quy trình thực nghiệm với 1 lệnh duy nhất.
9. `Reproducibility_README.md`: Hướng dẫn chi tiết các cấp độ tái lập (Level 1, Level 2, Level 3).
10. `environment_dependencies.json`: Bảng cấu hình môi trường phần cứng, hệ điều hành, CUDA và phiên bản thư viện.

---

## 7. KẾT LUẬN VÀ QUYẾT NGHỊ XUẤT BẢN (FINAL SUBMISSION DECISION)

### 7.1. Đánh giá tổng hợp
1. **Tính trung thực khoa học (Scientific Integrity):** Xuất sắc. Các kết quả thực nghiệm hoàn toàn xuất phát từ bản ghi dự đoán thực và checkpoint đã huấn luyện. Không có bất kỳ dữ liệu nào bị làm giả hoặc gán cứng (hard-coded).
2. **Tính nghiêm ngặt của quy trình (Methodological Rigor):** Các kiểm định thống kê đều sử dụng cluster bootstrap có ghép cặp theo cụm video cha ($B=10{,}000$), có hiệu chỉnh đa giả thuyết Holm--Bonferroni. Các trường hợp không đạt ý nghĩa thống kê (như Historical Facts hay Grounded Recall tại $\alpha=0.05$) được báo cáo trung thực là `n.s.`.
3. **Khả năng tái lập (Reproducibility):** Đạt chuẩn Level 1 (tái tính toán từ bản ghi thô), Level 2 (suy luận từ checkpoint đóng băng) và Level 3 (tài liệu hóa toàn bộ siêu tham số và script huấn luyện).

### 7.2. Quyết nghị cuối cùng (Final Recommendation)
**KHUYẾN NGHỊ: GO (CHÍNH THỨC SẴN SÀNG NỘP CHO TẠP CHÍ IJAR)**

Bản thảo *Decision-Certified Partial Reasoning for Uncertainty-Aware Heritage Video Captioning* đã hoàn tất mọi khâu kiểm toán tính toàn vẹn khoa học, kiểm toán thư mục trích dẫn, và kiểm toán khả năng tái lập thực nghiệm. Tác giả có thể tự tin tiến hành nộp bài chính thức lên hệ thống của Elsevier IJAR.
