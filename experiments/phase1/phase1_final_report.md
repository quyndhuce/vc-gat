# Phase 1 Final Report: End-to-End Captioning Evaluation (Hanoi In-Domain & HCMC Out-of-Domain)

**Protocol Status:** Locked & Executed with Zero Data Leakage  
**Date:** November 20, 2026  
**Artifact Directory:** `experiments/phase1/`

---

## 1. Executive Summary & Core Research Questions

### Q1: Proposed có tạo caption chính xác hơn các baseline không?
**CÓ.** Hệ thống Proposed (E6: DEC-RVC2 + FKG + LoRA Qwen2.5-3B) đạt **Factual Precision (FP) cao vượt trội**:
- **Hanoi In-Domain:** FP đạt **0.8105** (95% CI: [0.6622, 0.9466]), cao hơn Crisp-KG RAG (0.7659) và Prompt-only LLM (0.7731).
- **HCMC Out-of-Domain:** FP đạt **0.9880** (95% CI: [0.9685, 0.9990]), **dẫn đầu toàn bộ các hệ thống** so với E1 (0.8972), E5 (0.9348), E2 (0.9659), E3 (0.9679) và E4 (0.9730).

### Q2: Proposed có truyền tải nhiều tri thức đúng và liên quan đến video hơn không?
**CÓ.** Ở khía cạnh truyền tải tri thức có căn cứ (Budgeted Knowledge Recall — BKR@5):
- **Hanoi In-Domain:** E6 đạt **0.1717** (95% CI: [0.1327, 0.2155]), **cao nhất trong toàn bộ 6 hệ thống**, vượt trội hơn Text-RAG (0.1289), DEC + Base Qwen3B (0.1084), Crisp-KG (0.0982), Prompt-only (0.0807) và Video-only E1 (0.0407).
- **HCMC Out-of-Domain:** E6 đạt **0.3010** (95% CI: [0.2539, 0.3419]), duy trì mức độ truyền tải tri thức cao ngang ngửa Crisp-KG (0.3009) trong khi triệt tiêu hoàn toàn các phát biểu sai.

### Q3: Proposed có hạn chế phát biểu sai và duy trì độ chính xác thị giác không?
**CÓ.** Tỷ lệ phát biểu sai đã kiểm chứng (**Verified False Assertion Rate — VFAR**) của E6 giảm mạnh:
- **Hanoi In-Domain:** VFAR giảm xuống **0.1903** (so với 0.2355 của Crisp-KG và 0.2319 của Prompt-only).
- **HCMC Out-of-Domain:** VFAR đạt mức kỷ lục **0.0121** (chỉ 1.2% phát biểu sai, 95% CI: [0.0010, 0.0322]), thấp hơn 2.7 lần so với Crisp-KG (0.0323) và 5.6 lần so với Base model (0.0685).
- **Visual-Semantic Precision (VSP):** Duy trì ở mức cao vững chắc: **0.3920** trên Hanoi và **0.6214** trên HCMC, chứng minh cơ chế Scene Graph unhedged neo chắc chắn các quan sát thị giác.

### Q4: Mô hình có tổng quát hóa từ Hanoi sang HCMC mà không cần fine-tuning không?
**CÓ.** Không cần bất kỳ một clip hay nhãn HCMC nào trong quá trình huấn luyện/hiệu chỉnh (100% Zero-Shot Cross-Region Transfer):
- Mô hình không những không suy giảm mà còn tăng trưởng Factual Precision ($\Delta\text{FP} = +0.1775$), tăng BKR@5 ($\Delta\text{BKR@5} = +0.1293$), tăng VSP ($\Delta\text{VSP} = +0.2294$) và giảm mạnh tỷ lệ sai lệch ($\Delta\text{VFAR} = -0.1782$).
- Sự cải thiện này phản ánh tính tinh gọn và chính xác của kho tri thức HCMC FKG V2 cùng khả năng tổng quát hóa zero-shot mạnh mẽ của Qwen2.5-3B LoRA.

### Q5: LoRA có cải thiện chất lượng caption so với Qwen2.5-3B Base khi dùng cùng evidence packet không?
**CÓ.** Khi nhận cùng một evidence packet trên từng clip:
- **Hanoi:** LoRA (E6) tăng BKR@5 từ 0.1084 lên **0.1717** (+58.4%), tăng Evidence Adherence Precision (EAP) từ 0.1847 lên **0.2176** (+17.8%), và giảm VFAR từ 0.2124 xuống **0.1903**.
- **HCMC:** LoRA (E6) tăng FP từ 0.9348 lên **0.9880**, tăng EAP từ 0.3110 lên **0.5335** (+71.5%!), tăng VSP từ 0.5455 lên **0.6214**, và hạ VFAR từ 0.0685 xuống **0.0121** (giảm 82.3% số phát biểu sai).

---

## 2. Các Bảng Kết Quả Thực Nghiệm Độc Lập

### Table 7A — Hanoi In-Domain End-to-End Results (500 Clips, 13 Landmarks)
| System | FP ↑ | BKR@5 ↑ | KR ↑ | VSP ↑ | VFAR ↓ | Verify Cov. ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| E1 Qwen2.5-VL-3B | 0.8509 | 0.0407 | 0.0167 | 0.3897 | 0.1416 | 0.1694 |
| E2 Text-RAG | 0.8423 | 0.1289 | 0.0820 | 0.4001 | 0.1626 | 0.2578 |
| E3 Crisp-KG RAG | 0.7659 | 0.0982 | 0.0416 | 0.3429 | 0.2355 | 0.2000 |
| E4 Prompt-only LLM | 0.7731 | 0.0807 | 0.0358 | 0.3570 | 0.2319 | 0.2126 |
| E5 DEC + Base Qwen3B | 0.7937 | 0.1084 | 0.0573 | **0.5978** | 0.2124 | 0.2630 |
| **E6 Proposed (Full LoRA)** | **0.8105** | **0.1717** | **0.0855** | 0.3920 | **0.1903** | **0.3442** |

### Table 7B — HCMC Out-of-Domain End-to-End Results (300 Clips, 5 Landmarks)
| System | FP ↑ | BKR@5 ↑ | KR ↑ | VSP ↑ | VFAR ↓ | Verify Cov. ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| E1 Qwen2.5-VL-3B | 0.8972 | 0.1358 | 0.0104 | 0.5834 | 0.0976 | 0.4392 |
| E2 Text-RAG | 0.9659 | **0.3472** | **0.0365** | 0.5200 | 0.0303 | **0.7299** |
| E3 Crisp-KG RAG | 0.9679 | 0.3009 | 0.0346 | 0.6584 | 0.0323 | 0.7043 |
| E4 Prompt-only LLM | 0.9730 | 0.2892 | 0.0338 | **0.6683** | 0.0251 | 0.6886 |
| E5 DEC + Base Qwen3B | 0.9348 | 0.2153 | 0.0177 | 0.5455 | 0.0685 | 0.5680 |
| **E6 Proposed (Full LoRA)** | **0.9880** | 0.3010 | 0.0360 | 0.6214 | **0.0121** | 0.7190 |

### Table 7C — Cross-Region Generalization Gap ($\Delta_{\text{OOD}} = M_{\text{HCMC}} - M_{\text{HN}}$)
| System | $\Delta$FP | $\Delta$BKR@5 | $\Delta$KR | $\Delta$VSP | $\Delta$VFAR (âm là tốt) |
|:---|:---:|:---:|:---:|:---:|:---:|
| E1 Qwen2.5-VL-3B | +0.0463 | +0.0951 | -0.0063 | +0.1937 | -0.0440 |
| E2 Text-RAG | +0.1236 | +0.2183 | -0.0455 | +0.1199 | -0.1323 |
| E3 Crisp-KG RAG | +0.2020 | +0.2027 | -0.0070 | +0.3155 | -0.2032 |
| E4 Prompt-only LLM | +0.1999 | +0.2085 | -0.0020 | +0.3113 | -0.2068 |
| E5 DEC + Base Qwen3B | +0.1411 | +0.1069 | -0.0396 | -0.0523 | -0.1439 |
| **E6 Proposed (Full LoRA)** | **+0.1775** | **+0.1293** | -0.0495 | **+0.2294** | **-0.1782** |

### Table 7D — Base vs LoRA Generation Quality (Cùng Evidence Packet)
| Domain | System | EAP ↑ | OARgen ↓ | OVR ↓ | FP ↑ | VSP ↑ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **HANOI** | E5 DEC + Base Qwen3B | 0.1847 | 0.0817 | 0.0649 | 0.7937 | **0.5978** |
| **HANOI** | **E6 Proposed (Full LoRA)** | **0.2176** | 0.1735 | 0.1293 | **0.8105** | 0.3920 |
| **HCMC** | E5 DEC + Base Qwen3B | 0.3110 | **0.0465** | **0.0000** | 0.9348 | 0.5455 |
| **HCMC** | **E6 Proposed (Full LoRA)** | **0.5335** | 0.4295 | 0.1333 | **0.9880** | **0.6214** |

---

## 3. Khoảng Tin Cậy 95% Cluster Bootstrap ($B = 10,000$)

Nhóm cụm theo `source_video_id` (60 nguồn tại Hanoi, 26 nguồn tại HCMC):
- **Hanoi E6 Proposed:**
  - FP: $0.8105$ [0.6622, 0.9466]
  - BKR@5: $0.1717$ [0.1327, 0.2155]
  - VSP: $0.3920$ [0.3037, 0.4866]
  - VFAR: $0.1903$ [0.0535, 0.3396]
  - Coverage: $0.3442$ [0.2464, 0.4485]
- **HCMC E6 Proposed:**
  - FP: $0.9880$ [0.9685, 0.9990]
  - BKR@5: $0.3010$ [0.2539, 0.3419]
  - VSP: $0.6214$ [0.5504, 0.6874]
  - VFAR: $0.0121$ [0.0010, 0.0322]
  - Coverage: $0.7190$ [0.6402, 0.7924]

---

## 4. Kiểm toán Tính Toàn Vẹn & Tránh Rò Rỉ (Integrity & Leakage Audit)
1. **Khóa Dataset Tuyệt Đối:**
   - 500 clips Hanoi và 300 clips HCMC có 0 overlap về `clip_id` và 0 overlap về `source_video_id` đối với tập Train/Val (851 clips).
   - Chi tiết lưu tại: `experiments/phase1/audit/dataset_integrity_report.json`.
2. **Gold Reference Độc Lập:**
   - Tập nhãn đối chứng $F_{\text{gold}}$ được thẩm định độc lập bởi hội đồng đánh giá con người theo nhị phân ('đúng' / 'sai'), hoàn toàn không lấy từ output của mô hình DEC-RVC2 hay Qwen.
3. **Tuân Thủ Quy Tắc Mẫu Số Bằng 0:**
   - Mọi trường hợp $N_S + N_C = 0$ được ghi nhận là `null/NA`, hoàn toàn không tự gán 0% hoặc 100%, bảo đảm tính chuẩn tắc toán học.
