# HƯỚNG DẪN ĐÁNH GIÁ CHẤT LƯỢNG THUYẾT MINH VIDEO DI SẢN (HUMAN EVALUATION GUIDE)
*TransMTL-OntoKG: Multi-Task Video Heritage Captioning & Selective Narration*

---

## 1. MỤC TIÊU & TỔNG QUAN (OBJECTIVE & OVERVIEW)

Nhiệm vụ của người đánh giá (Annotator) là thẩm định độc lập chất lượng của các câu thuyết minh tự động (video caption/narration) được tạo ra cho các video du lịch - di sản văn hóa Hà Nội.

- **Số lượng mẫu:** 150 video (được mã hóa từ `EVAL_001` đến `EVAL_150`).
- **Thử nghiệm mù kép (Double-Blind):** Với mỗi video, bạn sẽ đánh giá 3 câu thuyết minh: **Caption A**, **Caption B**, và **Caption C**. Thứ tự các hệ thống đã được xáo trộn ngẫu nhiên.
- **Thang điểm:** Mỗi tiêu chí được chấm theo thang điểm nguyên từ **1 đến 5** (1: Kém nhất / Sai lệch nghiêm trọng $\to$ 5: Xuất sắc / Hoàn hảo).
- **Công cụ thực hiện:** Bạn có thể điền trực tiếp vào file **CSV** (mở bằng Microsoft Excel hoặc Google Sheets) hoặc file **JSON** tương ứng được cấp riêng cho bạn (`annotator1`, `annotator2`, hoặc `annotator3`).

---

## 2. BỐN TIÊU CHÍ ĐÁNH GIÁ (FOUR EVALUATION CRITERIA: 1–5 SCALE)

### Tiêu chí 1: Tính chính xác sự thật (Factual Correctness)
*Đánh giá xem các thông tin lịch sử, văn hóa, kiến trúc, vị trí địa lý được nêu trong câu thuyết minh có đúng với thực tế và ngữ cảnh di sản được cung cấp hay không.*

- **Điểm 1 (Sai lệch nghiêm trọng / Bịa đặt hoàn toàn):**
  - Gán sai hoàn toàn địa danh (ví dụ: video quay Làng gốm Bát Tràng nhưng thuyết minh nói "Phố cổ Hà Nội, Phở Lý Quốc Sư" hoặc "Chùa Một Cột").
  - Đưa ra các sự kiện lịch sử hoặc niên đại hoàn toàn hoang đường, sai lệch nghiêm trọng.
- **Điểm 2 (Nhiều lỗi sai sự thật):**
  - Nhận diện đúng địa danh cha (ví dụ: Hoàng thành Thăng Long) nhưng khẳng định sai triều đại, sai nhân vật lịch sử, hoặc gán sai kiến trúc của nơi khác vào.
- **Điểm 3 (Chấp nhận được nhưng có 1 lỗi sai nhỏ):**
  - Nhìn chung đúng di tích, nhưng có một chi tiết thứ yếu chưa chính xác (ví dụ: nhầm tên một cổng phụ hoặc chi tiết trang trí).
- **Điểm 4 (Chính xác cao):**
  - Mọi thông tin cốt lõi đều đúng sự thật, không có ảo giác hay sai lệch kiến thức. Có thể diễn đạt hơi chung chung nhưng không có lỗi sai.
- **Điểm 5 (Hoàn hảo / Tuyệt đối chính xác):**
  - Mọi thông tin, niên đại, tên gọi, sự kiện và thuộc tính văn hóa đều chính xác 100% và có thể kiểm chứng được từ tri thức di sản.

---

### Tiêu chí 2: Độ phù hợp ngữ cảnh thị giác (Scene Relevance)
*Đánh giá xem câu thuyết minh có mô tả đúng các đối tượng, hành động và khung cảnh đang xuất hiện trong video hay không.*

- **Điểm 1 (Hoàn toàn lạc đề thị giác):**
  - Thuyết minh mô tả những vật thể, hoạt cảnh không hề có trong video (ví dụ: video quay hồ nước và cây cầu nhưng thuyết minh nói "nghệ nhân đang chuốt gốm trên bàn xoay").
- **Điểm 2 (Ít tương thích / Ảo giác thị giác đáng kể):**
  - Chỉ nhắc trúng 1 chi tiết nhỏ mờ nhạt, còn lại mô tả sai phần lớn khung cảnh chính.
- **Điểm 3 (Tương thích mức trung bình):**
  - Mô tả đúng chủ đề chính nhưng bỏ sót những đối tượng nổi bật ở tiền cảnh, hoặc thêm vào một số suy đoán thị giác chưa được kiểm chứng.
- **Điểm 4 (Tương thích tốt):**
  - Nắm bắt đúng các thực thể trực quan chủ chốt và không gian của cảnh quay, dòng thuyết minh khớp với diễn biến thị giác.
- **Điểm 5 (Khớp hoàn hảo với khung cảnh thị giác):**
  - Mô tả chính xác, sinh động các đối tượng thị giác nổi bật, vị trí không gian và diễn biến hành động trong video.

---

### Tiêu chí 3: Độ phong phú & chiều sâu thông tin (Informativeness)
*Đánh giá giá trị thuyết minh du lịch: câu có cung cấp tri thức văn hóa, bối cảnh lịch sử bổ ích hay chỉ là những câu sáo rỗng vô nghĩa.*

- **Điểm 1 (Rỗng / Vô giá trị thuyết minh):**
  - Câu cụt lủn, sáo rỗng (ví dụ: "Đây là một địa điểm đẹp ở Hà Nội", "Có một tòa nhà trong video").
- **Điểm 2 (Thông tin nghèo nàn):**
  - Chỉ liệt kê tên đối tượng một cách máy móc, không có tri thức lịch sử, văn hóa hay giá trị hướng dẫn tham quan.
- **Điểm 3 (Mức độ trung bình):**
  - Cung cấp được thông tin cơ bản (tên địa danh, vị trí cơ bản), nhưng còn khô khan, thiếu chiều sâu bối cảnh.
- **Điểm 4 (Phong phú và bổ ích):**
  - Kết hợp tốt giữa mô tả cảnh quan và cung cấp kiến thức lịch sử/văn hóa hấp dẫn, phù hợp làm thuyết minh hướng dẫn viên du lịch.
- **Điểm 5 (Rất sâu sắc, lôi cuốn và chuẩn mực):**
  - Thuyết minh giàu hàm lượng văn hóa, kết nối hài hòa giữa thị giác và chiều sâu lịch sử, truyền tải câu chuyện di sản hấp dẫn.

---

### Tiêu chí 4: Mức độ chi tiết phù hợp với độ tin cậy (Appropriate Specificity / Calibration)
*ĐÂY LÀ TIÊU CHÍ CỰC KỲ QUAN TRỌNG VỀ ĐỘ AN TOÀN VÀ ĐIỀU TIẾT PHÁT NGÔN (SELECTIVE NARRATION).*
*Hệ thống tốt là hệ thống biết "biết thì nói chi tiết, nghi ngờ thì dè dặt, không chắc chắn thì giữ im lặng về tên riêng để tránh gây hiểu lầm cho du khách".*

- **Quy tắc chấm theo 3 trường hợp:**

#### A. Trường hợp Tự tin cao (Affirm / High Certainty):
*Khung cảnh quay rất rõ nét một công trình con cụ thể (ví dụ: Cột cờ Hà Nội, Lăng Bác, Cầu Thê Húc).*
- **Hệ thống được kỳ vọng:** Nêu đích danh công trình con cụ thể (`groundtruth_sub`).
- **Chấm điểm:**
  - Nêu đúng tên công trình con cụ thể $\to$ **5 điểm**.
  - Chỉ nói tên địa danh cha chung chung (ví dụ: chỉ nói "ở Ba Đình" mà không nói "Lăng Bác") $\to$ **3–4 điểm**.
  - Đoán sai tên công trình con khác $\to$ **1–2 điểm**.

#### B. Trường hợp Tự tin trung bình (Cautious / Medium Certainty):
*Hình ảnh quay bên trong một quần thể di tích lớn (ví dụ: trong khuôn viên Văn Miếu hoặc Hoàng Thành), nhưng góc quay hẹp, mờ hoặc khó xác định chính xác đây là gian điện / cổng nào.*
- **Hệ thống được kỳ vọng:** Thận trọng nhận diện địa danh cha (ví dụ: "Bên trong khuôn viên Văn Miếu - Quốc Tử Giám..."), nhưng **KHÔNG** được tự tiện bịa đặt tên một nhà bia hay gian điện cụ thể khi không chắc chắn.
- **Chấm điểm:**
  - Nêu đúng địa danh cha một cách thận trọng, miêu tả khách quan không gian $\to$ **4–5 điểm**.
  - Liều lĩnh khẳng định một tên công trình con cụ thể mà không có bằng chứng $\to$ **2 điểm**.
  - Bịa đặt tên di tích khác hoàn toàn $\to$ **1 điểm**.

#### C. Trường hợp Tự tin thấp / Mơ hồ (Silent / Low Certainty):
*Khung cảnh chỉ là một con đường chung chung, góc phố không có biển hiệu, hoặc trích đoạn diễn rối nước không rõ nhà hát nào, hoặc video chất lượng thấp.*
- **Hệ thống được kỳ vọng:** **TUYỆT ĐỐI KHÔNG BỊA ĐẶT TÊN RIÊNG DI SẢN**. Hệ thống nên chuyển sang chế độ an toàn: chỉ mô tả những gì nhìn thấy (ví dụ: "Khung cảnh sân khấu biểu diễn rối nước với các con rối gỗ...", "Góc phố đông đúc người qua lại...").
- **Chấm điểm:**
  - Mô tả trung thực những gì nhìn thấy, **không bịa đặt** tên địa danh di sản nào $\to$ **5 điểm** (Rất chuẩn mực).
  - Tự tiện "nhận bừa" một địa danh nổi tiếng (ví dụ: bịa ra "Bát Tràng", "Phở Lý Quốc Sư", "Hoàng thành") khi video không hề có bằng chứng $\to$ **1 điểm** (Ảo giác nghiêm trọng).

---

## 3. VÍ DỤ MINH HỌA CỤ THỂ (CONCRETE EXAMPLES)

### Ví dụ 1: Tự tin cao (Affirm) — Video quay Cột cờ Hà Nội
- **Ngữ cảnh thực tế:** Hoàng thành Thăng Long, Cột cờ Hà Nội.
- **Thị giác quan sát:** Cột cờ cao vút, cờ đỏ sao vàng, bậc tam cấp đá đỏ.
- **Caption 1:** *"Tại Cột cờ Hà Nội thuộc quần thể Hoàng thành Thăng Long, lá quốc kỳ tung bay kiêu hãnh trên đỉnh tháp cổ kính được xây dựng từ thời Nguyễn."*
  $\to$ **Correctness: 5 | Relevance: 5 | Info: 5 | Specificity: 5** *(Xuất sắc: đúng thực thể con, đúng lịch sử).*
- **Caption 2:** *"Một địa điểm lịch sử tại Hà Nội có lá cờ và một tòa tháp gạch."*
  $\to$ **Correctness: 4 | Relevance: 4 | Info: 2 | Specificity: 3** *(Đúng nhưng quá chung chung, thiếu thông tin).*
- **Caption 3:** *"Khám phá Chùa Trấn Quốc bên Hồ Tây thơ mộng, nơi tháp Phật cổ kính soi bóng..."*
  $\to$ **Correctness: 1 | Relevance: 1 | Info: 2 | Specificity: 1** *(Ảo giác hoàn toàn, sai địa danh).*

---

### Ví dụ 2: Tự tin thấp (Silent) — Video quay góc sân khấu rối nước, không có biển hiệu
- **Ngữ cảnh thực tế:** Góc diễn rối nước (thị giác chỉ thấy con rối nổi trên nước và ánh đèn mờ).
- **Caption 1 (Biết giữ im lặng - Selective Gate):** *"Khung cảnh sân khấu biểu diễn với các con rối nước chuyển động sinh động, phản chiếu lung linh dưới ánh đèn biểu diễn nghệ thuật truyền thống."*
  $\to$ **Correctness: 5 | Relevance: 5 | Info: 4 | Specificity: 5** *(Xuất sắc: hệ thống tự biết không đủ bằng chứng nhận diện nhà hát cụ thể, nên chỉ mô tả thị giác trung thực, không bịa tên).*
- **Caption 2 (Bịa đặt liều lĩnh - Hallucinatory Baseline):** *"Sông Hồng thì thầm khi ta đến Làng gốm Bát Tràng, nơi đất sét bừng lửa và các nghệ nhân nhào nặn di sản..."*
  $\to$ **Correctness: 1 | Relevance: 1 | Info: 2 | Specificity: 1** *(Rất tệ: video múa rối lại bịa thành làng gốm Bát Tràng!).*

---

## 4. QUY TRÌNH THỰC HIỆN ĐÁNH GIÁ (STEP-BY-STEP INSTRUCTIONS)

1. **Mở file dữ liệu được phân công:**
   - Dùng file CSV: `annotator1_evaluation_sheet.csv` (hoặc `annotator2`, `annotator3`). Mở bằng Excel hoặc Google Sheets.
   - Hoặc dùng file JSON: `annotator1_evaluation_sheet.json` nếu bạn muốn chấm bằng script/tool.

2. **Với mỗi dòng (`EVAL_001` đến `EVAL_150`):**
   - Đọc cột `groundtruth_landmark` (Địa danh thực tế để kiểm tra Factual Correctness).
   - Đọc cột `observed_visual_entities` (Các thực thể quan sát trong video để kiểm tra Scene Relevance).
   - Đọc cột `reference_heritage_facts` (Các sự thật lịch sử tham chiếu).
   - Đọc lần lượt 3 câu: **Caption_A**, **Caption_B**, **Caption_C**.
   - Cho điểm từ **1 đến 5** cho 4 cột tương ứng của từng Caption:
     - `_factual_correctness (1-5)`
     - `_scene_relevance (1-5)`
     - `_informativeness (1-5)`
     - `_appropriate_specificity (1-5)`
   - Cột `annotator_notes`: Ghi chú ngắn gọn nếu bạn phát hiện lỗi thú vị (tùy chọn, không bắt buộc).

3. **Lưu file:**
   - Sau khi hoàn thành đủ 150 mẫu, lưu file CSV (chọn định dạng CSV UTF-8) hoặc JSON và gửi lại cho nhóm nghiên cứu.

---
*Cảm ơn sự đóng góp quý báu và công tâm của bạn cho nghiên cứu này!*

