# HỆ THỐNG GIÁM ĐỊNH ẢNH SỐ ĐA TẦNG VÀ PHÁT HIỆN AI THẾ HỆ MỚI
Nâng cấp từ công trình nghiên cứu B-Free (CVPR 2025)

Web App Serverless: https://sangbv2206--bfree-forensics-web.modal.run  
Bài báo gốc: "A Bias-Free Training Paradigm for More General AI-generated Image Detection" (GRIP - Đại học Federico II Naples, CVPR 2025)

---

## 1. PHÂN TÍCH ĐIỂM YẾU CỦA BẢN GỐC TÁC GIẢ VÀ CÁC MODULE BỔ SUNG

Bản gốc của nhóm nghiên cứu GRIP công bố tại CVPR 2025 cung cấp mô hình B-Free (dựa trên backbone DINOv2 / ViT). Tuy nhiên, khi đưa vào kiểm thử thực tế và ứng dụng giám định thực tế, mã nguồn gốc bộc lộ 5 điểm yếu kỹ thuật nghiêm trọng. Để giải quyết, dự án đã xây dựng mới hoàn toàn 4 module độc lập:

### Điểm yếu 1: Mù can thiệp cục bộ (Inpainting Blindness)
* **Bản gốc:** B-Free được thiết kế như một bộ phân loại toàn ảnh (Global Image Classifier). Cơ chế hoạt động là co giãn toàn bộ bức ảnh về kích thước cố định và trích xuất vector đặc trưng toàn cảnh.
* **Hậu quả:** Đối với trường hợp ảnh chụp thật ngoài đời (chiếm 80% - 90% diện tích gồm bàn ghế, bối cảnh, con người) nhưng bị dùng AI đổi/chèn một vật thể (như Google Magic Editor, Generative Fill), các đặc trưng máy ảnh thật lấn át hoàn toàn vùng AI. Điểm logit bị kéo sâu về phía âm khiến bản gốc kết luận sai thành "ẢNH THẬT".
* **Khắc phục qua `run_pipeline.py`:** Tích hợp kỹ thuật Multi-Crop Patch Scanning. Hệ thống chia ảnh thành 5 phân vùng chiến lược (Center + 4 góc phân vùng) và suy luận theo lô (Batch Inference) qua mạng DINOv2. Nếu một vùng cục bộ bị AI vẽ đè, điểm logit của riêng vùng đó sẽ vọt lên dương, kích hoạt nhãn LOCAL INPAINTING.

### Điểm yếu 2: Không có khả năng nhận diện Chữ ký số và Dấu ấn AI (Watermark & Provenance)
* **Bản gốc:** Hoàn toàn bỏ qua tầng thông tin siêu dữ liệu (Metadata) và dấu vết đồ họa nhân tạo dán ở mép ảnh.
* **Hậu quả:** Các mô hình thế hệ mới (Google Gemini / Imagen 3, DALL-E 3, Adobe Firefly, TikTok AI) luôn tự động nhúng chứng chỉ số hoặc đóng dấu watermark bán trong suốt ở góc ảnh, nhưng bản gốc không đọc được và bỏ lọt.
* **Khắc phục qua `run_pipeline.py`:** Xây dựng hàm `check_ai_watermark_and_signatures` hoạt động theo nguyên lý chuẩn hóa:
  1. Quét container siêu dữ liệu chuẩn quốc tế mở: C2PA (Coalition for Content Provenance and Authenticity), chuẩn IPTC `trainedAlgorithmicMedia`, `synthid`.
  2. Áp dụng toán tử trường vi phân Sobel bậc 1 phân tích bất thường gradient và entropy màu tại 4 góc viền ảnh để bắt các vết đè đồ họa nhân tạo (Generic Graphic Overlay).

### Điểm yếu 3: Mô hình "Hộp đen" không có giải thích pháp chứng (Black-box Limitation)
* **Bản gốc:** Chỉ trả về duy nhất một con số logit vô hướng (ví dụ: -2.179 hay +1.450). Không chỉ ra được bằng chứng tại sao ảnh bị coi là giả và không định vị được vị trí bị sửa.
* **Hậu quả:** Không thể sử dụng trong công tác giám định pháp lý, báo chí hay kiểm duyệt nội dung vì thiếu tính minh bạch và khả năng giải thích (Explainable AI).
* **Khắc phục qua `forensic_analysis.py`:** Xây dựng bộ công cụ xử lý ảnh số phân tích dấu vết vật lý quang học (nhiễu cảm biến SRM, sai lệch nén ELA, phổ Fourier) và tự động khoanh vùng đỏ (Bounding Box) vị trí vật thể nghi vấn.

### Điểm yếu 4: Giao diện dòng lệnh thô sơ, không có khả năng tương tác
* **Bản gốc:** Chỉ có một file script chạy console (`main_bfree.py`), yêu cầu người dùng phải tự chuẩn bị file CSV, tự gõ lệnh terminal và kết quả trả về cũng chỉ là file CSV.
* **Hậu quả:** Người dùng thông thường hoặc các chuyên gia thẩm định không chuyên lập trình không thể sử dụng.
* **Khắc phục qua `app.py`:** Xây dựng giao diện Web Gradio 5 chuyên nghiệp, hỗ trợ kéo thả ảnh 1-chạm, bảng hiển thị kết quả phân loại 3 trạng thái, hiển thị 4 chỉ số giám định độc lập và trích xuất dữ liệu kỹ thuật JSON.

### 4. Khác biệt về Phạm vi ứng dụng: Từ Nghiên cứu thực nghiệm sang Dịch vụ thực tế
* **Bản gốc (Academic Benchmark):** Nhóm tác giả hướng tới việc đo đạc chỉ số nghiên cứu trên các tập dữ liệu offline trong phòng thí nghiệm. Mã nguồn chỉ hỗ trợ đọc danh sách file từ máy tính nội bộ, không có cơ chế đóng gói dịch vụ (API / Web Service).
* **Đóng góp của module `modal_app.py`:** Đóng gói toàn bộ mô hình thành dịch vụ Web Serverless chạy trên đám mây. Tự động hóa việc nạp trọng số, cấu hình Modal Persistent Volume chống mất dữ liệu tải lên và áp dụng cơ chế giữ ấm container giúp người dùng bên ngoài có thể truy cập thử nghiệm trực tiếp qua Internet.

---

## 2. BẢNG TỔNG KẾT CÁC MODULE ĐƯỢC XÂY DỰNG MỚI

| Tên File | Vai trò trong hệ thống | Khắc phục hạn chế gì |
| :--- | :--- | :--- |
| `run_pipeline.py` | Pipeline giám định đa tầng & Cây quyết định kết hợp | Khắc phục lỗi "Mù can thiệp cục bộ" (bổ sung Multi-Crop Patch Scanning) và bổ sung tầng quét chữ ký số C2PA / Watermark AI tổng quát. |
| `forensic_analysis.py` | Bộ phân tích xử lý ảnh số & Tự động khoanh vùng ROI | Khắc phục tính chất "Hộp đen" của AI: minh bạch hóa bằng chứng qua SRM, ELA, phổ Fourier và tự động đóng khung Bounding Box. |
| `app.py` | Giao diện Web tương tác trực quan (Gradio 5) | Thay thế script dòng lệnh CLI đơn sơ bằng giao diện Web kéo thả ảnh 1-chạm kèm bảng điều khiển trực quan. |
| `modal_app.py` | Cấu hình triển khai Serverless Cloud (Modal) | Đưa mô hình từ nghiên cứu nội bộ thành dịch vụ trực tuyến phục vụ người dùng thực tế qua Internet. |

---

## 3. GIẢI THÍCH CHI TIẾT ĐỒ THỊ ĐỐI CHIẾU 6 ẢNH (FORENSIC ANALYSIS REPORT)

Khi một bức ảnh được gửi vào hệ thống, module `forensic_analysis.py` sẽ thực thi một chuỗi thuật toán xử lý ảnh số và xuất ra bản đồ đối chiếu gồm 6 ảnh (bố cục 3 hàng x 2 cột). Dưới đây là ý nghĩa vật lý và nguyên lý của từng ảnh:

```
┌──────────────────────────────────────┬──────────────────────────────────────┐
│  Ảnh 1: Ảnh gốc & Khoanh vùng        │  Ảnh 2: Zoom cận cảnh vùng nghi vấn  │
│  (Toàn cảnh với Bounding Box)        │  (Crop chi tiết đối tượng AI)        │
├──────────────────────────────────────┼──────────────────────────────────────┤
│  Ảnh 3: ELA Toàn cảnh                │  Ảnh 4: ELA Vi mô vùng nghi vấn      │
│  (Sai lệch mức nén JPEG toàn bức)    │  (Biên độ suy hao nén tại vùng sửa)  │
├──────────────────────────────────────┼──────────────────────────────────────┤
│  Ảnh 5: Bản đồ nhiễu SRM toàn cảnh   │  Ảnh 6: SRM Vi mô vùng nghi vấn      │
│  (Dấu vân cảm biến quang học PRNU)   │  (Sự đứt gãy nhiễu hạt cảm biến)     │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

### Ảnh 1: Ảnh Gốc (Toàn cảnh & Khoanh vùng đối tượng)
* **Ý nghĩa:** Hiển thị bức ảnh gốc đầu vào kèm khung chữ nhật đỏ (Bounding Box) tự động bao quanh đối tượng bị nghi ngờ AI can thiệp.
* **Nguyên lý XLA:** Tọa độ khung được tính toán tự động thông qua thuật toán hợp nhất 3 tầng: Saliency phổ Fourier (đo vùng hội tụ tần số không gian), năng lượng phần dư SRM và độ tương phản màu nền, sau đó phân ngưỡng Otsu và lọc hình thái học.

### Ảnh 2: Zoom Chi Tiết Vùng Nghi Vấn (Target ROI Crop)
* **Ý nghĩa:** Trích xuất và phóng to cận cảnh khu vực bên trong khung Bounding Box.
* **Nguyên lý XLA:** Giúp chuyên gia quan sát trực tiếp các dị thường về mặt thị giác mà mắt thường ở góc nhìn toàn cảnh dễ bỏ sót: đường biên ghép nối thiếu tự nhiên, độ sắc nét không tương thích với độ sâu trường ảnh (DoF), các chi tiết biến dạng cấu trúc hoặc quang sai thấu kính không đồng nhất.

### Ảnh 3: Error Level Analysis - ELA Toàn cảnh (Sai lệch nén JPEG toàn bức)
* **Ý nghĩa:** Bản đồ trực quan hóa mức sai lệch lượng tử hóa nén JPEG trên toàn bức ảnh.
* **Nguyên lý XLA:** Ảnh gốc được nén lại ở mức chất lượng 90% (chuẩn nén bảng lượng tử hóa chuẩn), sau đó tính hiệu số tuyệt đối giữa ảnh gốc và ảnh nén lại: `Diff = |Original - Resaved| * Scale`. Trong một bức ảnh tự nhiên đồng nhất, mức sai lệch nén phân bố đều theo kết cấu bề mặt. Nếu một vùng bị chỉnh sửa hoặc chèn thêm sau này, vùng đó sẽ có mức ELA sáng rực hoặc tối khác thường so với nền xung quanh.

### Ảnh 4: ELA Vi Mô Vùng Nghi Vấn (Local ELA Anomaly)
* **Ý nghĩa:** Phóng to bản đồ ELA tại chính xác khu vực vật thể bị nghi ngờ.
* **Nguyên lý XLA:** Cho phép soi rõ đường viền biên giới tiếp giáp giữa vật thể AI và nền ảnh thật. Vùng biên tiếp giáp thường xuất hiện một quầng vi phân sáng sắc nét do sự không đồng nhất về chu kỳ khối 8x8 DCT của chuẩn nén JPEG.

### Ảnh 5: Bản Đồ Năng Lượng Nhiễu SRM Toàn cảnh (Spatial Rich Model Residual)
* **Ý nghĩa:** Trích xuất dấu vết nhiễu cảm biến quang học tần số cao trên toàn khung hình (hiển thị theo dải màu nhiệt Inferno).
* **Nguyên lý XLA:** Sử dụng bộ lọc nhân chập High-pass bậc 2 để triệt tiêu toàn bộ thông tin màu sắc và ngữ nghĩa của cảnh vật, chỉ giữ lại phần dư nhiễu cảm biến (PRNU - Photo-Response Non-Uniformity). Ảnh chụp từ máy ảnh thật luôn có một lớp hạt nhiễu cảm biến đồng nhất phủ kín từ góc này sang góc kia.

### Ảnh 6: SRM Vi Mô Vùng Nghi Vấn (Local Noise Discontinuity)
* **Ý nghĩa:** Cận cảnh cấu trúc nhiễu cảm biến tại vùng bị AI can thiệp.
* **Nguyên lý XLA:** Các công cụ tạo ảnh AI hoặc Inpainting không thể mô phỏng chính xác dấu vân cảm biến quang học ngẫu nhiên của camera vật lý. Tại vùng AI vẽ vào, năng lượng nhiễu thường bị phẳng hóa (màu tím tối - thiếu hạt nhiễu tự nhiên) hoặc mang cấu trúc tuần hoàn giả tạo lệch pha hoàn toàn so với hạt nhiễu máy ảnh của nền xung quanh.

---

## 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```
B-Free/
│
├── README.md                     <-- Tài liệu kỹ thuật chính của dự án
│
└── code/
    ├── app.py                    <-- [Mới] Giao diện Web Gradio 5
    ├── run_pipeline.py           <-- [Mới] Pipeline giám định kết hợp B-Free + Watermark/C2PA + Multi-Crop
    ├── forensic_analysis.py      <-- [Mới] Module trích xuất SRM, ELA, Bounding Box và vẽ đồ thị 6 Panel
    ├── modal_app.py              <-- [Mới] File triển khai Serverless Cloud trên nền tảng Modal
    ├── localize_fake.py          <-- Thuật toán sliding window gốc của B-Free
    ├── networks/                 <-- Kiến trúc mạng ViT DINOv2
    ├── utils/                    <-- Tiền xử lý và chuẩn hóa ảnh
    ├── weights/
    │   └── BFREE_dino2reg4/      <-- Trọng số mô hình B-Free DINOv2
    │       ├── config.yaml
    │       └── model_epoch_best.pth
    └── requirements.txt          <-- Danh sách thư viện phụ thuộc
```

---

## 5. HƯỚNG DẪN VẬN HÀNH

### Triển khai hệ thống lên Modal Cloud
```powershell
cd code
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"; modal deploy modal_app.py
```

### Chạy giao diện Web trên máy cục bộ
```bash
python app.py
```

### Chạy giám định 1 ảnh qua dòng lệnh
```bash
python run_pipeline.py -i <duong_dan_file_anh>
```
Kết quả giám định cùng đồ thị đối chiếu 6 panel sẽ được lưu tại thư mục `out/`.

---

## 6. HẠN CHẾ HIỆN TẠI VÀ HƯỚNG PHÁT TRIỂN (LIMITATIONS & FUTURE WORK)

* **Hạn chế về hạ tầng phần cứng thử nghiệm:**
  Hệ thống demo hiện tại đang được vận hành trên hạ tầng CPU Serverless miễn phí (2.0 vCPU trên Modal) nhằm tối ưu chi phí. Mặc dù độ chính xác toán học của mô hình được giữ nguyên vẹn 100%, thời gian suy luận (latency) mất khoảng 3 - 5 giây cho mỗi lượt phân tích đa tầng (chậm hơn so với ~0.3 giây khi chạy trên GPU chuyên dụng).
* **Giới hạn số lượng Crop phân tích cục bộ:**
  Do chạy trên tài nguyên CPU giới hạn, thuật toán Multi-Crop hiện đang quét 5 vùng cắt chiến lược (Center + 4 góc phân vùng). Đối với các đối tượng bị chỉnh sửa ở kích thước siêu vi mô (dưới 5% diện tích khung hình), mật độ 5 crop có thể chưa bao phủ triệt để.
* **Hướng phát triển:**
  1. Nâng cấp hạ tầng lên GPU đám mây (NVIDIA T4 hoặc A10G) để giảm thời gian phản hồi xuống dưới 0.5 giây.
  2. Mở rộng mật độ quét cửa sổ trượt (Sliding Window) lên 20 - 30 patch/ảnh để định vị chính xác từng pixel đối tượng bị chỉnh sửa.
  3. Tích hợp bổ sung mô hình phân đoạn chuyên biệt TruFor để kết hợp song song cùng B-Free.
