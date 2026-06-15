# Brain Connectivity Tensor Tracking & Community Dynamics 🧠

Dự án này triển khai pipeline xử lý tín hiệu và học máy nâng cao nhằm phân tích dữ liệu kết nối mạng não bộ (tần số lấy mẫu 1024Hz) từ dữ liệu EEG thô, phục vụ cho việc theo dõi biến đổi không gian-thời gian (subspace tracking) và cấu trúc cộng đồng (community dynamics) trong các pha Pre-ERN, ERN và Post-ERN.

---

## 🚀 Tính Năng Chính (Core Features)

1. **Preprocessing & Trial Balancing** (Xử lý thô & Cân bằng mẫu):
   - Lọc tín hiệu dải tần rộng (1-30Hz) cho phân tích ERP và lọc dải tần Theta (4-8Hz) cho phân tích kết nối PLV.
   - Cân bằng tỷ lệ số epochs (trials) giữa điều kiện Correct (CRN) và Incorrect (ERN) để đồng bộ hóa mức nhiễu nền (SNR).
   - Trực quan hóa ERP sóng trung bình (Grand Average ERP) tại điện cực trung tâm **FCz**.

2. **Phase Locking Value (PLV) Tensors**:
   - Sử dụng Hilbert Transform tính toán Phase Locking Value (PLV) giữa 30 kênh điện cực trên toàn bộ subjects.
   - Dựng tensor 4D kết nối: `Subjects x Nodes x Nodes x Timepoints`.

3. **Low-Rank Subspace Decomposition** (Phân tách cấp thấp):
   - Sử dụng thuật toán **Higher-Order Recursive Low-Rank + Sparse Decomposition (HO-RLSL)** hoặc **High-Order SVD (HoSVD)** để tách mạng cấu trúc nền tảng (Low-Rank component) khỏi các nhiễu quá độ (Sparse component).

4. **Subspace Change Point Detection** (Phát hiện điểm thay đổi):
   - Theo dõi sự dịch chuyển không gian con kết nối mạng qua 5 thuật toán: `HO-RLSL` (Subspace tracking), `HOSVD`, `PELT`, `DMD` (Dynamic Mode Decomposition), và `CP-Tracking`.
   - Chia ranh giới thời gian tối ưu cho Pre-ERN, ERN, và Post-ERN.

5. **Fiedler Consensus Community Splits (FCCA)**:
   - Dựng ma trận đồng xuất hiện tích hợp cộng đồng của các subjects (consensus co-occurrence matrix).
   - Phân hoạch nhị phân phổ đệ quy dựa trên vector Fiedler, chấm điểm chất lượng phân cụm ($U$), độ đồng nhất ($H$) và mô-đun hóa ($Q$) để tối ưu hóa số cụm cộng đồng ($k$).

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```bash
├── data/                                    # Dữ liệu EEG thô định dạng BIDS
│   └── ERN_Raw_Data_BIDS-Compatible/
├── outputs/                                 # Thư mục lưu kết quả đầu ra
│   ├── processed_1024/                      # Data epochs đã tiền xử lý & cân bằng
│   ├── tensor_4d_1024/                      # Tensor kết nối 4D PLV (.npy)
│   ├── fcca_results/                        # Kết quả dò Change Point & phân tách FCCA (.mat, .npy, .npz)
│   └── ho_rlsl_results.npz                  # Kết quả subspace low-rank của HO-RLSL
├── src/                                     # Mã nguồn chính của pipeline
│   ├── preprocessing/                       # Tiền xử lý, lọc dải tần & tính PLV
│   ├── low_rank_extraction/                 # Thuật toán HoSVD & HO-RLSL
│   ├── change_point_detection/              # Thuật toán dò tìm Change Point
│   ├── fcca/                                # Phân tách cộng đồng Fiedler Consensus
│   ├── main.py                              # Script chạy chính toàn bộ pipeline
│   ├── precompute_fcca_all.py               # Precompute toàn bộ FCCA cho các thuật toán
│   └── utils.py                             # Hàm bổ trợ chung
├── notebooks/                               # Giao diện trực quan hóa & Experiment
│   ├── streamlit_app.py                     # Giao diện dashboard tương tác Streamlit
│   └── notebook.ipynb                       # Notebook nghiên cứu vẽ ERP & Change Points
└── README.md
```

---

## 🛠️ Hướng Dẫn Sử Dụng (How to Run)

### 1. Cài đặt thư viện
Yêu cầu Python 3.10+ và các thư viện xử lý neuro-signal:
```bash
pip install mne matplotlib numpy scipy torch streamlit networkx pandas
```

### 2. Chạy Pipeline Tính Toán Chính
Để thực hiện tiền xử lý tín hiệu, cân bằng epochs, tính PLV, chạy tách low-rank HO-RLSL/HoSVD và xác định change points:
```bash
python src/main.py --low-rank-method ho-rlsl
```
* **Tham số hỗ trợ**:
  * `--low-rank-method`: Thuật toán phân tách (`ho-rlsl`, `hosvd`, hoặc `raw`). Mặc định: `ho-rlsl`.
  * `--skip-preprocessing`: Bỏ qua tiền xử lý thô và tải trực tiếp connectivity tensors đã lưu.
  * `--num-subjects`: Số lượng subjects giới hạn chạy test.
* **Tự động hóa**: Khi chạy với `--low-rank-method ho-rlsl`, pipeline sẽ tự động kích hoạt bước vẽ đồ thị mạng lưới FCCA của bài báo (`fcca.fcca_single_run`) ở cuối tiến trình.

### 3. Tiền tính toán (Precompute) FCCA
Để quét tất cả các khoảng thời gian change points của cả 5 phương pháp và lưu trữ sẵn kết quả phân cụm đồng nhất (FCCA) giúp tăng tốc độ phản hồi trên Dashboard:
```bash
python src/precompute_fcca_all.py
```
* **Chi tiết**: Script xử lý tuần tự qua từng điều kiện (`incorrect`, `correct`) và các phương pháp (`HOSVD`, `HO-RLSL`, `PELT`, `DMD`, `CP-Tracking`). Kết quả được xuất ra file nén tổng hợp `outputs/fcca_results/precomputed_fcca_all.npz`.

### 4. Chạy File Nghiên Cứu Jupyter Notebook
Mở và chạy file [notebooks/notebook.ipynb](file:///h:/CN12_AI/Neuroscience/Tensor-Tracking/notebooks/notebook.ipynb) để vẽ biểu đồ Grand Average ERP trên kênh **FCz** đè lên các vệt mốc thời gian Change Points.

### 5. Khởi chạy Dashboard Streamlit
Để xem walkthrough tương tác 5 bước từ tiền xử lý, PLV, tách low-rank đến phân tách consensus communities khớp 100% với paper:
```bash
streamlit run notebooks/streamlit_app.py
```

Giao diện sẽ chạy tại cổng mặc định: `http://localhost:8501`.
- **Lưu ý:** Dashboard sẽ tự động tải trước dữ liệu FCCA đã được tính toán sẵn từ file `outputs/fcca_results/precomputed_fcca_all.npz` (nếu có) để hiển thị tức thì, hoặc tự động tính toán mô phỏng live trực tiếp nếu không tìm thấy file precomputed.
