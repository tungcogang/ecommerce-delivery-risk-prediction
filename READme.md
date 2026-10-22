# Phân tích Hiệu suất Vận chuyển và Dự báo Rủi ro Giao hàng Trễ trong Thương mại Điện tử
### E-Commerce Delivery Risk Prediction

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Frontend](https://img.shields.io/badge/UI-Streamlit%201.65-red.svg)](https://streamlit.io/)
[![Interactive Charts](https://img.shields.io/badge/Charts-Plotly-brightgreen.svg)](https://plotly.com/)

---

## 1. Giới thiệu Đề tài

Trong ngành Thương mại điện tử (E-commerce), hiệu suất giao hàng đóng vai trò quyết định đến trải nghiệm và tỷ lệ giữ chân khách hàng. Giao hàng trễ hạn không chỉ gây phát sinh chi phí vận hành cho doanh nghiệp (lưu kho, hoàn hàng, tổng đài hỗ trợ) mà còn làm sụt giảm trực tiếp điểm đánh giá uy tín của nền tảng.

Dự án **“Phân tích hiệu suất vận chuyển và dự báo rủi ro giao hàng trễ trong Thương mại điện tử”** (`ecommerce-delivery-risk-prediction`) được phát triển nhằm:
- Khám phá các yếu tố chính chi phối thời gian giao hàng (mức chiết khấu sản phẩm, trọng lượng kiện hàng, phương thức vận chuyển, kho hàng xuất phát).
- Xây dựng quy trình học máy (Machine Learning Pipeline) khép kín từ tiền xử lý, huấn luyện đa mô hình, tinh chỉnh siêu tham số đến tối ưu hóa ngưỡng phân loại (Decision Threshold Tuning).
- Đưa mô hình vào ứng dụng thực tế thông qua giao diện Web tương tác trực quan (Streamlit) và chương trình giao diện dòng lệnh (CLI), hỗ trợ nhà quản lý logistics phát hiện sớm các đơn hàng có nguy cơ giao trễ để chủ động can thiệp.

---

## 2. Tính năng Chính của Dự án

Hệ thống cung cấp 5 chức năng chính trên giao diện Web tương tác:

1. **📊 Bảng điều khiển Tổng quan (Exploratory Data Analysis - EDA):**
   - Thống kê tỷ lệ đơn hàng giao đúng hạn và giao trễ theo thời gian thực.
   - Trực quan hóa tỷ lệ trễ hạn phân loại theo phương thức vận chuyển (`Ship`, `Flight`, `Road`), kho hàng (`A`, `B`, `C`, `D`, `F`), và mức độ ưu tiên của hàng hóa.
   - Phân tích phân phối của các biến số tác động mạnh: mức giảm giá (`Discount_offered`) và khối lượng kiện hàng (`Weight_in_gms`).
2. **🔮 Dự đoán Đơn hàng Đơn lẻ:**
   - Biểu mẫu nhập liệu 10 thông số đặc trưng của một đơn hàng.
   - Đồng hồ đo rủi ro tương tác (Plotly Gauge Chart) hiển thị xác suất giao trễ chính xác.
   - Đưa ra cảnh báo trực quan theo ngưỡng phân loại tối ưu (`Threshold = 0.35`).
   - Tùy chọn lưu kết quả dự đoán vào lịch sử hệ thống.
3. **📁 Dự đoán Hàng loạt (Batch Prediction):**
   - Hỗ trợ tải lên file danh sách đơn hàng định dạng CSV.
   - Tự động kiểm tra tính hợp lệ của dữ liệu (kiểm tra đủ 10 cột đặc trưng, kiểm tra ô trống và dải giá trị cho phép).
   - Dự đoán đồng thời toàn bộ danh sách, trực quan hóa biểu đồ phân phối xác suất và cho phép xuất kết quả dự báo ra file CSV (`.csv`).
4. **🕘 Quản lý Lịch sử Dự đoán:**
   - Lưu trữ và phân tách lịch sử dự đoán từ giao diện Web và từ chương trình dòng lệnh.
   - Hỗ trợ tải toàn bộ lịch sử phân tích phục vụ báo cáo.
5. **🧠 Bảng thông tin và Đối soát Mô hình:**
   - Cung cấp minh bạch cấu hình mô hình hiện hành (`Random Forest Tuned`, `Threshold = 0.35`).
   - Trực quan hóa ma trận nhầm lẫn (Confusion Matrix Heatmap) trên tập dữ liệu kiểm tra độc lập.
   - Biểu đồ so sánh đa chiều giữa các thuật toán và các mức ngưỡng quyết định.
6. **💻 Giao diện Dòng lệnh (CLI Module):**
   - Script chạy độc lập `src/predict_delivery.py` cho phép nhập liệu và dự báo đơn hàng trực tiếp ngay trên terminal.

---

## 3. Ảnh chụp Giao diện Ứng dụng

> *Gợi ý: Chụp ảnh màn hình giao diện ứng dụng khi chạy thực tế, lưu vào thư mục `docs/images/` và cập nhật đường dẫn bên dưới.*

| Tổng quan phân tích dữ liệu (EDA) | Dự báo nguy cơ đơn hàng đơn lẻ |
|:---:|:---:|
| ![EDA Dashboard](docs/images/dashboard_eda.png) | ![Single Prediction](docs/images/prediction_gauge.png) |

| Dự đoán hàng loạt từ file CSV | Ma trận nhầm lẫn & Đối soát mô hình |
|:---:|:---:|
| ![Batch Prediction](docs/images/batch_prediction.png) | ![Model Comparison](docs/images/model_evaluation.png) |

*(Đường cong Precision-Recall của các mô hình đã được trích xuất sẵn tại `models/random_forest_pr_curve.png` và `models/decision_tree_pr_curve.png`).*

---

## 4. Công nghệ và Thư viện Sử dụng

- **Ngôn ngữ lập trình:** Python (hỗ trợ tốt nhất trên Python 3.11 - 3.13)
- **Xử lý dữ liệu & Tính toán:**
  - `pandas` (>= 2.2 / 3.0): Đọc, làm sạch, biến đổi và thống kê dữ liệu.
  - `numpy` (>= 2.0): Xử lý ma trận và tính toán số học.
- **Học máy (Machine Learning):**
  - `scikit-learn` (>= 1.5 / 1.9): Xây dựng Pipeline tiền xử lý (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`), phân tách tập dữ liệu phân tầng (`StratifiedKFold`, `train_test_split`), thuật toán `RandomForestClassifier`, `DecisionTreeClassifier`, `LogisticRegression` và các thang đo đánh giá.
  - `xgboost` (>= 2.1 / 3.4): Thuật toán Gradient Boosting phục vụ huấn luyện thử nghiệm nâng cao.
  - `joblib`: Đóng gói (serialize) và tải Pipeline tiền xử lý cùng trọng số mô hình đã huấn luyện.
- **Giao diện Web & Trực quan hóa:**
  - `streamlit` (>= 1.40 / 1.65): Nền tảng xây dựng Web App ứng dụng dữ liệu.
  - `plotly` (>= 6.0 / 7.1): Biểu đồ tương tác (Gauge, Heatmap, Histogram, Bar chart).
  - `matplotlib` & `seaborn`: Khám phá dữ liệu chuyên sâu và vẽ đường cong hiệu năng.

---

## 5. Dữ liệu Nghiên cứu

- **Tệp dữ liệu:** `data/Train.csv`
- **Quy mô:** Đúng **10,999 dòng** và **12 cột**, không có giá trị bị thiếu (Missing values = 0).
- **Phân loại thuộc tính:**
  - `ID`: Mã định danh đơn hàng (bị loại bỏ trong quá trình huấn luyện mô hình).
  - **10 biến đặc trưng đầu vào (Features):**
    1. `Warehouse_block`: Kho lưu trữ xuất hàng (`A`, `B`, `C`, `D`, `F`).
    2. `Mode_of_Shipment`: Hình thức vận chuyển (`Ship`, `Flight`, `Road`).
    3. `Customer_care_calls`: Số lần gọi cho trung tâm CSKH (2 đến 7 cuộc gọi).
    4. `Customer_rating`: Đánh giá của khách hàng (thang điểm từ 1 đến 5).
    5. `Cost_of_the_Product`: Giá trị của sản phẩm (từ 96 đến 310 USD).
    6. `Prior_purchases`: Số đơn hàng đã mua trước đó của khách hàng (2 đến 10 lần).
    7. `Product_importance`: Độ ưu tiên của mặt hàng (`low`, `medium`, `high`).
    8. `Gender`: Giới tính khách hàng (`F`, `M`).
    9. `Discount_offered`: Mức giảm giá khuyến mãi (từ 1% đến 65%).
    10. `Weight_in_gms`: Khối lượng sản phẩm (từ 1,001 gam đến 7,846 gam).
  - **Biến mục tiêu (Target Label):**
    - `Reached.on.Time_Y.N`: Tình trạng giao hàng.
    - Trong bài toán này, quy ước nhãn được thiết lập:
      - **Nhãn `1` (Giao trễ):** Chiếm **6,563 mẫu (~59.67%)**.
      - **Nhãn `0` (Giao đúng hạn):** Chiếm **4,436 mẫu (~40.33%)**.
- **Tệp dữ liệu mẫu đi kèm:**
  - `data/sample_orders.csv`: 51 đơn hàng chuẩn dùng để thử nghiệm chức năng dự đoán file hàng loạt.
  - `data/test_loi.csv`: Tệp dữ liệu giả lập chứa lỗi (thiếu cột, sai định dạng) để kiểm thử bộ lọc bảo vệ ứng dụng.

---

## 6. Quy trình Machine Learning

Quy trình phát triển mô hình được tổ chức bài bản qua các bước:
data/Train.csv │ ▼ [ 1. Khám phá & Phân tích tương quan (EDA) ] ── (src/eda_visualization.py, src/feature_analysis.py) │ ▼ [ 2. Tiền xử lý & Đóng gói Pipeline ] ─────── (src/preprocessing.py) │ • Categorical: SimpleImputer(most_frequent) + OneHotEncoder │ • Numerical: SimpleImputer(median) + StandardScaler │ • Stratified Split (80% Train, 20% Test) │ ▼ [ 3. Huấn luyện đa mô hình cơ sở ] ────────── (Logistic Regression, Decision Tree, Random Forest, XGBoost) │ ▼ [ 4. Tinh chỉnh Siêu tham số (Tuning) ] ────── (src/tune_models.py với 5-Fold StratifiedKFold, chấm điểm Recall) │ ▼ [ 5. Tối ưu hóa Ngưỡng quyết định (Threshold) ] ─ (src/compare_threshold.py qua Out-Of-Fold Train) │ ▼ [ 6. Đóng gói & Triển khai ] ──────────────── (models/random_forest_tuned.joblib -> app.py & predict_delivery.py)


1. **Khám phá dữ liệu:** Phân tích ma trận tương quan chỉ ra rằng `Discount_offered` và `Weight_in_gms` là hai biến có mối tương quan mạnh nhất với khả năng giao hàng trễ (nhóm sản phẩm có mức chiết khấu cao kết hợp khối lượng nhẹ có tỷ lệ trễ hạn cao vượt trội).
2. **Tiền xử lý độc lập tránh rò rỉ dữ liệu (Data Leakage):**
   - Phân chia tập dữ liệu phân tầng (`stratify=y`): **80% Train** (8,799 đơn) và **20% Test** (2,200 đơn).
   - Bộ tiền xử lý `ColumnTransformer` chỉ được khớp (`fit`) trên tập Train, sau đó áp dụng biến đổi (`transform`) cho tập Test và lưu trữ vào `models/preprocessor.joblib`.
3. **Huấn luyện và Tinh chỉnh:** Sử dụng `RandomizedSearchCV` kết hợp `StratifiedKFold (n_splits=5)` tập trung tối ưu hóa `recall_score` cho lớp đơn giao trễ.
4. **Khảo sát ngưỡng quyết định (Threshold Tuning):**
   - Khảo sát các dải ngưỡng từ `0.20` đến `0.50` trên dự đoán Out-Of-Fold (OOF) của tập Train.
   - Lựa chọn ngưỡng tối ưu đạt tiêu chuẩn cân bằng giữa việc giữ độ bao phủ phát hiện đơn trễ cao nhất và mức cảnh báo sai chấp nhận được.

---

## 7. Kết quả Đánh giá Mô hình

### Bảng đối sánh chỉ số trên tập Kiểm tra Độc lập (Test Set - 2,200 đơn hàng)

Dữ liệu thực nghiệm được ghi nhận chính xác tại `models/final_model_comparison.csv`:

| Thuật toán | Ngưỡng (Threshold) | Accuracy | Precision (Trễ) | Recall (Trễ) | F1-Score (Trễ) | True Negative | False Positive | False Negative | True Positive |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Tuned)** *(Được chọn)* | **0.35** | **61.64%** | **62.75%** | **87.89%** | **73.22%** | **202** | **685** | **159** | **1,154** |
| Random Forest (Tuned) | 0.50 (Mặc định) | 66.86% | 80.93% | 58.19% | 67.70% | 707 | 180 | 549 | 764 |
| Decision Tree (Tuned) | 0.30 | 61.77% | 64.48% | 80.05% | 71.42% | 308 | 579 | 262 | 1,051 |
| Decision Tree (Tuned) | 0.50 (Mặc định) | 63.86% | 70.11% | 68.77% | 69.43% | 502 | 385 | 410 | 903 |

### Phân tích ý nghĩa Nghiệp vụ của việc Chọn Ngưỡng 0.35
- Trong bài toán Logistics TMĐT, **chi phí của việc bỏ sót một đơn hàng giao trễ (False Negative - FN)** gây hậu quả nghiêm trọng hơn việc cảnh báo nhầm một đơn hàng đúng hạn (False Positive - FP).
- Nếu áp dụng ngưỡng mặc định `0.50`, mô hình Random Forest bỏ sót tới **549 đơn hàng bị trễ** (Recall chỉ đạt 58.19%).
- Bằng việc hạ ngưỡng phân loại xuống **`0.35`**, mô hình phát hiện thành công **1,154 trên tổng số 1,313 đơn trễ thực tế**, nâng **Recall lên 87.89%** và giảm lượng đơn trễ bị bỏ sót xuống chỉ còn **159 đơn** (giảm hơn 71% lượng đơn trễ bị bỏ sót). Do đó, mô hình `Random Forest (Threshold = 0.35)` được chọn làm mô hình suy luận chính cho hệ thống.

---

## 8. Cấu trúc Thư mục Dự án

ecommerce-delivery-risk-prediction/ │ ├── .streamlit/ │ └── config.toml # Cấu hình giao diện Dark Theme cho Streamlit │ ├── data/ │ ├── Train.csv # Tập dữ liệu vận chuyển gốc (10,999 dòng, 12 cột) │ ├── sample_orders.csv # Dữ liệu 51 đơn hàng mẫu dùng để test dự đoán hàng loạt │ └── test_loi.csv # Dữ liệu thử nghiệm chứa lỗi định dạng │ ├── models/ │ ├── random_forest_tuned.joblib # Mô hình Random Forest tốt nhất (được nạp vào App) │ ├── decision_tree_tuned.joblib # Mô hình Decision Tree đã qua tinh chỉnh │ ├── logistic_regression_pipeline.joblib# Pipeline mô hình cơ sở Logistic Regression │ ├── preprocessor.joblib # Bộ tiền xử lý ColumnTransformer đã học từ tập Train │ ├── final_model_comparison.csv # Bảng số liệu đối soát hiệu năng và ngưỡng thực tế │ ├── decision_thresholds.json # File lưu trữ ngưỡng quyết định đã khảo sát │ ├── prediction_history_app.csv # Nhật ký lưu trữ lịch sử dự đoán từ giao diện Web │ ├── prediction_results.csv # Kết quả xuất ra từ script dự đoán dòng lệnh │ └── random_forest_pr_curve.png # Biểu đồ đường cong Precision-Recall │ ├── src/ │ ├── preprocessing.py # Script tiền xử lý và tách tập Train/Test độc lập │ ├── feature_analysis.py # Phân tích tương quan đặc trưng và ma trận tương tác │ ├── eda_visualization.py # Tạo biểu đồ phân phối dữ liệu │ ├── train_model.py # Huấn luyện mô hình cơ sở Logistic Regression │ ├── train_random_forest.py # Huấn luyện Random Forest chuẩn │ ├── train_decision_tree.py # Huấn luyện Decision Tree chuẩn │ ├── tune_models.py # Tinh chỉnh siêu tham số bằng RandomizedSearchCV │ ├── compare_threshold.py # Tối ưu hóa ngưỡng quyết định dựa trên OOF │ └── predict_delivery.py # Module suy luận và chương trình dự đoán qua CLI │ ├── app.py # Giao diện ứng dụng chính (Streamlit Web Dashboard) ├── requirements_backup.txt # Danh sách dependency chi tiết của môi trường └── README.md # Tài liệu giới thiệu và hướng dẫn sử dụng dự án


---

## 9. Yêu cầu Hệ thống

- **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux.
- **Phiên bản Python:** Khuyến nghị **Python 3.11, 3.12 hoặc 3.13**.
- **Phần mềm quản lý gói:** `pip` và công cụ tạo môi trường ảo `venv`.

---

## 10. Hướng dẫn Cài đặt và Chạy Ứng dụng

Thực hiện lần lượt các bước sau trong Terminal (Command Prompt / PowerShell trên Windows hoặc Terminal trên macOS/Linux):

### Bước 1: Di chuyển vào thư mục dự án
```bash
cd ecommerce-delivery-risk-prediction

Tạo và kích hoạt môi trường ảo Python
Trên Windows:
powershell
python -m venv .venv
.venv\Scripts\activate


Trên macOS / Linux:
bash
python3 -m venv .venv
source .venv/bin/activate


Bước 3: Cài đặt các thư viện phụ thuộc

Cài đặt các gói cốt lõi để chạy ứng dụng:

bash
pip install streamlit scikit-learn xgboost pandas numpy plotly joblib matplotlib seaborn
Bước 4: Khởi chạy Ứng dụng Web (Streamlit Dashboard)
bash
streamlit run app.py

Sau khi lệnh chạy, trình duyệt web sẽ tự động mở tại địa chỉ:

http://localhost:8501
(Tùy chọn) Chạy chương trình dự đoán qua Terminal (CLI)

Nếu muốn thực hiện dự đoán tương tác trực tiếp trên dòng lệnh mà không cần mở trình duyệt:

bash
python src/predict_delivery.py


Hướng dẫn Sử dụng Chi tiết
Khám phá dữ liệu: Chọn mục "📊 Tổng quan" tại thanh menu bên trái để quan sát tỷ lệ trễ hạn và các biểu đồ phân tích yếu tố ảnh hưởng.
Dự báo cho 1 đơn hàng:
Chọn mục "🔮 Dự đoán đơn hàng".
Điều chỉnh thông tin: Kho hàng, Phương thức vận chuyển, Cân nặng kiện hàng, Giá sản phẩm, Mức chiết khấu...
Nhấn 🔮 Dự đoán: Hệ thống sẽ trả về đồng hồ đo xác suất rủi ro. Nếu xác suất 
≥
35
%
≥35%, hệ thống lập tức hiển thị cảnh báo đỏ và cho phép nhấn 💾 Lưu vào lịch sử.
Dự báo theo file danh sách (Batch):
Chọn mục "📁 Dự đoán hàng loạt".
Bấm duyệt file và tải lên tệp data/sample_orders.csv.
Nhấn 🔮 Dự đoán hàng loạt, hệ thống tự động phân loại, vẽ biểu đồ rủi ro và cung cấp nút ⬇️ Tải kết quả (CSV) để tải báo cáo về máy.


12. Đánh giá và Hạn chế của Đề tài
Độ chính xác tổng thể (Accuracy ~61.6%): Do dữ liệu logistics thực tế chịu ảnh hưởng bởi nhiều biến số ngoại cảnh không có trong tập dữ liệu (như điều kiện thời tiết, mật độ giao thông, ngày lễ mua sắm cao điểm), mô hình hiện tại đạt mức phân loại vừa phải, nhưng có giá trị cảnh báo sớm cao nhờ Recall đạt xấp xỉ 88%.
Chênh lệch cảnh báo sai (False Positives): Việc ưu tiên bắt tối đa đơn trễ (hạ ngưỡng xuống 0.35) dẫn đến tỷ lệ cảnh báo nhầm (FP = 685 đơn) trên tập test, đòi hỏi quy trình hậu kiểm từ bộ phận điều hành logistics.
Dữ liệu dạng tĩnh: Mô hình hiện được huấn luyện trên tệp dữ liệu tĩnh, chưa được kết nối theo luồng thời gian thực với hệ thống quản lý đơn hàng (OMS/WMS).


13. Hướng Phát triển Tiếp theo
 Thu thập thêm các đặc trưng không gian - thời gian (tọa độ khoảng cách kho - điểm giao, thời gian tạo đơn trong ngày, mùa cao điểm mua sắm).
 Xây dựng giải pháp giải thích mô hình cục bộ bằng SHAP (SHapley Additive exPlanations) để giải thích rõ lý do từng đơn hàng cụ thể bị đánh giá là rủi ro.
 Đóng gói toàn bộ ứng dụng bằng Docker Container để đơn giản hóa việc triển khai lên môi trường máy chủ đám mây (Cloud Server / AWS / GCP).
 Xây dựng RESTful API (sử dụng FastAPI) để cho phép các hệ thống bán hàng khác tích hợp chức năng dự báo rủi ro tự động.