# Assignment 02 — Mã nguồn theo báo cáo `A02_CT_khoipv.447.pdf`

Ba hệ thống thông minh tương ứng ba ứng dụng của báo cáo. Mỗi thư mục dữ liệu có **một notebook**
chạy đầy đủ từ đọc dữ liệu → làm sạch → EDA → so sánh mô hình → lưu weight, kèm **một app Flask**
(REST API + giao diện web) nạp lại đúng weight đó.

```
src/Assignment 02/
├── diabetes/                          # Ứng dụng 1 - Dự đoán tiểu đường (phân loại nhị phân)
│   ├── data/diabetes.csv
│   ├── notebook/diabetes_prediction.ipynb
│   ├── models/                        # weight sinh ra từ mục 10 của notebook
│   ├── report/figures/                # hình sinh ra khi chạy notebook
│   ├── templates/index.html
│   └── REST_API.py                    # cổng 8080
├── house_price/                       # Ứng dụng 2 - Dự đoán giá nhà (hồi quy)
│   ├── data/vietnam_housing_dataset.csv
│   ├── notebook/house_price_prediction.ipynb
│   ├── models/                        # + preprocessor.sav
│   ├── report/figures/
│   ├── templates/index.html
│   └── REST_API.py                    # cổng 8081
└── customer_comments/                 # Ứng dụng 3 - Hành vi khách hàng (phân loại văn bản)
    ├── data/Womens Clothing E-Commerce Reviews.csv
    ├── notebook/customer_behavior_prediction.ipynb
    ├── models/                        # + tfidf_vectorizer.sav
    ├── report/figures/
    ├── templates/index.html
    └── REST_API.py                    # cổng 8082
```

## Tóm tắt ba hệ thống

| | Diabetes | House Price | Customer Behavior |
| --- | --- | --- | --- |
| Loại bài toán | Phân loại nhị phân | Hồi quy | Phân loại nhị phân |
| Biểu diễn | 6 đặc trưng số | 3 biến số + 2 biến phân loại (one-hot) | TF-IDF unigram/bigram, 5000 chiều |
| Model mặc định | **K-Nearest Neighbors** (K = 39) | **Decision Tree** | **Logistic Regression (Balanced)** |
| Model tốt nhất | KNN | Random Forest (RMSE 1.946, R² 0.194) | Naive Bayes (F1 92.9%) |
| Metric chính | Accuracy, F1 | MAE, RMSE, R² | F1, Recall, ROC-AUC |
| Kết quả | Acc 74.0%, P 65.2%, R 55.6%, F1 60.0% | RMSE 1.946, R² 0.194, MAE (SVR) 1.572 | Acc 87.9%, P 96.3%, R 88.6%, F1 92.3%, AUC 0.941 |
| Endpoint | `POST /diabetes/v1/predict` | `POST /house/v1/predict` | `POST /recommendation/v1/predict` |
| Cổng | 8080 | 8081 | 8082 |

## Yêu cầu môi trường

Máy này đã có sẵn conda env **`ISD`** với đủ thư viện (flask 3.1.3, scikit-learn 1.7.2,
pandas 3.0.5, neo4j 6.2.0). **Phải dùng env này** — `python3` mặc định không có pandas/sklearn/flask,
chạy sẽ báo `ModuleNotFoundError: No module named 'pandas'`.

```bash
conda activate ISD
```

Nếu dựng môi trường mới từ đầu:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn flask requests jupyter
pip install neo4j        # tuỳ chọn, chỉ cần cho phần tri thức y khoa của Ứng dụng 1
```

## 1. Chạy notebook

Mở notebook rồi `Run All`. Notebook tự tìm dữ liệu theo đường dẫn tương đối nên chạy được từ
thư mục `notebook/` hoặc từ gốc project.

| Notebook | Nội dung | Sinh ra |
| --- | --- | --- |
| `diabetes/notebook/diabetes_prediction.ipynb` | Mục 2 báo cáo: làm sạch + gỡ label leakage, EDA, dò K bằng CV, 4 mô hình phân loại | `diabetes/models/*.sav`, `metadata.json`, 8 hình |
| `house_price/notebook/house_price_prediction.ipynb` | Mục 3 báo cáo: làm sạch + cắt outlier, EDA, 4 mô hình hồi quy, khảo sát siêu tham số | `house_price/models/*.sav`, `preprocessor.sav`, `metadata.json`, 7 hình |
| `customer_comments/notebook/customer_behavior_prediction.ipynb` | Mục 4 báo cáo: làm sạch văn bản, TF-IDF, 4 mô hình phân loại, diễn giải trọng số từ | `customer_comments/models/*.sav`, `tfidf_vectorizer.sav`, `metadata.json`, 7 hình |

Chạy notebook **trước** khi khởi động web app, vì app nạp weight từ `models/`.

Chạy bằng dòng lệnh cũng được:

```bash
cd "src/Assignment 02/diabetes/notebook"
/opt/miniconda3/envs/ISD/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=isd diabetes_prediction.ipynb
```

## 2. Chạy ứng dụng web

Ba app độc lập nhau, chạy ở ba cổng khác nhau nên bật đồng thời được ở **ba cửa sổ Terminal
riêng**. Mỗi app phải chạy từ **đúng thư mục chứa file `REST_API.py`** (app tìm `models/` và
`templates/` theo vị trí file).

```bash
conda activate ISD

# Ứng dụng 1 — Dự đoán tiểu đường (cổng 8080)
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 02/diabetes"
python REST_API.py

# Ứng dụng 2 — Dự đoán giá nhà (cổng 8081)
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 02/house_price"
python REST_API.py

# Ứng dụng 3 — Hành vi khách hàng (cổng 8082)
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 02/customer_comments"
python REST_API.py
```

Mở trình duyệt vào `http://127.0.0.1:8080`, `:8081` hoặc `:8082`. Dừng server bằng `Ctrl + C`.

Cả ba giao diện đều có dropdown chọn 1 trong 4 mô hình đã huấn luyện, hiển thị metric của từng
mô hình và responsive trên mobile.

### Không muốn `conda activate`?

```bash
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 02/diabetes"
/opt/miniconda3/envs/ISD/bin/python REST_API.py
```

### Chạy nền (không chiếm Terminal)

```bash
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 02/diabetes"
nohup /opt/miniconda3/envs/ISD/bin/python REST_API.py > /tmp/diabetes.log 2>&1 &

tail -f /tmp/diabetes.log                                  # xem log

# dừng app theo cổng (pkill -f REST_API.py sẽ dừng cả app của Assignment 01)
for port in 8080 8081 8082; do lsof -ti tcp:$port | xargs kill; done
```

## 3. Gọi thử API

```bash
# Ứng dụng 1
curl -X POST http://127.0.0.1:8080/diabetes/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"knn","Pregnancies":3,"Glucose":150,"BMI":32,"Age":25,"SkinThickness":24,"Insulin":110}'
# -> {"prediction": 0, "confidence": 61.54, "model_name": "K-Nearest Neighbors", ...}

# Ứng dụng 2
curl -X POST http://127.0.0.1:8081/house/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"decision_tree","Floors":3,"Bedrooms":3,"Bathrooms":2,"Legal status":"Sale contract","Furniture state":"Basic"}'
# -> {"success": true, "price": 5.6, "unit": "tỷ VNĐ", ...}

# Ứng dụng 3
curl -X POST http://127.0.0.1:8082/recommendation/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"Title":"Love this dress","Review Text":"The fabric is soft and the fit is perfect."}'
# -> {"success": true, "prediction": 1, "label": "Khuyến nghị", "probability": 97.08, ...}
```

Mỗi app còn có endpoint `GET /<prefix>/v1/models` trả về danh sách model, thứ tự feature và
metric đã lưu trong `metadata.json`.

### Neo4j cho Ứng dụng 1 là tuỳ chọn

Khi dự đoán là dương tính, `REST_API.py` truy vấn đồ thị tri thức Neo4j để trả tư vấn chi tiết
(chế độ ăn, lối sống vận động, thông tin thuốc, biến chứng). Nếu chưa đặt biến môi trường
`NEO4J_PASSWORD`, API **vẫn trả dự đoán bình thường** và ghi lý do vào trường `knowledge_error`.

```bash
export NEO4J_PASSWORD='mat-khau-cua-ban'   # tuỳ chọn
export NEO4J_URI='bolt://localhost:7687'   # mặc định
```

## 4. Vì sao mỗi ứng dụng lưu weight khác nhau

| Ứng dụng | File lưu | Lý do |
| --- | --- | --- |
| Diabetes | `knn.sav`, … (mỗi file là một `Pipeline` trọn gói) | Chỉ có biến số, `StandardScaler` gói luôn vào pipeline nên API chỉ cần `predict` |
| House Price | `preprocessor.sav` + `decision_tree.sav`, … | Có biến phân loại, `ColumnTransformer` dùng chung cho cả 4 model nên tách riêng cho gọn |
| Customer Behavior | `tfidf_vectorizer.sav` + `logistic_regression.sav`, … | Thiếu vectorizer thì không có cách nào ánh xạ review mới về đúng 5000 chiều model mong đợi |

Điểm chung — và cũng là kết luận mục 5.1 của báo cáo: **preprocessor/vectorizer được fit rồi lưu
lại**, API không bao giờ fit lại lúc chạy. Đây là cách tránh rò rỉ dữ liệu (*data leakage*) khi
inference.

## 5. Lưu ý về dữ liệu `diabetes.csv`

Tệp `diabetes/data/diabetes.csv` đã được impute sẵn **theo từng nhãn `Outcome`** chứ không phải
bằng mean toàn cục: `Insulin = 102.5` xuất hiện 236 lần và toàn bộ là `Outcome = 0`, còn
`Insulin = 169.5` xuất hiện 138 lần và toàn bộ là `Outcome = 1`. Giá trị đã mang sẵn thông tin
nhãn (*label leakage*) nên nếu giữ nguyên, mô hình sẽ đạt điểm cao giả tạo (F1 lên tới 0.74).

Mục **3.2** của notebook phát hiện và gỡ các giá trị sentinel này (lặp ≥ 50 lần, thuần nhãn
≥ 85%, và **không phải số nguyên** — dấu vết của phép lấy trung bình chứ không phải số đo thật),
trả về `NaN` rồi impute lại bằng mean toàn cục. Sau khi gỡ, kết quả quay về đúng mức nêu trong
báo cáo: KNN đạt Accuracy 74.0%, F1 60.0%.

## 6. Lỗi thường gặp

| Thông báo | Nguyên nhân | Cách xử lý |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'pandas'` (hoặc `flask`, `sklearn`) | Đang chạy bằng `python3` mặc định | `conda activate ISD` rồi chạy lại |
| `FileNotFoundError: .../models/metadata.json` | Chưa chạy notebook nên chưa có weight | `Run All` notebook tương ứng trước |
| `Address already in use` / `Port 8080 is in use` | App cũ còn chạy | `lsof -ti tcp:8080 \| xargs kill` |
| Trang web báo `Chưa có weight...` | Thiếu file `.sav` trong `models/` | Chạy lại mục 10 của notebook |
| `knowledge_error: "Neo4j password has not been configured"` | **Không phải lỗi** — Neo4j là tuỳ chọn | Bỏ qua, dự đoán vẫn chạy bình thường |
| Ứng dụng 3 báo `Review không còn từ nào sau khi làm sạch` | Nhập tiếng Việt hoặc chỉ có số/ký hiệu | Nhập review tiếng Anh — mô hình học từ dữ liệu tiếng Anh |
