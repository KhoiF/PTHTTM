# Assignment 01 — Mã nguồn theo báo cáo `A01_CT_khoipv.447.pdf`

Hai hệ thống thông minh tương ứng hai phần của báo cáo, mỗi thư mục dữ liệu có một notebook
chạy đầy đủ từ đọc dữ liệu đến lưu weight, kèm ứng dụng web triển khai model.

```
src/Assignment 01/
├── diabetes/                         # PHẦN I - Dự đoán nguy cơ tiểu đường (phân loại)
│   ├── data/diabetes.csv
│   ├── notebook/diabetes_prediction.ipynb
│   ├── models/                       # weight sinh ra từ mục 18 của notebook
│   ├── report/figures/               # hình sinh ra khi chạy notebook
│   ├── templates/index.html
│   └── REST_API.py                   # Flask API + giao diện web (cổng 8080)
└── house_price/                      # PHẦN II - Dự đoán giá nhà (hồi quy)
    ├── data/vietnam_housing_dataset.csv
    ├── notebook/house_price_prediction.ipynb
    ├── models/
    ├── report/figures/
    ├── templates/index.html
    └── flask_app.py                  # Flask app + giao diện web (cổng 8081)
```

## Yêu cầu môi trường

Máy này đã có sẵn conda env **`ISD`** với đủ thư viện (flask 3.1.3, scikit-learn 1.7.2,
pandas 3.0.5, neo4j 6.2.0). **Phải dùng env này** — `python3` mặc định (`/usr/local/bin/python3`)
không có pandas/sklearn/flask, chạy sẽ báo `ModuleNotFoundError: No module named 'pandas'`.

```bash
conda activate ISD
```

Nếu dựng môi trường mới từ đầu:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn flask requests
pip install neo4j        # tuỳ chọn, chỉ cần cho phần tri thức y khoa của Phần I
```

## 1. Chạy notebook

Mở notebook rồi `Run All`. Notebook tự tìm dữ liệu theo đường dẫn tương đối nên chạy được từ
thư mục `notebook/` hoặc từ gốc project.

| Notebook | Nội dung | Sinh ra |
| --- | --- | --- |
| `diabetes/notebook/diabetes_prediction.ipynb` | Mục 1–22 Phần I: EDA, baseline, 4 mô hình phân loại, tuning K, thí nghiệm 3 vs 6 feature | `diabetes/models/*.sav`, `metadata.json`, 15 hình |
| `house_price/notebook/house_price_prediction.ipynb` | Mục 1–22 Phần II: EDA, baseline, 4 mô hình hồi quy, 3 thí nghiệm | `house_price/models/*.sav`, `preprocessor.sav`, `metadata.json`, 10 hình |

Chạy notebook **trước** khi khởi động web app, vì app nạp weight từ `models/`.

## 2. Chạy ứng dụng web (file `.py`)

Hai app độc lập nhau, chạy ở hai cổng khác nhau nên có thể bật đồng thời ở **hai cửa sổ
Terminal riêng**. Mỗi app phải chạy từ **đúng thư mục chứa file `.py`** (app tìm `models/` và
`templates/` theo vị trí file).

### Phần I — Dự đoán tiểu đường (cổng 8080)

```bash
conda activate ISD
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 01/diabetes"
python REST_API.py
```

### Phần II — Định giá nhà (cổng 8081)

```bash
conda activate ISD
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 01/house_price"
python flask_app.py
```

Khi chạy thành công sẽ thấy:

```
 * Serving Flask app 'REST_API'
 * Running on http://127.0.0.1:8080
```

Mở trình duyệt vào `http://127.0.0.1:8080` (Phần I) hoặc `http://127.0.0.1:8081` (Phần II).
Dừng server bằng `Ctrl + C`.

### Không muốn `conda activate`?

Gọi thẳng Python của env, không cần activate:

```bash
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 01/diabetes"
/opt/miniconda3/envs/ISD/bin/python REST_API.py
```

### Chạy nền (không chiếm Terminal)

```bash
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Assignment 01/diabetes"
nohup /opt/miniconda3/envs/ISD/bin/python REST_API.py > /tmp/diabetes.log 2>&1 &

# xem log
tail -f /tmp/diabetes.log

# dừng
pkill -f REST_API.py
pkill -f flask_app.py
```

Cả hai giao diện đều có dropdown chọn 1 trong 4 mô hình đã huấn luyện, giống ảnh chụp ở mục 20
của báo cáo, và responsive trên mobile.

### Lỗi thường gặp

| Thông báo | Nguyên nhân | Cách xử lý |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'pandas'` (hoặc `flask`, `sklearn`) | Đang chạy bằng `python3` mặc định | `conda activate ISD` rồi chạy lại |
| `FileNotFoundError: .../models/metadata.json` | Chưa chạy notebook nên chưa có weight | Chạy `Run All` notebook tương ứng trước |
| `Address already in use` / `Port 8080 is in use` | App cũ còn chạy | `pkill -f REST_API.py` (hoặc `pkill -f flask_app.py`) |
| Trang web báo `Chưa có weight...` | Thiếu file `.sav` trong `models/` | Chạy lại mục 18 của notebook |
| `knowledge_error: "Neo4j password has not been configured"` | **Không phải lỗi** — Neo4j là tuỳ chọn | Bỏ qua, dự đoán vẫn chạy bình thường |

### API Phần I

```bash
curl -X POST http://127.0.0.1:8080/diabetes/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"knn","Pregnancies":3,"Glucose":150,"BMI":32,"Age":25,"SkinThickness":24,"Insulin":110}'
# -> {"prediction": 1, "confidence": "53.85", "model_name": "K-Nearest Neighbors", ...}
```

Phần tri thức y khoa đọc từ Neo4j là **tuỳ chọn**: nếu chưa đặt biến môi trường
`NEO4J_PASSWORD` thì API vẫn trả dự đoán bình thường và ghi lý do vào `knowledge_error`.

### API Phần II

```bash
curl -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"model":"decision_tree","Floors":3,"Bedrooms":3,"Bathrooms":2,"Legal status":"Sale contract","Furniture state":"Basic"}'
# -> {"success": true, "price": 5.6, "unit": "tỷ VNĐ", ...}
```

## 3. Lưu ý về dữ liệu `diabetes.csv`

Tệp `diabetes/data/diabetes.csv` đã được impute sẵn **theo từng nhãn `Outcome`** chứ không phải
bằng mean toàn cục như báo cáo mô tả: `Insulin = 102.5` xuất hiện 236 lần và toàn bộ là
`Outcome = 0`, còn `Insulin = 169.5` xuất hiện 138 lần và toàn bộ là `Outcome = 1`. Giá trị đã
mang sẵn thông tin nhãn (*label leakage*), làm các mô hình dùng `Insulin`/`SkinThickness` đạt
điểm cao giả tạo (F1 lên tới 0.77 thay vì ~0.63).

Mục **6.1** của notebook phát hiện và gỡ các giá trị sentinel này, trả về `NaN` rồi impute lại
bằng mean toàn cục — đúng quy trình mục 4.3 của báo cáo. Sau khi gỡ, kết quả 6 feature
(F1 0.55–0.57) quay về sát kết quả 3 feature, khớp nhận xét "chênh lệch rất nhỏ" ở mục 16.6.
