# Hướng dẫn chạy deploy Assignment 06

Weights PyTorch, Keras và metadata đã được tạo sẵn trong thư mục `models` của từng dataset. Không cần
chạy lại notebook trước khi mở giao diện.

## 1. Mở Terminal tại thư mục project

```bash
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM"
conda activate ISD
```

Nếu không muốn activate Conda, thay `python` trong các lệnh bên dưới bằng
`/opt/miniconda3/envs/ISD/bin/python`.

## 2. Chạy website Amazon Stock Price trên localhost

```bash
python "src/Assignment 06/Amazon Stock Price/deploy.py" --web
```

Giữ Terminal đang chạy và mở trình duyệt tại [http://127.0.0.1:8001](http://127.0.0.1:8001).

1. Chọn `PyTorch` hoặc `Keras`.
2. Nhấn **Nạp 60 giá mới nhất** để đọc tự động từ `AMZN.csv`, hoặc nhập đúng 60 giá `Close` từ cũ
   đến mới, cách nhau bằng dấu phẩy.
3. Nhấn **Dự báo**. Kết quả là giá `Close` của phiên kế tiếp.
4. Nhấn `Ctrl+C` trong Terminal để dừng server.

## 3. Chạy website Avocado Price trên localhost

```bash
python "src/Assignment 06/Avocado Price/deploy.py" --web
```

Giữ Terminal đang chạy và mở trình duyệt tại [http://127.0.0.1:8002](http://127.0.0.1:8002).

1. Chọn framework, `region` và `type`.
2. Nhấn **Nạp 12 tuần mới nhất** để đọc dữ liệu của nhóm đã chọn, hoặc nhập đúng 12 giá
   `AveragePrice` từ tuần cũ đến tuần mới.
3. Nhấn **Dự báo**. Kết quả là `AveragePrice` của tuần kế tiếp cho đúng nhóm đó.
4. Nhấn `Ctrl+C` trong Terminal để dừng server.

Hai website dùng cổng khác nhau nên có thể chạy đồng thời trong hai cửa sổ Terminal. Có thể đổi cổng:

```bash
python "src/Assignment 06/Amazon Stock Price/deploy.py" --web --port 9001
```

## 4. Chạy bằng command line

Amazon:

```bash
python "src/Assignment 06/Amazon Stock Price/deploy.py" \
  --framework pytorch \
  --prices "p1,p2,...,p60"
```

Avocado:

```bash
python "src/Assignment 06/Avocado Price/deploy.py" \
  --framework keras \
  --region TotalUS \
  --type conventional \
  --prices "p1,p2,...,p12"
```

Kết quả CLI được in dưới dạng JSON. Xem toàn bộ tham số bằng `python deploy.py --help`.

## 5. Lưu ý

- Web server mặc định chỉ bind `127.0.0.1`, vì vậy chỉ máy hiện tại truy cập được.
- Chuỗi giá phải theo thứ tự từ cũ đến mới, không chứa giá âm và phải có đúng lookback.
- PyTorch tự ưu tiên MPS trên Apple Silicon và fallback CPU; Keras dùng thiết bị TensorFlow phát hiện.
