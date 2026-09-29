# CNN nhận dạng chữ số MNIST

Dự án huấn luyện một mô hình CNN bằng PyTorch trên MNIST và cung cấp giao diện
Gradio để vẽ chữ số, dự đoán ngay trên máy cục bộ.

## Thành phần

- `mnist_cnn_training.ipynb`: tải dữ liệu, huấn luyện 10 epochs, đánh giá và lưu weights.
- `model.py`: kiến trúc CNN được dùng chung khi huấn luyện và suy luận.
- `models/mnist_cnn.pth`: checkpoint đã huấn luyện.
- `app.py`: giao diện bảng vẽ và dự đoán trên localhost.

## Cài đặt

Dự án sử dụng môi trường Conda `ISD` với Python 3.12:

```bash
conda activate ISD
python -m pip install -r requirements.txt
```

## Huấn luyện lại mô hình

Mở notebook:

```bash
jupyter lab mnist_cnn_training.ipynb
```

Chạy lần lượt tất cả cell. MNIST được tải vào thư mục `data/` và checkpoint mới
được ghi vào `models/mnist_cnn.pth`. Notebook tự chọn CUDA, Apple MPS hoặc CPU.

## Chạy giao diện

```bash
python app.py
```

Sau đó mở [http://127.0.0.1:7860](http://127.0.0.1:7860), vẽ một chữ số bằng
nét trắng và bấm **Dự đoán**. Nút **Xóa** tạo lại bảng vẽ trống.

Nếu ứng dụng báo thiếu weights, hãy chạy toàn bộ notebook trước rồi khởi động lại
`app.py`.

## Định dạng checkpoint

Checkpoint là dictionary PyTorch gồm:

- `model_state_dict`: trọng số của `MNISTCNN`.
- `epochs`: số epoch đã huấn luyện.
- `test_accuracy`: độ chính xác trên tập test.
- `normalization`: mean và standard deviation dùng cho MNIST.
- `class_names`: danh sách nhãn từ `0` đến `9`.
