# Kịch bản thuyết trình chi tiết notebook CIFAR-100 - Assignment 05

Thời lượng gợi ý: 12-15 phút  
Notebook: `src/Assignment 05/CIFAR-100/notebook/cifar100_cnn_benchmark.ipynb`  
Chủ đề: So sánh Basic CNN, Compact VGG, Compact ResNet và Compact DenseNet trên CIFAR-100.

## 0. Cách dùng file này

File này được viết theo kiểu kịch bản nói. Khi thuyết trình, có thể đọc gần như trực tiếp, nhưng nên nói tự nhiên hơn bằng cách:

- Mở notebook CIFAR-100 song song.
- Đi theo từng section trong notebook.
- Khi đến phần kết quả, dừng lâu hơn ở bảng benchmark, learning curves, confusion matrix và error analysis.
- Nếu thiếu thời gian, ưu tiên trình bày: mục tiêu, dataset, bốn kiến trúc, pipeline huấn luyện, bảng kết quả và kết luận.

## 1. Mở đầu

Kính chào thầy và các bạn.  
Trong phần này em xin trình bày notebook CIFAR-100 của Assignment 05. Mục tiêu chính của notebook là so sánh bốn kiến trúc mạng tích chập, gồm Basic CNN, Compact VGG, Compact ResNet và Compact DenseNet, trên cùng một bộ dữ liệu và cùng một pipeline huấn luyện.

Điểm quan trọng của notebook này không chỉ là tìm mô hình có accuracy cao nhất, mà là hiểu vì sao các kiến trúc phát triển từ CNN cơ bản có thể hoạt động tốt hơn khi dữ liệu khó hơn.

CIFAR-100 là một dataset phù hợp cho mục tiêu này vì nó có 100 lớp fine-grained. Mỗi ảnh chỉ có kích thước 32x32 pixel, nhưng nội dung ảnh rất đa dạng. Do ảnh nhỏ và số lớp nhiều, mô hình cần học đặc trưng tốt hơn so với các dataset đơn giản như Fashion-MNIST.

Điểm cần nhấn mạnh:

- CIFAR-100 có 100 lớp, nhiều hơn Fashion-MNIST 10 lần.
- Mỗi lớp chỉ có 500 ảnh train gốc.
- Ảnh 32x32 khá nhỏ, nhiều chi tiết bị mất.
- Nhiều lớp có hình dạng hoặc texture gần giống nhau.

## 2. Mục tiêu của notebook

Notebook này có bốn mục tiêu chính.

Thứ nhất, notebook trình bày lại CNN dưới dạng một chuỗi hàm hợp. Cách nhìn này giúp ta hiểu CNN không phải là một khối đen, mà là sự kết hợp tuần tự của nhiều phép biến đổi như convolution, BatchNorm, ReLU, pooling và classifier.

Thứ hai, notebook hiện thực bốn kiến trúc CNN compact từ đầu bằng TensorFlow/Keras. Các mô hình đều không dùng pretrained weights, nghĩa là toàn bộ trọng số được học trực tiếp từ CIFAR-100.

Thứ ba, notebook xây dựng một pipeline dữ liệu đầy đủ: tải dữ liệu, kiểm tra checksum, đọc file pickle, reshape ảnh, chia train-validation-test, chuẩn hóa và augmentation.

Thứ tư, notebook huấn luyện và đánh giá bốn mô hình trên cùng test set, sau đó so sánh bằng Accuracy, Macro Precision, Macro Recall, Macro F1, test loss, số tham số, best epoch, thời gian train và thời gian inference.

Ở đây em dùng Macro F1 vì CIFAR-100 có nhiều lớp. Macro F1 tính F1 riêng cho từng lớp rồi lấy trung bình đều. Như vậy mỗi lớp có tầm quan trọng ngang nhau, không bị một vài lớp dễ hoặc khó chi phối toàn bộ kết luận.

## 3. Giới thiệu dataset CIFAR-100

Ở phần dữ liệu, notebook tải CIFAR-100 từ nguồn chính thức của University of Toronto. File tải về là `cifar-100-python.tar.gz`. Sau khi tải xong, notebook kiểm tra MD5 để đảm bảo file không bị lỗi hoặc tải thiếu.

Sau khi giải nén, dataset có ba file chính:

- `train`: chứa 50.000 ảnh train.
- `test`: chứa 10.000 ảnh test.
- `meta`: chứa tên các lớp.

Mỗi ảnh ban đầu được lưu dưới dạng vector 3.072 phần tử. Con số này đến từ 3 kênh màu nhân với 32 nhân 32 pixel:

`3 x 32 x 32 = 3072`

Notebook reshape vector này về dạng ảnh chuẩn:

`(N, 32, 32, 3)`

Trong đó `N` là số lượng ảnh, 32 là chiều cao, 32 là chiều rộng và 3 là ba kênh màu RGB.

Khi chỉ vào cell split dữ liệu, có thể nói:

Sau khi đọc dữ liệu, notebook chia tập train gốc thành 45.000 ảnh train và 5.000 ảnh validation bằng stratified split. Tập test 10.000 ảnh được giữ nguyên.

Việc chia validation bằng stratified split giúp mỗi lớp vẫn giữ tỷ lệ cân bằng. Điều này quan trọng vì CIFAR-100 có tới 100 lớp. Nếu chia ngẫu nhiên không stratify, một số lớp có thể bị thiếu hoặc lệch trong validation set.

Test set không được dùng để fit mean/std, không dùng để early stopping và không dùng để chọn mô hình. Test chỉ được dùng một lần sau khi mô hình đã huấn luyện xong. Đây là cách tránh data leakage.

## 4. Tiền xử lý và augmentation

Về tiền xử lý, ảnh được rescale từ khoảng 0 đến 255 về khoảng 0 đến 1. Sau đó notebook tính mean và standard deviation trên train set, rồi dùng hai thống kê này để chuẩn hóa train, validation và test.

Điểm quan trọng là mean và standard deviation chỉ được fit trên train set. Nếu ta tính mean/std trên cả test set, thông tin từ test sẽ rò rỉ vào quá trình huấn luyện. Khi đó kết quả test không còn khách quan nữa.

Notebook cũng dùng data augmentation, nhưng augmentation chỉ áp dụng khi training. Các phép augmentation gồm:

- Horizontal flip.
- Translation.
- Rotation nhẹ.
- Zoom.
- Contrast nhẹ.

Mục đích của augmentation là tạo thêm biến thể hợp lý của ảnh train. Ví dụ cùng một vật thể có thể hơi lệch vị trí, hơi xoay, hơi phóng to hoặc có ánh sáng khác nhau. Nhờ đó mô hình ít học thuộc ảnh train hơn và tổng quát hóa tốt hơn.

Nếu bị hỏi vì sao validation và test không augmentation, có thể trả lời: validation và test cần phản ánh dữ liệu gốc. Nếu augmentation cả validation hoặc test, metric sẽ dao động theo phép biến đổi ngẫu nhiên và không còn là một chuẩn đánh giá cố định.

## 5. CNN dưới dạng hàm hợp

Phần lý thuyết đầu notebook giải thích CNN như một hàm hợp:

`x -> Convolution -> BatchNorm -> ReLU -> Pooling -> Classifier -> Probability`

Đầu tiên, input `x` là ảnh CIFAR-100 kích thước 32x32x3.

Convolution dùng các kernel nhỏ để quét qua ảnh và phát hiện đặc trưng cục bộ. Ở tầng đầu, đặc trưng có thể là cạnh, góc, màu hoặc texture đơn giản. Ở các tầng sâu hơn, đặc trưng trở nên trừu tượng hơn, ví dụ bộ phận của vật thể hoặc hình dạng tổng quát.

BatchNorm chuẩn hóa activation trong mini-batch. Nhờ đó quá trình tối ưu ổn định hơn và mô hình có thể học nhanh hơn.

ReLU thêm tính phi tuyến. Nếu chỉ có convolution tuyến tính liên tiếp, toàn bộ mạng vẫn gần như là một phép biến đổi tuyến tính lớn. ReLU giúp mô hình học các quan hệ phức tạp hơn.

Pooling hoặc downsampling giúp giảm kích thước không gian và tăng receptive field. Khi đi sâu hơn, mỗi neuron nhìn được một vùng lớn hơn của ảnh gốc.

Cuối cùng, classifier chuyển feature thành xác suất 100 lớp bằng softmax.

Câu chuyển ý: Sau khi hiểu CNN cơ bản là một chuỗi hàm hợp, ta có thể xem VGG, ResNet và DenseNet là ba cách khác nhau để thiết kế phần feature extractor của CNN.

## 6. Kiến trúc 1 - Basic CNN

Mô hình đầu tiên là Basic CNN. Đây là baseline đơn giản nhất trong notebook. Mô hình dùng các block Conv-BN-ReLU-Pool theo chuỗi.

Ý tưởng là mỗi stage sẽ học một cấp đặc trưng:

- Stage đầu học đặc trưng thấp như cạnh và màu.
- Stage giữa học texture hoặc pattern nhỏ.
- Stage sau học đặc trưng trừu tượng hơn.

Sau các stage convolution, mô hình dùng Global Average Pooling để biến feature map thành vector, rồi đưa vào dense classifier.

Basic CNN có 123.332 tham số trên CIFAR-100. Vì vậy thời gian train và inference nhanh nhất. Tuy nhiên, do kiến trúc đơn giản, khả năng biểu diễn thấp hơn các mô hình còn lại.

Kết quả:

- Accuracy: 0.3156.
- Macro F1: 0.3035.
- Train time: 373.66 giây.
- Inference time: 0.9739 giây.

Kết quả này là baseline để thấy các kiến trúc sâu hơn cải thiện như thế nào.

## 7. Kiến trúc 2 - Compact VGG

Mô hình thứ hai là Compact VGG. Ý tưởng chính của VGG là dùng nhiều convolution kernel 3x3 xếp chồng lên nhau.

Thay vì dùng một convolution lớn, VGG dùng nhiều convolution nhỏ liên tiếp. Cách này có hai lợi ích.

Thứ nhất, nhiều lớp 3x3 giúp tăng receptive field. Ví dụ hai convolution 3x3 liên tiếp có thể nhìn vùng tương đương 5x5, nhưng thêm nhiều ReLU ở giữa nên mô hình học được quan hệ phi tuyến tốt hơn.

Thứ hai, thiết kế của VGG rất đồng nhất và dễ hiểu. Trong notebook, Compact VGG có ba stage:

- Stage 1: 2 convolution, 32 filter.
- Stage 2: 2 convolution, 64 filter.
- Stage 3: 3 convolution, 128 filter.

Sau mỗi stage có pooling để giảm kích thước không gian.

Kết quả:

- Accuracy: 0.4501.
- Macro F1: 0.4447.
- Params: 465.732.
- Train time: 988.18 giây.

So với Basic CNN, VGG cải thiện Macro F1 từ 0.3035 lên 0.4447. Điều này cho thấy tăng độ sâu bằng cách chồng convolution 3x3 giúp mô hình học đặc trưng tốt hơn trên CIFAR-100.

## 8. Kiến trúc 3 - Compact ResNet

Mô hình thứ ba là Compact ResNet. Đây là mô hình tốt nhất trong notebook.

Vấn đề của mạng sâu là khi tăng số tầng, việc tối ưu trở nên khó hơn. Gradient có thể suy giảm hoặc mô hình khó học ánh xạ tốt. ResNet giải quyết bằng residual connection.

Một residual block có công thức:

`y = ReLU(F(x) + P(x))`

Trong đó `F(x)` là phần mô hình cần học, còn `P(x)` là shortcut. Nếu input và output cùng shape thì shortcut có thể gần như là identity. Nếu đổi số channel hoặc stride, notebook dùng projection 1x1 để đưa shortcut về cùng shape với nhánh chính.

Ý nghĩa của residual connection là mô hình không cần học toàn bộ ánh xạ từ đầu. Nó chỉ cần học phần chênh lệch, tức phần residual. Điều này làm quá trình tối ưu dễ hơn và giúp gradient truyền qua mạng sâu ổn định hơn.

Trong notebook, Compact ResNet gồm:

- Stem convolution 32 filter.
- Ba stage với filter `[32, 64, 128]`.
- Mỗi stage có hai residual block.
- Dùng projection 1x1 khi đổi shape.

Kết quả:

- Accuracy: 0.5062.
- Macro Precision: 0.5665.
- Macro Recall: 0.5062.
- Macro F1: 0.5055.
- Params: 710.468.
- Train time: 2018.57 giây.

ResNet tốt hơn VGG khoảng 6 điểm phần trăm Macro F1. Điều này cho thấy residual connection có tác dụng rõ trên dataset khó như CIFAR-100.

## 9. Kiến trúc 4 - Compact DenseNet

Mô hình cuối cùng là Compact DenseNet. DenseNet có ý tưởng khác ResNet. Thay vì cộng shortcut, DenseNet nối feature từ các tầng trước.

Công thức là:

`x_l = H_l([x_0, x_1, ..., x_{l-1}])`

Nghĩa là lớp thứ `l` nhận input là toàn bộ feature map từ các lớp trước đó sau khi concatenate theo chiều channel.

Ưu điểm của DenseNet là tái sử dụng feature. Các tầng sau có thể truy cập trực tiếp đặc trưng tầng trước, nên mô hình không cần học lại cùng một đặc trưng nhiều lần.

Trong notebook, Compact DenseNet dùng:

- Ba dense block `[3, 4, 4]`.
- Growth rate 16.
- Transition block nén 0.5 và average pooling.

Kết quả:

- Accuracy: 0.3796.
- Macro F1: 0.3719.
- Params: 174.564.
- Train time: 2841.07 giây.

DenseNet có ít tham số hơn VGG và ResNet, nhưng trong cấu hình này lại train lâu nhất và kết quả không cao bằng VGG/ResNet. Có thể do phiên bản compact, batch size, CPU runtime và giới hạn 20 epoch chưa đủ để DenseNet phát huy tốt.

## 10. Cấu hình huấn luyện chung

Để so sánh công bằng, bốn mô hình dùng cùng cấu hình huấn luyện:

- Optimizer: AdamW.
- Learning rate ban đầu: `1e-3`.
- Weight decay: `1e-4`.
- Batch size: `64`.
- Epoch tối đa: `20`.
- Callback giảm learning rate: ReduceLROnPlateau.
- Early stopping theo validation loss.

AdamW khác Adam thường ở chỗ weight decay được tách riêng khỏi gradient update. Điều này giúp regularization rõ ràng hơn.

ReduceLROnPlateau theo dõi validation loss. Nếu validation loss không cải thiện trong một số epoch, learning rate sẽ giảm một nửa. Trong log huấn luyện, ta thấy nhiều mô hình được giảm learning rate ở các epoch giữa quá trình train.

Early stopping được cấu hình theo validation loss và restore best weights. Tuy nhiên với CIFAR-100, best epoch của cả bốn mô hình đều là 20, nghĩa là trong 20 epoch mô hình vẫn chưa có dấu hiệu dừng tối ưu rõ ràng.

## 11. Bảng kết quả benchmark

Đây là bảng quan trọng nhất của notebook.

| Mô hình | Accuracy | Macro Precision | Macro Recall | Macro F1 | Test Loss | Params | Best Epoch | Train Time | Inference Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Compact ResNet | 0.5062 | 0.5665 | 0.5062 | 0.5055 | 1.8785 | 710.468 | 20 | 2018.57s | 6.6999s |
| Compact VGG | 0.4501 | 0.5011 | 0.4501 | 0.4447 | 2.1184 | 465.732 | 20 | 988.18s | 2.5776s |
| Compact DenseNet | 0.3796 | 0.4288 | 0.3796 | 0.3719 | 2.3846 | 174.564 | 20 | 2841.07s | 6.7352s |
| Basic CNN | 0.3156 | 0.3639 | 0.3156 | 0.3035 | 2.7271 | 123.332 | 20 | 373.66s | 0.9739s |

Nhìn vào bảng, Compact ResNet đứng đầu ở Accuracy, Macro Precision, Macro Recall, Macro F1 và Test Loss. Vì vậy trong notebook này, ResNet là mô hình tốt nhất.

Compact VGG đứng thứ hai. Điều này cho thấy thiết kế chồng nhiều convolution 3x3 có hiệu quả, nhưng vẫn kém ResNet vì không có residual connection.

Basic CNN nhanh nhất nhưng kết quả thấp nhất. Điều này hợp lý vì mô hình đơn giản, ít tham số và khả năng biểu diễn thấp hơn.

DenseNet có số tham số khá thấp nhưng train lâu. Điều này có thể đến từ việc concatenate feature làm graph tính toán phức tạp hơn, dù số tham số không quá lớn.

## 12. Learning curves

Khi nhìn vào learning curves, ta cần chú ý hai đường chính: train và validation.

Nếu train loss giảm nhưng validation loss tăng, đó là dấu hiệu overfitting. Nếu cả train và validation đều chưa tốt, đó là dấu hiệu underfitting hoặc chưa train đủ lâu.

Với CIFAR-100, các mô hình vẫn tiếp tục cải thiện đến gần epoch cuối. Đặc biệt best epoch đều là 20. Vì vậy em nhận xét rằng notebook đang giới hạn thời gian train ở 20 epoch, và kết quả có thể cải thiện nếu tăng số epoch hoặc dùng learning rate schedule dài hơn.

ResNet có đường metric tốt nhất, phù hợp với bảng benchmark. Basic CNN thấp hơn rõ rệt, cho thấy kiến trúc baseline chưa đủ mạnh cho dataset 100 lớp.

## 13. Metric comparison

Biểu đồ metric comparison trực quan hóa bốn metric chính: Accuracy, Macro Precision, Macro Recall và Macro F1.

Ta thấy ResNet cao nhất ở cả bốn metric. VGG đứng thứ hai, DenseNet thứ ba và Basic CNN thấp nhất.

Điểm đáng chú ý là Macro Precision của ResNet cao hơn Macro Recall. Điều này có nghĩa là khi ResNet dự đoán một lớp, dự đoán đó tương đối chính xác hơn; nhưng vẫn còn nhiều mẫu của từng lớp chưa được bắt hết, dẫn tới recall chưa quá cao.

Với bài toán 100 lớp, Macro F1 là metric cân bằng hơn vì nó kết hợp cả precision và recall.

## 14. Confusion matrix

Confusion matrix của CIFAR-100 có kích thước 100x100. Mỗi hàng là lớp thật, mỗi cột là lớp dự đoán.

Nếu mô hình hoàn hảo, toàn bộ màu đậm sẽ nằm trên đường chéo. Trong thực tế, ta thấy vẫn có nhiều ô ngoài đường chéo, nghĩa là mô hình còn nhầm lẫn giữa các lớp.

Notebook ẩn nhãn tick vì 100 lớp quá nhiều, nếu hiển thị toàn bộ sẽ rất rối. Mục đích của heatmap ở đây là xem cấu trúc tổng thể: mô hình nào có đường chéo rõ hơn thì mô hình đó phân loại tốt hơn.

ResNet có đường chéo rõ hơn các mô hình còn lại, phù hợp với việc ResNet có Macro F1 cao nhất.

## 15. Error analysis

Ở phần error analysis, notebook chọn mô hình tốt nhất là Compact ResNet và hiển thị một số ảnh dự đoán sai.

Số ảnh dự đoán sai là 4.938 trên 10.000 ảnh test. Nghĩa là mô hình dự đoán đúng khoảng hơn một nửa test set. Với CIFAR-100, kết quả này có thể chấp nhận được trong cấu hình compact và chỉ 20 epoch.

Các lỗi thường đến từ ba nguyên nhân.

Thứ nhất, ảnh CIFAR-100 rất nhỏ, chỉ 32x32, nên nhiều chi tiết quan trọng không rõ.

Thứ hai, nhiều lớp fine-grained có hình dạng hoặc ngữ cảnh gần nhau. Ví dụ các loài động vật, các loại cây, đồ gia dụng hoặc phương tiện có thể khá giống nhau.

Thứ ba, mô hình train từ đầu, không dùng pretrained. Nếu dùng pretrained backbone hoặc train lâu hơn, kết quả có thể cao hơn.

## 16. Kết luận chính

Từ notebook CIFAR-100, em rút ra bốn kết luận.

Thứ nhất, CIFAR-100 là dataset khó. Basic CNN chỉ đạt Macro F1 0.3035, cho thấy CNN cơ bản chưa đủ mạnh cho bài toán 100 lớp.

Thứ hai, VGG cải thiện đáng kể so với Basic CNN nhờ tăng độ sâu bằng nhiều convolution 3x3. Macro F1 tăng từ 0.3035 lên 0.4447.

Thứ ba, ResNet là mô hình tốt nhất, đạt Macro F1 0.5055. Kết quả này cho thấy residual connection giúp mô hình sâu hơn học ổn định và hiệu quả hơn.

Thứ tư, DenseNet có ý tưởng tái sử dụng feature và ít tham số hơn ResNet, nhưng trong lần chạy này chưa đạt kết quả tốt nhất. Điều này nhắc lại rằng một kiến trúc tốt còn phụ thuộc cấu hình huấn luyện, số epoch, augmentation và tài nguyên tính toán.

Tóm lại, notebook đã hoàn thành mục tiêu: so sánh bốn kiến trúc CNN trên cùng một pipeline và cho thấy các kiến trúc phát triển như VGG và đặc biệt là ResNet có lợi thế rõ rệt trên dataset khó như CIFAR-100.

## 17. Câu trả lời nhanh nếu bị hỏi

### Vì sao ResNet tốt nhất?

Vì ResNet có residual connection, giúp gradient truyền tốt hơn qua mạng sâu. Mô hình học phần residual thay vì học toàn bộ ánh xạ, nên tối ưu dễ hơn VGG khi độ sâu tăng.

### Vì sao DenseNet ít tham số nhưng train lâu?

DenseNet nối feature map từ nhiều tầng trước. Số tham số có thể thấp, nhưng việc concatenate và xử lý nhiều feature map làm graph tính toán phức tạp hơn, nên thời gian train/inference không nhất thiết thấp.

### Vì sao Accuracy CIFAR-100 chỉ khoảng 50%?

CIFAR-100 có 100 lớp, ảnh nhỏ 32x32, mỗi lớp chỉ có 500 ảnh train gốc. Notebook dùng mô hình compact, train từ đầu và chỉ 20 epoch, nên 50% là hợp lý trong phạm vi assignment.

### Vì sao không dùng pretrained?

Yêu cầu của Assignment 05 là so sánh bốn kiến trúc compact huấn luyện từ đầu. Nếu dùng pretrained, kết quả sẽ phụ thuộc vào dữ liệu pretraining và không còn là so sánh trực tiếp giữa các kiến trúc compact trong cùng điều kiện.

### Vì sao dùng Macro F1?

Macro F1 tính trung bình F1 trên từng lớp, giúp đánh giá công bằng giữa 100 lớp. Accuracy chỉ cho biết tỷ lệ đúng tổng thể, còn Macro F1 cho biết mô hình có học tương đối đều giữa các lớp hay không.

### Nếu muốn cải thiện kết quả thì làm gì?

Có thể tăng số epoch, dùng cosine learning rate schedule, cutout, mixup, label smoothing, mô hình sâu hơn hoặc pretrained backbone. Ngoài ra có thể chạy nhiều seed để đánh giá độ ổn định.

## 18. Thứ tự demo notebook khi thuyết trình

1. Mở tiêu đề notebook và nói mục tiêu.
2. Mở phần dataset để nói CIFAR-100 có 50.000 train, 10.000 test, 100 lớp.
3. Mở phần CNN như hàm hợp để giải thích Conv-BN-ReLU-Pool-Classifier.
4. Mở phần model builders và chỉ bốn hàm:
   - `build_basic_cnn`
   - `build_vgg_compact`
   - `build_resnet_compact`
   - `build_densenet_compact`
5. Mở phần load data để chỉ shape `(50000, 32, 32, 3)` và split `45000/5000/10000`.
6. Mở phần training config để nói epoch 20, batch 64, AdamW, callbacks.
7. Mở bảng benchmark và nhấn mạnh Compact ResNet tốt nhất.
8. Mở learning curves để nói mô hình vẫn còn có thể cải thiện nếu train lâu hơn.
9. Mở confusion matrix để nói CIFAR-100 có nhiều nhầm lẫn do 100 lớp.
10. Mở error analysis để kết luận về độ khó của dataset.

