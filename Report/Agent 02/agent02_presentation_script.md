# Kịch bản thuyết trình chi tiết chương trình Agent 02

Thời lượng gợi ý: 15-18 phút
Chương trình: `src/Agent 02/agent02_experiment.py`
Kết quả: `src/Agent 02/agent02_results/`
Báo cáo: `Report/Agent 02/agent02_CT_03_khoipv.447.pdf`
Chủ đề: So sánh agent học tăng cường (Q-learning) và agent dựa trên luật trong môi trường thương mại điện tử hai người bán cạnh tranh giảm giá.

## 0. Cách dùng file này

File này được viết theo kiểu kịch bản nói. Khi thuyết trình, có thể đọc gần như trực tiếp, nhưng nên nói tự nhiên hơn bằng cách:

- Mở file `agent02_experiment.py` song song, chương trình có 19 section được đánh số trong comment.
- Mở sẵn thư mục `agent02_results/` để chiếu các hình PNG.
- Khi đến phần kết quả, dừng lâu hơn ở bảng so sánh từng cặp và biểu đồ kịch bản.
- Nếu thiếu thời gian, ưu tiên trình bày: mô hình môi trường, hai agent, kết quả thí nghiệm E, và câu 9-10.

## 1. Mở đầu

Kính chào thầy và các bạn.
Trong phần này em xin trình bày chương trình của thí nghiệm Agent 02. Thí nghiệm so sánh hai cách xây dựng agent ra quyết định trong cùng một môi trường: một agent học tăng cường dùng Q-learning dạng bảng, và một agent dựa trên luật if-else do người viết sẵn.

Bài toán là một sàn thương mại điện tử đơn giản có hai người bán cùng loại sản phẩm. Ở mỗi bước thời gian, mỗi người quyết định **giảm giá** hoặc **không giảm giá**. Giảm giá làm tăng xác suất khách mua nhưng làm giảm lợi nhuận trên mỗi sản phẩm. Agent phải cân bằng hai tác động trái ngược này.

Em xin nói trước kết luận chính, để thầy và các bạn dễ theo dõi: trong môi trường tham chiếu của đề bài, **agent dựa trên luật tình cờ đã là chính sách tối ưu**, nên agent RL tốt nhất chỉ có thể hòa chứ không thể thắng. Nhưng khi em thay đổi tham số thị trường, kết luận đảo ngược. Đây là điểm thú vị nhất của bài.

## 2. Mô hình môi trường

Trạng thái cục bộ của mỗi agent gồm hai biến:

$$s_i(t) = (d_t, c_i(t))$$

Trong đó `d` là nhu cầu thị trường, bằng 0 nếu thấp và 1 nếu cao. Còn `c` là hành động của đối thủ ở **bước trước**. Mỗi agent có 4 trạng thái và 2 hành động.

Một điểm quan trọng: hai agent chọn hành động **đồng thời**, nên không agent nào biết hành động hiện tại của đối thủ. Nó chỉ biết đối thủ đã làm gì ở bước trước.

Xác suất một khách tiềm năng mua hàng:

$$p_i = \text{clip}(0{,}30 + 0{,}25 d + 0{,}15 a_i - 0{,}10 a_{-i};\ 0{,}05;\ 0{,}95)$$

Đọc công thức này theo nghĩa kinh doanh: nhu cầu cao làm tăng xác suất mua thêm 0,25. Mình giảm giá thì tăng thêm 0,15. Nhưng đối thủ giảm giá thì mình **mất** 0,10.

Số khách tiềm năng là 10 khi nhu cầu thấp và 20 khi nhu cầu cao. Số đơn bán được rút từ phân phối nhị thức. Với giá gốc 10, giá vốn 5 và mức giảm 20 phần trăm, phần thưởng là lợi nhuận:

$$r_i = S_i \cdot [P(1 - \delta a_i) - C]$$

Em xin lưu ý: dữ liệu ở đây **do chương trình tự sinh ra**, không dùng dataset nào. Đề bài yêu cầu thiết kế một mô phỏng nhiều agent, và bản thân công thức xác suất mua là mô hình minh họa, không được kiểm chứng bằng dữ liệu thương mại điện tử thật.

## 3. Phân tích lợi nhuận trước khi chạy code

Trước khi chạy thí nghiệm, em đã tính tay lợi nhuận kỳ vọng một bước cho mọi tổ hợp. Đây là bước quan trọng vì nó cho ta biết **đáp án đúng là gì**, để sau này đối chiếu.

Lợi nhuận mỗi sản phẩm: không giảm giá được 10 trừ 5 bằng 5. Giảm giá 20 phần trăm thì giá bán còn 8, lợi nhuận còn 8 trừ 5 bằng 3.

| Nhu cầu `d` | Đối thủ `b` | Không giảm giá | Giảm giá | Tốt hơn |
|---|---|---|---|---|
| 0 | 0 | 15,0 | 13,5 | không giảm |
| 0 | 1 | 10,0 | 10,5 | giảm |
| 1 | 0 | 55,0 | 42,0 | không giảm |
| 1 | 1 | 45,0 | 36,0 | không giảm |

Nhìn bảng này ta thấy ba trong bốn trường hợp thì không giảm giá tốt hơn. Biên lợi nhuận giảm 40 phần trăm, nên giảm giá chỉ có lợi khi số đơn tăng hơn 1,67 lần. Thực tế chỉ tăng 1,5 lần khi nhu cầu thấp và 1,27 lần khi nhu cầu cao.

Còn một lý do thứ hai, tinh tế hơn: nếu agent RL giảm giá ở bước này, agent luật sẽ **giảm giá đáp trả** ở bước sau khi nhu cầu cao, kéo lợi nhuận của RL từ 55 xuống 45. Nên không giảm giá vừa lợi trước mắt, vừa không kích hoạt phản ứng của đối thủ.

Kết luận: khi đối thủ là agent luật, chính sách tối ưu là **không bao giờ giảm giá**, đạt lợi nhuận kỳ vọng chính xác 35,00 mỗi bước.

## 4. Hai agent

**Agent dựa trên luật** có chính sách cố định:

```python
def act(self, state, explore=False):
    demand, competitor_previous_action = state
    if demand == 0:
        return 0
    elif demand == 1 and competitor_previous_action == 1:
        return 1
    else:
        return 0
```

Ba luật: nhu cầu thấp thì không giảm giá; nhu cầu cao và đối thủ vừa giảm giá thì giảm giá; nhu cầu cao mà đối thủ không giảm thì cũng không giảm.

Agent này **không học gì cả**. Nó vẫn là một agent đầy đủ, vì nó quan sát trạng thái, ra quyết định và tác động lên môi trường. Đây là ví dụ rõ nhất cho thấy khái niệm agent không đòi hỏi phải có học máy.

**Agent Q-learning** giữ bảng Q gồm 4 trạng thái nhân 2 hành động bằng 8 giá trị:

```python
def learn(self, state, action, reward, next_state, terminal=False):
    old_value = self.q[state + (action,)]
    if terminal:
        target = reward
    else:
        target = reward + self.gamma * np.max(self.q[next_state])
    self.q[state + (action,)] = old_value + self.alpha * (target - old_value)
```

Đây đúng là công thức Q-learning đã dùng ở Agent 01, chỉ khác là trạng thái bây giờ mô tả thị trường thay vì tọa độ trong mê cung.

Hành động chọn theo ε-greedy, với epsilon giảm từ 0,20 xuống sàn 0,02. Với hệ số giảm 0,995 mỗi episode, epsilon chạm sàn từ khoảng episode 460.

## 5. Các thay đổi so với mã tham khảo

Đề bài cho phép sửa mã nếu ghi rõ thay đổi và giữ so sánh hợp lệ. Em đã làm bốn thay đổi, xin trình bày vì chúng ảnh hưởng trực tiếp đến độ tin cậy của kết quả.

**Thứ nhất, số ngẫu nhiên chung.** Môi trường dùng một bộ sinh số riêng cho nhu cầu và một bộ riêng cho doanh số của từng agent, cùng tách ra từ một seed.

```python
streams = np.random.SeedSequence(actual_seed).spawn(3)
self.demand_rng = np.random.default_rng(streams[0])
self.sales_rngs = [np.random.default_rng(s) for s in streams[1:]]
```

Nhờ vậy, khi em đánh giá cặp RL-luật và cặp luật-luật với cùng seed, hai lần chạy thấy **đúng cùng một chuỗi nhu cầu**. Chênh lệch đo được đến từ chính sách, không phải từ may rủi của chuỗi ngẫu nhiên.

**Thứ hai, giao diện agent thống nhất.** Mọi agent đều có `observe`, `act`, `learn`, nên cùng một vòng huấn luyện dùng được cho RL đấu luật, RL đấu RL, và các agent mới.

**Thứ ba, đánh giá chính xác.** Đây là phần em tâm đắc nhất. Hàm `exact_evaluation` lan truyền phân phối xác suất của trạng thái chung qua 20 bước, tính lợi nhuận kỳ vọng mà **không cần lấy mẫu**:

```python
distribution = {(0, 0, 0): 0.5, (1, 0, 0): 0.5}
for _ in range(T):
    for (demand, p0, p1), mass in distribution.items():
        for actions in product((0, 1), repeat=2):
            weight = mass * ...
            reward[i] += weight * customers * conversion * margin
```

Và hàm `best_response` liệt kê toàn bộ 16 chính sách tất định để tìm mức tối ưu. Nhờ hai hàm này, em có một **mốc chuẩn không nhiễu** để so sánh, thay vì chỉ so hai con số đo được với nhau.

**Thứ tư, thí nghiệm bổ sung:** epsilon cố định, 10 seed độc lập, hai agent cùng học, thay đổi mức giảm giá, hàm thưởng và đối thủ.

## 6. Quy trình đánh giá tách biệt

Đề bài nêu rõ mã gốc chưa có giai đoạn đánh giá tách biệt, và sinh viên phải tự cài đặt trước khi kết luận về chất lượng chính sách.

Trong `evaluate_pair`, em đảm bảo ba điều:

- Bảng Q được **đóng băng**, không cập nhật.
- Khám phá bị **tắt hoàn toàn**, epsilon bằng 0.
- Dùng **seed khác** với lúc huấn luyện, cụ thể là seed huấn luyện cộng 500.000 hoặc 800.000.

Mỗi lần đánh giá chạy 500 episode nhân 20 bước, tức 10.000 bước.

Vì sao điều này quan trọng? Vì phần thưởng trong lúc huấn luyện bao gồm cả chi phí của các bước khám phá ngẫu nhiên, và chính sách khi đó vẫn đang thay đổi. Hai đại lượng này đo hai thứ khác nhau.

## 7. Thí nghiệm A — Baseline

Chạy với seed 42, kết quả đánh giá:

```
RL vs Rule    | RL (Q-learning)  | mean reward/step = 34.5565 | discount = 0 | no discount = 10000
RL vs Rule    | Rule-based       | mean reward/step = 34.6720 | discount = 0
Rule vs Rule  | Rule-based       | mean reward/step = 34.5565 | discount = 0
```

Chính sách học được là **không giảm giá ở cả bốn trạng thái**. Agent luật cũng không giảm giá lần nào.

Điểm đáng chú ý: hai lần chạy cho kết quả **trùng đến từng đơn vị**, 34,5565 ở cả hai. Đó là nhờ số ngẫu nhiên chung — cùng chính sách thì cùng chuỗi nhu cầu sẽ cho cùng doanh số.

Chênh lệch giữa ghế 1 là 34,56 và ghế 2 là 34,67 **không phải** do chính sách khác nhau, vì cả hai đều không giảm giá. Nó chỉ là nhiễu doanh số ngẫu nhiên. Giá trị kỳ vọng của cả hai đều đúng bằng 35,00.

Bảng Q sau huấn luyện:

```
State=(0, 0) | Q0=209.03 | Q1=200.01 | Greedy action=0
State=(0, 1) | Q0=227.10 | Q1=181.17 | Greedy action=0
State=(1, 0) | Q0=266.32 | Q1=228.50 | Greedy action=0
State=(1, 1) | Q0=228.33 | Q1=219.32 | Greedy action=0
```

Mọi trạng thái đều có Q0 lớn hơn Q1 nên chính sách tham lam chọn không giảm giá.

Nếu có ai hỏi vì sao giá trị Q lên tới 200-270 trong khi phần thưởng mỗi bước chỉ khoảng 35: vì Q ước lượng tổng thưởng chiết khấu của nhiều bước còn lại, gần 35 chia cho 1 trừ 0,9 bằng 350 ở đầu episode.

## 8. Thí nghiệm B — Quá trình học

Chiếu hình `episode_rewards.png`.

Thưởng huấn luyện dao động quanh 33-34 suốt 5.000 episode và **không tăng rõ rệt** sau vài trăm episode đầu.

Điều đáng chú ý hơn: tỷ lệ giảm giá trong huấn luyện vẫn ở mức 0,13 đến 0,23 dù epsilon đã chạm sàn 0,02 từ episode 460. Phần khám phá ngẫu nhiên chỉ giải thích được khoảng 0,01. Vậy phần lớn các lần giảm giá đến từ **chính sách tham lam tạm thời chọn giảm giá**.

Chiếu hình `policy_value_during_training.png`.

Hình này là đóng góp riêng của em. Nó đo giá trị kỳ vọng **chính xác** của chính sách tham lam, cứ 25 episode một lần. Ta thấy nó liên tục nhảy giữa 35,0 là mức tối ưu, và các mức 27 đến 33.

Nghĩa là chính sách của agent **không hội tụ**, nó dao động suốt quá trình huấn luyện. Chính sách cuối cùng đạt tối ưu một phần là nhờ may mắn: nếu dừng ở episode khác, kết quả có thể kém hơn.

Chiếu hình `q_value_trajectories.png`. Tám giá trị Q dao động với biên độ vài chục đơn vị, lớn hơn khoảng cách giữa Q0 và Q1 tại trạng thái `(0,0)` vốn chỉ khoảng 9. Ba nguyên nhân hợp lý: doanh số là biến nhị thức nên nhiễu lớn; alpha cố định 0,1 nên chỉ trung bình trên khoảng 10 mẫu gần nhất; và trạng thái không chứa vị trí bước trong episode.

## 9. Thí nghiệm C — Tốc độ học

Chạy ba cấu hình với 10 seed mỗi cấu hình:

| alpha | Đánh giá | Đổi chính sách (nửa sau) | Số seed tối ưu |
|---|---|---|---|
| 0,01 | 34,546 ± 1,328 | 17,1 | 9/10 |
| 0,10 | 34,724 ± 0,747 | 256,3 | 9/10 |
| 0,50 | 32,912 ± 2,457 | 827,7 | 5/10 |

Cột "đổi chính sách" là số lần chính sách tham lam thay đổi trong 2.500 episode cuối. Đây là chỉ số em thêm vào để **đo trực tiếp độ ổn định**, thay vì chỉ nhìn đồ thị.

Giải thích: alpha nhỏ làm ước lượng ít nhiễu, chỉ 17 lần đổi so với 256 lần. Nhưng đổi lại agent học rất chậm, đường học đứng quanh 30,6 trong 2.500 episode đầu. Alpha lớn làm mỗi cập nhật thay một nửa giá trị cũ bằng một mẫu nhiễu, nên chính sách đổi hơn 800 lần và chỉ 5 trên 10 seed đạt tối ưu.

Chiếu hình `learning_rate_q_trajectory.png` để minh họa: với alpha bằng 0,5, hiệu Q0 trừ Q1 dao động trong khoảng cộng trừ 200 và âm 50 phần trăm thời gian.

## 10. Thí nghiệm D — Mức khám phá

| Cấu hình | Đánh giá | Tỷ lệ giảm giá | Số seed tối ưu |
|---|---|---|---|
| Lịch 0,20 xuống 0,02 | 34,724 ± 0,747 | 0,040 | 9/10 |
| Cố định 0,00 | 29,719 ± 3,852 | 0,415 | 3/10 |
| Cố định 0,05 | 33,775 ± 2,267 | 0,130 | 7/10 |
| Cố định 0,30 | 29,476 ± 3,104 | 0,461 | 2/10 |
| Lịch 0,50 xuống 0,05 | 32,411 ± 2,656 | 0,300 | 4/10 |

Em xin phân tích hai thái cực.

**Ngừng khám phá hoàn toàn**, epsilon bằng 0: chỉ 3 trên 10 seed tối ưu, và chính sách **không đổi lần nào**. Lý do rất cụ thể: mọi lợi nhuận đều dương và bảng Q khởi tạo bằng 0. Nên hành động được thử đầu tiên ở mỗi trạng thái lập tức có Q lớn hơn 0 và được chọn mãi mãi. Agent bị khóa vào lựa chọn ngẫu nhiên ban đầu. Tỷ lệ giảm giá 0,415 gần đúng bằng kết quả tung đồng xu.

**Khám phá quá nhiều**, epsilon cố định 0,30: thưởng huấn luyện chỉ 30,91 vì khoảng 15 phần trăm số bước là giảm giá ngẫu nhiên. Nhưng điều đáng nói là chính sách tham lam cuối cùng **cũng kém**, chỉ 29,48.

Ở đây em xin trình bày trung thực: em có một giả thuyết cho hiện tượng này, rằng vì trạng thái không chứa hành động trước của chính agent, nên điểm hội tụ của Q phụ thuộc vào tần suất agent tự giảm giá trong lúc huấn luyện. Nhưng **thí nghiệm hiện tại chưa đủ để chứng minh** cơ chế đó, cần một thí nghiệm riêng.

## 11. Thí nghiệm E — So sánh công bằng

Đây là phần quan trọng nhất về mặt phương pháp. Chiếu bảng `multi_seed_summary.csv`:

| Seed | RL | Luật | RL trừ Luật | Chính sách RL |
|---|---|---|---|---|
| 11 | 35,189 | 35,189 | 0,000 | 0100 |
| 22 | 34,881 | 34,881 | 0,000 | 0100 |
| 33 | 32,641 | 35,258 | **-2,617** | 1000 |
| 44 | 34,931 | 34,931 | 0,000 | 0000 |
| ... | ... | ... | 0,000 | ... |

Ở **9 trên 10 seed, hai agent cho kết quả giống hệt nhau**.

Có một chi tiết cần giải thích. Các chính sách ghi `0100` nghĩa là agent chọn giảm giá ở trạng thái `(0,1)`, tức nhu cầu thấp và đối thủ vừa giảm giá. Nhưng trạng thái này **không bao giờ xuất hiện** nếu agent RL không giảm giá, vì agent luật chỉ giảm giá sau khi RL giảm giá trước. Nên hành vi thực tế vẫn là không bao giờ giảm giá.

Seed 33 là trường hợp duy nhất học sai: chính sách `1000` nghĩa là giảm giá khi nhu cầu thấp, làm lợi nhuận tụt còn 32,64.

Kết quả thống kê: hiệu cặp RL trừ Luật có trung bình -0,262, độ lệch chuẩn 0,828, khoảng tin cậy 95 phần trăm là **từ -0,854 đến 0,330**, có chứa số 0.

Kết luận đúng mực: **dữ liệu không ủng hộ nhận định bên nào tốt hơn** trong môi trường tham chiếu. Đây không phải là RL thất bại, mà là vì luật tham chiếu tình cờ đã trùng với chính sách tối ưu, nên RL tốt nhất chỉ có thể hòa.

## 12. Câu 8 — Khi hai agent cùng học

Em cho hai agent Q-learning độc lập cùng học với nhau:

| Thiết lập | Thưởng đánh giá | Tỷ lệ giảm giá | Đổi chính sách |
|---|---|---|---|
| RL đấu luật cố định | 34,724 ± 0,747 | 0,040 | 256,3 |
| RL đấu RL | 33,266 ± 2,469 | 0,187 | 284,4 |

Khi cùng học, **5 trên 10 seed** kết thúc ở cặp chính sách kém, so với chỉ 1 trên 10 khi đối thủ cố định. Lợi nhuận giảm 1,46 mỗi bước và độ lệch chuẩn tăng hơn ba lần.

Ở seed 44 và 99, cả hai agent cùng giảm giá ở nhiều trạng thái, tạo thành một dạng "chiến tranh giá" nhỏ làm cả hai cùng mất lợi nhuận.

Lý do lý thuyết: khi cả hai cùng học, chính sách đối thủ thay đổi theo thời gian, nên cùng một cặp trạng thái và hành động lại có phân phối phần thưởng khác nhau ở các giai đoạn khác nhau. Giả định Markov dừng mà Q-learning cần để hội tụ bị vi phạm. Mỗi agent đang đuổi theo một mục tiêu đang dịch chuyển.

Em cũng xin nói thật một hạn chế: số lần đổi chính sách chỉ tăng nhẹ, từ 256 lên 284. Phần lớn dao động đã có sẵn do nhiễu, nên thí nghiệm này **chưa tách riêng được** phần suy giảm nào thực sự do tính không dừng.

Đề xuất của em để nghiên cứu sâu hơn: học luân phiên, đóng băng agent 2 trong M episode trong khi agent 1 học rồi đổi vai. Mỗi pha khi đó là bài toán dừng, nên so sánh sẽ tách được nguyên nhân.

## 13. Câu 9 — Thay đổi môi trường

Đây là phần cho thấy kết luận ở trên **phụ thuộc hoàn toàn vào tham số**. Em thay đổi mức giảm giá, hàm thưởng và hành vi đối thủ.

Chiếu bảng `scenario_exact_benchmarks.csv`:

| Kịch bản | Không giảm | Luôn giảm | Luật | Tối ưu | Chính sách tối ưu |
|---|---|---|---|---|---|
| Giảm 20% (gốc) | 35,00 | 24,90 | 35,00 | 35,00 | 0000 |
| Giảm 5% | 35,00 | 37,35 | 35,00 | **37,35** | **1111** |
| Giảm 10% | 35,00 | 33,20 | 35,00 | 35,00 | 0000 |
| Giảm 30% | 35,00 | 16,60 | 35,00 | 35,00 | 0000 |

Và kết quả đo được:

| Kịch bản | Luật | RL tham chiếu | Dựa trên mô hình |
|---|---|---|---|
| Giảm 20% | 34,99 | 34,72 (9/10) | 34,99 (10/10) |
| **Giảm 5%** | 34,99 | **36,72** (2/10) | **37,33** (9/10) |
| **Giảm 10%** | 34,99 | **33,85** (1/10) | 34,99 (10/10) |
| Giảm 30% | 34,99 | 34,54 (9/10) | 34,99 (10/10) |

Hai kịch bản đáng nói nhất:

**Giảm 5 phần trăm:** lợi nhuận mỗi đơn khi giảm giá còn 4,5. Khi nhu cầu thấp, giảm giá cho 20,25 so với 15. Luôn giảm giá trở thành tối ưu. Agent luật không đổi hành vi nên **mất 2,35 mỗi bước**, tức 6,3 phần trăm. Agent RL tự chuyển sang giảm giá 75 phần trăm thời gian và đạt 36,72, **vượt agent luật**. Đây là kịch bản thể hiện rõ lợi thế thích ứng của RL.

**Giảm 10 phần trăm:** đây là kịch bản tinh tế nhất. Giảm giá có lợi trước mắt, khi nhu cầu thấp được 18 so với 15. Nhưng nó kích hoạt đối thủ giảm giá đáp trả ở bước sau. Tối ưu vẫn là không giảm giá. Agent RL chỉ thấy lợi ích trước mắt, vì hậu quả xảy ra ở một trạng thái mà nó **không phân biệt được**. Kết quả: RL giảm giá 50 phần trăm thời gian và **thua agent luật**.

Em cũng thử đổi hàm thưởng thành doanh thu thay vì lợi nhuận. Agent RL nâng tỷ lệ giảm giá từ 0,04 lên 0,50, và lợi nhuận thực tế **tụt từ 34,72 xuống 29,07**. Bài học: agent tối ưu chính xác thứ ta đo, nên hàm thưởng phải phản ánh đúng mục tiêu kinh doanh.

## 14. Câu 10 — Agent cải tiến

Từ phân tích trên, em xác định hai điểm yếu của agent RL tham chiếu: trạng thái thiếu thông tin, và cập nhật TD với alpha cố định quá nhiễu.

Agent đề xuất xử lý cả hai:

- **Trạng thái** thêm hành động trước của chính agent, thành `(d, c, own)`, 8 trạng thái. Điều này làm trạng thái trở thành Markov khi đối thủ phản ứng theo hành động trước.
- **Học mô hình** thay vì học thẳng Q. Agent đếm số lần thăm, tổng thưởng và tần suất trạng thái kế tiếp, từ đó ước lượng hàm thưởng và **mô hình phản ứng của đối thủ**.
- **Lập kế hoạch** bằng lặp giá trị trên mô hình đã học, cứ 50 episode một lần.

```python
def plan(self):
    visits = np.maximum(self.visits, 1)
    reward = self.reward_sum / visits
    transition = self.transitions / visits[..., None]
    value = np.zeros(reward.shape[0])
    for _ in range(1000):
        q = reward + self.plan_gamma * transition @ value
        ...
```

Để đánh giá độ bền vững, em dùng **regret**, bằng mức tối ưu trừ giá trị đạt được, rồi lấy **regret lớn nhất** trên toàn bộ sáu kịch bản:

| Kịch bản | Luật | RL tham chiếu | RL mở rộng | Dựa trên mô hình |
|---|---|---|---|---|
| Giảm 20% | 0,000 | 0,255 | 1,887 | 0,000 |
| Giảm 5% | 2,350 | 0,630 | 0,868 | 0,020 |
| Giảm 10% | 0,000 | 1,149 | 1,121 | 0,000 |
| Doanh thu | 0,000 | 2,298 | 2,242 | 0,000 |
| **Regret lớn nhất** | **2,350** | **2,298** | **2,242** | **0,085** |

Agent đề xuất có regret lớn nhất chỉ 0,085, so với khoảng 2,3 của cả agent luật lẫn agent RL tham chiếu. Chính sách của nó gần như không đổi trong nửa sau huấn luyện, chỉ 0 đến 2,3 lần.

Một kết quả đối chứng em thấy rất đáng nói: **chỉ thêm trạng thái mà vẫn dùng Q-learning thì không cải thiện**, thậm chí kém hơn ở kịch bản gốc, chỉ 33,07 và 4 trên 10 seed. Vì bảng Q có 16 giá trị nên mỗi ô nhận ít dữ liệu hơn, trong khi nhiễu không giảm. Vậy cải thiện đến từ việc **kết hợp** trạng thái đầy đủ với ước lượng ít nhiễu và lập kế hoạch, không đến từ riêng phần trạng thái.

Em cũng xin nêu hạn chế của chính đề xuất này: em thiết kế nó **sau khi** đã thấy điểm yếu của agent tham chiếu trên chính các kịch bản này, nên có nguy cơ khớp quá mức với tập kiểm tra. Cần thêm kịch bản mới chưa dùng khi thiết kế để kiểm tra lại.

## 15. Tính tái lập

Toàn bộ chương trình chạy khoảng 40 giây trên CPU. Em đã chạy lại hai lần và **nhật ký giống hệt nhau**, nghĩa là kết quả tái lập hoàn toàn với cùng môi trường phần mềm: Python 3.12.13, NumPy 2.5.2, Matplotlib 3.11.1.

Mọi seed, tham số, số episode và giao thức đánh giá đều được ghi trong báo cáo.

## 16. Kết luận chính

Em xin tóm tắt bốn kết luận.

Thứ nhất, trong môi trường tham chiếu, dữ liệu **không ủng hộ** nhận định RL tốt hơn hay kém hơn agent luật. Luật trùng chính sách tối ưu, RL đạt tối ưu ở 9 trên 10 seed, khoảng tin cậy của hiệu có chứa 0.

Thứ hai, kết luận đổi khi môi trường đổi. Ở mức giảm 5 phần trăm, RL thích ứng và vượt luật. Ở mức 10 phần trăm, RL thua luật vì trạng thái không thể hiện hậu quả trễ.

Thứ ba, siêu tham số ảnh hưởng mạnh. Alpha bằng 0,5 và epsilon cố định 0,30 đều cho chính sách kém hơn nhiều. Epsilon bằng 0 khóa agent vào lựa chọn ngẫu nhiên ban đầu.

Thứ tư, agent đề xuất đạt regret lớn nhất 0,085 trên sáu kịch bản, so với khoảng 2,3 của hai agent còn lại.

Bài học phương pháp luận quan trọng nhất: **không thể chọn agent dựa vào một lần chạy có phần thưởng cao nhất**. Cần một quy trình đánh giá tách biệt, có mốc tối ưu để đối chiếu, lặp nhiều seed, và thử trên nhiều điều kiện môi trường.

## 17. Câu trả lời nhanh nếu bị hỏi

### Vì sao agent luật lại tối ưu trong môi trường gốc?

Vì ba trong bốn trạng thái thì không giảm giá có lợi hơn, và trạng thái còn lại không bao giờ xảy ra khi đối thủ là agent luật. Agent luật không bao giờ giảm giá trong tình huống này, nên nó trùng với chính sách tối ưu. Em đã kiểm chứng bằng cách liệt kê toàn bộ 16 chính sách tất định.

### Vì sao RL thua ở mức giảm 10 phần trăm?

Vì giảm giá ở mức này có lợi trước mắt nhưng kích hoạt đối thủ giảm giá đáp trả ở bước sau. Hậu quả đó xảy ra ở một trạng thái mà agent không phân biệt được, do trạng thái `(d, c)` không chứa hành động trước của chính nó. Agent dựa trên mô hình sửa được lỗi này.

### Regret là gì và vì sao dùng nó?

Regret bằng mức tối ưu trừ giá trị đạt được. Em dùng nó vì so sánh hai con số thưởng với nhau không cho biết cả hai còn cách tối ưu bao xa. Có regret, ta biết agent còn bao nhiêu dư địa cải thiện.

### Vì sao cần 10 seed?

Vì một lần chạy không đủ. Ở seed 33, RL thua 2,62; nếu chỉ chạy seed đó, ta sẽ kết luận RL kém hẳn. Ở seed 11 thì ngược lại. Mười seed cho phép tính khoảng tin cậy và nói về độ ổn định.

### Số ngẫu nhiên chung giúp gì?

Nó loại bỏ nhiễu do chuỗi nhu cầu khác nhau. Nhờ vậy, ở 9 trên 10 seed, hai agent có cùng hành vi cho kết quả trùng đến từng đơn vị, nên mọi chênh lệch quan sát được đều đến từ chính sách.

### Vì sao giá trị Q lên tới 200-270?

Vì Q ước lượng tổng thưởng chiết khấu của nhiều bước còn lại, không phải thưởng một bước. Với gamma bằng 0,9 và thưởng khoảng 35 mỗi bước, giá trị ở đầu episode gần 350, còn ở bước cuối chỉ khoảng 35.

### Vì sao không dùng dataset từ Kaggle?

Vì Q-learning phải tự thử hành động rồi nhận phản hồi. Dataset tĩnh chỉ chứa những gì người bán thật đã làm, không có kết quả của hành động khác, cũng không có phản ứng của đối thủ. Đề bài cũng yêu cầu thiết kế một mô phỏng nhiều agent với lớp environment thực hiện chuyển trạng thái. Nếu muốn dùng dữ liệu thật, cách hợp lý là ước lượng các hệ số 0,15 và 0,10 từ dữ liệu rồi đưa vào mô phỏng.

### Agent dựa trên luật có phải là AI không?

Nó là agent, và thuộc nhánh AI cổ điển dựa trên luật, nhưng nó **không phải học máy** vì không học từ dữ liệu. Đây chính là điểm cho thấy khái niệm agent rộng hơn khái niệm học máy.

### Hạn chế lớn nhất của thí nghiệm là gì?

Mô hình xác suất mua chỉ là công thức minh họa, không được kiểm chứng bằng dữ liệu thật. Ngoài ra agent cải tiến được thiết kế sau khi đã quan sát điểm yếu trên chính tập kịch bản dùng để kiểm tra nó, nên có nguy cơ khớp quá mức.

## 18. Thứ tự demo khi thuyết trình

1. Mở `agent02_experiment.py`, chỉ 19 section đánh số trong comment để nói về cấu trúc.
2. Mở section 2 và 3, chỉ hàm `conversion_probability` và lớp `ECommerceEnvironment`.
3. Chiếu bảng lợi nhuận kỳ vọng một bước, nhấn mạnh tối ưu là không giảm giá, bằng 35,00.
4. Mở section 4 và 5, đặt cạnh nhau `QLearningAgent.learn` và `RuleBasedAgent.act` để thấy một bên học, một bên không.
5. Mở section 6, giới thiệu `exact_evaluation` và `best_response` như đóng góp riêng.
6. Mở section 8, nhấn mạnh ba điều kiện của đánh giá tách biệt.
7. Chiếu `episode_rewards.png` rồi `policy_value_during_training.png`, nói chính sách dao động chứ không hội tụ.
8. Chiếu bảng thí nghiệm C và D, nhấn mạnh cột "số seed tối ưu".
9. Chiếu `multi_seed_summary.csv`, nhấn mạnh 9 trên 10 seed trùng khớp và khoảng tin cậy chứa 0.
10. Chiếu `simultaneous_learning.png` cho câu 8.
11. Chiếu `scenario_comparison.png`, nhấn mạnh hai cột giảm 5% và giảm 10%.
12. Kết lại bằng bảng regret và bài học phương pháp luận.
