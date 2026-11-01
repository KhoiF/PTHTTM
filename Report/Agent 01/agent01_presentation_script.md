# Kịch bản thuyết trình chi tiết notebook Q-learning mê cung - Agent 01

Thời lượng gợi ý: 10-12 phút
Notebook: `src/Agent 01/reinforcement_learning.ipynb`
Báo cáo: `Report/Agent 01/agent01_CT_03_khoipv.447.pdf`
Chủ đề: Khái niệm AI Agent, học tăng cường và thực hành Q-learning cho robot tìm đường trong mê cung 10x10.

## 0. Cách dùng file này

File này được viết theo kiểu kịch bản nói. Khi thuyết trình, có thể đọc gần như trực tiếp, nhưng nên nói tự nhiên hơn bằng cách:

- Mở notebook `reinforcement_learning.ipynb` song song.
- Notebook có 21 cell, trong đó 7 cell code nằm ở phần "Ví dụ thực hành", đi theo đúng bảy bước.
- Khi đến phần kết quả, dừng lâu hơn ở hình mê cung và đồ thị phần thưởng.
- Nếu thiếu thời gian, ưu tiên trình bày: khái niệm agent, công thức Q-learning, vòng huấn luyện và hai hình kết quả.

## 1. Mở đầu

Kính chào thầy và các bạn.
Trong phần này em xin trình bày notebook Q-learning của Agent 01. Notebook gồm hai phần: phần lý thuyết diễn giải các khái niệm của học tăng cường, và phần thực hành cài đặt Q-learning cho một robot tìm đường trong mê cung.

Mục tiêu của notebook không phải là giải mê cung bằng mọi giá. Nếu chỉ cần tìm đường đi, thuật toán BFS hoặc A\* đã làm được ngay và còn đảm bảo tối ưu. Mục tiêu ở đây là cho thấy một agent có thể **tự học** cách đi mà **không ai nói trước cho nó đường nào đúng**. Nó chỉ nhận một con số phần thưởng sau mỗi bước và phải tự suy ra hành vi tốt.

Đây là điểm khác biệt quan trọng so với học có giám sát mà chúng ta đã làm ở các assignment trước.

## 2. Agent là gì và khác gì học máy thông thường

Trước khi vào code, em xin làm rõ khái niệm agent, vì đây là trọng tâm của môn học.

**Agent** là một hệ thống quan sát môi trường, ra quyết định, rồi hành động lên môi trường đó để theo đuổi một mục tiêu, và lặp lại vòng này.

Ba dấu hiệu nhận biết một agent:

- Có mục tiêu rõ ràng.
- Hành động của nó làm thay đổi môi trường, chứ không chỉ in ra một con số.
- Có vòng lặp phản hồi: hành động bây giờ ảnh hưởng đến những gì nó quan sát ở bước sau.

So sánh với CNN phân loại ảnh ở Assignment 04 và 05:

| Tiêu chí | CNN phân loại ảnh | Agent Q-learning trong notebook |
|---|---|---|
| Dữ liệu đầu vào | Tập ảnh cho sẵn, cố định | Chuỗi trạng thái do chính agent đi qua |
| Đầu ra | Nhãn dự đoán | Hành động di chuyển, làm đổi vị trí |
| Tín hiệu học | Nhãn đúng cho từng ảnh | Một con số phần thưởng, không nói hướng nào mới đúng |
| Quan hệ thời gian | Mỗi ảnh độc lập | Tuần tự, bước này ảnh hưởng bước sau |

Điểm cần nhấn mạnh: học tăng cường là **một nhánh của học máy**, nên agent Q-learning này vừa là agent, vừa là một mô hình học máy. Hai khái niệm không mâu thuẫn, chỉ là chúng trả lời hai câu hỏi khác nhau. Học máy trả lời "học một hàm số bằng cách nào", còn agent trả lời "tổ chức một hệ thống biết hành động ra sao".

## 3. Năm khái niệm nền tảng

Notebook trình bày năm khái niệm, em xin ánh xạ trực tiếp sang ví dụ mê cung:

| Thuật ngữ | Ý nghĩa | Trong mê cung |
|---|---|---|
| Agent | Chủ thể ra quyết định và học từ kết quả | Robot tìm đường |
| Environment | Hệ thống mà agent tương tác | Mê cung có ô trống và vật cản |
| State | Tình huống hiện tại | Tọa độ `(hàng, cột)` |
| Action | Lựa chọn có thể thực hiện | Đi trái, phải, lên, xuống |
| Reward | Phản hồi dạng số cho hành động | +50 tới đích, -10 đụng vật cản, -1 mỗi bước |

Ngoài ra còn bốn thành phần của một bài toán RL: policy là quy tắc ánh xạ trạng thái sang hành động; reward signal diễn tả mục tiêu; value function ước lượng lợi ích dài hạn; và model dự đoán trạng thái kế tiếp. Trong notebook này, agent **không dùng model** — nó học thẳng giá trị của từng cặp trạng thái và hành động từ kinh nghiệm. Đây gọi là phương pháp model-free.

## 4. Bước 1 — Tạo mê cung và xác định điểm đầu, đích

Bây giờ em xin đi vào code.

Cell đầu tiên nạp thư viện và khai báo mê cung bằng một mảng NumPy kích thước 10x10. Giá trị `0` là ô đi được, giá trị `1` là vật cản.

```python
maze = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    ...
])

start = (0, 0)
goal = (9, 9)
```

Robot xuất phát tại góc trên bên trái `(0, 0)` và cần đến góc dưới bên phải `(9, 9)`.

Cell này chỉ khai báo dữ liệu nên không in ra output.

Điểm cần lưu ý: mê cung này không phải là lưới trống. Nhìn vào mảng, ta thấy chỉ có một số hành lang hẹp nối từ trên xuống dưới. Vì vậy robot không thể đi thẳng theo đường chéo, mà phải học cách đi vòng theo các hành lang đó.

## 5. Bước 2 — Đặt tham số và khởi tạo bảng Q

```python
num_episodes = 5000
alpha = 0.1
gamma = 0.9
epsilon = 0.5

reward_fire = -10
reward_goal = 50
reward_step = -1

actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
Q = np.zeros(maze.shape + (len(actions),))
```

Em xin giải thích từng tham số:

- `num_episodes = 5000`: số lượt agent thử đi qua mê cung.
- `alpha = 0.1`: tốc độ học, quyết định mức tin vào quan sát mới.
- `gamma = 0.9`: hệ số chiết khấu, quyết định mức coi trọng phần thưởng tương lai.
- `epsilon = 0.5`: ban đầu một nửa số lần agent sẽ đi ngẫu nhiên để khám phá.

Danh sách `actions` là bốn vector dịch chuyển, theo thứ tự trái, phải, lên, xuống.

Bảng Q có shape `(10, 10, 4)`, tức 400 giá trị. Mỗi ô lưu bốn giá trị tương ứng bốn hướng. Toàn bộ khởi tạo bằng `0`, nghĩa là ban đầu agent không biết gì cả.

Về cách đặt phần thưởng, em muốn nhấn mạnh ý nghĩa của `reward_step = -1`. Mỗi bước hợp lệ đều bị trừ một điểm. Nếu không có hình phạt này, agent đi đường dài hay đường ngắn đều nhận được +50 như nhau, nên nó sẽ không có lý do gì để tìm đường ngắn. Chi phí mỗi bước chính là thứ khuyến khích agent đi nhanh tới đích.

## 6. Bước 3 — Kiểm tra ô hợp lệ và chọn hành động

```python
def is_valid(pos):
    row, col = pos
    return (
        0 <= row < maze.shape[0]
        and 0 <= col < maze.shape[1]
        and maze[row, col] == 0
    )


def choose_action(state):
    if np.random.random() < epsilon:
        return np.random.randint(len(actions))
    return np.argmax(Q[state])
```

Hàm `is_valid` loại hai trường hợp: ô nằm ngoài biên mê cung, và ô là vật cản.

Hàm `choose_action` cài đặt chiến lược **ε-greedy**. Với xác suất `epsilon`, agent chọn một hướng ngẫu nhiên trong bốn hướng, đây là **khám phá**. Với xác suất còn lại, agent chọn hướng có giá trị Q cao nhất tại ô hiện tại, đây là **khai thác**.

Vì sao cần khám phá? Vì bảng Q ban đầu toàn số `0`. Nếu agent luôn chọn giá trị cao nhất, `np.argmax` sẽ luôn trả về hướng đầu tiên, và agent sẽ lặp đi lặp lại đúng một hành vi mà không bao giờ biết các hướng khác tốt hay xấu. Khám phá là thứ phá vỡ bế tắc này.

## 7. Bước 4 — Vòng huấn luyện Q-learning

Đây là phần trọng tâm của notebook. Công thức cập nhật:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\left[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\right]$$

```python
rewards_all_episodes = []

for episode in range(num_episodes):
    state = start
    total_rewards = 0
    done = False

    while not done:
        action_index = choose_action(state)
        move = actions[action_index]
        next_state = (state[0] + move[0], state[1] + move[1])

        if not is_valid(next_state):
            reward = reward_fire
            done = True
            next_max = 0
        elif next_state == goal:
            reward = reward_goal
            done = True
            next_max = 0
        else:
            reward = reward_step
            next_max = np.max(Q[next_state])

        old_value = Q[state][action_index]
        Q[state][action_index] = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        if is_valid(next_state):
            state = next_state
        total_rewards += reward

    epsilon = max(0.01, epsilon * 0.995)
    rewards_all_episodes.append(total_rewards)
```

Em xin giải thích ý nghĩa toán học của công thức trước, rồi mới nói về code.

Phần trong ngoặc vuông gọi là **sai số TD**. Nó là hiệu giữa "mục tiêu" và "giá trị đang tin". Mục tiêu bằng phần thưởng nhận ngay cộng với giá trị tốt nhất có thể đạt được từ ô kế tiếp, nhân hệ số chiết khấu. Nếu sai số dương, nghĩa là hành động này tốt hơn agent tưởng, nên giá trị Q được nâng lên. Nếu âm thì hạ xuống.

Hệ số `alpha` quyết định dịch bao nhiêu phần trăm quãng đường về phía mục tiêu. Với `alpha = 0.1`, mỗi lần chỉ dịch 10 phần trăm, nên một quan sát nhiễu không làm đảo lộn hiểu biết của agent.

Trong code có ba nhánh xử lý:

- Nhánh một, nếu bước tiếp theo không hợp lệ: nhận -10, kết thúc episode, và quan trọng là `next_max = 0` vì không còn tương lai nào nữa.
- Nhánh hai, nếu tới đích: nhận +50, kết thúc episode, `next_max = 0`.
- Nhánh ba, bước hợp lệ bình thường: nhận -1 và `next_max` lấy giá trị lớn nhất tại ô kế tiếp.

Một chi tiết tinh tế cần nói: khi đụng vật cản, dòng `if is_valid(next_state): state = next_state` không chạy, nên agent **không di chuyển vào ô cấm**. Nhưng bảng Q vẫn được cập nhật với phần thưởng -10 cho hướng đó tại ô hiện tại. Nói cách khác, agent học được rằng "đi hướng này từ ô này là xấu" mà không cần thực sự bước vào vật cản.

Cuối mỗi episode, `epsilon` được nhân với 0.995 nhưng không xuống dưới 0.01. Ban đầu agent khám phá nhiều để thu thập thông tin, về sau giảm dần để khai thác những gì đã học. Với hệ số 0.995, epsilon giảm từ 0.5 xuống sàn 0.01 sau khoảng 780 episode.

Cell này không in output, nhưng nó là nơi toàn bộ việc học diễn ra.

## 8. Bước 5 — Trích xuất đường đi từ bảng Q

```python
def get_optimal_path(Q, start, goal, actions, maze, max_steps=200):
    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break
        visited.add(state)
        best_action = None
        best_value = -float('inf')

        for index, move in enumerate(actions):
            next_state = (state[0] + move[0], state[1] + move[1])
            if (... hợp lệ và chưa thăm ... and Q[state][index] > best_value):
                best_value = Q[state][index]
                best_action = index

        if best_action is None:
            break
        ...
```

Sau khi huấn luyện xong, bảng Q chứa đánh giá của agent về từng hướng tại từng ô. Hàm này đi từ điểm xuất phát, mỗi bước chọn hướng có Q cao nhất trong số các hướng dẫn tới ô hợp lệ và **chưa thăm**.

Hai cơ chế an toàn:

- Tập `visited` ngăn agent đi vòng tròn.
- `max_steps=200` là giới hạn cứng, tránh vòng lặp vô hạn.

Output đã lưu trong notebook:

```
Số ô trên đường đi: 25
Đã tới đích: True
```

Nghĩa là agent đi qua 25 ô và tới được đích. Điểm cần nói thật: đây là một phép đi tham lam theo bảng Q, **không đảm bảo** là đường ngắn nhất tuyệt đối. Vì việc huấn luyện có yếu tố ngẫu nhiên, chạy lại notebook có thể cho số ô khác.

## 9. Bước 6 — Hiển thị mê cung và đường đi

Hình này đánh dấu điểm xuất phát bằng hình tròn, đích bằng ngôi sao, và vẽ đường mà agent chọn theo bảng Q.

Khi chiếu hình, có thể nói:

Ta thấy đường đi bám theo các hành lang hẹp của mê cung. Agent không đi thẳng theo đường chéo được, vì phần lớn ô ở giữa là vật cản. Nó học được rằng phải đi xuống theo cột bên trái, rẽ ngang ở các hàng trống, rồi mới xuống được tới đích.

Điều đáng chú ý là agent không hề được cho bản đồ. Nó chỉ biết sau khi đâm vào vật cản thì bị trừ 10 điểm. Qua 5000 lượt thử, nó dần xây được một bảng giá trị phản ánh đúng cấu trúc mê cung.

## 10. Bước 7 — Đồ thị phần thưởng theo episode

Đồ thị này vẽ tổng phần thưởng của từng episode trong 5000 lượt.

Khi chiếu hình, có thể nói:

Giai đoạn đầu phần thưởng chủ yếu âm, vì agent đang đi ngẫu nhiên nhiều và thường đâm vào vật cản ngay sau vài bước, nhận -10.

Về sau xuất hiện nhiều lượt có phần thưởng cao hơn, cho thấy agent đã học được cách tránh vật cản và tới đích.

Tuy nhiên đường biểu diễn **vẫn dao động** đến cuối. Lý do là epsilon không bao giờ bằng 0, luôn còn sàn 0.01, nên agent vẫn thỉnh thoảng đi ngẫu nhiên. Một bước ngẫu nhiên sai hướng có thể làm cả episode kết thúc sớm với -10.

Em muốn nhấn mạnh một điểm phương pháp luận: **phần thưởng trong lúc huấn luyện không phải là thước đo chất lượng chính sách**. Nó bao gồm cả chi phí khám phá. Muốn đánh giá đúng chính sách đã học, phải tắt khám phá và chạy đánh giá riêng. Đây chính là điều em đã làm ở Agent 02.

## 11. Hai khái niệm lý thuyết cần làm rõ

Notebook có hai phần lý thuyết dễ gây nhầm, em xin nói rõ.

**Tăng cường dương và tăng cường âm.** Tăng cường âm nói về việc *loại bỏ một điều kiện bất lợi* sau hành vi, làm hành vi đó tăng lên. Nó **không đồng nghĩa** với việc cộng một phần thưởng âm để phạt. Trong mê cung, `reward_fire = -10` là một hình phạt, không phải tăng cường âm theo nghĩa gốc.

**Online RL và offline RL.** Notebook này là online RL: agent tự tương tác với môi trường trong lúc học. Offline RL thì agent chỉ học từ một tập dữ liệu tương tác đã ghi lại trước, không tương tác mới. Phân biệt này quan trọng khi ai đó hỏi "sao không dùng dataset có sẵn".

## 12. Kết luận chính

Từ notebook Agent 01, em rút ra bốn kết luận.

Thứ nhất, agent Q-learning học được đường đi tới đích chỉ từ tín hiệu phần thưởng, không cần biết trước bản đồ và không cần nhãn đáp án.

Thứ hai, cách đặt hàm thưởng quyết định hành vi học được. Chi phí -1 mỗi bước là thứ khiến agent ưu tiên đường ngắn.

Thứ ba, cân bằng khám phá và khai thác là bắt buộc. Nếu epsilon bằng 0 ngay từ đầu, agent sẽ kẹt ở hành động được thử đầu tiên.

Thứ tư, phần thưởng huấn luyện dao động không có nghĩa là agent học kém. Cần tách riêng giai đoạn đánh giá để kết luận về chất lượng chính sách.

Notebook này là nền tảng cho Agent 02, nơi em mở rộng sang môi trường có hai agent cùng tương tác.

## 13. Câu trả lời nhanh nếu bị hỏi

### Vì sao không dùng BFS hoặc A\* cho mê cung?

Vì mục tiêu là minh họa cách agent học từ phản hồi, không phải giải mê cung. BFS cần biết trước toàn bộ bản đồ. Q-learning chỉ cần tín hiệu phần thưởng, nên áp dụng được cả khi môi trường chưa biết hoặc có yếu tố ngẫu nhiên.

### Bảng Q có bao nhiêu giá trị?

400 giá trị, từ 10 nhân 10 ô nhân 4 hướng. Nhiều ô là vật cản nên giá trị tại đó không bao giờ được dùng.

### Vì sao `next_max = 0` khi kết thúc episode?

Vì sau khi episode kết thúc, không còn trạng thái tương lai nào để lấy giá trị. Nếu vẫn cộng `gamma * max Q(s')`, agent sẽ tính thêm phần thưởng tương lai không tồn tại.

### Gamma bằng 0.9 có ý nghĩa gì?

Phần thưởng ở bước sau được tính với trọng số 0.9, bước sau nữa là 0.81, và cứ thế giảm dần. Gamma gần 1 làm agent nhìn xa hơn. Gamma bằng 0 làm agent chỉ quan tâm phần thưởng trước mắt, khi đó nó sẽ không bao giờ chịu trả giá -1 để đi tới đích.

### Q-learning khác SARSA ở chỗ nào?

Q-learning dùng `max` trên hành động kế tiếp, nên nó học giá trị của chính sách tham lam dù đang thực hiện hành động khám phá. Đây gọi là off-policy. SARSA dùng chính hành động thực sự được chọn ở bước sau, nên là on-policy.

### Vì sao cần tập `visited` khi trích đường đi?

Vì bảng Q có thể tạo ra một vòng lặp, ví dụ ô A chỉ sang ô B và ô B lại chỉ về ô A. Tập `visited` đảm bảo mỗi ô chỉ đi qua một lần, nên hàm luôn kết thúc.

### Đường đi 25 ô có phải ngắn nhất không?

Không đảm bảo. Đây là đường tham lam theo bảng Q sau huấn luyện, và quá trình huấn luyện có yếu tố ngẫu nhiên. Muốn biết đường ngắn nhất tuyệt đối thì phải chạy BFS để đối chiếu.

### Nếu tăng số episode thì kết quả tốt hơn không?

Có thể, nhưng không chắc chắn. Vì epsilon đã chạm sàn 0.01 từ khoảng episode 780, các episode về sau đóng góp ít thông tin mới. Muốn cải thiện thì nên điều chỉnh lịch giảm epsilon hoặc tốc độ học, chứ không chỉ tăng số lượt.

## 14. Thứ tự demo notebook khi thuyết trình

1. Mở phần lý thuyết, chỉ bảng năm khái niệm agent, environment, state, action, reward.
2. Mở Bước 1, chỉ mảng `maze` và nói về các hành lang hẹp.
3. Mở Bước 2, chỉ `Q = np.zeros(maze.shape + (4,))` và nhấn mạnh bảng Q ban đầu toàn số 0.
4. Mở Bước 3, chỉ hàm `choose_action` và giải thích ε-greedy.
5. Mở Bước 4, dừng lâu nhất ở đây, giải thích công thức cập nhật và ba nhánh phần thưởng.
6. Mở Bước 5, chỉ output `Số ô trên đường đi: 25`.
7. Mở Bước 6, chiếu hình mê cung và đường đi.
8. Mở Bước 7, chiếu đồ thị phần thưởng và giải thích vì sao vẫn dao động.
9. Kết lại bằng ý: cần đánh giá tách biệt mới biết chính sách tốt hay không, dẫn sang Agent 02.
