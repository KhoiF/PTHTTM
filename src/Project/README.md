# Hệ đa agent dự đoán nguy cơ bệnh tiểu đường

Đồ án môn Phát triển hệ thống thông minh — Phan Văn Khôi (B23DCCN447).

Hệ thống hiện thực sơ đồ trên bảng: nhánh **DB → ML/DL → Prediction/Classification** và nhánh
**PDF/web → KB → RAG → Agent**, hợp lại thành một câu trả lời có trích nguồn.

| Trên bảng | Trong hệ thống | Ở đâu |
|---|---|---|
| DB | `data/diabetes_prediction_dataset.csv` (100.000 dòng; còn 96.128 sau khi loại trùng) | `ml/preprocess.py` |
| ML/DL | Logistic Regression, Random Forest, XGBoost, MLP PyTorch; 2 bộ đặc trưng *full* / *screening* | `ml/train.ipynb` |
| Prediction / Classification | Xác suất đã hiệu chỉnh + nhãn + SHAP + what-if | `ml/predictor.py` |
| PDF/web | 20 trang của Nhà thuốc Long Châu (bệnh, bài viết, thuốc) | `kb/seed_urls.csv` |
| KB | **Knowledge Graph** trên Neo4j (ontology đóng, mọi quan hệ có câu trích + URL) | `kb/build_kg.ipynb` |
| RAG | **GraphRAG**: tìm thực thể → duyệt đồ thị → đoạn văn gốc | `tools/kg_tools.py` |
| Agent | 6 agent điều phối bằng LangGraph, LLM local `qwen3.5:9b` qua Ollama | `agents/` |
| Hình tròn "+" | Câu trả lời tiếng Việt, mỗi ý có trích dẫn [n], đã qua kiểm duyệt | `app/app.py` |

## Kiến trúc đa agent

```
người dùng → Supervisor ─ask──────────────────────────────────────┐
                 └─assess/both→ Intake ─thiếu dữ liệu→ hỏi lại người dùng
                                  ↑  └→ Risk ─vùng xám, thiếu xét nghiệm→ Intake
                                  └──────┘  └→ Knowledge ⟲ chưa đủ bằng chứng
                                                 └→ Advisor ⇄ Critic → Finalize → câu trả lời + nguồn
```

| Agent | Việc | Công cụ / quyết định |
|---|---|---|
| Supervisor | Phân loại ý định: `assess` / `ask` / `both` | Câu hỏi kiến thức đi thẳng sang Knowledge |
| Intake | Gom hồ sơ từ form + tin nhắn (tự tính BMI, đổi mmol/L → mg/dL), kiểm tra miền giá trị | Thiếu trường bắt buộc → hỏi lại người dùng |
| Risk | Gọi mô hình ML | `predict_risk`, `explain_risk`, `what_if`; thiếu xét nghiệm → mô hình screening; vùng xám → nhờ Intake tìm thêm |
| Knowledge | Tìm bằng chứng trong KG; đối chiếu chỉ số xét nghiệm với ngưỡng chẩn đoán | `link_entities`, `neighbors`, `drug_info`, `explain_feature`, `search_chunks`; chưa đủ → tự lặp với truy vấn khác (≤ 3 vòng) |
| Advisor | Viết câu trả lời có trích dẫn [n] | Chỉ dùng số liệu của mô hình và bằng chứng |
| Critic | Kiểm duyệt | Code: số liệu, trích dẫn, liều thuốc, câu khẳng định bệnh, mâu thuẫn ngưỡng, giả định "đo lúc đói". LLM: khuyên gặp bác sĩ, khẳng định thiếu nguồn. Không đạt → trả Advisor (≤ 2 vòng) |

## Cấu trúc thư mục

Logic dùng lại nằm trong file `.py` (được import ở nơi khác và có test); các bước chạy tuần tự, cần giải thích
thì viết thành notebook.

```
config.py               cấu hình chung (đọc .env)
nbstyle.py              định dạng bảng trong notebook (màu cố định, đọc rõ ở giao diện sáng/tối)
ml/        eda.ipynb, train.ipynb       ← notebook
           preprocess.py, evaluation.py, mlp.py, predictor.py, train_mlp.py
kb/        build_kg.ipynb               ← notebook
           seed_urls.csv, seed_urls_full.csv, open_urls.py, import_html.py (thu thập trang)
           parse.py, extract.py, schema.py, textutil.py, normalize.py, feature_bridge.py, load_neo4j.py
tools/     ml_tools.py, kg_tools.py     công cụ cho agent
agents/    demo_agents.ipynb            ← notebook
           llm.py, state.py, prompts.py, supervisor.py, intake.py, risk.py, knowledge.py, advisor.py,
           critic.py, graph.py, single_agent.py
app/       app.py                       giao diện Gradio
eval/      evaluation.ipynb             ← notebook
           scenarios.jsonl (24 kịch bản), metrics.py
tests/     44 test (pytest)
```

## Cài đặt (một lần)

```bash
cd "/Users/macos/Docs/Kì 1 năm 4/PTHTTM/src/Project"
conda activate ISD
pip install -r requirements.txt
brew install ollama libomp          # libomp cần cho XGBoost trên macOS
brew services start ollama
ollama pull qwen3.5:9b              # ~7,6 GB
ollama pull bge-m3                  # ~1,2 GB, embedding đa ngôn ngữ
neo4j start                         # Neo4j 2026.x đã cài qua Homebrew
cp .env.example .env                # rồi điền NEO4J_PASSWORD
```

## Chạy từng bước

**1. Mô hình ML.** Mở `ml/eda.ipynb` và `ml/train.ipynb` (kernel *Python 3 (ISD)*), chạy từ trên xuống, chưa tới 1 phút.

**2. Thu thập trang Long Châu.** Long Châu chặn request từ script (Cloudflare trả 403) dù robots.txt cho phép các mục
`/benh/`, `/bai-viet/`, `/thuoc/`, `/thanh-phan/`, nên trang được mở bằng Chrome và lưu thủ công:

```bash
python -m kb.open_urls        # mở 20 trang, mỗi lượt 10 tab; lưu bằng SingleFile "Save all tabs" hoặc Cmd+S
python -m kb.import_html      # nhập vào kb/raw_html/, báo trang còn thiếu hoặc bị thu gọn
```

Trang `/benh/` thu gọn nửa sau bài (Nguy cơ, Chẩn đoán & điều trị, Phòng ngừa) sau nút **"Xem thêm"**, và phần đó chỉ
được tải khi bấm nút. Phải bấm "Xem thêm" rồi mới lưu; trang lưu thiếu bị `import_html` báo `[thu gọn]` và
`python -m kb.open_urls --missing` sẽ mở lại.

`kb/seed_urls.csv` gồm 20 trang cốt lõi: đủ cho 8 đặc trưng của dataset và các câu hỏi trong `eval/scenarios.jsonl`.
Danh sách đầy đủ 65 trang nằm ở `kb/seed_urls_full.csv`; muốn KG lớn hơn thì chép thêm dòng từ file đó sang.

`kb/raw_html/` và `kb/processed/pages.jsonl` (toàn văn đã tách mục) không được commit vì là nội dung có bản quyền
của Long Châu; repo chỉ giữ dữ liệu đã trích (quan hệ kèm câu trích và URL nguồn).

**3. Dựng Knowledge Graph.** Mở `kb/build_kg.ipynb`. Lần đầu mất khoảng 45 phút ở bước LLM trích xuất; kết quả được cache trong
`kb/cache/` nên chạy lại rất nhanh. Xem thống kê KG đang có trong Neo4j: `python -m kb.load_neo4j --stats`.
Bản sao lưu toàn bộ KG (node, quan hệ, embedding, chỉ mục) là `kb/DB_diabet.cypher`; sinh lại bằng
`python -m kb.load_neo4j --export kb/DB_diabet.cypher`, nạp vào Neo4j khác bằng
`cypher-shell -u neo4j -p <mật khẩu> -f kb/DB_diabet.cypher` (khoảng 1 phút, không cần Ollama hay trang Long Châu).
Cuối notebook có file `kb/processed/precision_sample.csv` (50 quan hệ ngẫu nhiên) để chấm precision: cột `correct` là
1/0, cột `ly_do` ghi lý do.

**4. Kiểm thử.** `pytest` (cần Ollama + Neo4j cho phần test KG; test định tuyến dùng LLM giả lập).

**5. Thử hệ đa agent.** Mở `agents/demo_agents.ipynb`, hoặc dùng dòng lệnh:

```bash
python -m agents.graph --question "Triệu chứng của tiểu đường tuýp 2 là gì?"
python -m agents.graph --scenario full_kg_conflict
```

**6. Giao diện.** `python -m app.app` rồi mở [http://127.0.0.1:8010](http://127.0.0.1:8010). Bên trái là form hồ sơ
(ô số nhận cả dấu phẩy, vd. `24,5`), bên phải là khung chat; phần "Bên trong hệ thống" có 4 tab: dòng thời gian các
agent (cập nhật trực tiếp), thẻ nguy cơ + SHAP, đồ thị con của KG tô màu theo loại quan hệ, và bằng chứng kèm bảng
đối chiếu ngưỡng. Giao diện theo chế độ sáng/tối của máy; ép chế độ bằng `?__theme=light` hoặc `?__theme=dark`.

**7. Đánh giá.** `python -m eval.metrics` (khoảng 1 giờ, chạy dưới nền được) rồi mở `eval/evaluation.ipynb`.

## Kết quả mô hình ML (tập test, 14.420 dòng)

| Mô hình | Full: PR-AUC / F1 | Screening: PR-AUC / F1 |
|---|---|---|
| Logistic Regression | 0,801 / 0,721 | 0,289 / 0,365 |
| Random Forest | 0,863 / 0,807 | 0,253 / 0,331 |
| **XGBoost** (dùng cho agent) | **0,875 / 0,807** | 0,293 / 0,365 |
| MLP PyTorch | 0,862 / 0,801 | 0,305 / 0,379 |
| Assignment 03 (Decision Tree, chưa loại trùng) | 0,851 / 0,804 | — |

Hiệu chỉnh isotonic giảm Brier của XGBoost full từ 0,058 xuống 0,024, nên xác suất mà agent đọc cho người dùng đúng nghĩa xác suất.

## Kết quả Knowledge Graph (20 trang Long Châu)

| | Số lượng |
|---|---|
| Trang / đoạn văn (chunk) | 20 trang (5 bệnh, 12 bài viết, 3 thuốc) → 312 chunk |
| Thực thể | 816: Disease 188, Lifestyle 172, Symptom 147, RiskFactor 112, SideEffect 91, Drug 46, Test 36, DrugClass 16, Feature 8 |
| Quan hệ ngữ nghĩa | 1.052: HAS_RISK_FACTOR 218, HAS_SYMPTOM 170, TREATED_BY 166, PREVENTED_BY 139, HAS_COMPLICATION 106, HAS_SIDE_EFFECT 95, DIAGNOSED_BY 79 (38 có ngưỡng số), CONTRAINDICATED_IN 43, BELONGS_TO 4 |
| Bản ghi LLM trích được | 1.467, giữ 1.202 (81,9%); trích theo mục có hướng dẫn 100%, tổng quát 69%, ngưỡng 78% |
| Precision (50 quan hệ ngẫu nhiên) | **0,94** (47/50). Claude chấm nháp kèm lý do, người viết kiểm tra lại (`kb/processed/precision_sample.csv`) |

Ba lỗi còn lại đều là lỗi phân loại của model 9B: một loại bệnh tim bị ghi là yếu tố nguy cơ, "Huyết áp" (thay vì tăng
huyết áp) bị ghi là biến chứng, tổn thương cầu thận bị ghi là triệu chứng. Mẫu này cũng làm lộ một lỗi có hệ thống ở
ngưỡng chẩn đoán (thiếu cận trên của "≥100 và <126 mg/dL", ghi "dưới 140" thành `<=`); lỗi được sửa bằng code cho toàn bộ
38 ngưỡng (`kb/extract.py: refine_threshold`). Trước khi sửa precision là 0,90, nên con số 0,94 hơi lạc quan vì lỗi được
phát hiện chính trên mẫu này.

54 cặp thực thể gần trùng (cosine 0,92–0,95) được duyệt trong `kb/merge_decisions.csv`: 48 cặp gộp (vd. "Bỏ hút thuốc" /
"Ngừng hút thuốc lá"), 6 cặp giữ riêng vì khác nghĩa (vd. "Tổn thương cầu thận" / "Tổn thương thận"). Bản duyệt hiện tại do
Claude làm nháp; sửa cột `merge` rồi chạy lại `kb/build_kg.ipynb` (mọi kết quả LLM đã cache, chỉ mất khoảng 2 phút).

## Kết quả đánh giá hệ thống (24 kịch bản × 4 cấu hình)

Chạy bằng `python -m eval.metrics`, phân tích trong `eval/evaluation.ipynb`. Mọi chỉ số tính bằng code, không nhờ LLM chấm.

| Chỉ số | A. Đầy đủ | B. Bỏ Critic | C. Vector RAG thay KG | D. Một agent |
|---|---|---|---|---|
| Đúng tuyến | 100% | 100% | 100% | — (không có định tuyến) |
| Số liệu trung thực | 100% | 100% | 100% | 58,3% |
| Trích dẫn hợp lệ | 100% | 95,5% | 100% | 12,5% |
| Nêu mâu thuẫn ngưỡng (HbA1c đạt ngưỡng, mô hình báo thấp) | 100% | 100% | 100% | 100% |
| Vi phạm liều thuốc / chẩn đoán | 0 / 0 | 0 / 0 | 0 / 0 | **1** / 0 |
| Bằng chứng trung bình mỗi câu | 8,5 | 8,5 | 5,7 | 1,6 |
| Lượt LLM / thời gian trung bình | 8,5 / 53s | 7,0 / 36s | 7,8 / 37s | 1,7 / 15s |

- **Đa agent hơn hẳn một agent.** Với model 9B, một agent cầm cả 6 công cụ gần như không tra cứu (1,6 bằng chứng mỗi câu),
  hầu như không trích nguồn, 42% câu trả lời có con số tự bịa, và có một lần nêu liều thuốc ("500mg đến 2000mg/ngày")
  khi người dùng đòi kê đơn. Chia vai trò, mỗi agent 3–5 công cụ: 100% ở mọi chỉ số và 0 vi phạm, đổi lại chậm hơn khoảng
  3,5 lần.
- **Critic** bác bản nháp ở 7/24 kịch bản, nâng tỉ lệ trích dẫn hợp lệ từ 95,5% lên 100%, tốn thêm khoảng 17 giây mỗi câu.
- **KG so với vector RAG:** ngang nhau trên các chỉ số tự động (đã gần trần với 24 kịch bản). KG cho nhiều bằng chứng hơn
  mỗi câu (8,5 so với 5,7) và đối chiếu chỉ số với ngưỡng bằng code; nhờ đó Critic có thêm luật để kiểm, bác bản nháp
  nhiều hơn (7 so với 3 lần) nên A chậm hơn C. Cần thêm kịch bản sát ngưỡng và chấm tay chất lượng giải thích mới đủ cơ sở
  kết luận KG hơn RAG.
- **Định tuyến động:** cấu hình A đi qua 8 đường khác nhau. Các vòng quay lui đều xuất hiện: Critic → Advisor (7 kịch bản),
  Risk → Intake (2), Intake → Finalize để hỏi lại (2), Knowledge → Knowledge (1).

## Ghi chú kỹ thuật

- **Mâu thuẫn dữ liệu và tri thức.** Trong dataset, HbA1c < 5,7% có 0% mắc bệnh, ≥ 7% có 100%, nhưng nhóm 6,5–7,0% chỉ
  11,5% được gán nhãn mắc bệnh, dù theo Long Châu HbA1c ≥ 6,5% đã là ngưỡng chẩn đoán. Knowledge Agent đối chiếu chỉ số
  với ngưỡng trong KG; Critic buộc câu trả lời phải nêu điều này kể cả khi mô hình báo nguy cơ thấp.
- **Hai bản OpenMP.** PyTorch và XGBoost mỗi bên mang một `libomp.dylib`; nạp cả hai vào một tiến trình gây deadlock trên
  macOS. Vì vậy MLP được huấn luyện ở tiến trình riêng (`ml/train_mlp.py`).
- **Structured output.** Ollama chưa hỗ trợ `format=` (structured output) cho `qwen3.5:9b`, còn tool calling thì ổn định,
  nên LLM nộp kết quả có cấu trúc qua một tool `submit_*` có schema Pydantic (`agents/llm.py`).
- **Tốc độ.** Model 9B sinh khoảng 20 token/giây nên mỗi token đều đáng kể. Knowledge và Risk chỉ cần kết quả gọi
  công cụ, nên `LLM.run_tools(need_text=False)` ngắt ngay khi model bắt đầu viết đoạn tóm tắt (tiết kiệm 5–15 giây mỗi
  agent). Trích dẫn giữ chỗ `[n]` được xóa bằng code thay vì để Critic bắt viết lại; Critic nhận tên đồng nghĩa của xét
  nghiệm (đường huyết / glucose). Advisor gửi chữ dần lên giao diện qua luồng `custom` của LangGraph, và app nạp sẵn
  model khi khởi động. Kết quả: câu hỏi kiến thức khoảng 22–26 giây (chữ đầu tiên hiện sau khoảng 8–12 giây), đánh giá có
  xét nghiệm khoảng 40–55 giây (trước đây 70–95 giây); trung bình cấu hình A giảm từ 60s xuống 53s mỗi câu.
- **Thinking tắt mặc định.** Bật thinking làm Advisor sinh ~5.200 token (154s) so với ~310 token (11s) khi tắt. Bật lại
  bằng `LLM_THINK=1`.
- **An toàn.** KG không lưu mục "Liều dùng" của trang thuốc; Critic chặn mọi con số kèm mg/viên/lần dùng và mọi câu
  khẳng định người dùng đã mắc bệnh. Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.
- **Đường huyết của dataset.** Cột `blood_glucose_level` không ghi đo lúc đói hay bất kỳ, và không phải số đo 2 giờ sau
  nghiệm pháp dung nạp glucose. Vì vậy `explain_feature` chỉ so với ngưỡng lúc đói / bất kỳ (bỏ ngưỡng dung nạp, sau ăn),
  và Critic bắt câu trả lời nói rõ "nếu đo lúc đói…" khi người dùng không nói thời điểm đo.
- **Neo4j dùng chung.** Mọi node của đồ án mang nhãn `DKG`; nạp lại KG chỉ xóa các node này. Chỉ mục fulltext/vector đặt
  trên nhãn riêng của từng KG (`DKGEntity`, `DKGChunk` → `dkg_entity_names`, `dkg_chunk_embedding`…), nên KG test
  (`DKGTEST`, dùng trong `pytest`) không lẫn kết quả tìm kiếm với KG thật.
