# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Công Chiến
**Nhóm:** Nhóm 1
**Ngày:** 03/08/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

> *Viết 1-2 câu:* Độ tương tự cosine cao có nghĩa là hai vector embedding hướng về gần cùng một hướng trong không gian vector, cho thấy ngữ nghĩa (semantic) của hai đoạn văn bản tương ứng rất giống nhau.

**Ví dụ có độ tương tự CAO:**

- Câu A: "Tôi rất thích ăn phở bò."
- Câu B: "Món phở bò là món ăn yêu thích của tôi."
- Tại sao tương đồng: Cả hai câu đều mang cùng một ý nghĩa về sở thích ăn uống đối với món phở (Thực tế khi chạy qua OpenAI text-embedding-3-small, Cosine Similarity đạt mức rất cao: **0.8394**).

**Ví dụ có độ tương tự THẤP:**

- Câu A: "Hôm nay trời nắng đẹp."
- Câu B: "Thị trường chứng khoán đang giảm điểm mạnh."
- Tại sao khác: Hai câu nói về hai chủ đề hoàn toàn không liên quan đến nhau (thời tiết và tài chính). Khi đo đạc thực tế qua model OpenAI, độ tương đồng chỉ ở mức rất thấp là **0.3444**.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

> *Viết 1-2 câu:* Cosine similarity chỉ đo lường góc giữa hai vector mà không phụ thuộc vào độ lớn (magnitude) của chúng, giúp so sánh chính xác mức độ tương đồng về ngữ nghĩa giữa hai văn bản bất kể sự chênh lệch về độ dài.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> *Trình bày phép tính:* Số chunk = 1 + ceiling((Tổng số ký tự - chunk_size) / (chunk_size - overlap)) = 1 + ceiling((10000 - 500) / (500 - 50)) = 1 + ceiling(9500 / 450) = 1 + ceiling(21.11) = 23 chunks.
> *Đáp án:* 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

> *Viết 1-2 câu:* Nếu overlap tăng lên 100, số chunk sẽ tăng lên thành 25 (vì mẫu số giảm còn 400). Ta muốn độ chồng chéo nhiều hơn để đảm bảo không bị cắt đứt mạch ngữ nghĩa ở ranh giới giữa các chunk, giúp bảo toàn trọn vẹn ngữ cảnh khi truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:

> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?*
> Sử dụng regex như `(?<=[.!?])\s+` để tách câu dựa trên các dấu câu kết thúc. Cần xử lý các trường hợp ngoại lệ như từ viết tắt (vd: "Mr.", "Dr.") hoặc chuỗi có nhiều khoảng trắng để tránh việc cắt xén câu sai logic.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:

> *Viết 2-3 câu: thuật toán hoạt động thế nào? Base case (trường hợp cơ sở) là gì?*
> Thuật toán đệ quy chia văn bản theo các dấu phân cách (separators) từ lớn đến nhỏ (ví dụ: `\n\n`, `\n`, khoảng trắng). Trường hợp cơ sở là khi đoạn văn bản nhỏ hơn giới hạn `chunk_size` hoặc không còn dấu phân cách nào có thể dùng để chia tiếp.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:

> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?*
> Tài liệu đầu vào sẽ được nhúng (embed) thành vector và lưu trong bộ nhớ hoặc cơ sở dữ liệu vector cùng với metadata tương ứng. Hàm search sẽ tính toán độ tương tự Cosine giữa vector của câu hỏi truy vấn và vector của các tài liệu trong kho để trả về Top K tài liệu liên quan nhất.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:

> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?*
> Việc lọc thường được thực hiện kết hợp cùng quá trình tìm kiếm (lọc trước thông qua metadata) nhằm tối ưu hoá không gian tìm kiếm. Hàm xóa được thực thi bằng cách lấy ID của document và loại bỏ phần tử/vector tương ứng ra khỏi hệ thống lưu trữ.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:

> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?*
> Prompt được thiết kế sao cho có phần chỉ dẫn (system instruction) yêu cầu agent trả lời dựa trên ngữ cảnh. Sau đó, ngữ cảnh (các chunk liên quan từ kết quả truy xuất) sẽ được inject thẳng vào trong prompt trước hoặc sau câu hỏi của người dùng.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
(day07-2a202601053-trancongchien) tcc3281@tcc3281-ThinkPad-T14s-Gen-3:/data/vinai/labs/DAY07_2A202601053_TranCongChien$ pytest tests/ -v
======================================================================== test session starts =========================================================================
platform linux -- Python 3.11.0rc1, pytest-9.1.1, pluggy-1.6.0 -- /data/vinai/labs/DAY07_2A202601053_TranCongChien/.venv/bin/python
cachedir: .pytest_cache
rootdir: /data/vinai/labs/DAY07_2A202601053_TranCongChien
configfile: pyproject.toml
collected 42 items                                                                                                                                                 

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                          [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                   [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                            [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                             [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                  [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                  [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                        [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                         [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                       [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                         [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                         [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                    [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                          [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                 [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                     [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                               [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                     [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                         [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                           [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                             [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                   [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                        [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                          [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                              [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                           [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                    [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                   [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                              [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                          [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                     [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                         [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                               [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                         [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                      [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                    [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                   [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                       [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                  [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                           [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                 [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                     [100%]

========================================================================= 42 passed in 0.07s =========================================================================
(day07-2a202601053-trancongchien) tcc3281@tcc3281-ThinkPad-T14s-Gen-3:/data/vinai/labs/DAY07_2A202601053_TranCongChien$ 
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Mèo là động vật dễ thương" | "Con mèo này rất đáng yêu" | cao | 0.6020 | Đúng |
| 2 | "Tôi đi học bằng xe đạp" | "Trời hôm nay rất đẹp" | thấp | 0.3145 | Đúng |
| 3 | "Trí tuệ nhân tạo phát triển mạnh" | "AI đang thay đổi thế giới" | cao | 0.4334 | Đúng |
| 4 | "Anh ấy thích uống cà phê đen" | "Cà phê sữa là món tôi thích" | cao | 0.6139 | Đúng |
| 5 | "Bầu trời màu xanh dương" | "Cỏ cây có màu xanh lá" | thấp | 0.5723 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

> *Viết 2-3 câu:* Kết quả bất ngờ nhất có thể là khi hai câu sử dụng từ vựng hoàn toàn khác nhau nhưng mô hình vẫn cho điểm tương tự cao (như "Trí tuệ nhân tạo" và "AI"). Điều này cho thấy embeddings thực sự bắt được ý nghĩa ngữ nghĩa tiềm ẩn (semantic meaning) chứ không chỉ đơn thuần là so khớp từ khóa (keyword matching).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | "RAG là gì?" | RAG kết hợp truy xuất và sinh văn bản... | 0.5282 | Có | RAG là kỹ thuật kết hợp truy... |
| 2 | "Cosine similarity khác gì?" | Phép đo góc giữa hai vector... | 0.7757 | Có | Nó đo lường góc giữa hai vector... |
| 3 | "Vector Database là gì?" | Cơ sở dữ liệu chuyên lưu trữ và tìm... | 0.8021 | Có | Vector Database dùng để lưu trữ... |
| 4 | "Recursive chunking?" | Kỹ thuật cắt nhỏ tài liệu đệ quy qua... | 0.5576 | Có | Là việc chia văn bản theo nhiều... |
| 5 | "Ứng dụng của embedding?" | Embedding giúp AI hiểu được ngữ nghĩa... | 0.5470 | Có | Giúp chuyển văn bản thành vector... |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

> *Viết 2-3 câu:* Điều hay nhất tôi học được là cách các nhóm khác tinh chỉnh tham số `chunk_size` và `overlap` phù hợp với từng loại tài liệu cụ thể để đạt được kết quả truy xuất ngữ cảnh chính xác nhất. Việc thiết kế prompt rõ ràng cũng giúp agent trả lời mượt mà hơn.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                           | Điểm tự đánh giá |
| ---------------------------------------------------- | ---------------------- |
| Khởi động (Warm-up)                               | 5 / 5                  |
| Hướng tiếp cận của tôi (My Approach)           | 10 / 10                |
| Hoàn thiện code (Core Implementation — tests)     | 30 / 30                |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5                  |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10                |
| **Tổng phần cá nhân**                      | **60 / 60**      |
