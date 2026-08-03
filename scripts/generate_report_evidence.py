import sys
import os

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings import LocalEmbedder, MockEmbedder
from src.chunking import compute_similarity
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document
from src.chunking import RecursiveChunker

def evaluate_part_4():
    print("="*50)
    print("PHẦN 4: DỰ ĐOÁN ĐỘ TƯƠNG TỰ (SIMILARITY PREDICTIONS)")
    print("="*50)
    pairs = [
        ("Mèo là động vật dễ thương", "Con mèo này rất đáng yêu", "cao"),
        ("Tôi đi học bằng xe đạp", "Trời hôm nay rất đẹp", "thấp"),
        ("Trí tuệ nhân tạo phát triển mạnh", "AI đang thay đổi thế giới", "cao"),
        ("Anh ấy thích uống cà phê đen", "Cà phê sữa là món tôi thích", "cao"),
        ("Bầu trời màu xanh dương", "Cỏ cây có màu xanh lá", "thấp")
    ]

    try:
        embedder = LocalEmbedder()
        print("Sử dụng LocalEmbedder (sentence-transformers).")
    except Exception as e:
        print(f"Không thể load LocalEmbedder: {e}")
        print("Sử dụng MockEmbedder (kết quả sẽ mang tính ngẫu nhiên do hash).")
        embedder = MockEmbedder()

    for idx, (a, b, expected) in enumerate(pairs, 1):
        vec_a = embedder(a)
        vec_b = embedder(b)
        sim = compute_similarity(vec_a, vec_b)
        print(f"Cặp {idx}:")
        print(f"  - Câu A: {a}")
        print(f"  - Câu B: {b}")
        print(f"  - Dự đoán: {expected}")
        print(f"  - Điểm thực tế: {sim:.4f}")
        print("-" * 30)

def evaluate_part_5():
    print("\n" + "="*50)
    print("PHẦN 5: KẾT QUẢ TRUY XUẤT (COMPETITION RESULTS)")
    print("="*50)
    print("Mô phỏng 5 câu hỏi truy xuất với một kho tài liệu mẫu.")
    
    # Tạo store với MockEmbedder (hoặc LocalEmbedder nếu có)
    try:
        embedder = LocalEmbedder()
    except Exception:
        embedder = MockEmbedder()

    store = EmbeddingStore(embedding_fn=embedder)
    
    # Các câu hỏi giả định
    queries = [
        "RAG là gì?",
        "Cosine similarity khác gì?",
        "Vector Database là gì?",
        "Recursive chunking?",
        "Ứng dụng của embedding?"
    ]

    # Sample documents to populate the store
    sample_docs = [
        Document(id="1", content="RAG kết hợp truy xuất và sinh văn bản để AI trả lời chính xác hơn.", metadata={}),
        Document(id="2", content="Phép đo góc giữa hai vector được gọi là Cosine similarity.", metadata={}),
        Document(id="3", content="Vector Database là cơ sở dữ liệu chuyên lưu trữ và tìm kiếm vector nhúng.", metadata={}),
        Document(id="4", content="Kỹ thuật cắt nhỏ tài liệu đệ quy qua dấu câu được gọi là Recursive chunking.", metadata={}),
        Document(id="5", content="Embedding giúp AI hiểu được ngữ nghĩa từ vựng trong văn bản.", metadata={})
    ]

    store.add_documents(sample_docs)

    # Dummy LLM
    def dummy_llm(prompt: str) -> str:
        return "[Câu trả lời sinh ra từ LLM dựa vào context trên]"

    agent = KnowledgeBaseAgent(store=store, llm_fn=dummy_llm)

    for idx, query in enumerate(queries, 1):
        # Trích xuất top 1 chunk để làm report
        results = store.search(query, top_k=1)
        top1_content = results[0]["content"] if results else "Không tìm thấy"
        top1_score = results[0].get("score", 0.0) if results else 0.0
        
        # Agent trả lời
        agent_answer = agent.answer(query, top_k=3)

        print(f"Câu hỏi {idx}: {query}")
        print(f"  - Top-1 Chunk: {top1_content}")
        print(f"  - Score: {top1_score:.4f}")
        print(f"  - Câu trả lời Agent: {agent_answer}")
        print("-" * 30)

if __name__ == "__main__":
    evaluate_part_4()
    evaluate_part_5()
