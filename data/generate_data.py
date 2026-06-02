import os
import json
import random
import csv
import time
import sys

try:
    import ollama
except ImportError:
    print("\n[SYSTEM WARNING] Missing dependency 'ollama'. Attempting automatic installation...")
    os.system('pip install ollama')
    import ollama

try:
    from tqdm import tqdm
except ImportError:
    print("\n[SYSTEM WARNING] Missing dependency 'tqdm'. Attempting automatic installation...")
    os.system('pip install tqdm')
    from tqdm import tqdm


class LocalAdversarialGenerator:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        print(f"\n--- [INIT] Launching Local Curation Engine via Ollama ---")
        self.model_name = model_name
        # Kiểm tra xem mô hình đã được tải về máy chưa
        try:
            ollama.show(self.model_name)
            print(f"-> Local Model '{self.model_name}' detected and ready. GPU/Unlimited Mode Engaged.")
        except Exception:
            print(f"[ERROR] Model '{self.model_name}' not found in Ollama.")
            print(f"Please run 'ollama run {self.model_name}' in your terminal first to download the weights.")
            sys.exit(1)

        def generate_adversarial_trap(self, positive_chunk: str, province: str, title: str) -> dict:
            """
            Advanced English Prompt optimized for Qwen 2.5 (7B) to extract
            high-fidelity adversarial geo-spatial training triplets.
            """
            prompt = f"""You are an expert AI data engineer specialized in generating high-quality adversarial datasets for training neural Cross-Encoder rerankers.
            Your goal is to build a hard-negative sample to teach a model how to detect geo-spatial hallucinations and lexical traps in Vietnamese RAG systems.
        
            [CONTEXT DATA]
            - Target Province: {province}
            - Article Title: {title}
            - Positive Chunk (True Context): 
            \"\"\"{positive_chunk}\"\"\"
        
            [YOUR TASKS]
            You are creating hard-negative training data for a neural Cross-Encoder reranker used in a Vietnamese tourism RAG system.
        
            1. Generate a natural Vietnamese user query.
                - The query must be derived from the Positive Chunk.
                - Preserve the original user intent and factual information contained in the Positive Chunk.
                - Inject ONLY ONE incorrect geographic entity (province, city, attraction, district, landmark, or tourism destination).
                - Keep all other facts unchanged.
                - The query should remain highly relevant to the Positive Chunk except for the geographic substitution.
                - Avoid repetitive templates such as "Có phải bạn đang nói về...".
                - Use diverse styles including search keywords, information requests, travel questions, ticket inquiries, transportation questions, festival questions, accommodation questions, etc.
            2. Generate a hard-negative passage in Vietnamese.
                - The negative passage must describe the incorrect geographic entity used in the query.
                - Reuse as many tourism-related attributes as possible from the Positive Chunk, including:
                    - ticket prices
                    - transportation details
                    - visitor statistics
                    - tourism activities
                    - accommodations
                    - festivals
                    - services
                    - schedules
                    - attractions
                - Preserve most semantic content from the Positive Chunk while relocating those facts to the incorrect location.
                - The passage should sound realistic and convincing.
                - The passage must be lexically similar to the Positive Chunk so that BM25 and dense retrieval systems may mistakenly rank it highly.
                - The passage must contradict the true geographic information contained in the Positive Chunk.
                - Do NOT create a completely unrelated tourism description.
            3. Quality Constraints
                - The query should still be answerable by the Positive Chunk if the incorrect location name were replaced with the correct one.
                - The negative passage should be difficult for a retrieval model to distinguish from the Positive Chunk.
                - The generated negative passage must be a hard negative, not an unrelated negative.
                - Focus on geographic hallucination and location confusion only.
            
                [OUTPUT REQUIREMENT]
                
                Return only a valid JSON object:
                
                {{
                "query": "...",
                "neg": "..."
                }}
                """

            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.5}  # Giữ temperature tầm 0.5 để mô hình sáng tạo câu hỏi đa dạng hơn
                )

                content = response['message']['content'].strip()

                # Trích xuất chuỗi JSON an toàn bằng định vị dấu ngoặc nhọn
                first_brace = content.find("{")
                last_brace = content.rfind("}")
                if first_brace != -1 and last_brace != -1:
                    content = content[first_brace:last_brace + 1]

                content = content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
                return json.loads(content)

            except Exception as e:
                return None

def load_chunks_from_csv(file_path: str, sample_size: int = 1000) -> list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target dataset CSV not found at: {file_path}")

    all_rows = []
    seen_texts = set() # Bộ nhớ đệm lưu các văn bản đã xuất hiện

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text_content', '').strip()
            # Nếu đoạn text không rỗng và chưa từng xuất hiện trong file CSV
            if text and text not in seen_texts:
                seen_texts.add(text) # Đánh dấu đã thấy
                all_rows.append({
                    "text": text,
                    "province": row.get('province', 'Unknown').strip(),
                    "title": row.get('title', 'Unknown').strip()
                })

    # Bốc ngẫu nhiên từ tập hợp các chunk độc nhất 100%
    return random.sample(all_rows, min(sample_size, len(all_rows)))


# =========================================================================
# EXPERIMENT EXECUTION INTERFACE
# =========================================================================
if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    # Trỏ đúng vào thư mục chứa dữ liệu
    SOURCE_CSV_PATH = os.path.join(CURRENT_DIR, "processed", "vietnam_tourism_rag_dataset.csv")
    PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Đổi tên file kết quả thành bge_train_data_local.json để phân biệt với bản API cũ
    OUTPUT_JSON_PATH = os.path.join(PROCESSED_DIR, "bge_train_data_local.json")

    # Chạy trước 10 quăng file cho gemini coi thử đúng chưa rồi mới chạy 1000 nha ní
    TOTAL_SAMPLES_TO_GENERATE = 10

    try:
        sampled_data = load_chunks_from_csv(SOURCE_CSV_PATH, sample_size=TOTAL_SAMPLES_TO_GENERATE)

        # Assign the local adversarial generator with the specified model (Qwen 2.5:7B) for GPU-accelerated processing
        generator = LocalAdversarialGenerator(model_name="qwen2.5:7b")

        # Chế độ chạy tiếp sức tận dụng lại dữ liệu cũ nếu có
        if os.path.exists(OUTPUT_JSON_PATH):
            try:
                with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                    finetuning_dataset = json.load(f)
                print(f"-> Phát hiện file kết quả cũ đang có {len(finetuning_dataset)} mẫu. Đang chạy tiếp sức...")
            except:
                finetuning_dataset = []
        else:
            finetuning_dataset = []

        print(f"\n--- Local Pipeline Engaged: Generating {len(sampled_data)} Triplets ---")
        start_time = time.time()

        pbar = tqdm(total=len(sampled_data), desc="Local Curation Progress", unit="sample")
        pbar.update(len(finetuning_dataset))

        for item in sampled_data:
            # Kiểm tra trùng lặp để không sinh lại câu đã có trong file
            if any(item["text"] in pos_list for sample in finetuning_dataset for pos_list in sample["pos"]):
                continue

            generated_trap = generator.generate_adversarial_trap(
                positive_chunk=item["text"],
                province=item["province"],
                title=item["title"]
            )

            if generated_trap and "query" in generated_trap and "neg" in generated_trap:
                triplet_sample = {
                    "query": generated_trap["query"],
                    "pos": [item["text"]],
                    "neg": [generated_trap["neg"]]
                }
                finetuning_dataset.append(triplet_sample)

                # Ghi đè cập nhật xuống ổ đĩa thời gian thực để bảo toàn dữ liệu
                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(finetuning_dataset, f, ensure_ascii=False, indent=2)

                pbar.update(1)
            else:
                pbar.update(1)

        pbar.close()
        print(f"\n[SUCCESS] Hoàn thành! Đã bảo mật được {len(finetuning_dataset)} mẫu tại: '{OUTPUT_JSON_PATH}'")
        print(f"Tổng thời gian xử lý cục bộ trên GPU: {(time.time() - start_time) / 60:.1f} phút.")

    except Exception as e:
        print(f"\n[FATAL PIPELINE ERROR] Hệ thống dừng: {str(e)}")