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
    def __init__(self, model_name: str = "qwen2.5:14b-instruct-q4_K_M"):
        print(f"\n--- [INIT] Launching Local Curation Engine via Ollama ---")
        self.model_name = model_name
        try:
            ollama.show(self.model_name)
            print(f"-> Local Model '{self.model_name}' detected and ready. GPU/Unlimited Mode Engaged.")
        except Exception:
            print(f"[ERROR] Model '{self.model_name}' not found in Ollama.")
            print(f"Please run 'ollama run {self.model_name}' in your terminal first to download the weights.")
            sys.exit(1)

    def generate_adversarial_trap(self, positive_chunk: str, province: str, title: str, trap_strategy: str) -> dict:
        prompt = f"""You are an Elite AI Data Scientist specializing in Adversarial Machine Learning for Vietnamese RAG systems.
Your task is to generate a 'Hard-Negative' training sample that will trick baseline search algorithms but be penalized by a Cross-Encoder.

[CONTEXT]
Target Province: {province}
True Content: \"\"\"{positive_chunk}\"\"\"

[INJECTED TRAP STRATEGY]
You MUST exclusively apply this trap:
-> {trap_strategy}

[STRICT CONSTRAINTS]
1. OUTPUT FORMAT: Respond ONLY with a valid JSON object. No markdown, no conversational text.
2. "query": A natural Vietnamese user question targeting a core fact. MUST end with '?'. Embed the logic of the trap into the query.
3. "neg" (The Trap): A highly deceptive Vietnamese passage answering the query.
   - FATAL RULE 1 (Lexical): You MUST retain at least 95% of the original vocabulary, tone, and sentence structure (High Lexical Overlap).
   - FATAL RULE 2 (NER Preservation): You MUST strictly COPY-PASTE all proper nouns (people, specific locations, organizations), exact numbers, and dates EXACTLY as they appear in the True Content. ONLY alter the single specific detail dictated by the TRAP STRATEGY.
   - LENGTH: The "neg" passage must be strictly identical in length to the True Content. Do NOT shorten it.

[EXAMPLE]
Target Province: Phú Thọ
True Content: "Lễ hội bánh chưng tại đền Hùng, Phú Thọ thu hút 10.000 du khách do ông Nguyễn Văn A tổ chức vào năm 2023."
{{
  "query": "Lễ hội bánh chưng ở đền Trần, Nam Định năm 2023 do ai tổ chức và hút bao nhiêu khách?",
  "neg": "Lễ hội bánh chưng tại đền Trần, Nam Định thu hút 10.000 du khách do ông Nguyễn Văn A tổ chức vào năm 2023.",
  "margin": 2.0
}}

[GENERATE JSON FOR CONTEXT]
"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a JSON-only API. You strictly output raw JSON without markdown formatting."},
                    {"role": "user", "content": prompt}
                ],
                format='json', # Ép cứng JSON
                options={"temperature": 0.2, "top_p": 0.9, "frequency_penalty": 0.2} 
            )

            # Lấy chuỗi JSON sạch trực tiếp từ Ollama
            content = response['message']['content'].strip()

            # Chuyển ngay thành Dictionary mà không cần cắt gọt gì thêm
            result = json.loads(content)
            
            if "query" in result and "neg" in result:
                return result
            return None

        except Exception as e:
            # In ra lỗi nếu có để dễ debug (Tùy chọn)
            print(f"\n[LỖI JSON] {e}")
            return None


def load_chunks_from_csv(file_path: str, sample_size: int = 1000) -> list:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target dataset CSV not found at: {file_path}")

    all_rows = []
    seen_texts = set() 

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get('text_content', '').strip()
            
            # Lọc rác ngắn và footer
            if len(text) < 100: continue
            junk_keywords = ["Giấy phép số", "Email:", "Tổng biên tập", "Địa chỉ:", "Hotline:", "Bản quyền thuộc về"]
            if any(junk in text for junk in junk_keywords):
                continue

            if text and text not in seen_texts:
                seen_texts.add(text)
                all_rows.append({
                    "text": text,
                    "province": row.get('province', 'Unknown').strip(),
                    "title": row.get('title', 'Unknown').strip()
                })

    return random.sample(all_rows, min(sample_size, len(all_rows)))

# =========================================================================
# EXPERIMENT EXECUTION INTERFACE
# =========================================================================
if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    SOURCE_CSV_PATH = os.path.join(CURRENT_DIR, "processed", "vietnam_tourism_rag_dataset.csv")
    PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    OUTPUT_JSON_PATH = os.path.join(PROCESSED_DIR, "bge_train_data_local.json")

    # Mục tiêu chuẩn xác cần thu về
    # 1. Khai báo mục tiêu
    TARGET_SAMPLES = 1000
    
    # 2. Hệ số dự phòng (Buffer Multiplier) cho mô hình Quantization (nén)
    BUFFER_MULTIPLIER = 3 

    try:
        # 3. Code sẽ tự động tính toán: 1000 * 3 = 3000 mẫu gốc được load lên
        sampled_data = load_chunks_from_csv(SOURCE_CSV_PATH, sample_size=(TARGET_SAMPLES * BUFFER_MULTIPLIER))
        generator = LocalAdversarialGenerator(model_name="qwen2.5:14b-instruct-q4_K_M")

        existing_pos_chunks = set()
        if os.path.exists(OUTPUT_JSON_PATH):
            try:
                with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
                    finetuning_dataset = json.load(f)
                
                for sample in finetuning_dataset:
                    for pos in sample.get("pos", []):
                        existing_pos_chunks.add(pos)
                        
                print(f"-> Phát hiện file kết quả cũ đang có {len(finetuning_dataset)} mẫu. Đang chạy tiếp sức...")
            except:
                finetuning_dataset = []
        else:
            finetuning_dataset = []

        print(f"\n--- Local Pipeline Engaged: Generating Target {TARGET_SAMPLES} Triplets ---")
        start_time = time.time()

        # Tìm dòng khởi tạo pbar và thêm tham số ncols=100 (hoặc dynamic_ncols=True)
        pbar = tqdm(total=TARGET_SAMPLES, desc="Local Curation Progress", unit="sample")
        pbar.update(len(finetuning_dataset))

        TRAP_STRATEGIES = [
            "1. LOCATION SWAP: Completely change the primary province/city to a geographically distant one, while keeping the specific event/description identical.",
            "2. NEAR-ENTITY CONFUSION: Replace a famous landmark/entity with a highly similar or confusable one (e.g., 'Vịnh Hạ Long' -> 'Vịnh Lan Hạ', 'Đền Hùng' -> 'Đền Trần').",
            "3. TEMPORAL SHIFT: Significantly alter the dates, years, or timeframes (e.g., change past events to future plans like '2023' to '2026'). Keep everything else factual.",
            "4. NUMERIC DISTORTION: Subtly manipulate key statistics, prices, tourist volumes, or dimensions. Do not change locations or dates.",
            "5. MULTI-HOP CONTRADICTION: Introduce a geographical or environmental contradiction (e.g., asking about or describing beach activities in a mountain province).",
            "6. HALLUCINATION INJECTION: Keep 95% of the facts perfectly true, but flawlessly insert one completely fake, believable detail (e.g., a fake VIP guest, a fake new architectural feature)."
        ]

        for item in sampled_data:
            # CHỐT CHẶN: Dừng chính xác khi đạt target
            if len(finetuning_dataset) >= TARGET_SAMPLES:
                break

            # Kiểm tra trùng lặp tốc độ cao O(1)
            if item["text"] in existing_pos_chunks:
                continue

            current_trap = random.choice(TRAP_STRATEGIES)
            generated_trap = generator.generate_adversarial_trap(
                positive_chunk=item["text"],
                province=item["province"],
                title=item["title"],
                trap_strategy=current_trap
            )

            if generated_trap and "query" in generated_trap and "neg" in generated_trap:
                triplet_sample = {
                    "query": generated_trap["query"],
                    "pos": [item["text"]],
                    "neg": [generated_trap["neg"]],
                    "margin": 2.0
                }
                finetuning_dataset.append(triplet_sample)
                existing_pos_chunks.add(item["text"])

                with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                    json.dump(finetuning_dataset, f, ensure_ascii=False, indent=2)

                pbar.update(1)

        pbar.close()
        print(f"\n[SUCCESS] Hoàn thành! Đã bảo mật được {len(finetuning_dataset)} mẫu tại: '{OUTPUT_JSON_PATH}'")
        print(f"Tổng thời gian xử lý cục bộ trên GPU: {(time.time() - start_time) / 60:.1f} phút.")

    except Exception as e:
        print(f"\n[FATAL PIPELINE ERROR] Hệ thống dừng: {str(e)}")