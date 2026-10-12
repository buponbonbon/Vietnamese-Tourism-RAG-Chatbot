import os
import requests


class LanguageGenerator:
    def __init__(self, api_key: str = None, base_url: str = "http://localhost:11434/v1/chat/completions"):
        print("\n--- [5/5] Initializing Language Generation Engine (Ollama Local - Qwen2.5 14B Q3_K_L) ---")

        # Local Ollama endpoint bypasses external API rate limits (HTTP 429)
        self.api_key = api_key or "ollama-local"
        self.base_url = base_url
        print(f"-> Generation client successfully targeted to Ollama Local Endpoint")

    def format_prompt(self, query: str, retrieved_chunks: list) -> str:
        """
        Constructs a structured context-augmented prompt layout tailored for RAG.
        """
        context_str = ""
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_str += f"[Document Reference {i}]: {chunk['text']}\n\n"

        prompt_template = f"""You are an elite, highly knowledgeable Vietnamese tourism assistant. 
Your objective is to answer the user's query comprehensively and accurately based strictly on the provided context references.

[CONTEXT DATA]
{context_str}
[END OF CONTEXT DATA]

[USER QUERY]
{query}

[INSTRUCTIONAL CONSTRAINTS]
1. Answer in fluent, natural Vietnamese.
2. Rely exclusively on the facts stated within the [CONTEXT DATA]. 
3. If the context does not contain relevant information to answer the query, state clearly that the information is unavailable in the database—do not hallucinate.
4. Maintain a professional, welcoming, and helpful tone.

RESPONSE:"""
        return prompt_template

    def generate_answer(self, query: str, retrieved_chunks: list,
                        model_name: str = "qwen2.5:14b-instruct-q4_K_M") -> str:

        if not retrieved_chunks:
            return "Xin lỗi, hệ thống không tìm thấy dữ liệu phù hợp trong kho lưu trữ để trả lời câu hỏi này."

        final_prompt = self.format_prompt(query, retrieved_chunks)

        headers = {
            "Authorization": f"Bearer {self.api_key.strip()}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": 0.3,  # Enforce factual alignment and reduce creativity
            "max_tokens": 2048   # Allocating sufficient token budget for production responses
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)

            if response.status_code == 200:
                result_json = response.json()
                return result_json['choices'][0]['message']['content']
            else:
                print(f"[API ERROR] Ollama Gateway returned status code {response.status_code}: {response.text}")
                return "Hệ thống gặp sự cố khi kết nối với máy chủ xử lý ngôn ngữ local. Vui lòng kiểm tra lại Ollama."

        except Exception as e:
            print(f"[API ERROR] Failed to fetch generation from LLM Gateway: {str(e)}")
            return "Hệ thống gặp sự cố khi kết nối với máy chủ xử lý ngôn ngữ local."


# =========================================================================
# SYSTEM VERIFICATION & UNIT TESTING (Independent Execution)
# =========================================================================
if __name__ == "__main__":
    mock_query = "Kinh nghiệm đi Vịnh Hạ Long"
    mock_reranked_chunks = [
        {
            "chunk_id": "HP_008_chunk_2",
            "text": "Trên Vịnh Hạ Long có nhiều hành trình tham quan đặc sắc, trong đó hành trình kết nối hang Sửng Sốt và bãi tắm Ti Tốp được nhiều du khách lựa chọn nhất. Vé tham quan ban ngày có giá niêm yết theo tuyến.",
            "rerank_score": 4.1250
        }
    ]

    generator = LanguageGenerator()
    compiled_prompt = generator.format_prompt(mock_query, mock_reranked_chunks)
    print("\n==================================================")
    print("         COMPILED RAG PROMPT SPECIFICATION        ")
    print("==================================================")
    print(compiled_prompt)

    print("\n--- Testing Response Generation via Ollama Local ---")
    response_test = generator.generate_answer(mock_query, mock_reranked_chunks)
    print("RESPONSE:")
    print(response_test)