import os
import json
import pandas as pd
from underthesea import word_tokenize


class VietnamesePreprocessor:
    def __init__(self, csv_path):
        """
        Initialize the preprocessor with the path to the CSV dataset.
        """
        self.csv_path = csv_path
        self.df = None
        self.inverted_index = {}

    def load_data(self):
        """
        Load the RAG dataset from the specified CSV path.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset file not found at: {self.csv_path}")

        print(f"--- Loading data from {self.csv_path} ---")
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def clean_and_tokenize(self, text):
        """
        Preprocess text: Convert to lowercase, tokenize Vietnamese words,
        and remove predefined stop words and special characters.
        """
        if not isinstance(text, str):
            return []

        # Expanded Vietnamese stop words list based on domain requirements
        vietnamese_stopwords = {
            "và", "của", "các", "những", "cho", "được", "bởi",
            "tại", "trong", "trên", "dưới", "để", "với",
            "như", "khi", "sau", "trước", "từ", "đến",
            "này", "đó", "kia", "thì", "mà", "hoặc",
            "là", "có", "đang", "đã", "sẽ", "bị",
            "về", "theo", "ra", "vào", "lại", "nên",
            "một", "nhiều", "ít", "rằng"
        }

        # Convert to lowercase and strip whitespace
        text = text.lower().strip()

        # Perform word segmentation using underthesea
        raw_tokens = word_tokenize(text, format="text").split()

        # Remove stop words and non-alphanumeric tokens
        cleaned_tokens = []
        for token in raw_tokens:
            if token not in vietnamese_stopwords and token.isalnum():
                cleaned_tokens.append(token)

        return cleaned_tokens

    def build_inverted_index(self):
        """
        Build an Inverted Index for BM25 keyword search.
        """
        if self.df is None:
            self.load_data()

        print("--- Text preprocessing and Inverted Index construction in progress ---")

        for index, row in self.df.iterrows():
            doc_id = row['chunk_id']
            text_content = row['text_content']

            tokens = self.clean_and_tokenize(text_content)

            for position, token in enumerate(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = {}

                if doc_id not in self.inverted_index[token]:
                    self.inverted_index[token][doc_id] = []

                self.inverted_index[token][doc_id].append(position)

        print(f" Successfully built inverted index with {len(self.inverted_index)} distinct tokens.")
        return self.inverted_index

    def save_inverted_index(self, output_path):
        """
        Save the constructed inverted index into a JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, ensure_ascii=False, indent=4)
        print(f" Inverted index successfully saved to: {output_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_file_path = os.path.join(BASE_DIR, "data", "processed", "vietnam_tourism_rag_dataset.csv")
    index_file_path = os.path.join(BASE_DIR, "data", "processed", "inverted_index.json")

    try:
        preprocessor = VietnamesePreprocessor(csv_path=csv_file_path)
        preprocessor.build_inverted_index()
        preprocessor.save_inverted_index(output_path=index_file_path)

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        print("Tip: Please verify that 'data/processed/vietnam_tourism_rag_dataset.csv' exists.")