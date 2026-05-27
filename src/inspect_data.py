import os
import json


def inspect_inverted_index(index_path):
    """
    Inspect the constructed inverted index to find potential noise/garbage tokens
    by combining low document frequency (occurs in only 1 chunk) and vocabulary validation.
    """
    if not os.path.exists(index_path):
        print(f"Error: Cannot find index file at {index_path}")
        return

    with open(index_path, 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)

    print("==================================================")
    print("       INVERTED INDEX QUALITY INSPECTION          ")
    print("==================================================")
    print(f"Total distinct tokens: {len(inverted_index)}")

    # 1. Base analysis from previous setup
    short_tokens = [token for token in inverted_index.keys() if len(token) <= 3 and token.isalpha()]
    numeric_tokens = [token for token in inverted_index.keys() if token.isdigit()]

    print(f"\n[Analysis 1] Number of short tokens (<= 3 chars): {len(short_tokens)}")
    print(f"Sample short tokens: {short_tokens[:20]}")

    print(f"\n[Analysis 2] Number of pure numeric tokens: {len(numeric_tokens)}")
    print(f"Sample numeric tokens: {numeric_tokens[:20]}")

    # =========================================================================
    # NEW INTEGRATION: COMBINING METHOD 1 (LOW DF) & METHOD 2 (DICT VALIDATION)
    # =========================================================================

    # Define a valid structural vocabulary base commonly found in Vietnamese tourism domain.
    # This prevents valid multi-word fragments or English terms from being falsely flagged.
    valid_tourism_vocab = {
        # Common English tourism terms
        "tour", "tourism", "travel", "hotel", "resort", "homestay", "video", "media", "map", "guide",
        "booking", "flight", "trip", "cruise", "bus", "taxi", "website", "online", "link", "page",
        # Common non-accented names or specific brands/acronyms in data
        "sapa", "hanoi", "halong", "phuquoc", "danang", "saigon", "tpo", "vietnam", "unesco", "facebook"
    }

    # Step A: Filter tokens that appear in exactly ONE chunk (Method 1: Hapax Legomena)
    single_occurrence_tokens = [token for token, doc_map in inverted_index.items() if len(doc_map) == 1]

    # Step B: From the single-occurrence list, isolate true noise (Method 2)
    # True noise: Not a number, contains no Vietnamese underscores, not in valid vocabulary,
    # and has structural patterns of corrupted text (like 'lir', 'htmlt', 'href')
    detected_garbage_tokens = []

    for token in single_occurrence_tokens:
        # If it's pure numbers, skip (already tracked in Analysis 2)
        if token.isdigit():
            continue

        # If it contains underscores '_', it's a valid Vietnamese compound word from Underthesea -> skip
        if "_" in token:
            continue

        # If it's a known valid tourism/technical term -> skip
        if token in valid_tourism_vocab:
            continue

        # If it passes all checks, it's highly likely to be a corrupted/noise token
        detected_garbage_tokens.append(token)

    print(f"\n[Analysis 3] Combined Garbage Token Detection (Low DF + Vocabulary Check)")
    print(f"Total suspected garbage/noise tokens: {len(detected_garbage_tokens)}")
    print(f"Percentage of noise in vocabulary: {(len(detected_garbage_tokens) / len(inverted_index)) * 100:.2f}%")
    print(f"Sample detected noise tokens: {detected_garbage_tokens[:40]}")

    # =========================================================================

    # 3. Check the document frequency of a specific weird token (e.g., 'lir')
    target_token = "lir"
    if target_token in inverted_index:
        docs_containing_target = list(inverted_index[target_token].keys())
        print(f"\n[Target Check] Token '{target_token}' appears in {len(docs_containing_target)} chunk(s).")
        print(f"Associated Chunk IDs: {docs_containing_target}")
    else:
        print(f"\n[Target Check] Token '{target_token}' does not exist or was filtered out.")
    print("==================================================")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_file_path = os.path.join(BASE_DIR, "data", "processed", "inverted_index.json")

    inspect_inverted_index(index_file_path)