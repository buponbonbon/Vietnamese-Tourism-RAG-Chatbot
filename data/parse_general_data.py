import json
import random
import os

def convert_squad_to_adaptive_triplet():
    INPUT_PATH = "train_vietnam_tourism.json"
    OUTPUT_PATH = os.path.join("processed", "other_team_general_data.json")

    if not os.path.exists(INPUT_PATH):
        print(f"[ERROR] Input file not found at {INPUT_PATH}!")
        return

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)

    # Group all contexts globally to serve as a pool for Level 1 (Easy Negatives)
    all_articles = squad_data['data']
    all_contexts_global = []

    for article in all_articles:
        for paragraph in article['paragraphs']:
            if paragraph['context'].strip():
                all_contexts_global.append(paragraph['context'].strip())

    general_triplets = []

    for article in all_articles:
        title = article['title']
        # Extract all local contexts within the same article topic
        local_contexts = [p['context'].strip() for p in article['paragraphs'] if p['context'].strip()]

        for paragraph in article['paragraphs']:
            pos_context = paragraph['context'].strip()

            for qa in paragraph['qas']:
                query = qa['question'].strip()

                # ADAPTIVE GEOGRAPHIC DIFFICULTY ASSIGNMENT
                # 50/50 probability split to maintain structural balance between Level 1 and Level 2
                if random.random() > 0.5 and len(local_contexts) > 1:
                    # LEVEL 2/3: In-domain negative from the SAME article (Regional/Semantic context)
                    possible_negs = [c for c in local_contexts if c != pos_context]
                    if possible_negs:
                        neg_context = random.choice(possible_negs)
                        margin_value = 1.2
                    else:
                        neg_context = random.choice([c for c in all_contexts_global if c != pos_context])
                        margin_value = 0.5
                else:
                    # LEVEL 1: Out-of-domain negative from a DIFFERENT article (Easy Negative)
                    possible_negs = [c for c in all_contexts_global if c not in local_contexts]
                    neg_context = random.choice(possible_negs) if possible_negs else random.choice(all_contexts_global)
                    margin_value = 0.5

                general_triplets.append({
                    "query": query,
                    "pos": [pos_context],
                    "neg": [neg_context],
                    "margin": margin_value  # Persist dynamic margin metadata directly into JSON
                })

    os.makedirs("processed", exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(general_triplets, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Converted and stratified {len(general_triplets)} triplets successfully!")


if __name__ == "__main__":
    convert_squad_to_adaptive_triplet()