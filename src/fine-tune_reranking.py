import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path



class DynamicGeoDataset(Dataset):
    def __init__(self, adv_json_path: str, general_data_path: str, tokenizer, max_length=512):
        self.encoded_triplets = []
        raw_items = []

        # 1. Load raw textual data from JSON files
        if os.path.exists(adv_json_path):
            with open(adv_json_path, 'r', encoding='utf-8') as f:
                adv_data = json.load(f)
            for item in adv_data:
                raw_items.append({"query": item["query"], "pos": item["pos"][0], "neg": item["neg"][0], "margin": 2.0})
            print(f"[DATA] Successfully loaded {len(adv_data)} Level 4 adversarial samples (Margin = 2.0)")

        if general_data_path and os.path.exists(general_data_path):
            with open(general_data_path, 'r', encoding='utf-8') as f:
                gen_data = json.load(f)
            for item in gen_data:
                raw_items.append(
                    {"query": item["query"], "pos": item["pos"][0], "neg": item["neg"][0], "margin": item["margin"]})
            print(f"[DATA] Successfully loaded {len(gen_data)} Level 1 & 2 adaptive samples")

        # 2. 🌟 PRE-TOKENIZATION: Process text to input IDs on CPU before training loop
        # This completely resolves the data transfer bottleneck between RAM and VRAM
        print("[DATA] Pre-tokenizing entire corpus to free up GPU bandwidth...")
        for item in tqdm(raw_items, desc="Tokenizing"):
            pos_pair = [item["query"], item["pos"]]
            neg_pair = [item["query"], item["neg"]]

            # Pass query and document as separate arguments to create a properly combined single sequence
            pos_enc = tokenizer(item["query"], item["pos"], padding="max_length", truncation=True,
                                max_length=max_length, return_tensors="pt")
            neg_enc = tokenizer(item["query"], item["neg"], padding="max_length", truncation=True,
                                max_length=max_length, return_tensors="pt")

            self.encoded_triplets.append({
                "pos_input_ids": pos_enc["input_ids"].squeeze(0),
                "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
                "neg_input_ids": neg_enc["input_ids"].squeeze(0),
                "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
                "margin": item["margin"]
            })

    def __len__(self):
        return len(self.encoded_triplets)

    def __getitem__(self, idx):
        return self.encoded_triplets[idx]


def train_pipeline():
    print("\n========================================================")
    print("  LAUNCHING DYNAMIC ADVERSARIAL MARGIN CALIBRATION (DAGMC) ")
    print("========================================================")

    # Hyperparameters setup matching standard Transformer-based tuning
    MODEL_NAME = "BAAI/bge-reranker-base"
    EPOCHS = 3
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1

    # Automatic absolute path routing independent of PyCharm working directory settings
    BASE_DIR = Path(__file__).resolve().parent.parent
    ADV_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "bge_train_data_local.json")
    GENERAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "other_team_general_data.json")
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "vietnam_tourism_reranker_dagmc")

    # Hardware binding checking
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> Target Device bound to: [{str(device).upper()}]")

    # WAKE UP GPU: Create a dummy tensor on GPU immediately to force the driver to stay active
    if torch.cuda.is_available():
        _ = torch.zeros(1).to(device)

    # Initialize weights and tokenizer from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

    # Initialize pre-tokenized dataset
    dataset = DynamicGeoDataset(ADV_DATA_PATH, GENERAL_DATA_PATH if os.path.exists(GENERAL_DATA_PATH) else None,
                                tokenizer)

    # Enable multi-threaded data loading and page-locked memory for high-throughput VRAM transfer
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # Optimizer and linear learning rate scheduler configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps
    )

    # Main optimization loop
    # Main optimization loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        # Clear CUDA cache ONCE at the beginning of each epoch instead of inside the batch loop
        torch.cuda.empty_cache()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for batch in progress_bar:
            # REMOVED: torch.cuda.empty_cache() from here to eliminate high latency bottleneck
            optimizer.zero_grad()

            # Directly feed pre-computed 2D numerical tensors into the GPU device
            pos_inputs = {
                "input_ids": batch["pos_input_ids"].to(device),
                "attention_mask": batch["pos_attention_mask"].to(device)
            }
            neg_inputs = {
                "input_ids": batch["neg_input_ids"].to(device),
                "attention_mask": batch["neg_attention_mask"].to(device)
            }

            batch_margins = batch["margin"].to(device, dtype=torch.float32)

            # Compute cross-encoder relevance scores
            pos_scores = model(**pos_inputs).logits.view(-1)
            neg_scores = model(**neg_inputs).logits.view(-1)

            # Dynamic Adversarial Margin Calibration (DAGMC) loss function calculation
            loss_tensor = torch.clamp(batch_margins - (pos_scores - neg_scores), min=0)
            loss = loss_tensor.mean()

            # Backpropagation & Gradient Clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Batch_Loss": f"{loss.item():.4f}"})

        print(f"--> Epoch [{epoch}] Completed. Average Loss: {total_loss / len(dataloader):.5f}")

    # Export best model checkpoints and artifacts
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n[✔️ SUCCESS] DAGMC Model successfully checkpointed at: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    train_pipeline()