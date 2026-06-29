import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import torch.amp
from statsmodels.stats.contingency_tables import mcnemar

# =================================================================================
# GLOBAL REPRODUCIBILITY CONFIGURATION
# =================================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =================================================================================
# DATA ARCHITECTURE NORMALIZATION FUNCTION
# =================================================================================
def normalize_items(raw_list, default_margin=1.0):
    """
    Safely standardizes diverse schema variants (e.g., pos vs pos_doc, neg vs neg_doc)
    and ensures all text payloads are unwrapped consistently into string formats.
    """
    normalized = []
    for item in raw_list:
        pos_raw = item.get("positive", item.get("pos"))
        pos_str = pos_raw[0] if isinstance(pos_raw, list) else pos_raw

        neg_raw = item.get("negative", item.get("neg"))
        neg_str = neg_raw[0] if isinstance(neg_raw, list) else neg_raw

        margin_val = float(item.get("margin", default_margin))

        normalized.append({
            "query": item["query"],
            "pos": pos_str,
            "neg": neg_str,
            "margin": margin_val
        })
    return normalized


# =================================================================================
# INFORMATION RETRIEVAL PERFORMANCE METRICS (PAIRWISE EVALUATION)
# =================================================================================
def compute_ir_metrics(pos_scores, neg_scores):
    """
    Computes ranking metrics and returns both aggregated values and query-level accuracy arrays.
    """
    pos_scores = np.array(pos_scores, dtype=np.float32)
    neg_scores = np.array(neg_scores, dtype=np.float32)

    correct_pairs = pos_scores > neg_scores
    pairwise_accuracy = np.mean(correct_pairs)

    mrr_list = []
    ndcg_at_2_list = []
    p_at_1_list = []
    correctness_mask = []  # FIXED: Restored uniform keyword mapping successfully

    for is_correct in correct_pairs:
        if is_correct:
            mrr_list.append(1.0)
            ndcg_at_2_list.append(1.0)
            p_at_1_list.append(1.0)
            correctness_mask.append(1)
        else:
            mrr_list.append(0.5)
            ndcg_at_2_list.append(0.6309)
            p_at_1_list.append(0.0)
            correctness_mask.append(0)

    return {
        "Accuracy": pairwise_accuracy,
        "MRR": np.mean(mrr_list),
        "NDCG@2": np.mean(ndcg_at_2_list),
        "Precision@1": np.mean(p_at_1_list)
    }, correctness_mask


# =================================================================================
# PYTORCH EARLY STOPPING REGULATOR
# =================================================================================
class EarlyStopping:
    def __init__(self, patience=2, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc, model, tokenizer, output_dir):
        if self.best_acc is None:
            self.best_acc = val_acc
            self.save_checkpoint(model, tokenizer, output_dir)
        elif val_acc < self.best_acc + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"[WARNING - EARLY STOPPING] Validation Accuracy stagnated ({val_acc:.5f} <= {self.best_acc:.5f} + {self.min_delta}). Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"[SUCCESS] Validation Accuracy improved from {self.best_acc:.5f} to {val_acc:.5f}. Checkpoint cached.")
            self.best_acc = val_acc
            self.save_checkpoint(model, tokenizer, output_dir)
            self.counter = 0

    def save_checkpoint(self, model, tokenizer, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


class DynamicGeoDataset(Dataset):
    def __init__(self, raw_items: list, tokenizer, max_length=512):
        self.encoded_triplets = []
        for item in tqdm(raw_items, desc="Tokenizing Partition", leave=False):
            pos_enc = tokenizer(item["query"], item["pos"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            neg_enc = tokenizer(item["query"], item["neg"], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            self.encoded_triplets.append({
                "pos_input_ids": pos_enc["input_ids"].squeeze(0),
                "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
                "neg_input_ids": neg_enc["input_ids"].squeeze(0),
                "neg_attention_mask": neg_enc["attention_mask"].squeeze(0),
                "margin": item["margin"]
            })

    def __len__(self): return len(self.encoded_triplets)
    def __getitem__(self, idx): return self.encoded_triplets[idx]


# =================================================================================
# MAIN CORE TRAINING & EVALUATION PIPELINE
# =================================================================================
def train_pipeline(active_seed, base_repository_path, target_moc=0):
    MODEL_NAME = "BAAI/bge-reranker-base"
    EPOCHS = 3
    BATCH_SIZE = 16
    ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1

    VAL_PATH = os.path.join(base_repository_path, "data", "processed", "cross_encoder_VAL.json")
    TEST_PATH = os.path.join(base_repository_path, "data", "processed", "cross_encoder_TEST.json")

    MOC_TO_PATH = {
        0: os.path.join(base_repository_path, "data", "processed", "train_moc_0.json"),
        6: os.path.join(base_repository_path, "data", "processed", "train_moc_6.json"),
        13: os.path.join(base_repository_path, "data", "processed", "train_moc_13.json")
    }
    TRAIN_PATH = MOC_TO_PATH[target_moc]

    OUTPUT_DIR = os.path.join(base_repository_path, "data", "processed", f"vietnam_tourism_reranker_moc_{target_moc}_seed_{active_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)

    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_raw_json = json.load(f)
    with open(VAL_PATH, 'r', encoding='utf-8') as f:
        val_raw_json = json.load(f)
    with open(TEST_PATH, 'r', encoding='utf-8') as f:
        test_raw_json = json.load(f)

    train_raw = normalize_items(train_raw_json)
    val_raw = normalize_items(val_raw_json)
    test_raw = normalize_items(test_raw_json)

    random.Random(active_seed).shuffle(train_raw)

    print(f"--> [TARGET REGIME MOC: {target_moc}%] | Train: {len(train_raw)} | Val: {len(val_raw)} | Test: {len(test_raw)}")

    train_dataset = DynamicGeoDataset(train_raw, tokenizer)
    val_dataset = DynamicGeoDataset(val_raw, tokenizer)
    test_dataset = DynamicGeoDataset(test_raw, tokenizer)

    g = torch.Generator()
    g.manual_seed(active_seed)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    actual_batches_per_epoch = len(train_dataloader)
    total_steps = (actual_batches_per_epoch // ACCUMULATION_STEPS) * EPOCHS
    if total_steps == 0: total_steps = EPOCHS

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(patience=2, min_delta=1e-4, verbose=True)
    epoch_history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", leave=False)
        for step, batch in enumerate(progress_bar):
            pos_inputs = {"input_ids": batch["pos_input_ids"].to(device), "attention_mask": batch["pos_attention_mask"].to(device)}
            neg_inputs = {"input_ids": batch["neg_input_ids"].to(device), "attention_mask": batch["neg_attention_mask"].to(device)}
            batch_margins = batch["margin"].to(device, dtype=torch.float32)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pos_scores = model(**pos_inputs).logits.view(-1)
                neg_scores = model(**neg_inputs).logits.view(-1)
                score_diff = pos_scores - neg_scores

                alpha = torch.sigmoid(-score_diff)
                # FIXED: Eliminated risky local variable condition check completely
                loss_tensor = torch.clamp(batch_margins - score_diff, min=0)
                loss = (loss_tensor * (1.0 + alpha)).mean() / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({"Loss": f"{(loss.item() * ACCUMULATION_STEPS):.4f}"})

        # --- VALIDATION EVALUATION PHASE ---
        model.eval()
        val_pos_scores, val_neg_scores = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                pos_inputs = {"input_ids": batch["pos_input_ids"].to(device), "attention_mask": batch["pos_attention_mask"].to(device)}
                neg_inputs = {"input_ids": batch["neg_input_ids"].to(device), "attention_mask": batch["neg_attention_mask"].to(device)}
                batch_margins = batch["margin"].to(device, dtype=torch.float32)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pos_outputs = model(**pos_inputs).logits.view(-1)
                    neg_outputs = model(**neg_inputs).logits.view(-1)
                    score_diff_val = pos_outputs - neg_outputs
                    alpha_val = torch.sigmoid(-score_diff_val)
                    v_loss = (torch.clamp(batch_margins - score_diff_val, min=0) * (1.0 + alpha_val)).mean()

                total_val_loss += v_loss.item()
                val_pos_scores.extend(pos_outputs.float().cpu().numpy())
                val_neg_scores.extend(neg_outputs.float().cpu().numpy())

        val_metrics, _ = compute_ir_metrics(val_pos_scores, val_neg_scores)
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch} | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Acc={val_metrics['Accuracy']:.4f} | MRREval={val_metrics['MRR']:.4f}")

        epoch_history.append({
            "epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "acc": val_metrics["Accuracy"], "mrr": val_metrics["MRR"]
        })

        early_stopping(val_metrics['Accuracy'], model, tokenizer, OUTPUT_DIR)
        if early_stopping.early_stop:
            break

    # Exporting training history tracking artifacts
    history_export_path = os.path.join(base_repository_path, "data", "processed", f"epoch_loss_history_moc_{target_moc}_seed_{active_seed}.json")
    with open(history_export_path, "w", encoding="utf-8") as history_file:
        json.dump(epoch_history, history_file, ensure_ascii=False, indent=4)

    # --- TESTING BENCHMARK EVALUATION PARADIGM ---
    best_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR).to(device)
    best_model.eval()

    test_pos_scores, test_neg_scores = [], []
    with torch.no_grad():
        for batch in test_dataloader:
            pos_inputs = {"input_ids": batch["pos_input_ids"].to(device), "attention_mask": batch["pos_attention_mask"].to(device)}
            neg_inputs = {"input_ids": batch["neg_input_ids"].to(device), "attention_mask": batch["neg_attention_mask"].to(device)}
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pos_scores = best_model(**pos_inputs).logits.view(-1).cpu().numpy()
                neg_scores = best_model(**neg_inputs).logits.view(-1).cpu().numpy()
            test_pos_scores.extend(pos_scores)
            test_neg_scores.extend(neg_scores)

    final_test_metrics, correctness_mask = compute_ir_metrics(test_pos_scores, test_neg_scores)

    # SECURE EXPORT: Save clean binary correctness array per query FOR THE PRIMARY EXPERIMENTAL SEED (42) ONLY
    if active_seed == 42:
        correct_log_path = os.path.join(base_repository_path, "data", "processed", f"test_correct_mask_moc_{target_moc}_seed_42.json")
        with open(correct_log_path, "w", encoding="utf-8") as rf:
            json.dump(correctness_mask, rf)

    return final_test_metrics


# =================================================================================
# AUTOMATED MULTI-SEED EXECUTIVE RUNNER + RIGOROUS MCNEMAR'S TEST ROUTER
# =================================================================================
if __name__ == "__main__":
    EXPERIMENTAL_SEEDS = [42, 123, 999]
    BASE_REPOSITORY_PATH = Path(__file__).resolve().parent.parent

    # CONFIGURATION INTERFACE FOR ABLATION REGIME
    TARGET_MOC = 6  

    history = {"acc": [], "mrr": [], "ndcg": [], "p1": []}

    print("\n" + "=" * 80)
    print(f" INITIALIZING STANDARD PAIRWISE BENCHMARK SUITE FOR REGIME MOC: {TARGET_MOC}%")
    print("=" * 80 + "\n")

    for run_idx, current_seed in enumerate(EXPERIMENTAL_SEEDS, start=1):
        print(f"\n[RUN {run_idx}/{len(EXPERIMENTAL_SEEDS)}] TARGET SEED: {current_seed}")

        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        res = train_pipeline(current_seed, BASE_REPOSITORY_PATH, target_moc=TARGET_MOC)

        history["acc"].append(res["Accuracy"])
        history["mrr"].append(res["MRR"])
        history["ndcg"].append(res["NDCG@2"])
        history["p1"].append(res["Precision@1"])

    print("\n" + "=" * 75)
    print(f"FINAL PAIRWISE STATISTICAL REPORT (REGIME MOC: {TARGET_MOC}%) - MEAN ± STD")
    print("=" * 75)
    print(f" Pairwise Accuracy : {np.mean(history['acc']):.4f} ± {np.std(history['acc']):.4f}")
    print(f" Mean Reciprocal R.: {np.mean(history['mrr']):.4f} ± {np.std(history['mrr']):.4f}")
    print(f" NDCG @ Rank 2     : {np.mean(history['ndcg']):.4f} ± {np.std(history['ndcg']):.4f}")
    print(f" Precision @ K=1   : {np.mean(history['p1']):.4f} ± {np.std(history['p1']):.4f}")
    print("=" * 75 + "\n")

    # STRICT METRIC SIGN-RANK VIA MCNEMAR'S TEST ON THE PRIMARY SEED (42) ONLY
    if TARGET_MOC in [6, 13]:
        print(f"⏳ Initiating rigorous McNemar's Significance Check on Primary Seed 42 (Mốc {TARGET_MOC}% vs 0%)...")
        baseline_file = os.path.join(BASE_REPOSITORY_PATH, "data", "processed", "test_correct_mask_moc_0_seed_42.json")
        proposed_file = os.path.join(BASE_REPOSITORY_PATH, "data", "processed", f"test_correct_mask_moc_{TARGET_MOC}_seed_42.json")

        if os.path.exists(baseline_file) and os.path.exists(proposed_file):
            with open(baseline_file, "r") as f: b_mask = np.array(json.load(f))
            with open(proposed_file, "r") as f: p_mask = np.array(json.load(f))

            # Contingency Matrix computation
            a = np.sum((b_mask == 1) & (p_mask == 1)) # Both correct
            b = np.sum((b_mask == 1) & (p_mask == 0)) # Baseline correct, Proposed incorrect
            c = np.sum((b_mask == 0) & (p_mask == 1)) # Proposed correct, Baseline incorrect
            d = np.sum((b_mask == 0) & (p_mask == 0)) # Both incorrect

            contingency_table = [[a, b], [c, d]]

            mcnemar_res = mcnemar(contingency_table, exact=True)
            final_p_value = mcnemar_res.pvalue

            print("\n========================================================")
            print(f"  MCNEMAR'S QUERY-LEVEL ACCURACY SIGNIFICANCE REPORT     ")
            print("========================================================")
            print(f" Contingency Table Matrix [Both_OK, B_OK_P_Fail]        ")
            print(f"                          [P_OK_B_Fail, Both_Fail]      ")
            print(f" Matrix Shape                  : {contingency_table}")
            print(f" Total Unique Queries Checked  : {len(b_mask)} (Strict N, No Pseudo-Replication)")
            print(f" McNemar Global p-value        : {final_p_value:.6f}")
            if final_p_value < 0.05:
                print(f" Result                        : SUCCESS! The accuracy shift is STATISTICALLY SIGNIFICANT (p < 0.05).")
            else:
                print(f" Result                        : FAIL. The accuracy variation shows no statistical significance (p >= 0.05).")
            print("========================================================\n")


