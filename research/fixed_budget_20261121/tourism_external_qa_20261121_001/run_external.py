"""Run the external MLQA Vietnamese retrieval experiment on frozen artifacts."""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[variable] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path(os.environ.get("TOURISM_REPO", "/workspace/Vietnamese-Tourism-RAG-Chatbot"))
ARCHIVE = REPO / "data/oracle_recovery_20261113_001/evidence_selection_reinference_001"
DENSE = Path(os.environ.get("TOURISM_DENSE_OUTPUT", ROOT.parent / "tourism_retrieval_budget_20261120_001"))
ARMS = ("bm25", "dense", "rrf")
STOP = set(
    "và của các những cho được bởi tại trong trên dưới để với như khi sau trước từ "
    "đến này đó kia thì mà hoặc là có đang đã sẽ bị về theo ra vào lại nên một nhiều ít rằng".split()
)


def digest(file):
    with file.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def save(name, data):
    with (ROOT / name).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def load_source():
    protocol = load("PROTOCOL.json")
    package = ROOT / "MLQA_V1.zip"
    assert digest(package) == protocol["source_archive_sha256"]
    with zipfile.ZipFile(package) as z:
        blob = z.read(protocol["member"])
    assert hashlib.sha256(blob).hexdigest() == protocol["source_member_sha256"]
    docs = []
    questions = []
    for article in json.loads(blob)["data"]:
        for paragraph in article["paragraphs"]:
            text = paragraph["context"]
            doc = len(docs)
            docs.append(text)
            for qa in paragraph["qas"]:
                spans = sum(text[a["answer_start"]:a["answer_start"] + len(a["text"])] == a["text"]
                            for a in qa["answers"] if a["text"])
                questions.append({"id": qa["id"], "text": qa["question"], "doc": doc,
                                  "article": article["title"], "valid_offsets": spans})
    assert len(docs) == len(set(docs)) == 4795
    assert len(questions) == len({row["id"] for row in questions}) == 5495
    return docs, questions


def choose_cohort(docs, questions):
    train = ARCHIVE / "source_inputs/train_moc_0.json"
    tourism = ARCHIVE / "source_inputs/vietnam_tourism_rag_dataset.csv"
    records = json.loads(train.read_text())
    assert len(records) == 7618
    train_queries = {normalized(pair["query"]) for pair in records}
    train_texts = {normalized(pair[field]) for pair in records for field in ("positive", "negative")}
    with tourism.open(encoding="utf-8-sig", newline="") as f:
        tourism_texts = {normalized(row["text_content"]) for row in csv.DictReader(f)}
    docs_normalized = [normalized(doc) for doc in docs]
    groups = defaultdict(list)
    query_match = gold_train_match = gold_tourism_match = invalid = 0
    for item in questions:
        gold = docs_normalized[item["doc"]]
        qmatch = normalized(item["text"]) in train_queries
        pmatch = gold in train_texts
        query_match += qmatch
        gold_train_match += pmatch
        gold_tourism_match += gold in tourism_texts
        invalid += not item["valid_offsets"]
        if item["valid_offsets"] and not qmatch and not pmatch:
            groups[item["article"]].append(item["id"])

    def order(field, value):
        return hashlib.sha256(f"tourism-external-qa-v1|{field}|{value}".encode()).hexdigest()

    articles = sorted(groups, key=lambda article: order("article", article))[:140]
    ids = [min(groups[article], key=lambda q: order("question", q)) for article in articles]
    assert len(ids) == 140
    return ids, {"test_articles": len({q["article"] for q in questions}),
                 "eligible_articles": len(groups),
                 "test_docs": len(docs), "test_queries": len(questions),
                 "test_queries_without_one_valid_offset": invalid,
                 "test_exact_normalized_query_matches_to_M0": query_match,
                 "test_gold_exact_normalized_passage_matches_to_M0": gold_train_match,
                 "test_gold_exact_normalized_passage_matches_to_tourism_corpus": gold_tourism_match,
                 "M0_base_sha256": digest(train), "tourism_corpus_sha256": digest(tourism)}


def bm25(docs, queries, variant):
    from underthesea import word_tokenize

    def simple(text):
        return [word for word in re.findall(r"[^\W_]+", text.lower()) if word not in STOP]

    def document_terms(text):
        if variant == "unicode":
            return simple(text)
        return [word for word in word_tokenize(text.lower().strip(), format="text").split()
                if word.isalnum() and word not in STOP]

    def query_terms(text):
        if variant == "unicode":
            return simple(text)
        return [word for token in word_tokenize(text.lower().strip()) if token not in STOP
                for word in token.replace("_", " ").split() if word.isalnum() and word not in STOP]

    lengths = np.asarray([len(doc.split()) for doc in docs])
    constants = 1.5 * (0.25 + 0.75 * lengths / lengths.mean())
    inverted = defaultdict(list)
    for i, doc in enumerate(docs):
        for term, tf in Counter(document_terms(doc)).items():
            inverted[term].append((i, tf))
        if i % 1000 == 0:
            print("BM25", variant, "indexed", i, flush=True)
    contributions = {}
    for term, posts in inverted.items():
        rows, tf = np.asarray(posts).T
        idf = math.log(1 + (len(docs) - len(rows) + .5) / (len(rows) + .5))
        contributions[term] = (rows, idf * (tf * 2.5 / (tf + constants[rows])))

    all_rows = np.arange(len(docs))
    ranks = {}
    under50 = 0
    for q in queries:
        scores = np.zeros(len(docs))
        for term in query_terms(q["text"]):
            if term in contributions:
                rows, values = contributions[term]
                scores[rows] += values
        under50 += np.count_nonzero(scores) < 50
        ranks[q["id"]] = np.lexsort((all_rows, -scores))[:50].tolist()
    print("BM25", variant, "done", flush=True)
    return ranks, under50


def dense(docs, queries):
    import faiss
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    model = DENSE / "dense_model"
    manifest = json.loads((DENSE / "dense_download_manifest.json").read_text())
    for file, record in manifest.items():
        assert digest(model / file) == record["sha256"], file
    embedding = load_file(str(model / "model.safetensors"))["embeddings"]
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    assert embedding.shape == (tokenizer.get_vocab_size(), 256) and embedding.dtype == np.float32
    unk = tokenizer.token_to_id("<unk>")

    def encode(texts):
        features = np.zeros((len(texts), 256), dtype=np.float32)
        for i, item in enumerate(tokenizer.encode_batch_fast(texts, add_special_tokens=False)):
            ids = [token for token in item.ids if token != unk]
            if ids:
                v = embedding[ids].mean(axis=0)
                features[i] = v / (np.linalg.norm(v) + 1e-32)
        assert np.isfinite(features).all()
        return features

    index = faiss.IndexFlatIP(256)
    faiss.omp_set_num_threads(4)
    for start in range(0, len(docs), 512):
        index.add(encode(docs[start:start + 512]))
    assert index.ntotal == len(docs)
    ranks = {}
    for start in range(0, len(queries), 256):
        batch = queries[start:start + 256]
        vec = encode([q["text"] for q in batch])
        assert np.all(np.linalg.norm(vec, axis=1) > .99)
        scores, rows = index.search(vec, 51)
        assert np.isfinite(scores).all() and (rows >= 0).all()
        for q, v, d in zip(batch, scores, rows, strict=True):
            if v[49] == v[50]:
                raise ValueError(f"Dense cutoff has a tie for {q['id']}")
            order = np.lexsort((d, -v))
            ranks[q["id"]] = d[order[:50]].tolist()
    print("Dense done", flush=True)
    return ranks, {name: record["sha256"] for name, record in manifest.items()}


def combine(lexical, semantic):
    a = {doc: i for i, doc in enumerate(lexical, 1)}
    b = {doc: i for i, doc in enumerate(semantic, 1)}

    def order(doc):
        s = (1 / (60 + a[doc]) if doc in a else 0) + (1 / (60 + b[doc]) if doc in b else 0)
        return -s, a.get(doc, 51), b.get(doc, 51), doc

    return sorted(a.keys() | b.keys(), key=order)[:50]


def retrieve():
    import importlib.metadata

    started = time.monotonic()
    docs, qs = load_source()
    ids, stats = choose_cohort(docs, qs)
    save("COHORT.json", {"protocol_sha256": digest(ROOT / "PROTOCOL.json"), "ids": ids, "audit": stats})
    a, n_short = bm25(docs, qs, "project")
    b, hashes = dense(docs, qs)
    save("RANKS.json", {q["id"]: {"gold": q["doc"], "article": q["article"],
                                 "bm25": a[q["id"]], "dense": b[q["id"]],
                                 "rrf": combine(a[q["id"]], b[q["id"]])} for q in qs})
    simp, n_simple_short = bm25(docs, qs, "unicode")
    save("BM25_UNICODE.json", simp)
    save("RETRIEVAL_RUN.json", {"elapsed_seconds": time.monotonic() - started,
                               "bm25_below_50_nonzero": int(n_short),
                               "bm25_unicode_below_50_nonzero": int(n_simple_short),
                               "dense_files_sha256": hashes,
                               "underthesea_version": importlib.metadata.version("underthesea"),
                               "faiss_version": importlib.metadata.version("faiss-cpu"),
                               "protocol_sha256": digest(ROOT / "PROTOCOL.json")})


def score():
    import gc

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    docs, all_qs = load_source()
    qs = {item["id"]: item["text"] for item in all_qs}
    ids = load("COHORT.json")["ids"]
    ranks = load("RANKS.json")
    keys = sorted({(q, d) for q in ids for arm in ARMS for d in ranks[q][arm][:5]})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    settings = load("PROTOCOL.json")
    frozen = json.loads((ARCHIVE / "protocol.json").read_text())
    print("Unique pairs per checkpoint", len(keys), flush=True)
    for name in settings["reranker"]:
        target = ROOT / (name + "_scores.json")
        if target.is_file():
            continue
        info = next(row for row in frozen["models"] if row["name"] == name)
        folder = Path(info["path"])
        for file, expected in info["file_sha256"].items():
            assert digest(folder / file) == expected, (name, file)
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        enc = tokenizer([qs[q] for q, _ in keys], [docs[d] for _, d in keys],
                        truncation=True, max_length=512, padding=False)
        rows = []
        start = time.monotonic()
        for i in range(0, len(keys), 4):
            indices = range(i, min(i + 4, len(keys)))
            batch = tokenizer.pad([{field: enc[field][j] for field in enc} for j in indices],
                                  return_tensors="pt")
            with torch.inference_mode():
                scores = model(**batch).logits.float().reshape(-1).tolist()
            assert all(math.isfinite(value) for value in scores)
            for j, value in zip(indices, scores, strict=True):
                q, d = keys[j]
                rows.append({"q": q, "d": d, "score": value, "tokens": len(enc["input_ids"][j])})
            if (i + 4) % 112 == 0 or i + 4 >= len(keys):
                print(name, len(rows), "/", len(keys), "elapsed", round(time.monotonic() - start, 1),
                      flush=True)
        save(target.name, {"checkpoint_hashes": info["file_sha256"], "scores": rows,
                           "torch": torch.__version__, "transformers": transformers.__version__,
                           "forward_seconds": time.monotonic() - start,
                           "cohort_sha256": digest(ROOT / "COHORT.json"),
                           "ranks_sha256": digest(ROOT / "RANKS.json")})
        del model
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("retrieve", "score"))
    if parser.parse_args().stage == "retrieve":
        retrieve()
    else:
        score()
