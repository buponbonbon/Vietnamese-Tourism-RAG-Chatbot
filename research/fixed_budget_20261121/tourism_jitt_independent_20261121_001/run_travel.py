"""Rank the full historical forum corpus and rerank five answers per question."""

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import re
import time

for setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[setting] = "4"
os.environ["HF_HUB_OFFLINE"] = "1"

import faiss
import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

from prepare_travel import ROOT, load, parse, save, sha

PRIOR = ROOT.parent / "tourism_retrieval_budget_20261120_001"
REPO = Path(os.environ.get("TOURISM_REPO", "/workspace/Vietnamese-Tourism-RAG-Chatbot"))
ARCHIVE = REPO / "data/oracle_recovery_20261113_001/evidence_selection_reinference_001"
MODELS = {"BGE": "BGE_pretrained", "M0": "M0_fixed_1p0_weighted_matched_seed_42"}
WORD = re.compile(r"[^\W_]+")
ARMS = ("bm25", "dense", "rrf")


def indexed_bm25(texts, queries, full):
    lengths = np.asarray([len(text.split()) for text in texts], dtype=float)
    normalizer = 1.5 * (.25 + .75 * lengths / lengths.mean())
    wanted_terms = {w for q in queries for w in WORD.findall(q["text"].lower())}
    inverted = defaultdict(list)
    for i, text in enumerate(texts):
        for w, tf in Counter(WORD.findall(text.lower())).items():
            if w in wanted_terms:
                inverted[w].append((i, tf))
    contributions = {}
    for w, rows in inverted.items():
        ids, tf = np.asarray(rows).T
        idf = math.log(1 + (len(texts) - len(ids) + .5) / (len(ids) + .5))
        contributions[w] = (ids, idf * (tf * 2.5 / (tf + normalizer[ids])))
    result = {}
    rows = np.arange(len(texts))
    for field in full:
        result[field] = {}
        for q in queries:
            scores = np.zeros(len(texts))
            for w in WORD.findall(q[field].lower()):
                if w in contributions:
                    ids, impact = contributions[w]
                    scores[ids] += impact
            result[field][q["qid"]] = np.lexsort((rows, -scores))[:50].tolist()
    return result


def dense_rank(texts, queries, full):
    model = PRIOR / "dense_model"
    manifest = json.loads((PRIOR / "dense_download_manifest.json").read_text())
    for file, expected in manifest.items():
        assert sha(model / file) == expected["sha256"], file
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=512)
    unk = tok.token_to_id("<unk>")
    emb = load_file(str(model / "model.safetensors"))["embeddings"]
    assert emb.shape == (tok.get_vocab_size(), 256) and emb.dtype == np.float32

    def encode(strings):
        vecs = np.zeros((len(strings), 256), dtype=np.float32)
        for i, row in enumerate(tok.encode_batch_fast(strings, add_special_tokens=False)):
            ids = [token for token in row.ids if token != unk]
            if ids:
                vec = emb[ids].mean(axis=0)
                vecs[i] = vec / (np.linalg.norm(vec) + 1e-32)
        assert np.isfinite(vecs).all()
        return vecs

    faiss.omp_set_num_threads(4)
    index = faiss.IndexFlatIP(256)
    for start in range(0, len(texts), 512):
        index.add(encode(texts[start:start + 512]))
    assert index.ntotal == len(texts)
    result = {}
    for field in full:
        ranks = {}
        for start in range(0, len(queries), 16):
            block = queries[start:start + 16]
            qv = encode([q[field] for q in block])
            assert (np.linalg.norm(qv, axis=-1) > .99).all()
            sim, ids = index.search(qv, len(texts))
            for q, scores, rows in zip(block, sim, ids, strict=True):
                assert (rows >= 0).all()
                ranks[q["qid"]] = rows[np.lexsort((rows, -scores))[:50]].tolist()
        result[field] = ranks
    return result, {name: rec["sha256"] for name, rec in manifest.items()}


def fuse(a, b):
    ra = {d: i for i, d in enumerate(a, 1)}
    rb = {d: i for i, d in enumerate(b, 1)}

    def key(d):
        s = (1 / (60 + ra[d]) if d in ra else 0.) + (1 / (60 + rb[d]) if d in rb else 0.)
        return -s, ra.get(d, 51), rb.get(d, 51), d

    return sorted(ra.keys() | rb.keys(), key=key)[:50]


def retrieve():
    t = time.monotonic()
    answers, queries, meta = parse()
    assert meta == load("CORPUS_AUDIT.json")
    ids, texts = list(answers), [x[2] for x in answers.values()]
    fields = ("text", "title")
    a = indexed_bm25(texts, queries, fields)
    print("BM25 finished", round(time.monotonic() - t, 1), flush=True)
    b, hashes = dense_rank(texts, queries, fields)
    print("dense finished", round(time.monotonic() - t, 1), flush=True)
    result = {}
    for field in fields:
        ranks = {}
        for q in queries:
            key = q["qid"]
            lex, sem = ([ids[row] for row in r[field][key]] for r in (a, b))
            ranks[key] = {"gold": q["gold"], "bm25": lex, "dense": sem, "rrf": fuse(lex, sem)}
        result[field] = ranks
        print(field, {arm: sum(row["gold"] in row[arm][:5] for row in ranks.values()) for arm in ARMS},
              flush=True)
    save("RANKS.json", result)
    save("RANK_RUN.json", {"seconds": time.monotonic() - t, "dense_input_hashes": hashes,
                           "protocol_hash": sha(ROOT / "PROTOCOL.json"),
                           "manifest_hash": sha(ROOT / "CORPUS_AUDIT.json"),
                           "faiss": faiss.__version__})


def score():
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    docs, qs, meta = parse()
    assert meta == load("CORPUS_AUDIT.json")
    query_text = {q["qid"]: q["text"] for q in qs}
    rank = load("RANKS.json")["text"]
    pairs = sorted({(int(q), d) for q, r in rank.items() for arm in ARMS for d in r[arm][:5]})
    left, right = zip(*[(query_text[q], docs[d][2]) for q, d in pairs])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    fixed = json.loads((ARCHIVE / "protocol.json").read_text())
    for kind, name in MODELS.items():
        path = ROOT / (kind + "_scores.json")
        if path.is_file():
            continue
        info = next(row for row in fixed["models"] if row["name"] == name)
        folder = Path(info["path"])
        for file, expected in info["file_sha256"].items():
            assert sha(folder / file) == expected, file
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        enc = tok(list(left), list(right), max_length=512, truncation=True, padding=False)
        raw = tok(list(left), list(right), truncation=False, padding=False)["input_ids"]
        size = list(map(len, enc["input_ids"]))
        order = sorted(range(len(pairs)), key=lambda i: (size[i], i))
        rows = []
        start = time.monotonic()
        for i in range(0, len(order), 4):
            batch = order[i:i + 4]
            inputs = tok.pad([{f: enc[f][j] for f in enc} for j in batch], return_tensors="pt")
            with torch.inference_mode():
                logits = model(**inputs).logits.float().reshape(-1).tolist()
            assert all(math.isfinite(x) for x in logits)
            for j, val in zip(batch, logits, strict=True):
                q, d = pairs[j]
                rows.append({"q": q, "d": d, "logit": val, "tokens": size[j], "truncated": len(raw[j]) > 512})
            if (i + 4) % 112 == 0 or i + 4 >= len(order):
                print(kind, len(rows), "/", len(order), round(time.monotonic() - start, 1), "s", flush=True)
        save(path.name, {"ranks_sha256": sha(ROOT / "RANKS.json"), "checkpoint": info["file_sha256"],
                         "runtime_s": time.monotonic() - start, "torch": torch.__version__,
                         "transformers": transformers.__version__, "rows": rows})
        del model
        gc.collect()


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=("retrieve", "score"))
    {"retrieve": retrieve, "score": score}[cli.parse_args().stage]()
