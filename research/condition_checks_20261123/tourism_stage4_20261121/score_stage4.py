"""Inference on all five reviewed reader-question candidates, CPU FP32."""
import gc
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/workspace/Vietnamese-Tourism-RAG-Chatbot/data/oracle_recovery_20261113_001/'
               'evidence_selection_reinference_001/protocol.json')
FAMILIES = ('M0', 'M1')
SEEDS = (42, 123, 999)


def digest(path):
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def infer(meta, pairs):
    checkpoint = Path(meta['path'])
    for name, expected in meta['file_sha256'].items():
        assert digest(checkpoint / name) == expected, name
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, torch_dtype=torch.float32, attn_implementation='sdpa',
        local_files_only=True, trust_remote_code=False).eval()
    encoded = tokenizer([x[2] for x in pairs], [x[3] for x in pairs],
                        truncation=False, padding=False)
    lens = list(map(len, encoded['input_ids']))
    assert max(lens) <= 512, max(lens)

    def batch(indices):
        inputs = tokenizer.pad({k: [encoded[k][i] for i in indices] for k in encoded},
                               padding=True, return_tensors='pt')
        with torch.inference_mode():
            logits = model(**inputs).logits.flatten().tolist()
        assert len(logits) == len(indices) and all(math.isfinite(x) for x in logits)
        return logits

    fixed_order = sorted(range(len(pairs)), key=lambda i: (lens[i], i))
    scores = {}
    for j in range(0, len(pairs), 8):
        indices = fixed_order[j:j+8]
        scores.update(zip(indices, batch(indices), strict=True))
    check = fixed_order[::max(1, len(pairs)//5)][:5]
    max_drift = max(abs(scores[i]-batch([i])[0]) for i in check)
    assert max_drift < 0.001
    return {'checkpoint': meta['name'], 'checkpoint_files': meta['file_sha256'],
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'dtype': 'float32', 'device': 'CPU', 'max_tokens': max(lens),
            'dynamic_padding': True, 'batch_size': 8, 'batch_consistency_max_abs_delta': max_drift,
            'scores': [[*pairs[i][:2], scores[i]] for i in range(len(pairs))]}


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    for line in (ROOT / 'freeze.sha256').read_text().splitlines():
        expected, name = line.split()
        assert expected == digest(ROOT / name), name
    benchmark_path = ROOT / 'benchmark.json'
    questions = json.loads(benchmark_path.read_text())['questions']
    corpus = {p['id']: p for p in json.loads((ROOT / 'corpus.json').read_text())}
    pairs = [(q['id'], pid, q['query'], corpus[pid]['text'])
             for q in questions for pid in q['candidate_ids']]
    assert len({p[:2] for p in pairs}) == len(pairs)
    manifest = {m['name']: m for m in json.loads(ARCHIVE.read_text())['models']}
    conditions = {'BGE': 'BGE_pretrained'}
    for family in FAMILIES:
        for seed in SEEDS:
            name = f'{family}_fixed_1p0_weighted_matched_seed_{seed}'
            conditions[f'{family}_{seed}'] = name
    for key, model in conditions.items():
        output = ROOT / f'scores_{key}.json'
        start = time.monotonic()
        if output.exists():
            previous = json.loads(output.read_text())
            assert previous['benchmark_sha256'] == digest(benchmark_path)
            assert previous['checkpoint'] == model
            continue
        result = infer(manifest[model], pairs)
        result['benchmark_sha256'] = digest(benchmark_path)
        with output.open('x') as stream:
            print(json.dumps(result, ensure_ascii=False, indent=2), file=stream)
        print(f'{key}: {len(pairs)} scores, {time.monotonic()-start:.1f}s', flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
