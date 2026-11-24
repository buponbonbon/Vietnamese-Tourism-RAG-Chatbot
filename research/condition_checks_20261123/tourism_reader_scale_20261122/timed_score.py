"""Hash-checked inference on the two fixed pools; cache each scored pair once."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'tourism_stage4_20261121'))
from score_stage4 import ARCHIVE, digest, infer, torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', choices=['BGE', 'M0_42','M1_42','M0_123','M1_123','M0_999','M1_999'])
    args = parser.parse_args()

    for line in (ROOT/'timed_freeze.sha256').read_text().splitlines():
        expected, name = line.split()
        assert digest(ROOT/name) == expected, name
    path = ROOT/'timed_eval.json'
    data = json.loads(path.read_text())
    pairs = [
        (q['id'], p, q['text'], data['candidates'][p]['text'])
        for q in data['questions']
        for p in sorted({p for pool in q['pools'].values() for p in pool['candidates']})
    ]
    assert len({(q,p) for q,p,*_ in pairs}) == len(pairs)
    print(f'{len(pairs)} unique pairs', flush=True)

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    manifest = {m['name']:m for m in json.loads(ARCHIVE.read_text())['models']}
    for name in args.models:
        if name == 'BGE':
            model = 'BGE_pretrained'
        else:
            family, seed = name.split('_')
            model = f'{family}_fixed_1p0_weighted_matched_seed_{seed}'
        output = ROOT/f'timed_scores_{name}.json'
        if output.exists():
            existing = json.loads(output.read_text())
            assert existing['benchmark_sha256'] == digest(path) and existing['checkpoint'] == model
            continue
        start = time.monotonic()
        print(f'{name} starting', flush=True)
        result = infer(manifest[model],pairs)
        result['benchmark_sha256'] = digest(path)
        result['run_seconds'] = round(time.monotonic()-start,2)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)
        print(f'{name}: {len(pairs)} logits in {result["run_seconds"]:.1f} seconds', flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
