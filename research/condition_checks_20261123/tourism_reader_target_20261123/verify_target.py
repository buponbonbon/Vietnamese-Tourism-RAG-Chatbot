"""Verify score lineage, case-level diagnostics, and archived-training exact overlap."""
from pathlib import Path
import re
import unicodedata
from summarize_target import ROOT, digest, read, CONDITIONS


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()))


def main():
    repo = Path('/workspace/Vietnamese-Tourism-RAG-Chatbot')
    archive = repo/'data/oracle_recovery_20261113_001/evidence_selection_reinference_001'
    manifest = {m['name']:m for m in read(archive/'protocol.json')['models']}
    training = {}
    for path in sorted((archive/'source_inputs').glob('train_moc_*.json')):
        rows = read(path)
        query = {normalize(r['query']) for r in rows}
        passage = {normalize(r[x]) for r in rows for x in ('positive','negative')}
        training[path.name] = dict(sha256=digest(path), n=len(rows), q=query, p=passage)
    assert sorted(x['n'] for x in training.values()) == [7618,8668]
    frozen = read(ROOT/'reader_final.json')
    results = read(ROOT/'reader_results.json')
    docs = {r['id']:r for r in read(ROOT/'reader_corpus.json')}
    trace = {}
    for key in CONDITIONS[1:]:
        logged = read(ROOT/f'reader_scores_{key}.json')
        m = manifest[logged['checkpoint']]
        expected_name = 'BGE_pretrained' if key=='BGE' else (
            f'{key.split("_")[0]}_fixed_1p0_weighted_matched_seed_{key.split("_")[1]}')
        assert m['name'] == expected_name and logged['checkpoint_files'] == m['file_sha256']
        assert logged['device']=='CPU' and logged['dtype']=='float32'
        assert logged['max_tokens'] == 491 and logged['batch_consistency_max_abs_delta'] < .001
        scores = {(q,p):s for q,p,s in logged['scores']}
        q = frozen[0]
        assert q['id']=='T01' and not q['native_sufficient']
        ordered = sorted(q['oracle'], key=lambda p:(-scores['T01',p],p))
        assert ordered[:2] == ['vna_pet_booking','rail_pet']
        r = results['models'][key]
        trace[key] = dict(margin=scores['T01',ordered[0]]-scores['T01',ordered[1]],
                          condition=r['native_hit'], replacement=r['replacement_hit'])
    all_candidates = {p for q in frozen for p in q['candidate_ids']}
    anchors = {p for q in frozen for p in q['anchors']}
    audit = {}
    for f, t in training.items():
        audit[f] = dict(archive_sha256=t['sha256'], training_rows=t['n'],
                        exact_reader_quote_matches=[q['id'] for q in frozen if normalize(q['quote']) in t['q']],
                        exact_anchor_body_matches=[p for p in sorted(anchors) if normalize(docs[p]['body']) in t['p']],
                        exact_candidate_body_matches=[p for p in sorted(all_candidates) if normalize(docs[p]['body']) in t['p']])
    output = {'checkpoint_audit':trace,'training_exact_overlap':audit,
              'limits':'Exact normalized strings only. No semantic or base pretraining contamination audit.'}
    import json
    (ROOT/'reader_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
