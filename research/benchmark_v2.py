"""Corpus-anchored, review-gated V2 benchmark preparation. No model training.

Build drafts, mine UNLABELED BM25 candidates, then validate human annotations.
Existing research datasets and application behavior are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'tourism-evidence-v2.1'
SPLITS = ('train', 'validation', 'calibration', 'test')
BEHAVIORS = ('answer', 'correct', 'refuse')
REVIEW_FIELDS = ['reviewer_1_id', 'reviewer_1_label', 'reviewer_2_id',
                 'reviewer_2_label', 'adjudicator_id', 'adjudication_notes']
NUMBER = re.compile(r'(?<!\w)(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|(?:19|20)\d{2}|\d+(?:[.,]\d+)?\s*(?:triệu|nghìn|ngàn|đồng|km|ha|lượt|%))(?!\w)', re.I)
TOURISM = ('du lịch', 'du khách', 'tham quan', 'lễ hội', 'di tích', 'di sản',
           'ẩm thực', 'điểm đến', 'khách sạn', 'bảo tàng', 'vườn quốc gia')
BAD = ('giấy phép số', 'chịu trách nhiệm nội dung', 'contact@', 'thủ tục hành chính')


def normalize(text):
    if isinstance(text, list):
        text = text[0] if text else ''
    return ' '.join(unicodedata.normalize('NFC', str(text or '')).split())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def short_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:20]


def canonical_url(value):
    value = normalize(value)
    u = urlsplit(value)
    if u.scheme.lower() not in ('http', 'https') or not u.hostname or u.username or u.password:
        raise ValueError('Source must be an HTTP(S) URL without embedded credentials')
    host = u.hostname.lower()
    if host.startswith('www.'):
        host = host[4:]
    port = u.port
    if port and port not in (80, 443):
        host += ':' + str(port)
    query = sorted((k, v) for k, v in parse_qsl(u.query, keep_blank_values=True)
                   if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid'))
    return urlunsplit(('https', host, u.path.rstrip('/') or '/', urlencode(query), ''))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            # A deterministic representative also makes input row order irrelevant.
            a, b = sorted((a, b))
            self.parent[b] = a


def load_corpus(path):
    rows = read_csv(path)
    required = {'chunk_id', 'doc_id', 'text_content', 'url', 'title', 'province'}
    if not rows or not required <= rows[0].keys():
        raise ValueError('Corpus missing required fields: ' + ', '.join(sorted(required)))
    ids = set()
    for row in rows:
        cid = row['chunk_id']
        if not cid or cid in ids:
            raise ValueError('Blank or duplicate corpus chunk_id: ' + cid)
        ids.add(cid)
        row['text_content'] = normalize(row['text_content'])
        try:
            row['canonical_url'] = canonical_url(row['url'])
        except ValueError:
            row['canonical_url'] = ''
        row['text_sha256'] = hashlib.sha256(row['text_content'].encode()).hexdigest()
    return rows


def group_sources(rows, reviewed_links=()):
    uf = UnionFind()
    by_text, by_doc = {}, {}
    for row in rows:
        url = row['canonical_url']
        if not url:
            continue  # No synthetic provenance for unknown URLs.
        uf.find(url)
        for index, key in ((by_text, row['text_sha256'] if row['text_content'] else ''),
                           (by_doc, row['doc_id'])):
            if key:
                if key in index:
                    uf.union(url, index[key])
                index[key] = url
    by_chunk = {row['chunk_id']:row for row in rows} if reviewed_links else {}
    for link in reviewed_links:
        check_review(link, link.get('duplicate_decision'), ('duplicate','independent'),
                     'source link ' + link.get('anchor_chunk_id',''))
        a, b = by_chunk.get(link['anchor_chunk_id']), by_chunk.get(link['candidate_chunk_id'])
        if not a or not b or not a['canonical_url'] or not b['canonical_url']:
            raise ValueError('Reviewed source link contains unknown/unresolved corpus provenance')
        if link['duplicate_decision']=='duplicate':
            uf.union(a['canonical_url'], b['canonical_url'])
    members = defaultdict(set)
    for url in list(uf.parent):
        members[uf.find(url)].add(url)
    identifiers = {url: 'SRC_' + short_hash('\n'.join(sorted(urls)))
                   for urls in members.values() for url in urls}
    for row in rows:
        row['source_group_id'] = identifiers.get(row['canonical_url'], '')
    return members


def legacy_exposure(rows, paths):
    by_doc, by_text = defaultdict(set), defaultdict(set)
    for row in rows:
        if row['source_group_id']:
            by_doc[row['doc_id']].add(row['source_group_id'])
            by_text[row['text_sha256']].add(row['source_group_id'])
    exposed = set()
    for path in paths:
        raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if not isinstance(raw, list):
            raise ValueError('Expected a legacy JSON list: ' + str(path))
        for row in raw:
            exposed.update(by_doc.get(normalize(row.get('doc_id')), ()))
            for key, fallback in [('positive', 'pos'), ('negative', 'neg')]:
                text = normalize(row.get(key, row.get(fallback)))
                if text:
                    exposed.update(by_text.get(hashlib.sha256(text.encode()).hexdigest(), ()))
    return exposed


def assign_splits(rows, seed, train_exposed, evaluation_exposed, holdout=()):
    forced_test = {r['source_group_id'] for r in rows
                   if normalize(r['province']).casefold() in {normalize(x).casefold() for x in holdout}}
    # Contaminated geography holdouts are excluded, never relabeled as unseen.
    partitions = {}
    for group in sorted({r['source_group_id'] for r in rows if r['source_group_id']}):
        if group in forced_test and group in train_exposed | evaluation_exposed:
            split = 'quarantine'
        elif group in train_exposed:
            split = 'train'
        elif group in evaluation_exposed:
            split = 'legacy_evaluation'
        elif group in forced_test:
            split = 'test'
        else:
            fraction = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest(), 16) / 2**256
            split = 'train' if fraction < .60 else 'validation' if fraction < .75 else 'calibration' if fraction < .85 else 'test'
        partitions[group] = split
    for row in rows:
        group = row['source_group_id']
        row['split'] = partitions.get(group, 'quarantine')
        row['legacy_train_exposed'] = str(int(group in train_exposed))
        row['legacy_evaluation_exposed'] = str(int(group in evaluation_exposed))
    return partitions


def usable(row):
    text = row['text_content']
    return (420 <= len(text) <= 1700 and len(text.split()) >= 65
            and any(term in text.casefold() for term in TOURISM)
            and not any(term in text.casefold() for term in BAD)
            and NUMBER.search(text) is not None)


def make_family(row):
    text = row['text_content']
    match = NUMBER.search(text)
    original = match.group()
    # A candidate perturbation only. Human reviewers must verify it is false.
    date = re.fullmatch(r'(\d{1,2})([/-])(\d{1,2})(?:([/-])(\d{2,4}))?', original)
    if date:
        day = int(date[1])
        replacement = str(day + 1 if day < 28 else day - 1) + original[len(date[1]):]
    else:
        replacement = re.sub(r'\d+', lambda m: str(int(m.group()) + 1), original, count=1)
    boundary = text.rfind('. ', 0, match.start())
    start = boundary + 2 if boundary >= 0 else 0
    end = text.find('. ', match.end())
    end = len(text) if end == -1 else end + 1
    fact = text[start:end].strip()
    rel_start = match.start() - start
    corrupted = fact[:rel_start] + replacement + fact[rel_start + len(original):]
    masked = fact[:rel_start] + '[GIÁ TRỊ CẦN XÁC MINH]' + fact[rel_start + len(original):]
    title = normalize(row['title'])[:180]
    proposals = {
        'answer': f'Theo bài “{title}”, giá trị chính xác ở vị trí còn thiếu trong thông tin “{masked}” là gì?',
        'correct': f'Theo bài “{title}”, nhận định “{corrupted}” có chính xác không? Nếu sai, hãy sửa và dẫn bằng chứng.',
        'refuse': f'Dữ liệu hiện có có nêu điều kiện ưu đãi dành riêng cho sinh viên liên quan đến “{title}” không? Nếu có, điều kiện cụ thể là gì?',
    }
    family = 'FAM_' + short_hash(row['source_group_id'] + ':' + row['chunk_id'])
    result = []
    for behavior in BEHAVIORS:
        result.append(dict(
            benchmark_id=family + '_' + behavior, family_id=family,
            split=row['split'], source_group_id=row['source_group_id'],
            proposed_behavior=behavior, target_behavior='',
            query=proposals[behavior], reference_answer='',
            anchor_chunk_id=row['chunk_id'], anchor_evidence=text,
            source_url=row['canonical_url'], province=row['province'],
            task_type='TODO', trap_type='numeric_or_date_candidate' if behavior == 'correct' else '',
            trap_original_value=original if behavior == 'correct' else '',
            trap_replacement_value=replacement if behavior == 'correct' else '',
            refusal_scope='', evidence_search_notes='',
            annotation_status='TODO',
            **{key: '' for key in REVIEW_FIELDS}))
    return result


def identity_digest(questions):
    keys = ('benchmark_id','family_id','split','source_group_id','anchor_chunk_id','proposed_behavior')
    identities = sorted(tuple(q[k] for k in keys) for q in questions)
    return hashlib.sha256(json.dumps(identities,ensure_ascii=False).encode()).hexdigest()


def immutable_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Output exists; choose a new version directory: ' + str(path))
    path.mkdir(parents=True)
    return path


def build(args):
    rows = load_corpus(args.corpus)
    source_link_paths = getattr(args, 'source_links', [])
    reviewed_links = [row for path in source_link_paths for row in read_csv(path)]
    groups = group_sources(rows, reviewed_links)
    train = legacy_exposure(rows, args.legacy_train)
    evaluation = legacy_exposure(rows, args.legacy_evaluation)
    partitions = assign_splits(rows, args.seed, train, evaluation, args.holdout_province)
    rng = random.Random(args.seed)
    selected = []
    for split in SPLITS:
        options = defaultdict(list)
        for row in rows:
            if row['split'] == split and usable(row):
                options[row['source_group_id']].append(row)
        keys = sorted(options)
        rng.shuffle(keys)
        if len(keys) < args.groups_per_split:
            raise ValueError(f'{split}: only {len(keys)} eligible source groups; need {args.groups_per_split}')
        for key in keys[:args.groups_per_split]:
            candidates = sorted(options[key], key=lambda r: r['chunk_id'])
            selected.append(rng.choice(candidates))
    drafts = [q for row in selected for q in make_family(row)]
    out = immutable_output(args.out)
    manifest_fields = ['chunk_id', 'doc_id', 'canonical_url', 'source_group_id', 'split',
                       'province', 'text_sha256', 'legacy_train_exposed', 'legacy_evaluation_exposed']
    write_csv(out / 'corpus_manifest.csv', ({k:r[k] for k in manifest_fields} for r in rows))
    write_csv(out / 'benchmark_draft.csv', drafts)
    # Labels are blank, including the source anchor. A proposal is not a qrel.
    qrels = [dict(benchmark_id=q['benchmark_id'], chunk_id=q['anchor_chunk_id'],
                  relevance_grade='', evidence_quote='', annotation_status='TODO',
                  **{key:'' for key in REVIEW_FIELDS}) for q in drafts]
    write_csv(out / 'qrels_review.csv', qrels)
    configuration = dict(seed=args.seed, groups_per_split=args.groups_per_split,
                         holdout_province=args.holdout_province,
                         legacy_train=[str(p.resolve()) for p in args.legacy_train],
                         legacy_evaluation=[str(p.resolve()) for p in args.legacy_evaluation],
                         source_links=[str(p.resolve()) for p in source_link_paths])
    meta = dict(schema=SCHEMA, paper_ready=False, corpus_sha256=digest(args.corpus),
                legacy_input_hashes={str(p.resolve()):digest(p) for p in args.legacy_train + args.legacy_evaluation},
                source_link_hashes={str(p.resolve()):digest(p) for p in source_link_paths},
                config=configuration, corpus_chunks=len(rows),
                canonical_source_urls=sum(len(x) for x in groups.values()),
                source_components=len(groups), largest_component_urls=max(map(len, groups.values())),
                group_partition_counts=dict(Counter(partitions.values())),
                rows=len(drafts), draft_behavior_counts=dict(Counter(q['proposed_behavior'] for q in drafts)),
                draft_identity_sha256=identity_digest(drafts),
                output_hashes={name:digest(out/name) for name in ('corpus_manifest.csv','benchmark_draft.csv','qrels_review.csv')},
                limitations=['Draft templates are not natural user queries or verified labels.',
                             'Refusal proposals are NOT proven unanswerable in the corpus.',
                             'Exact duplicate grouping does not establish absence of semantic near-duplicates.',
                             'Legacy source exposure is observable only through known IDs or exact text.',
                             'Initial annotation batch; no statistical power claim.'])
    write_json(out/'build_manifest.json', meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def check_review(row, final_label, allowed, where):
    if row['annotation_status'] != 'VERIFIED':
        raise ValueError(where + ': human annotation is not VERIFIED')
    one, two = normalize(row.get('reviewer_1_id')), normalize(row.get('reviewer_2_id'))
    if not one or not two or one == two:
        raise ValueError(where + ': two distinct reviewers are required')
    a, b = row.get('reviewer_1_label'), row.get('reviewer_2_label')
    if a not in allowed or b not in allowed or final_label not in allowed:
        raise ValueError(where + ': invalid or missing review labels')
    if a != b:
        adjudicator = normalize(row.get('adjudicator_id'))
        if (not adjudicator or adjudicator in (one,two)
                or not normalize(row.get('adjudication_notes'))):
            raise ValueError(where + ': disagreement needs independent adjudication')
    elif final_label != a:
        raise ValueError(where + ': final label contradicts agreeing reviewers')


def validate_bundle(bundle, corpus, allow_draft=False):
    bundle = Path(bundle)
    meta = json.loads((bundle/'build_manifest.json').read_text())
    if meta.get('schema') != SCHEMA or digest(corpus) != meta['corpus_sha256']:
        raise ValueError('Corpus/schema differs from the frozen build')
    for path, expected in meta['legacy_input_hashes'].items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError('Legacy provenance input missing or changed: ' + path)
    for path, expected in meta.get('source_link_hashes',{}).items():
        if not Path(path).is_file() or digest(path) != expected:
            raise ValueError('Reviewed source-link input missing or changed: ' + path)
    manifest_path = bundle/'corpus_manifest.csv'
    if digest(manifest_path) != meta['output_hashes']['corpus_manifest.csv']:
        raise ValueError('Source/split manifest changed after build')
    source_rows = read_csv(manifest_path)
    if len(source_rows) != meta['corpus_chunks'] or len({r['chunk_id'] for r in source_rows}) != len(source_rows):
        raise ValueError('Incomplete or duplicate corpus manifest')
    by_chunk = {r['chunk_id']:r for r in source_rows}
    for key in ('source_group_id', 'canonical_url', 'text_sha256', 'doc_id'):
        seen = defaultdict(set)
        for row in source_rows:
            if row[key] and row['split'] != 'quarantine':
                seen[row[key]].add(row['split'])
        if any(len(v) > 1 for v in seen.values()):
            raise ValueError('Cross-split source/text identity: ' + key)
    if any(r['split'] in SPLITS[1:] and (r['legacy_train_exposed']=='1' or r['legacy_evaluation_exposed']=='1') for r in source_rows):
        raise ValueError('Legacy-exposed evidence in a held-out partition')
    questions = read_csv(bundle/'benchmark_draft.csv')
    if not questions or len({q['benchmark_id'] for q in questions}) != len(questions):
        raise ValueError('Empty or duplicate benchmark IDs')
    if identity_digest(questions) != meta['draft_identity_sha256']:
        raise ValueError('Benchmark membership/source identities changed after build; create a new version')
    indexed = {q['benchmark_id']:q for q in questions}
    corpus_rows = {r['chunk_id']:r for r in load_corpus(corpus)}
    families, query_splits = defaultdict(set), defaultdict(set)
    for q in questions:
        anchor = by_chunk.get(q['anchor_chunk_id'])
        if (not anchor or q['split'] not in SPLITS or q['split'] != anchor['split']
                or q['source_group_id'] != anchor['source_group_id']):
            raise ValueError(q['benchmark_id'] + ': wrong anchor/source/split')
        if normalize(q['anchor_evidence']) != corpus_rows[q['anchor_chunk_id']]['text_content']:
            raise ValueError(q['benchmark_id'] + ': anchor evidence was altered')
        families[q['family_id']].add(q['split'])
        query_splits[normalize(q['query'])].add(q['split'])
        if not normalize(q['query']):
            raise ValueError(q['benchmark_id'] + ': empty query')
        if not allow_draft:
            behavior = q['target_behavior']
            check_review(q, behavior, BEHAVIORS, q['benchmark_id'])
            if not normalize(q['task_type']) or q['task_type']=='TODO' or not normalize(q['reference_answer']):
                raise ValueError(q['benchmark_id'] + ': missing tourism task/reference response')
            if not normalize(q['evidence_search_notes']):
                raise ValueError(q['benchmark_id'] + ': missing pooled evidence review notes')
            if behavior == 'refuse' and q['refusal_scope'] != 'corpus_snapshot':
                raise ValueError(q['benchmark_id'] + ': refusal must specify corpus_snapshot scope')
            if behavior == 'correct':
                a, b = q['trap_original_value'], q['trap_replacement_value']
                if not a or not b or normalize(a)==normalize(b) or normalize(a) not in normalize(q['anchor_evidence']):
                    raise ValueError(q['benchmark_id'] + ': unsupported or empty correction anchor')
    if any(len(s)>1 for s in families.values()) or any(len(s)>1 for s in query_splits.values()):
        raise ValueError('Counterfactual family or identical query crosses partitions')
    sufficient, seen = defaultdict(set), set()
    for qrel in read_csv(bundle/'qrels_review.csv'):
        qid, cid = qrel['benchmark_id'], qrel['chunk_id']
        if qid not in indexed or cid not in by_chunk or (qid,cid) in seen:
            raise ValueError('Unknown or duplicate qrel identity')
        seen.add((qid,cid))
        if not allow_draft:
            grade = qrel['relevance_grade']
            check_review(qrel, grade, ('0','1','2'), qid + '/' + cid)
            if grade in ('1','2'):
                quote = normalize(qrel['evidence_quote'])
                if not quote or quote not in corpus_rows[cid]['text_content']:
                    raise ValueError(qid + ': qrel quote missing from corpus passage')
                if grade == '2':
                    if by_chunk[cid]['split'] != indexed[qid]['split']:
                        raise ValueError(qid + ': answer-supporting evidence crosses source partitions')
                    sufficient[qid].add(cid)
    if not allow_draft:
        for q in questions:
            if q['target_behavior']=='refuse' and sufficient[q['benchmark_id']]:
                raise ValueError(q['benchmark_id'] + ': refusal conflicts with relevant qrels')
            if q['target_behavior']!='refuse' and not sufficient[q['benchmark_id']]:
                raise ValueError(q['benchmark_id'] + ': answer/correction has no relevant evidence')
        for split in SPLITS:
            present={q['target_behavior'] for q in questions if q['split']==split}
            if present != set(BEHAVIORS):
                raise ValueError(split + ': all three behaviors must be represented')
        # Recheck the current anchors; an old audit file is not a readiness flag.
        audit_rows = list(corpus_rows.values())
        for row in audit_rows:
            row.update(split=by_chunk[row['chunk_id']]['split'],
                       source_group_id=by_chunk[row['chunk_id']]['source_group_id'])
        independent = set()
        for path in meta.get('source_link_hashes',{}):
            for link in read_csv(path):
                check_review(link,link.get('duplicate_decision'),('duplicate','independent'),'source review')
                if link['duplicate_decision']=='independent':
                    independent.add(frozenset((link['anchor_chunk_id'],link['candidate_chunk_id'])))
        unresolved = [r for r in near_matches(audit_rows,{q['anchor_chunk_id'] for q in questions},.85)
                      if frozenset((r['anchor_chunk_id'],r['candidate_chunk_id'])) not in independent]
        if unresolved:
            raise ValueError(f'{len(unresolved)} unresolved cross-partition near-duplicate flags; review and build a new version')
    return dict(schema=SCHEMA, review_gate_passed=not allow_draft, rows=len(questions),
                split_counts=dict(Counter(q['split'] for q in questions)),
                family_count=len(families),
                annotation_status_counts=dict(Counter(q['annotation_status'] for q in questions)),
                benchmark_sha256=digest(bundle/'benchmark_draft.csv'), qrels_sha256=digest(bundle/'qrels_review.csv'),
                scope='Structure only' if allow_draft else 'Review-record and provenance checks; human truthfulness cannot be certified by code')


def mine(args):
    validate_bundle(args.bundle, args.corpus, allow_draft=args.allow_draft)
    sys.path.insert(0, str(ROOT))
    from ESWA_candidate_set_preflight import bm25_topk, read_corpus
    from underthesea import word_tokenize
    corpus, by_id, _, _, warnings = read_corpus(args.corpus)
    if warnings:
        raise ValueError('Fix corpus warnings before mining: ' + '; '.join(warnings[:3]))
    inverted = json.loads(args.inverted_index.read_text())
    questions = read_csv(args.bundle/'benchmark_draft.csv')
    partitions = {r['chunk_id']:r['split'] for r in read_csv(args.bundle/'corpus_manifest.csv')}
    questions = sorted(questions, key=lambda q:q['benchmark_id'])
    random.Random(args.seed).shuffle(questions)
    questions = questions[:args.sample]
    average = sum(r['word_count'] for r in corpus)/len(corpus)
    results = []
    for q in questions:
        hits = bm25_topk(q['query'], inverted, by_id, len(corpus), average, args.top_k, word_tokenize)
        for rank, (cid, score) in enumerate(hits,1):
            results.append(dict(benchmark_id=q['benchmark_id'], query=q['query'], query_split=q['split'],
                                candidate_chunk_id=cid, candidate_split=partitions[cid],
                                train_partition_eligible=str(int(q['split']=='train' and partitions[cid]=='train')),
                                retriever='BM25', rank=rank, score=score,
                                candidate_text=by_id[cid]['text'], label_status='UNLABELED',
                                relevance_grade='', false_negative_review='', annotation_status='TODO',
                                **{key:'' for key in REVIEW_FIELDS}))
    out=immutable_output(args.out)
    fields=['benchmark_id','query','query_split','candidate_chunk_id','candidate_split',
            'train_partition_eligible','retriever','rank','score',
            'candidate_text','label_status','relevance_grade','false_negative_review','annotation_status']+REVIEW_FIELDS
    write_csv(out/'candidate_review.csv', results, fields)
    write_json(out/'mining_manifest.json', dict(schema=SCHEMA, paper_ready=False, label_status='UNLABELED',
               corpus_sha256=digest(args.corpus), index_sha256=digest(args.inverted_index),
               benchmark_sha256=digest(args.bundle/'benchmark_draft.csv'),
               seed=args.seed, top_k=args.top_k, queries=len(questions), candidates=len(results),
               input_drafts=args.allow_draft, retrieval='BM25 only; dense/RRF pooling outstanding',
               candidate_review_sha256=digest(out/'candidate_review.csv')))
    print(f'Saved {len(results)} UNLABELED candidates for {len(questions)} queries to {out}')


def shingles(text):
    tokens = re.findall(r'\w+', normalize(text).casefold(), flags=re.UNICODE)
    return {tuple(tokens[i:i+5]) for i in range(max(0,len(tokens)-4))}


def near_matches(rows, anchor_ids, threshold):
    """Exact Jaccard verification with a recall-preserving prefix candidate filter.

    For J(A,B)>=t, at most (1-t)*|A| shingles from A may be absent in B.
    Thus every matching B contains a shingle in any floor((1-t)*|A|)+1 subset.
    This covers token-5-gram similarity for the selected anchors, not semantic copies.
    """
    if not 0 < threshold <= 1:
        raise ValueError('Jaccard threshold must be in (0,1]')
    anchors = {r['chunk_id']:(r,shingles(r['text_content'])) for r in rows if r['chunk_id'] in anchor_ids}
    prefix = defaultdict(set)
    for cid, (_, tokens) in anchors.items():
        if not tokens:
            continue
        size = min(len(tokens), math.floor((1-threshold)*len(tokens)+1e-9)+1)
        for token in sorted(tokens)[:size]:
            prefix[token].add(cid)
    matches=[]
    for row in rows:
        tokens=shingles(row['text_content'])
        candidates=set()
        for token in tokens:
            candidates.update(prefix.get(token,()))
        for cid in sorted(candidates):
            anchor, a = anchors[cid]
            if anchor['split']==row['split'] or not a or not tokens:
                continue
            if min(len(a),len(tokens))/max(len(a),len(tokens)) + 1e-12 < threshold:
                continue
            jaccard=len(a & tokens)/len(a | tokens)
            if jaccard + 1e-12 >= threshold:
                matches.append(dict(anchor_chunk_id=cid, anchor_split=anchor['split'],
                                    anchor_group=anchor['source_group_id'], candidate_chunk_id=row['chunk_id'],
                                    candidate_split=row['split'], candidate_group=row['source_group_id'],
                                    token_5gram_jaccard=jaccard, anchor_evidence=anchor['text_content'],
                                    candidate_evidence=row['text_content'], duplicate_decision='',
                                    annotation_status='TODO', **{k:'' for k in REVIEW_FIELDS}))
    return matches


def audit_near(args):
    validate_bundle(args.bundle,args.corpus,allow_draft=True)
    rows=load_corpus(args.corpus)
    manifest={r['chunk_id']:r for r in read_csv(args.bundle/'corpus_manifest.csv')}
    for row in rows:
        row.update(split=manifest[row['chunk_id']]['split'],
                   source_group_id=manifest[row['chunk_id']]['source_group_id'])
    anchors={q['anchor_chunk_id'] for q in read_csv(args.bundle/'benchmark_draft.csv')}
    matches=near_matches(rows, anchors, args.threshold)
    out=immutable_output(args.out)
    fields=['anchor_chunk_id','anchor_split','anchor_group','candidate_chunk_id','candidate_split',
            'candidate_group','token_5gram_jaccard','anchor_evidence','candidate_evidence',
            'duplicate_decision','annotation_status']+REVIEW_FIELDS
    write_csv(out/'near_duplicate_review.csv',matches,fields)
    write_json(out/'near_duplicate_manifest.json',dict(schema=SCHEMA,paper_ready=False,
               corpus_sha256=digest(args.corpus), benchmark_sha256=digest(args.bundle/'benchmark_draft.csv'),
               source_manifest_sha256=digest(args.bundle/'corpus_manifest.csv'),
               threshold=args.threshold, anchors=len(anchors), compared_corpus_chunks=len(rows),
               cross_partition_matches=len(matches),
               cross_partition_source_pairs=len({tuple(sorted((r['anchor_group'],r['candidate_group']))) for r in matches}),
               review_sha256=digest(out/'near_duplicate_review.csv'),
               limitations=['Only selected anchors are checked against the full corpus.',
                            'Token five-gram Jaccard does not cover semantic paraphrases.',
                            'Flags require human review; no automatic duplicate labels or split mutation.']))
    print(f'Saved {len(matches)} cross-partition similarity flags for {len(anchors)} anchors to {out}')


def packets(args):
    validate_bundle(args.bundle,args.corpus,allow_draft=True)
    questions=read_csv(args.bundle/'benchmark_draft.csv')
    # Hide the proposed behavior, source split, and every other reviewer's judgment.
    rows=[dict(benchmark_id=q['benchmark_id'],query=q['query'],anchor_evidence=q['anchor_evidence'],
               source_url=q['source_url'],task_type='',behavior_label='',reference_answer='',
               query_revision='',evidence_search_notes='',reviewer_id='',review_status='TODO')
          for q in questions]
    question_by_id={q['benchmark_id']:q for q in questions}
    corpus_by_id={r['chunk_id']:r for r in load_corpus(args.corpus)}
    pairs={(r['benchmark_id'],r['chunk_id']) for r in read_csv(args.bundle/'qrels_review.csv')}
    candidates=getattr(args,'candidates',None)
    if candidates:
        pairs.update((r['benchmark_id'],r['candidate_chunk_id']) for r in read_csv(candidates))
    evidence_rows=[]
    for qid,cid in sorted(pairs):
        if qid not in question_by_id or cid not in corpus_by_id:
            raise ValueError('Review packet contains an unknown query or corpus passage')
        passage=corpus_by_id[cid]
        evidence_rows.append(dict(benchmark_id=qid,query=question_by_id[qid]['query'],chunk_id=cid,
                                  candidate_evidence=passage['text_content'],source_url=passage['canonical_url'],
                                  relevance_grade='',evidence_quote='',reviewer_id='',review_status='TODO'))
    out=immutable_output(args.out)
    for name in ('reviewer_1_questions.csv','reviewer_2_questions.csv'):
        write_csv(out/name,rows)
    for name in ('reviewer_1_qrels.csv','reviewer_2_qrels.csv'):
        write_csv(out/name,evidence_rows)
    write_json(out/'packet_manifest.json',dict(schema=SCHEMA,paper_ready=False,
               benchmark_sha256=digest(args.bundle/'benchmark_draft.csv'),rows_per_reviewer=len(rows),
               qrel_rows_per_reviewer=len(evidence_rows),
               candidate_input_sha256=digest(candidates) if candidates else None,
               hidden=['proposed_behavior','target_behavior','split','other_reviewer_labels'],
               note='Separate packets support independent review; they are not access-control or proof of actual human participation.'))
    print(f'Saved two independent packets of {len(rows)} questions / {len(evidence_rows)} passage reviews to {out}')


def audit_pairs(args):
    summaries, quarantined = [], []
    for role, paths in [('train',args.train),('evaluation',args.evaluation)]:
        for path in paths:
            rows=json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(rows,list):
                raise ValueError('Expected pair list: ' + str(path))
            reasons=Counter()
            invalid=set()
            for index,row in enumerate(rows):
                p=normalize(row.get('positive',row.get('pos')))
                n=normalize(row.get('negative',row.get('neg')))
                q=normalize(row.get('query'))
                problems=[]
                if not q: problems.append('empty_query')
                if not p: problems.append('empty_positive')
                if not n: problems.append('empty_negative')
                if p and p==n: problems.append('identical_normalized_positive_negative')
                if problems:
                    invalid.add(index)
                    reasons.update(problems)
                    quarantined.append(dict(dataset=str(path.resolve()),role=role,zero_based_index=index,
                                            doc_id=normalize(row.get('doc_id')),reasons=';'.join(problems),
                                            margin=row.get('margin',''),query=q,positive=p,negative=n))
            summaries.append(dict(path=str(path.resolve()),sha256=digest(path),role=role,rows=len(rows),
                                  mechanically_flagged_rows=len(invalid),reason_counts=dict(reasons),
                                  margin_counts=dict(Counter(str(r.get('margin')) for r in rows))))
    out=immutable_output(args.out)
    fields=['dataset','role','zero_based_index','doc_id','reasons','margin','query','positive','negative']
    write_csv(out/'quarantine_review.csv',quarantined,fields)
    write_json(out/'pair_audit_manifest.json',dict(schema=SCHEMA,paper_ready=False,inputs=summaries,
               quarantined_rows=len(quarantined),output_sha256=digest(out/'quarantine_review.csv'),
               note='Review ledger only. Originals retained; no clean training set, relabeling or replacement TEST exported.'))
    print(f'Saved {len(quarantined)} mechanical-quality flags across {len(summaries)} datasets to {out}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build_p=sub.add_parser('build')
    build_p.add_argument('--corpus',type=Path,default=ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
    build_p.add_argument('--legacy-train',type=Path,nargs='+',default=[ROOT/'data/processed/train_moc_13.json'])
    build_p.add_argument('--legacy-evaluation',type=Path,nargs='+',default=[ROOT/'data/processed/cross_encoder_VAL.json',ROOT/'data/processed/cross_encoder_TEST.json'])
    build_p.add_argument('--out',type=Path,default=ROOT/'data/research_v2/batch_001')
    build_p.add_argument('--groups-per-split',type=int,default=25)
    build_p.add_argument('--seed',type=int,default=42)
    build_p.add_argument('--holdout-province',action='append',default=[])
    build_p.add_argument('--source-links',type=Path,nargs='*',default=[],help='Fully reviewed near-duplicate source-link CSVs')
    valid_p=sub.add_parser('validate')
    mine_p=sub.add_parser('mine')
    for child in (valid_p,mine_p):
        child.add_argument('--bundle',type=Path,required=True)
        child.add_argument('--corpus',type=Path,default=ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
        child.add_argument('--allow-draft',action='store_true')
    mine_p.add_argument('--inverted-index',type=Path,default=ROOT/'data/processed/inverted_index.json')
    mine_p.add_argument('--out',type=Path,required=True)
    mine_p.add_argument('--sample',type=int,default=20)
    mine_p.add_argument('--top-k',type=int,default=10)
    mine_p.add_argument('--seed',type=int,default=42)
    near_p=sub.add_parser('near-duplicates')
    near_p.add_argument('--bundle',type=Path,required=True)
    near_p.add_argument('--corpus',type=Path,default=ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
    near_p.add_argument('--out',type=Path,required=True)
    near_p.add_argument('--threshold',type=float,default=.85)
    packet_p=sub.add_parser('packets')
    packet_p.add_argument('--bundle',type=Path,required=True)
    packet_p.add_argument('--corpus',type=Path,default=ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
    packet_p.add_argument('--out',type=Path,required=True)
    packet_p.add_argument('--candidates',type=Path,help='Optional candidate_review.csv to include in blinded qrel packets')
    pairs_p=sub.add_parser('audit-pairs')
    pairs_p.add_argument('--train',type=Path,nargs='+',default=[ROOT/f'data/processed/train_moc_{n}.json' for n in (0,6,13)])
    pairs_p.add_argument('--evaluation',type=Path,nargs='+',default=[ROOT/f'data/processed/cross_encoder_{n}.json' for n in ('VAL','TEST')])
    pairs_p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='build':
        if args.groups_per_split < 1:
            parser.error('--groups-per-split must be positive')
        build(args)
    elif args.command=='mine':
        if args.sample<1 or args.top_k<1:
            parser.error('--sample/--top-k must be positive')
        mine(args)
    elif args.command=='near-duplicates':
        if not 0 < args.threshold <= 1:
            parser.error('--threshold must be in (0,1]')
        audit_near(args)
    elif args.command=='packets':
        packets(args)
    elif args.command=='audit-pairs':
        audit_pairs(args)
    else:
        print(json.dumps(validate_bundle(args.bundle,args.corpus,args.allow_draft),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
