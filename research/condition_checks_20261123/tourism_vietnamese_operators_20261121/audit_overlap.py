"""Report exact query/source overlap and character-level near duplication."""
import csv
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
INPUTS = Path('/workspace/Vietnamese-Tourism-RAG-Chatbot/data/oracle_recovery_20261113_001/'
              'evidence_selection_reinference_001/source_inputs')


def norm(s):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', s).casefold()))


def ngrams(s):
    tokens = s.split()
    return set(zip(tokens, tokens[1:], tokens[2:]))


def key(address):
    parts = urlsplit(address)
    return parts.netloc.lower().removeprefix('www.'), parts.path.rstrip('/')


def main():
    plans = [INPUTS / f'train_moc_{level}.json' for level in (0, 13)]
    unique = set()
    for path in plans:
        unique.update(norm(row['query']) for row in json.loads(path.read_text()))
    train = [(s, ngrams(s)) for s in sorted(unique)]
    corpus = INPUTS / 'vietnam_tourism_rag_dataset.csv'
    with corpus.open() as stream:
        rows = list(csv.DictReader(stream))
    urls = {key(row['url']) for row in rows}
    normalized_corpus = [norm(row['text_content']) for row in rows]
    benchmark = json.loads((ROOT / 'benchmark.json').read_text())
    result = []
    for row in benchmark['questions']:
        q = norm(row['query'])
        grams = ngrams(q)
        def similarity(t):
            union = grams | t[1]
            return len(grams & t[1]) / len(union) if union else 0
        top = max(train, key=similarity)
        story_url = key(row['article_url'])
        train_relative = max((similarity(t) for t in train if t[0] in q), default=0)
        result.append({'case': row['id'], 'source_url_in_index': story_url in urls,
                       'reference_urls_in_index': sum(key(d['url']) in urls for d in row['candidates']),
                       'exact_training_query': q in unique,
                       'full_excerpt_in_index': any(q in text for text in normalized_corpus),
                       'max_training_query_token_trigram_jaccard': similarity(top),
                       'max_train_query_substring_token_trigram_jaccard': train_relative})
    print(json.dumps({'basis': 'Recovered M0/M1 input pair files plus archived corpus CSV; not pretraining coverage.',
                      'hashes': {x.name: sha256(x.read_bytes()).hexdigest() for x in [*plans, corpus]},
                      'unique_training_queries': len(unique), 'indexed_rows': len(rows),
                      'case_details': result}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
