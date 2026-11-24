"""Create a closed official corpus and deterministic native BM25 pools."""
from collections import Counter
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import unicodedata

from lxml import html

ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = [ROOT.parent / 'vietnam_travel_natural_20261121',
               ROOT.parent / 'tourism_vietnamese_operators_20261121', ROOT]
CHUNK_WORDS = 105
WINDOW_STEP = 75


def load(name):
    return json.loads((ROOT / name).read_text())


def save(name, value):
    output = ROOT / name
    with output.open('w') as w:
        print(json.dumps(value, ensure_ascii=False, indent=2), file=w)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def normalize(s):
    return ' '.join(unicodedata.normalize('NFC', s).replace('\u200b', '').replace('\ufeff', '').split())


def read_html(path):
    raw = path.read_bytes()
    if raw.startswith(b'\x1f\x8b'):
        raw = gzip.decompress(raw)
    tree = html.fromstring(raw, parser=html.HTMLParser(encoding='utf-8'))
    for node in tree.xpath('//script|//style|//noscript'):
        node.drop_tree()
    return tree


def source_index():
    found = {}
    for directory in SOURCE_DIRS:
        for line in (directory / 'sources' / 'fetch_log.jsonl').read_text().splitlines():
            row = json.loads(line)
            if 'sha256' in row:
                found[row['id']] = (row, directory / 'sources' / (row['id'] + '.html'))
    return found


def span(s, start, end):
    i = s.index(start) if start else 0
    j = s.index(end, i) + len(end) if end else len(s)
    return s[i:j]


def select_source(recipe, sources):
    meta, path = sources[recipe['source']]
    assert digest(path) == meta['sha256']
    tree = read_html(path)
    nodes = tree.xpath(recipe.get('xpath', '//body'))
    assert nodes
    content = normalize(' '.join(' '.join(n.itertext()) for n in nodes))
    text = normalize(span(content, recipe.get('start'), recipe.get('end')))
    assert 100 <= len(text) <= 42000
    return text, meta


def documents():
    sources = source_index()
    recipes = load('corpus_plan.json')
    output = []
    seen_content = set()
    for recipe in recipes:
        text, meta = select_source(recipe, sources)
        words = text.split()
        for i in range(0, len(words), WINDOW_STEP):
            chunk = ' '.join(words[i:i+CHUNK_WORDS])
            if i and len(words) - i < 20:
                break
            if normalize(chunk).casefold() in seen_content:
                continue
            seen_content.add(normalize(chunk).casefold())
            output.append({
                'id': f"{recipe['id']}.{i//WINDOW_STEP:02d}",
                'source': recipe['source'], 'service': recipe['service'],
                'header': recipe['title'], 'url': meta['final_url'],
                'snapshot_sha256': meta['sha256'], 'word_offset': i, 'body': chunk,
                'text': recipe['title'] + '\n' + chunk,
            })
            if i + CHUNK_WORDS >= len(words):
                break
    assert len(output) > 5 and len(output) == len({p['id'] for p in output})
    return output


def tokenize(text):
    return re.findall(r'[^\W_]+', normalize(text).casefold())


class BM25:
    def __init__(self, documents):
        self.bags = {p['id']: Counter(tokenize(p['text'])) for p in documents}
        self.df = Counter()
        for bag in self.bags.values():
            self.df.update(bag.keys())
        self.n = len(documents)
        self.avg = sum(sum(b.values()) for b in self.bags.values()) / self.n

    def scores(self, query):
        query = set(tokenize(query))
        ranks = []
        for pid, bag in self.bags.items():
            k = 1.5*(.25+.75*sum(bag.values())/self.avg)
            score = sum(math.log1p((self.n-self.df[w]+.5)/(self.df[w]+.5))
                        * (2.5*bag[w]/(bag[w]+k))
                        for w in sorted(query & bag.keys()))
            ranks.append((pid,score))
        return sorted(ranks, key=lambda a:(-a[1],a[0]))


if __name__ == '__main__':
    result = documents()
    save('corpus.json', result)
    print(Counter(row['service'] for row in result))
    print(f'{len(result)} passages')
