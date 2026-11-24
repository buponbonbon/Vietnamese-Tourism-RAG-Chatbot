"""Build the prescored all-Vietnamese extension from frozen source locators."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import gzip
import json
import re
from pathlib import Path

from lxml import html

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'vietnam_travel_natural_20261121'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(text.replace('\u200b', '').replace('\ufeff', '').split())


def read_html(path):
    raw = path.read_bytes()
    if raw.startswith(b'\x1f\x8b'):
        raw = gzip.decompress(raw)
    doc = html.fromstring(raw, parser=html.HTMLParser(encoding='utf-8'))
    for node in doc.xpath('//script|//style|//noscript'):
        node.drop_tree()
    return doc


def choose(container, span):
    first, last = span
    where = container.find(first)
    assert where >= 0, first
    stop = container.find(last, where)
    assert stop >= 0, last
    return container[where:stop + len(last)]


def lookup_comment(items, id_val):
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item['comment_id']) == str(id_val):
            return item
        found = lookup_comment(item.get('replys') or [], id_val)
        if found is not None:
            return found
    return None


def load_sources():
    index = {}
    for folder in (ORIGINAL, ROOT):
        for line in (folder / 'sources' / 'fetch_log.jsonl').read_text().splitlines():
            row = json.loads(line)
            if 'sha256' in row:
                index[row['id']] = row, folder / 'sources' / (row['id'] + '.html')
    return index


def query(recipe, index):
    key, locator = recipe['story'], recipe['query_loc']
    meta, file = index[key]
    assert digest(file) == meta['sha256'], key
    if recipe.get('comment_source'):
        meta_comment, comment_file = index[recipe['comment_source']]
        assert digest(comment_file) == meta_comment['sha256']
        comments = json.loads(comment_file.read_bytes())['data']['items']
        record = lookup_comment(comments, recipe['comment_id'])
        assert record is not None, recipe['comment_id']
        body = html.fromstring('<div>'+record['content']+'</div>').text_content()
        text = choose(normalize(body), locator)
        when = datetime.fromtimestamp(int(record['creation_time']), tz=timezone(timedelta(hours=7))).isoformat()
        title = normalize(' '.join(read_html(file).xpath('//h1//text()')))
        assert title, key
        return title + '\n' + text, when, {**meta, 'comment_id': recipe['comment_id'],
                            'comment_snapshot_sha256': meta_comment['sha256'],
                            'comment_endpoint': meta_comment['final_url']}
    doc = read_html(file)
    paragraphs = [normalize(' '.join(n.itertext())) for n in
                  doc.xpath('//article[contains(@class,"fck_detail")]//p')]
    date_labels = [normalize(x) for x in doc.xpath('//*[contains(@class,"date")]/text()') if 'GMT' in x]
    assert len(date_labels) == 1, key
    assert paragraphs, key
    text = choose(normalize(' '.join(paragraphs[1:])), locator)
    return text, date_labels[0], meta


def evidence(item, index, cid):
    key, heading, locators, rationale = item
    meta, source_path = index[key]
    assert digest(source_path) == meta['sha256'], key
    doc = read_html(source_path)
    text = normalize(' '.join(doc.xpath('//body')[0].itertext()))
    if heading:
        assert heading in text or heading in normalize(' '.join(doc.xpath('//title/text()'))), (key, heading)
    selected = [choose(text, loc) for loc in locators]
    combined = '\n'.join(([heading] if heading else [])+selected)
    assert 50 <= len(combined) <= 2100, (key, len(combined))
    return {'id':cid,'text':combined,'source':key,'url':meta['final_url'],
            'snapshot_sha256':meta['sha256'],'selectors':locators,'decision_reason':rationale}


def main():
    index = load_sources()
    cases = json.loads((ROOT / 'case_recipes.json').read_text())
    previous = {normalize(r['query']) for r in
                json.loads((ORIGINAL / 'benchmark.json').read_text())['questions']}
    questions = []
    for c in cases:
        text, date, source = query(c, index)
        assert text not in previous and ('?' in text or 'muốn biết' in text or
                                            'có đúng không' in text), c['id']
        candidates = [evidence(x,index,f"{c['id']}.{i}") for i,x in enumerate(c['candidates'])]
        assert len({x['text'] for x in candidates}) == len(candidates) == 5
        assert all(x['url'] != source['final_url'] for x in candidates)
        questions.append({'id':c['id'],'query':text,'article_url':source['final_url'],
                          'reader_source':source,'publication':date,'query_locator':c['query_loc'],
                          'operator':c['operator'],'scope':c['scope'],'evidence_language':'vi',
                          'decision':c['decision'],'sufficient':candidates[0]['id'],
                          'near':candidates[1]['id'],'candidates':sorted(candidates,key=lambda x:sha256(x['id'].encode()).hexdigest())})
    assert len({normalize(x['query']) for x in questions}) == len(questions)
    assert len({x['operator'] for x in questions}) >= 2
    result = {'protocol_sha256':digest(ROOT / 'protocol.md'),'recipe_sha256':digest(ROOT / 'case_recipes.json'),
              'questions':questions}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
