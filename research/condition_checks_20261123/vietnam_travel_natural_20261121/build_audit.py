"""Create the locked benchmark from exact snippets of archived web sources."""
from hashlib import sha256
import json
from pathlib import Path

from lxml import html

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'sources'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(text.replace('\u200b', '').replace('\ufeff', '').split())


def excerpts(source):
    path = SOURCE / f'{source}.html'
    doc = html.fromstring(path.read_bytes(), parser=html.HTMLParser(encoding='utf-8'))
    for node in doc.xpath('//script|//style|//noscript'):
        node.drop_tree()
    nodes = doc.xpath(
        '//body//*[self::h1 or self::h2 or self::h3 or self::h4 or self::p or '
        'self::li or self::tr or self::div[not(descendant::*)]]'
    )
    blocks = list(dict.fromkeys(text for node in nodes if (text := ' '.join(node.text_content().split()))))
    return doc, blocks


def extract(source, selectors, story=False):
    doc, blocks = excerpts(source)
    if story:
        blocks = [normalize(node.text_content()) for node in
                  doc.xpath('//article[contains(@class,"fck_detail")]//p')
                  if normalize(node.text_content())]
    text = []
    for selector in selectors:
        if isinstance(selector, int):
            text.append(normalize(blocks[selector]))
        else:
            start, end = selector
            if story:
                container = '\n'.join(blocks)
            else:
                container = normalize(doc.xpath('string(//body)'))
            assert container.count(start) == 1, (source, start, container.count(start))
            s = container.index(start)
            e = container.index(end, s)
            text.append(normalize(container[s:e + len(end)]))
    assert text and all(text) and len(text) == len(set(text)), (source, selectors)
    return '\n'.join(text)


def main():
    frame = json.loads((ROOT / 'screening_frame.json').read_text())
    sources = {}
    for line in (SOURCE / 'fetch_log.jsonl').read_text().splitlines():
        item = json.loads(line)
        if 'sha256' in item:
            sources[item['id']] = item
    recipes = json.loads((ROOT / 'case_recipes.json').read_text())
    questions = []
    for recipe in recipes:
        story = recipe['article']
        story_doc, _ = excerpts(story)
        when = [normalize(x) for x in story_doc.xpath('//*[contains(@class,"date")]/text()') if 'GMT' in x]
        assert len(when) == 1, story
        prompt = extract(story, recipe['question'], story=True)
        candidates = []
        for index, entry in enumerate(recipe['candidates']):
            sid, selectors, reason = entry
            link = sources[sid]
            assert link['sha256'] == digest(SOURCE / (sid + '.html')), sid
            text = extract(sid, selectors)
            assert 60 < len(text) < 2400, (sid, len(text))
            candidates.append({'id': f'{story}.{index}', 'text': text, 'source': sid,
                               'url': link['final_url'], 'snapshot_sha256': link['sha256'],
                               'selectors': selectors, 'decision_reason': reason})
        assert len(set(x['text'] for x in candidates)) == len(candidates) == 5
        candidate_order = sorted(candidates, key=lambda x: sha256(x['id'].encode()).hexdigest())
        questions.append({'id': story, 'query': prompt,
                          'article_url': sources[story]['final_url'],
                          'publication': when[0], 'query_selector': recipe['question'],
                          'story_sha256': digest(SOURCE / (story + '.html')),
                          'decision': recipe['decision'], 'scope': recipe['scope'],
                          'evidence_language': recipe['evidence_language'],
                          'sufficient': candidates[0]['id'], 'near': candidates[1]['id'],
                          'candidates': candidate_order})
    assert len({q['id'] for q in questions}) == len(questions)
    assert {q['id'] for q in questions}.issubset({r['id'] for r in frame})
    output = {'protocol_sha256': digest(ROOT / 'protocol.md'),
              'frame_sha256': digest(ROOT / 'screening_frame.json'),
              'recipe_sha256': digest(ROOT / 'case_recipes.json'), 'questions': questions}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
