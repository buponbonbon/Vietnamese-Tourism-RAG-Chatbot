"""Freeze attributable Vietnamese questions, separately published evidence, and natural pools."""
from collections import Counter
from pathlib import Path
import csv
import gzip
from hashlib import sha256
import json
import re
import sys
from urllib.parse import urlsplit

from lxml import html

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'tourism_stage4_20261121'
sys.path.insert(0,str(PREVIOUS))
from corpus_stage4 import BM25, normalize


def get(name):
    return json.loads((ROOT/name).read_text())


def write(name,data):
    (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def digest(blob):
    return sha256(blob).hexdigest()


def sources():
    result={}
    for line in (ROOT/'sources/fetch_log.jsonl').read_text().splitlines():
        record=json.loads(line)
        if 'sha256' in record:
            result[record['id']]=record
    return result


def text(source,manifest,ocr=None):
    raw=(ROOT/'sources'/f'{source}.html').read_bytes()
    assert digest(raw)==manifest[source]['sha256'],source
    if ocr:
        return normalize((ROOT/'sources'/ocr).read_text())
    if raw.startswith(b'\x1f\x8b'):
        raw=gzip.decompress(raw)
    tree=html.fromstring(raw,parser=html.HTMLParser(encoding='utf-8'))
    for node in tree.xpath('//script|//style|//noscript'):
        node.drop_tree()
    return normalize(' '.join(tree.xpath('//body')[0].itertext()))


def extract(content,start,end):
    first=content.index(start)
    last=content.index(end,first)+len(end)
    return content[first:last]


def main():
    manifest=sources()
    plan=get('stage4_extension_plan.json')
    corpus=json.loads((PREVIOUS/'corpus.json').read_text())
    errors=[]
    for ident,source,title,topic,start,end,*ocr in plan['source_documents']:
        try:
            body=extract(text(source,manifest,*ocr),start,end)
        except ValueError as fail:
            errors.append((ident,str(fail)))
            continue
        assert len(body.split())<=370,ident
        corpus.append({'id':ident,'body':body,'header':title,'service':topic,
                       'source':source,'url':manifest[source]['final_url'],
                       'snapshot_sha256':manifest[source]['sha256'],
                       'ocr_file':ocr[0] if ocr else None,'text':title+'\n'+body})
    print('unresolved documents',errors)
    print('library',len(corpus))
    passages={p['id']:p for p in corpus}
    bm25=BM25(corpus)
    rows=[]
    for case,source,start,end,decision,period,sufficient,rationale in plan['cases']:
        try:
            query=extract(text(source,manifest),start,end)
        except ValueError as fail:
            print(case, 'question extraction failed',fail)
            continue
        for pid in sufficient:
            assert pid in passages
            assert urlsplit(passages[pid]['url']).netloc!=urlsplit(manifest[source]['final_url']).netloc
        if case=='E05':
            # The original paragraph says "đường bộ" but requires its publication context.
            query='Đi lại qua cửa khẩu đường bộ Việt - Trung còn hạn chế dịp 2/9?\n'+query
            assert query.splitlines()[0] in text(source,manifest)
        ranking=bm25.scores(query)[:5]
        native=[pid for pid,_ in ranking]
        found=sorted(set(native)&set(sufficient))
        rows.append({
            'id':case,'query':query,'query_url':manifest[source]['final_url'],
            'query_source_sha256':manifest[source]['sha256'],'decision':decision,'period':period,
            'native_ids':native,'proven_sufficient':sufficient,'native_sufficient':found,
            'oracle_ids':native if found else [*native[:4],sufficient[0]],
            'rationale':rationale,'bm25_scores':ranking,
        })
        print(case, decision, 'native',native, 'sufficient',sufficient)
    for row in rows:
        row['candidate_ids']=list(dict.fromkeys(row['native_ids']+row['oracle_ids']))
    write('stage4_extension_corpus.json',corpus)
    write('stage4_extension_worksheet.json',{'questions':rows, 'unresolved_source_sections':errors})
    with (ROOT/'stage4_extension_worksheet.tsv').open('w',encoding='utf-8',newline='') as dest:
        w=csv.writer(dest,delimiter='\t')
        w.writerow(['qid','native_rank','candidate','source','published_text','grade','decision'])
        for row in rows:
            for rank,pid in enumerate(row['native_ids'],1):
                doc=passages[pid]
                w.writerow([row['id'],rank,pid,doc['url'],doc['text'],2 if pid in row['proven_sufficient'] else '',row['decision']])


if __name__=='__main__':
    main()
