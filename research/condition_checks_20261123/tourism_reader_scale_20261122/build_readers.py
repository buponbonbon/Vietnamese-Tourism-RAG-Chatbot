"""Build the frozen reader-question candidate library from attributable sources."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from collections import Counter
from lxml import html
import csv
import json
import sys

from inspect_sources import body, clean
from fetch_collections import interpret
from screen_questions import nested
from reader_scale_plan import CASES,DOCS

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'tourism_stage4_stage5_20261122'
sys.path.insert(0,str(ROOT.parent/'tourism_stage4_20261121'))
from corpus_stage4 import BM25


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def sources():
    found={}
    for path in sorted(ROOT.glob('*_downloads.jsonl')):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if 'file' in r:
                found[r['id']]=(ROOT/'sources'/r['file'],r)
    for base in (ROOT,PREV):
        for line in (base/'sources/fetch_log.jsonl').read_text().splitlines():
            r=json.loads(line)
            if 'sha256' in r:
                snapshot=base/'sources'/(r['id']+'.html')
                if snapshot.is_file() and digest(snapshot)==r['sha256']:
                    source=(snapshot,{'url':r['final_url'],**r})
                    if base==ROOT:
                        found[r['id']]=source
                    else:
                        found.setdefault(r['id'],source)
    return found


def excerpt(text,start,end):
    i=text.casefold().index(start.casefold())
    j=text.casefold().index(end.casefold(),i)+len(end)
    return text[i:j]


def question_text(source,manifest,comments):
    if source.startswith('C'):
        ref=comments[source]
        path=ROOT/'sources'/ref['file']
        assert digest(path)==ref['sha256']
        raw=json.loads(interpret(path))
        item=next(r for r in nested(raw['data']['items']) if 'C'+str(r.get('comment_id'))==source)
        txt=clean(html.fromstring('<div>'+item['content']+'</div>').text_content())
        when=datetime.fromtimestamp(int(ref['timestamp']),timezone.utc).isoformat()
        return txt,ref['title'],ref['url'],when,digest(path)
    path,meta=manifest[source]
    assert digest(path)==meta['sha256']
    tree=html.fromstring(interpret(path),parser=html.HTMLParser(encoding='utf-8'))
    title=clean(tree.xpath('string(//h1)'))
    when=clean(tree.xpath('string(//*[contains(@class,"date")])'))
    contents=clean(' '.join(x.text_content() for x in tree.xpath(
        '//article[contains(@class,"fck_detail")]//p|//*[contains(@class,"detail-content")]//p')))
    assert title and contents,source
    return contents,title,meta['url'],when,digest(path)


def main():
    manifest=sources()
    corpus=json.loads((PREV/'stage4_extension_corpus.json').read_text())
    comments={q['id']:q for q in json.loads((ROOT/'candidate_questions.json').read_text()) if q['kind']=='comment'}
    for name,(ref,title,start,end) in DOCS.items():
        key=Path(ref).stem
        path,meta=manifest[ref if ref in manifest else key]
        assert digest(path)==meta['sha256'],name
        try: span=excerpt(body(path),start,end)
        except ValueError:
            print('unresolved document',name)
            continue
        assert 13<len(span.split())<=185,(name,len(span.split()))
        corpus.append({'id':name,'source':key,'url':meta['url'],'sha256':meta['sha256'],
                       'header':title,'body':span,'text':title+'\n'+span})
    docs={p['id']:p for p in corpus}
    bm25=BM25(corpus)
    data=[]
    for qid,qid0,start,end,scope,reason,sufficient in CASES:
        if not set(sufficient)<=set(docs):
            print('missing document for',qid); continue
        try:
            text,title,url,when,qsha=question_text(qid0,manifest,comments)
            quote=excerpt(text,start,end)
        except (ValueError,KeyError):
            print('unresolved question',qid,qid0); continue
        if qid0.startswith('C'):
            quote=title+'\n'+quote
        expected=[p for p in sufficient if p in docs]
        assert all(docs[p]['url']!=url for p in expected)
        ranked=bm25.scores(quote)[:5]
        data.append({'id':qid,'source_id':qid0,'query':quote,'question_url':url,
                     'source_timestamp':when,'source_sha256':qsha,'scope':scope,
                     'scope_note':reason,'anchors':expected,'native':[p for p,s in ranked],
                     'bm25':[s for p,s in ranked]})
        print(qid,'top5',data[-1]['native'],'anchor',expected)
    (ROOT/'reader_corpus.json').write_text(json.dumps(corpus,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'reader_worksheet.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    with (ROOT/'reader_worksheet.tsv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter='\t')
        w.writerow(['case','question','rank','passage_id','passage','anchor','url'])
        for q in data:
            for n,p in enumerate(q['native'],1):
                d=docs[p]
                w.writerow([q['id'],q['query'],n,p,d['text'],p in q['anchors'],d['url']])
    print(len(data),'questions',len(corpus),'passages')

if __name__=='__main__': main()
