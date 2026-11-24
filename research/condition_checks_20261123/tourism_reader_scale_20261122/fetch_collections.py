"""Download article and comments from the publisher's publicly exposed endpoints."""
from hashlib import sha256
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from lxml import html
import gzip
import json
import re
import sys

ROOT=Path(__file__).resolve().parent
SOURCES=ROOT/'sources'


def digest(data): return sha256(data).hexdigest()


def get(key,url,suffix='html'):
    path=SOURCES/f'{key}.{suffix}'
    if path.is_file():
        return {'id':key,'file':path.name,'url':url,'sha256':digest(path.read_bytes()),'cached':True}
    try:
        req=Request(url,headers={'User-Agent':'Mozilla/5.0','Accept':'*/*'})
        with urlopen(req,timeout=16) as stream:
            raw=stream.read(10*1024*1024)
            end=stream.url
        path.write_bytes(raw)
        return {'id':key,'file':path.name,'url':url,'resolved':end,'sha256':digest(raw)}
    except HTTPError as exc:
        return {'id':key,'url':url,'http_error':exc.code,'retry_after':exc.headers.get('Retry-After')}
    except OSError as exc:
        return {'id':key,'url':url,'error':str(exc)}


def interpret(path):
    raw=path.read_bytes()
    if raw.startswith(b'\x1f\x8b'):
        raw=gzip.decompress(raw)
    return raw


def locate_comment(article,aid):
    for expr in (r'cate_id\s*[:=]\s*([0-9]+)',r'pageCate\s*[:=]\s*([0-9]+)',r'category_id\s*[:=]\s*([0-9]+)'):
        if m:=re.search(expr,article): return m.group(1)
    return '1001005'


def collect_comments(key,link):
    article=get(key,link)
    if 'file' not in article: return [article]
    file=interpret(SOURCES/article['file']).decode('utf-8','replace')
    match=re.search(r'-(\d+)\.html',link)
    if not match: return [article]
    aid=match.group(1)
    cate=locate_comment(file,aid)
    results=[article]
    for offset in (0,50):
        api='https://usi-saas.vnexpress.net/index/get?'+urlencode(
            {'offset':offset,'limit':50,'objectid':aid,'objecttype':1,
             'siteid':1000000,'categoryid':cate,'sort':'time','format':'json'})
        r=get(f'comments_{aid}_{offset}',api,'json')
        results.append({**r,'article_key':key,'article_url':link,'article_id':aid})
        if 'file' not in r: break
        try: count=len(json.loads(interpret(SOURCES/r['file']))['data']['items'])
        except (ValueError,KeyError,TypeError): break
        if count<50: break
    return results


def main():
    kind=sys.argv[1]
    if kind=='previous_articles':
        frame=json.loads((ROOT/'article_frame.json').read_text())
        source={k:r['url'] for k,r in frame.items()}
    else:
        entries=[json.loads(line) for line in (ROOT/f'{kind}_links.jsonl').read_text().splitlines()]
        urls=set()
        source={}
        for k,title,url in entries:
            if not url.startswith(('http://','https://')) or url in urls: continue
            urls.add(url)
            source[k]=url
    with ThreadPoolExecutor(max_workers=6) as pool:
        function=collect_comments if kind=='story' else lambda k,u:[get(k,u)]
        with (ROOT/f'{kind}_downloads.jsonl').open('a') as out:
            for result in pool.map(lambda kv:function(*kv),source.items()):
                for r in result: print(json.dumps(r,ensure_ascii=False),file=out,flush=True)


if __name__=='__main__': main()
