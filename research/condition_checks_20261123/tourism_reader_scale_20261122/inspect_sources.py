"""Expose short source passages around terms; used only for provenance checking."""
from pathlib import Path
from lxml import html
from fetch_collections import interpret
import sys, re, unicodedata

ROOT=Path(__file__).resolve().parent

def clean(s): return ' '.join(unicodedata.normalize('NFC',s).split())

def body(file):
    r=interpret(file)
    if r.startswith(b'%PDF'):
        return clean(file.with_suffix('.txt').read_text())
    d=html.fromstring(r,parser=html.HTMLParser(encoding='utf-8'))
    for item in d.xpath('//script|//style|//noscript|//footer'):
        item.drop_tree()
    return clean(' '.join(d.xpath('//body')[0].itertext()))

if __name__=='__main__':
    for entry in sys.argv[1:]:
        key,expr=entry.split(':',1)
        p=ROOT/'sources'/f'{key}.html'
        if not p.is_file():
            print(key,'missing')
            continue
        print(key)
        text=body(p)
        for m in list(re.finditer(expr,text,re.I))[:14]:
            print(text[max(0,m.start()-65):min(len(text),m.end()+135)])
