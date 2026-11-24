"""Keep only potentially actionable fragments, without identities and unrelated replies."""
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
from lxml import html
from fetch_collections import interpret
import json
import re
import unicodedata

ROOT=Path(__file__).resolve().parent
SOURCES=ROOT/'sources'
PERSON=re.compile(r'\b(tôi|em|mình|cháu|chúng tôi|nhà mình|nhà em|gia đình |bạn đọc |ông .* hỏi)\b',re.I)
ASK=re.compile(r'\?|cho (mình|em|tôi|chị|anh) hỏi|xin hỏi|thắc mắc|có ai biết|ai biết',re.I)
ACTION=re.compile(
    r'ở đâu|khi nào|bao (lâu|nhiêu)|mấy (giờ|tuổi)|thế nào|như thế|cách nào|'
    r'cho .+ hỏi|có (được|cần|phải|thể)|được không|mua vé|đi đâu|làm sao|'
    r'phí|thời gian|giấy tờ|bắt buộc|tính như|điều kiện|hồ sơ|visa|hộ chiếu|'
    r'cách (đi|mua|đổi)|có .* không\?|phải .*\?',re.I)
SCOPE=re.compile(r'vé|ga |tàu|sân bay|hành lý|du lịch|hộ chiếu|visa|phà|visa|di tích|bảo tàng|'
                 r'đường bay|quá cảnh|cửa khẩu|cáp treo|lăng bác|metro|buýt|lịch trình|qua biên|'
                 r'vn[ae]|việt nam airline|nhà ga|máy bay|lên máy bay|lên tàu|trẻ em|nội địa',re.I)
SENSITIVE=re.compile(r'[A-Za-z0-9_.+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]+|(?:\+?\d[ .()\-]?){8,}|\b(?:https?://|www\.)',re.I)


def norm(text):
    return ' '.join(unicodedata.normalize('NFC',text).replace('\u200b',' ').split())


def collect_from_articles():
    prior_ids=set()
    for folder in ['vietnam_travel_natural_20261121','tourism_vietnamese_operators_20261121','tourism_stage4_20261121','tourism_stage4_stage5_20261122']:
        file='stage4_extension_worksheet.json' if folder.endswith('1122') else 'benchmark.json'
        for q in json.loads((ROOT.parent/folder/file).read_text())['questions']:
            url=q.get('query_url') or q.get('article_url')
            if url and (m:=re.search(r'-(\d+)\.html?$',url)):
                prior_ids.add(m.group(1))
    seen=set()
    for mode in ('authored','previous_articles','story'):
        for line in (ROOT/f'{mode}_downloads.jsonl').read_text().splitlines():
            record=json.loads(line)
            if 'file' not in record or not record['file'].endswith('.html'): continue
            url=record['url']
            if url in seen: continue
            seen.add(url)
            m=re.search(r'-(\d+)\.html?$',url)
            if m and m.group(1) in prior_ids: continue
            try: page=html.fromstring(interpret(SOURCES/record['file']),parser=html.HTMLParser(encoding='utf-8'))
            except(ValueError,TypeError): continue
            title=norm(page.xpath('string(//h1)') or page.xpath('string(//title)'))
            selectors='//article[contains(@class,"fck_detail")]//p|//*[contains(@class,"detail-content")]//p'
            paras=[norm(p.text_content()) for p in page.xpath(selectors)]
            lead=[]
            for p in paras[:12]:
                if re.match(r'(-\s*)?trả lời\b|luật sư .+ trả lời',p,re.I):
                    break
                if 20<len(p)<850 and not SENSITIVE.search(p):
                    lead.append(p)
            if not lead:continue
            text=' '.join(lead[:5])
            if not (PERSON.search(text) and ASK.search(title+' '+text) and ACTION.search(text)
                    and SCOPE.search(title+' '+text)):
                continue
            time=page.xpath('//meta[@itemprop="datePublished"]/@content|//meta[@property="article:published_time"]/@content|//time/@datetime')
            yield {'id':record['id'],'kind':'article','file':record['file'],'sha256':record['sha256'],
                   'url':url,'title':title,'timestamp':time[0] if time else None,
                   'candidate_paragraphs':lead[:5]}


def nested(items):
    for item in items:
        if not isinstance(item,dict):continue
        yield item
        yield from nested(item.get('replys') or [])


def collect_from_comments():
    titles={}
    for line in (ROOT/'story_links.jsonl').read_text().splitlines():
        ident,title,url=json.loads(line)
        if (m:=re.search(r'-(\d+)\.html',url)):
            titles[m.group(1)]=norm(title.removesuffix(' - VnExpress'))
    seen=set()
    for line in (ROOT/'story_downloads.jsonl').read_text().splitlines():
        record=json.loads(line)
        if 'file' not in record or 'article_id' not in record: continue
        try: comments=json.loads(interpret(SOURCES/record['file']))['data']['items']
        except (ValueError,KeyError,TypeError): continue
        for item in nested(comments):
            cid=str(item.get('comment_id'))
            if not cid.isdigit() or cid in seen: continue
            seen.add(cid)
            body=norm(html.fromstring('<div>'+str(item.get('content') or '')+'</div>').text_content())
            title=titles[record['article_id']]
            if not (13<len(body)<530 and ASK.search(body) and ACTION.search(body)
                    and SCOPE.search(title+' '+body) and not SENSITIVE.search(body)):
                continue
            yield {'id':'C'+cid,'kind':'comment','timestamp':item.get('creation_time'),
                   'title':title,'url':record['article_url'],'file':record['file'],
                   'sha256':record['sha256'],'candidate_paragraphs':[body]}


def main():
    candidates=[*collect_from_articles(),*collect_from_comments()]
    (ROOT/'candidate_questions.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2)+'\n')
    print(Counter(q['kind'] for q in candidates))
    for r in candidates:
        print(r['id'],r['timestamp'],r['title'],'|',' / '.join(r['candidate_paragraphs'])[:460])

if __name__=='__main__':main()
