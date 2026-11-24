"""Extract traceable evidence cards and replay the mechanical BM25 pools."""
import hashlib
import json
from pathlib import Path
import sys
from lxml import html

from inspect_sources import body, clean
from target_plan import QUESTIONS

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'tourism_reader_scale_20261122'
OLD = ROOT.parent / 'tourism_stage4_stage5_20261122'
sys.path.insert(0, str(ROOT.parent / 'tourism_stage4_20261121'))
from corpus_stage4 import BM25

RAIL06 = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2011/01/105478_nd6cp.doc'
CASH = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2011/10/tt-15.doc'
COMP = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2015/05/14-tt.signed.pdf'
LAW49 = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/49.signed.pdf'
LAW51 = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/51.signed.pdf'
ANNEX13 = 'https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/04/13-bgtvt-kem.pdf'


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def span(text, a, b):
    i = text.casefold().index(clean(a).casefold())
    j = text.casefold().index(clean(b).casefold(), i)
    return text[i:j + len(clean(b))]


def manifests():
    html_sources = {}
    for path in sorted(ROOT.glob('*_downloads.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('file'):
                html_sources[row['id']] = row
    urls = {key: row.get('resolved', row['url']) for key, row in html_sources.items()}
    urls.update(read_json(ROOT/'exact_attachments.json'))
    urls.update(read_json(ROOT/'law_docs.json'))
    return urls, html_sources


def cards(urls):
    cards = []

    def make(key, label, parts):
        fulltext = '\n\n'.join(f"{p['locator']}\n{p['excerpt']}" for p in parts)
        assert 12 < len(fulltext.split()) < 420, key
        cards.append(dict(id=key, header=label, body=fulltext, text=f'{label}\n{fulltext}',
                          url=parts[0]['url'], provenance=parts))

    def web(key, a, b, locator):
        path = ROOT/'sources'/f'{key}.html'
        return dict(excerpt=span(body(path), a, b), url=urls[key], locator=locator,
                    snapshot_sha256=sha(path), rendering='whitespace-normalized HTML excerpt')

    def page(path, orig, url, lo, hi, locator, fixes=(), ocr=True):
        text = clean(' '.join(path.read_text().splitlines()[lo-1:hi]))
        for a, b in fixes:
            assert a in text, (locator, a)
            text = text.replace(a, b)
        return dict(excerpt=text, url=url, locator=locator, snapshot_sha256=sha(orig),
                    rendering='OCR transcription' if ocr else 'text conversion')

    def pdf(key, n, lo, hi, locator, fixes=(), url=None, base=ROOT, ocr_key=None):
        ocr = base/'sources'/'ocr'/f'{ocr_key or key}-{n}.txt'
        return page(ocr, base/'sources'/f'{key}.html', url or urls[key], lo, hi, locator, fixes)

    def txt(name, origin, url, a, b, locator):
        return dict(excerpt=span(clean((ROOT/'sources'/f'{name}.txt').read_text()), a, b), url=url,
                    locator=locator, snapshot_sha256=sha(ROOT/'sources'/origin),
                    rendering='whitespace-normalized source text')

    def passport(n, lo, hi, loc, fixes=()):
        if n in ('19', '20'):
            return pdf('law49_pdf',n,lo,hi,loc,fixes,LAW49,OLD)
        return pdf('law49pdf',n,lo,hi,loc,fixes,LAW49)

    make('rail_pet','Thông tư 09/2018/TT-BGTVT — vận tải hành lý xách tay trên tàu khách',[
        web('tvpl_00_resolved_15','e) Động vật sống','không gây ảnh hưởng tới người xung quanh);','Điều 13, khoản 3, điểm e')])
    make('rail_discount_overlap','Nghị định 65/2018/NĐ-CP — nhiều chế độ giảm giá vé tàu',[
        pdf('old_rail65_resolved_1_file0','16',33,34,'Điều 25, khoản 2')])
    make('rail_discount_senior','Giảm giá vé tàu hỏa đối với người cao tuổi — Luật 39/2009/QH12 và Nghị định 06/2011/NĐ-CP',[
        web('rail_eld_full','Người cao tuổi được quy định trong Luật này','60 tuổi trở lên.','Luật 39/2009/QH12, Điều 2'),
        txt('rail06','rail06.doc',RAIL06,'1. Người cao tuổi được giảm ít nhất mười lăm','máy bay chở khách.','Nghị định 06/2011/NĐ-CP, Điều 5, khoản 1'),
        txt('rail06','rail06.doc',RAIL06,'3. Để được giảm giá vé, giá dịch vụ','chứng minh là người cao tuổi.','Nghị định 06/2011/NĐ-CP, Điều 5, khoản 3')])
    make('passport_later_office','Luật 49/2019/QH14 — cấp hộ chiếu phổ thông trong nước từ lần thứ hai',[
        passport('10',49,51,'Điều 15, khoản 5',(('Công an cắp tỉnh','Công an cấp tỉnh'),))])
    make('air_delay_refund','Thông tư 19/2023/TT-BGTVT — nghĩa vụ khi chuyến bay bị chậm do người vận chuyển',[
        pdf('air_delay19_resolved_0_file0','02',29,47,'Điều 1, khoản 2 — Điều 8, khoản 4 (trang 2)',(('hành khácb','hành khách'),('văn phòng đại điện, chỉ nhánh','văn phòng đại diện, chi nhánh'))),
        pdf('air_delay19_resolved_0_file0','03',3,6,'Điều 8, khoản 4, điểm c (tiếp trang 3)'),
        pdf('law_compensation_resolved_1_doc_55','2',7,10,'Thông tư 14/2015/TT-BGTVT, Điều 2, khoản 4',url=COMP)])
    make('air_weather','Thông tư 14/2015/TT-BGTVT — miễn bồi thường ứng trước không hoàn lại',[
        pdf('law_compensation_resolved_1_doc_55','3',47,54,'Điều 6, khoản 1',(('vận chuyễn','vận chuyển'),('chậm kéo đài','chậm kéo dài')),COMP)])
    make('vat_departure','Thông tư 72/2014/TT-BTC — số tiền và thời điểm hoàn thuế GTGT khi xuất cảnh',[
        pdf('law_vat_foreign_resolved_1_file0','08',6,9,'Điều 12, khoản 2',(('đo người','do người'),('cửa khâu','cửa khẩu'))),
        pdf('law_vat_foreign_resolved_1_file0','11',38,42,'Điều 17, khoản 1',(('thủ tụo','thủ tục'),('chuyên bay','chuyến bay')))])
    make('passport_first_office','Luật 49/2019/QH14 — nơi nộp hồ sơ cấp hộ chiếu phổ thông lần đầu',[
        passport('10',29,32,'Điều 15, khoản 3')])
    make('card_passport_expiry','Luật 51/2019/QH14 — thời hạn thẻ tạm trú khi hộ chiếu sắp hết hạn',[
        pdf('law51_pdf','8',9,12,'Điều 1, khoản 16 — Điều 38, khoản 1',
            (('thể tạm trú','thẻ tạm trú'),('. 1.','1.')),LAW51,OLD)])
    make('visa_substitute','Luật 47/2014/QH13 — giấy tờ có thể thay thế thị thực',[
        pdf('old_visa47_resolved_0_file0','06',45,49,'Điều 12, khoản 2')])
    make('visa_de_15','Nghị quyết 32/NQ-CP — 15/03/2022, bản gốc trước sửa đổi tháng 8/2023',[
        web('resolution32_legal','Điều 1. Miễn thị thực cho công dân các nước','điều kiện theo quy định của pháp luật Việt Nam.','Điều 1, bản ban hành ngày 15/03/2022')])
    make('evisa_limit_90','Luật 23/2023/QH15 — thời hạn tối đa của thị thực điện tử, hiệu lực 15/08/2023',[
        txt('law_pass_amendment_resolved_0_doc_55','law_pass_amendment_resolved_0_doc_55.html',
            urls['law_pass_amendment_resolved_0_doc_55'],'2. Sửa đổi, bổ sung các khoản 1, 2, 3 và 4 Điều 9','3. Thị thực ký hiệu VR có thời hạn không quá 180 ngày.',
            'Điều 2, khoản 2 — Điều 9; EV là ký hiệu của thị thực điện tử')])
    make('evisa_entry_types','Báo Chính phủ, 30/08/2023 — triển khai thị thực điện tử có giá trị một lần hoặc nhiều lần',[
        web('visa23_txt_resolved_1','Nhóm nội dung sửa đổi, bổ sung để tiếp tục tạo điều kiện hơn nữa','(quy định trước đây thị thực điện tử có giá trị một lần).','Triển khai Luật 23/2023/QH15')])
    make('border_permit_period','Luật 49/2019/QH14 — thời hạn giấy thông hành',[
        passport('04',33,33,'Điều 7, khoản 3')])
    make('border_permit_loss_abroad','Nghị định 76/2020/NĐ-CP — mất giấy thông hành còn thời hạn ở nước ngoài',[
        pdf('law_pass_border_resolved_0_file0','05',36,41,'Điều 11, khoản 2',
            (('bị mắt','bị mất'),('việc mắt','việc mất'),('cửa khâu','cửa khẩu')))])
    make('vn_departure_old','Luật 49/2019/QH14 — điều kiện xuất cảnh của công dân Việt Nam, nguyên bản trước 15/08/2023',[
        passport('19',48,57,'Điều 33, khoản 1',(('©)','c)'),)),
        passport('20',3,6,'Điều 33, khoản 2',(('hành vỉ','hành vi'),('đại điện','đại diện')))])
    make('foreign_cash_border','Thông tư 15/2011/TT-NHNN — ngoại tệ tiền mặt khi xuất cảnh qua cửa khẩu quốc tế',[
        page(ROOT/'sources'/'cash15.txt',ROOT/'sources'/'cash15.doc',CASH,21,24,'Điều 2, khoản 1',ocr=False),
        page(ROOT/'sources'/'cash15.txt',ROOT/'sources'/'cash15.doc',CASH,28,31,'Điều 3, khoản 1–2',ocr=False)])
    make('passport_child_duration','Luật 49/2019/QH14 — thời hạn hộ chiếu phổ thông theo tuổi tại thời điểm cấp',[
        passport('04',22,28,'Điều 7, khoản 2, điểm a–b')])
    make('passport_restore','Luật 49/2019/QH14 — điều kiện khôi phục hộ chiếu phổ thông đã bị hủy do báo mất',[
        passport('19',14,19,'Điều 32, khoản 1',(('phố thông','phổ thông'),))])
    make('air_child_fare','Luật Hàng không dân dụng Việt Nam — cước vận chuyển trẻ em',[
        web('air_child_link','Trẻ em dưới mười hai tuổi đi tàu bay được miễn','mức ghi trong Điều lệ vận chuyển.','Luật 66/2006/QH11, Điều 147, khoản 6')])
    make('air_disability_fare','Nghị định 28/2012/NĐ-CP — giảm giá vé vận tải nội địa đối với người khuyết tật',[
        web('disability_link','2. Người khuyết tật đặc biệt nặng, người khuyết tật nặng được giảm','Giấy xác nhận khuyết tật.','Điều 12, khoản 2–4')])
    make('air_senior_fare','Nghị định 06/2011/NĐ-CP — giảm giá vé máy bay cho người cao tuổi',[
        txt('rail06','rail06.doc',RAIL06,'1. Người cao tuổi được giảm ít nhất mười lăm','máy bay chở khách.','Điều 5, khoản 1'),
        txt('rail06','rail06.doc',RAIL06,'3. Để được giảm giá vé, giá dịch vụ','chứng minh là người cao tuổi.','Điều 5, khoản 3')])
    add_id_cards(make, pdf)
    return cards


def add_id_cards(make, pdf):
    def annex(lo, hi, locator, fixes):
        return pdf('law_pass_caa_resolved_0_doc_57', '49', lo, hi, locator, fixes, ANNEX13,
                   ocr_key='annex_scan/a')

    minor = annex(3, 8, 'Phụ lục XIV, Mục I, khoản 3, điểm a (trang PDF 49)',
                  (('giấy tỜ','giấy tờ'),('một. trong','một trong'),('3) Giấy','a) Giấy')))
    make('air_infant_birth','Thông tư 13/2019/TT-BGTVT — nhân thân trẻ em trên chuyến bay nội địa',[minor])
    copy = annex(18, 23, 'Phụ lục XIV, Mục I, khoản 5, điểm a–b (trang PDF 49)',
                 (('của ¡pháp luật','của pháp luật'),))
    make('air_certified_copy','Thông tư 13/2019/TT-BGTVT — bản sao giấy khai sinh đi chuyến bay nội địa',[copy])


def main():
    urls, sources = manifests()
    previous = read_json(PREVIOUS/'reader_corpus.json')
    new_docs = cards(urls)
    corpus = previous + new_docs
    docs = {p['id']: p for p in corpus}
    assert len(corpus) == len(docs)
    bm25 = BM25(corpus)
    qs = []
    # Freeze a historical scope separate from the original quote. Expose it
    # uniformly to both BM25 and all rerankers rather than granting oracle-only context.
    effective = {'T06':'2023-09-01', 'T13':'2023-08-15'}
    for qid,key,a,b,scope,anchor in QUESTIONS:
        path = ROOT/'sources'/sources[key]['file']
        source = body(path)
        if qid == 'T17':
            source = source[source.index('Cho tôi hỏi: Hộ chiếu Việt Nam'):]
        if qid == 'T21':
            source = source[source.index('Tôi có thắc mắc: Trẻ em'):]
        quote = span(source,a,b)
        assert len(quote) < 475
        assert urls[key] not in {p['url'] for p in docs[anchor]['provenance']}
        cutoff = effective.get(qid, scope[:10])
        context = f'Mốc áp dụng quy định: {cutoff}.'
        service = {'T02':'Tàu hỏa nội địa.', 'T04':'Cấp hộ chiếu trong nước.',
                   'T23':'Chuyến bay nội địa, trẻ 5 tuổi.'}
        if qid in service:
            context += ' ' + service[qid]
        query = f'{context}\n{quote}'
        top = bm25.scores(query)[:5]
        qs.append(dict(id=qid, quote=quote, query=query, source_id=key, question_url=urls[key],
                       question_sha256=sha(path), published=scope[:10], cutoff=cutoff, scope=scope,
                       anchors=[anchor], native=[p for p,s in top], bm25=[s for p,s in top]))
    assert len({q['question_url'] for q in qs}) == len(qs) == 24
    write_json(ROOT/'reader_corpus.json',corpus)
    write_json(ROOT/'reader_worksheet.json',qs)
    write_json(ROOT/'reader_new_evidence.json',new_docs)
    for q in qs:
        print('\n',q['id'],q['quote'], '->', q['anchors'][0])
        for pid in q['native']:
            print(pid,docs[pid]['header'],docs[pid]['body'][:175])

if __name__ == '__main__':
    main()
