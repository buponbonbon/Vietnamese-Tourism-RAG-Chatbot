from corpus_stage4 import *

idx=source_index()
plan=load('corpus_plan.json')
extra=[
 ('park_pack','x_pack_resolved_0','park','Combo Vinpearl Phú Quốc 3 ngày 2 đêm hiện đang nhận được', 'Chính sách giải quyết tranh chấp'),
 ('air_seat','qnews_vna_childseat_resolved_0','air','Tôi có thể đặt trước chỗ ngồi khi đi cùng trẻ em không?', None),
 ('venue_hall','qnews_cityhall2023_resolved_1','venue','Di tích kiến trúc nghệ thuật cấp Quốc gia trụ sở HĐND - UBND TPHCM mở cửa đón khách tham quan dịp Quốc khánh 2/9','Minh Hiệp'),
 ('venue_fest','qnews_hanoifest2023_resolved_1','venue','17 – 26/11/2023','Lễ hội thiết kế sáng tạo Hà Nội năm 2026'),
]
for key,source,group,start,end in extra:
    d=read_html(idx[source][1])
    title=normalize(' '.join(d.xpath('//title/text()')))
    text=normalize(' '.join(d.xpath('//body')[0].itertext()))
    if end and (end not in text or text.find(end) < text.find(start)):
        end=None
    candidate={'id':key,'source':source,'title':title,'service':group,'start':start,'end':end}
    try:
        body,_=select_source(candidate,idx)
        print(key,len(body))
    except ValueError as exc:
        print(key,exc)
        continue
    plan.append(candidate)
save('corpus_plan.json',plan)
