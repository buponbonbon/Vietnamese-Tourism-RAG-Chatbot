"""Reader submission excerpts and independently issued evidence anchors."""

# ID, article snapshot, opening and closing strings, applicability, primary evidence ID.
QUESTIONS = [
    ('T01','tvpl_16_resolved_0','Cho hỏi khi về quê','theo hay không?','2023-01-02. Domestic rail conditions at the question date.', 'rail_pet'),
    ('T02','tvpl_02_resolved_5','Nội tôi 80 tuổi','cả 02 chế độ hay không?','2023-04-21. Domestic railway social fare discounts can overlap in eligibility.', 'rail_discount_overlap'),
    ('T03','tvpl_01_resolved_12','Tôi có thắc mắc: Người 60 tuổi khi đi tàu hỏa','sẽ được giảm bao nhiêu % giá vé?','2023-09-28. Minimum legal rate for qualifying Vietnamese senior rail passengers.', 'rail_discount_senior'),
    ('T04','tvpl_08_resolved_6','Làm hộ chiếu lần 2','ở đâu?','2023-11-04. Domestic application from the second issuance onward.', 'passport_later_office'),
    ('T05','tvpl_09_resolved_1','Vợ tôi mới sinh con','có được đi máy bay không?','2023-07-26. Three-week-old infant on a domestic flight and no birth certificate.', 'air_infant_birth'),
    ('T06','tvpl_10_resolved_7','Bị delay máy bay','hoàn tiền vé máy bay hay không?','2023-07-21. As discussed in the source, the forthcoming 19/2023 rule, effective 2023-09-01. Carrier-attributable delay, rerouting at two hours, refund option at five hours, advance compensation for prolonged delay.', 'air_delay_refund'),
    ('T07','tvpl_10_resolved_9','Cho tôi hỏi hãng hàng không','do thời tiết xấu hay không?','2023-08-08. Advance non-refundable compensation for a weather-related cancellation under the then-applicable rule.', 'air_weather'),
    ('T08','tvpl_18_resolved_20','Xin hỏi thời điểm và số tiền','Việt Nam được quy định thế nào?','2022-12-31. Amount and timing for qualifying VAT refunds on departure. The evidence card joins two separately marked subsections of one instrument.', 'vat_departure'),
    ('T09','tvpl_22_resolved_6','Cho tôi hỏi làm hộ chiếu lần đầu','làm hộ chiếu lần đầu không?','2022-08-26. First passport application office for ordinary in-country applications.', 'passport_first_office'),
    ('T10','tvpl_23_resolved_6','Cho tôi hỏi hết hạn hộ chiếu','hộ chiếu của tôi sắp hết hạn.','2022-07-08. Temporary residence card expiry relative to the passport. The later apostille subquestion is excluded.', 'card_passport_expiry'),
    ('T11','tvpl_24_resolved_12','Những loại giấy tờ nào có thể thay thế','nhập cảnh vào Việt Nam?','2024-01-13. Alternative entry documents for foreign citizens under the statutory provisions defining the two cards.', 'visa_substitute'),
    ('T12','tvpl_26_resolved_18','Tôi có thắc mắc là công dân','không quá bao nhiêu ngày?','2023-05-31. Original policy snapshot before the August 2023 change to 45 days.', 'visa_de_15'),
    ('T13','tvpl_28_resolved_12','Tôi muốn biết thời hạn visa điện tử','theo quy định mới nhất?','2023-07-20. The archived article expressly asks about the August 2023 regime, not the rule then in force.', 'evisa_limit_90'),
    ('T14','tvpl_28_resolved_13','Tôi có câu hỏi Người nước ngoài','nhập cảnh nhiều lần vào Việt Nam hay không?','2023-10-25. Multiple entry is possible only under a multiple-entry e-visa.', 'evisa_entry_types'),
    ('T15','tvpl_30_resolved_22','Tôi có một câu hỏi liên quan đến giấy thông hành như sau:','có thời hạn sử dụng bao lâu?','2023-08-08. Maximum validity for a Vietnam–China border pass. The quotation is the exact question following the authorial lead.', 'border_permit_period'),
    ('T16','tvpl_30_resolved_34','Trường hợp trong thời gian ở Campuchia','cần xử lý như thế nào?','2023-03-27. Loss of a border pass in Cambodia, including local reporting and reporting to Vietnam at return.', 'border_permit_loss_abroad'),
    ('T17','tvpl_31_resolved_5','Công dân Việt Nam muốn xuất cảnh','cần đáp ứng điều kiện gì?','2023-05-05. Historical ordinary departure requirements. The separately phrased destinations question is outside scope.', 'vn_departure_old'),
    ('T18','tvpl_32_resolved_4','Cho tôi hỏi trường hợp số tiền ngoại tệ','khi mang theo ngoại tệ?','2022-05-08. Foreign cash at an international border checkpoint, departing on a passport, including required declaration and supporting paperwork.', 'foreign_cash_border'),
    ('T19','reserve_05_resolved_9','Xin hỏi gia đình em','hộ chiếu phổ thông trước đó còn hiệu lực không?','2023-04-21. Passport issued to a 13-year-old, with seven years since issuance.', 'passport_child_duration'),
    ('T20','reserve_08_resolved_0','Hộ chiếu phổ thông của em bị mất','khôi phục được không ạ?','2024-01-06. Eligibility to consider restoration of a canceled passport lost in Vietnam and found intact with an unexpired foreign visa. Office/procedure subquestions are excluded.', 'passport_restore'),
    ('T21','reserve_13_resolved_0','Trẻ em bao nhiêu tuổi được','miễn, giảm giá vé máy bay?','2023-12-25. Statutory under-12 fare provision, with amounts determined by the carrier. The separately phrased fare ceiling question is excluded.', 'air_child_fare'),
    ('T22','reserve_13_resolved_3','Cho hỏi: Người khuyết tật','được giảm giá vé máy bay không?','2023-07-11. Passenger fare reduction on domestic flights under 28/2012 for qualifying disability categories.', 'air_disability_fare'),
    ('T23','tvpl_08_resolved_3','Trường hợp sử dụng bản sao giấy khai sinh','để đi máy bay được không?','2023-11-23. The documentation question for a five-year-old on a domestic flight. Chaperone and aviation family-escort rules are excluded.', 'air_certified_copy'),
    ('T24','reserve_13_resolved_1','Cho tôi hỏi là người cao tuổi','có được giảm giá vé máy bay không?','2022-07-27. Applicability of the senior concession to passenger aviation under 06/2011. No price quote for a particular airline.', 'air_senior_fare'),
]
