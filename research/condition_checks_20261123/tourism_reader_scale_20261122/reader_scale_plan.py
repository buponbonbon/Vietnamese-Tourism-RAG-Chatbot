"""Explicit documentary adjudication plan for new human-authored reader questions."""
DOCS={
 'condao_season':(
  'meta_con_dao_resolved_1','Côn Đảo mùa nào đẹp — Cổng thông tin du lịch Côn Đảo',
  'Từ tháng 3 đến tháng 9 là thời gian đẹp nhất', 'là mùa gió chướng , vùng biển vịnh Côn Sơn có sóng lớn.'),
 'sontra_night':(
  'meta_da_nang_night_resolved_0','Chợ đêm Sơn Trà — Cổng thông tin du lịch thành phố Đà Nẵng',
  'Địa chỉ: Đường Lý Nam Đế giao với Mai Hắc Đế','Giờ hoạt động: 18-24h mỗi ngày, vào cổng tự do.'),
 't3_bus':(
  'tsn_shuttle_vna_resolved_0','Chuyển đổi nhà ga T3 Tân Sơn Nhất — Vietnam Airlines (tháng 4 năm 2025)',
  'Lưu ý, giữa nhà ga T1 (nội địa cũ), T2 (quốc tế) và T3','hoặc các loại xe công nghệ, taxi.'),
 't3_2025_operator':(
  'tsn_shuttle_vna_resolved_0','Nhà ga khai thác nội địa tại Tân Sơn Nhất — Vietnam Airlines',
  'Ngày 17/04/2025 – Vietnam Airlines chính thức khai thác','tại nhà ga T1 cho đến khi có thông báo mới.'),
 'airport_id_policy':(
  'official_checkin_app_resolved_2','Thủ tục VNeID hoặc kiosk kể từ ngày 01/12/2025 — Vietnam Airlines',
  'từ ngày 1/12/2025, hành khách đi máy bay nội địa','tiếp tục làm thủ tục tại quầy như trước đây.'),
 'airport_human_counter':(
  'kiosk_fallback_resolved_0','Làm thủ tục tại T1 Nội Bài — Báo Điện tử Chính phủ',
  'Với những hành khách chưa quen sử dụng công nghệ','bảo đảm phục vụ mọi hành khách.'),
 'airport_liquid_boundary':(
  'duty_free_100_resolved_0','Hành lý chất lỏng trên chuyến bay quốc tế năm 2016 — Cục Hàng không Việt Nam',
  'Chỉ được mang không quá 01 lít chất lỏng theo người','túi nhựa an ninh được dán kín.'),
 'vna_pet_booking':(
  'air_early_pet_resolved_0','Đăng ký vận chuyển chó và mèo trên khoang hành khách — Vietnam Airlines',
  'Quý khách liên hệ phòng vé của Vietnam Airlines để đặt dịch vụ ít nhất 24 giờ trước thời gian khởi hành dự kiến.',
  'không gây ảnh hưởng đến hành khách và tài sản của họ.'),
 'vna_medication':(
  '../tourism_stage4_stage5_20261122/sources/evidence_rss1_resolved_2.html',
  'Chất lỏng là thuốc chữa bệnh trong hành lý — Vietnam Airlines',
  'Thuốc chữa bệnh có kèm theo đơn thuốc ghi rõ họ và tên','phù hợp với họ và tên trên vé hành khách.'),
}
CASES=[
 ('U01','a067','Tôi định du lịch Côn Đảo vào đầu năm 2021','Nếu không phải đầu năm, thì mùa nào ở đây đẹp nhất?',
  'mùa du lịch Côn Đảo','Cửa sổ thời tiết theo mùa, không đánh giá điều kiện đi lại trong đại dịch.',
  ['condao_season']),
 ('U02','a078','Chúng tôi đặt khách sạn ở trung tâm thành phố','chuyến du lịch của chúng tôi thêm đáng nhớ.',
  'điểm vui chơi tại Đà Nẵng sau 22 giờ','Một phương án cụ thể thỏa thời giờ, theo cẩm nang của thành phố.',
  ['sontra_night']),
 ('U03','C59430574','Nếu một hành khách bay từ Bali','di chuyển từ nhà ga T1-T2 qua T3 không?',
  'di chuyển giữa nhà ga T2 và T3 khi nối chuyến','Chỉ xét di chuyển mặt đất giữa ga; câu hỏi còn nhắc đến hành lý.',
  ['t3_bus']),
 ('U04','a130','Có lần tôi bị yêu cầu bỏ lại chai mỹ phẩm','mang từ bên ngoài và mua tại sân bay.',
  'ngoại lệ chất lỏng mua sau kiểm tra an ninh','Điều kiện túi an ninh niêm phong trên chuyến quốc tế; không suy đoán nguyên nhân riêng của chai 90 ml.',
  ['airport_liquid_boundary']),
 ('U05','a104','Sắp tới, tôi có chuyến du lịch Phú Quốc','mang chó về bằng máy bay cần thủ tục gì',
  'đặt dịch vụ mang chó đi trên chuyến bay','Một phương án có điều kiện theo Vietnam Airlines, không bảo đảm mọi chuyến có chỗ.',
  ['vna_pet_booking']),
 ('U06','C61761133','Mẹ tôi già rồi ko biết dùng công nghệ','thì phải làm như nào?',
  'thủ tục bay cho người cao tuổi không dùng smartphone','Quầy tiếp nhận người cao tuổi và các nhóm đặc biệt.',
  ['airport_id_policy']),
 ('U07','C61763498','Nếu đã làm thủ tục check in online trên trang của hãng','sao phải cần tích hợp vào VNEID ?',
  'kiosk là hình thức thay thế VNeID cho chuyến nội địa','Chỉ xét các luồng làm thủ tục được công bố, không bảo đảm in lại thẻ ở mọi sân bay.',
  ['airport_id_policy']),
 ('U08','a128','Cuối năm ba má tôi đi nước ngoài để thăm cháu','chứng minh nguồn gốc thuốc, liều dùng... hay không?',
  'đơn thuốc đối với thuốc nước hành khách sử dụng','Điều kiện đối với ngoại lệ chất lỏng là thuốc. Không đánh giá giới hạn nhập khẩu của quốc gia đến.',
  ['vna_medication']),
]
