# Ghi chú sửa bản thảo và hướng phát triển đóng góp

Ngày cập nhật: **31/10/2026**. Tài liệu này dành cho tác giả, không phải nội dung tuyên bố kết quả trong bài.

## 1. Bản sửa đã làm gì?

Đổi tiêu đề thành **The Tourist Trap: Auditing Hard-Negative Reranking for Vietnamese Tourism Question Answering** để phản ánh đối tượng thực sự được đánh giá. Viết lại abstract, introduction, contributions, related work, discussion và conclusion; bổ sung trạng thái dữ liệu hiện tại và bảng chẩn đoán đã lưu. Giữ số liệu của bốn bảng gốc: kết quả chính, từng seed/McNemar, sensitivity và legacy runs.

Những điểm được phân biệt rõ:

- Kết quả phân biệt hai passage: B 0.8826, C 0.9654, D 0.9683. Các giá trị này là kết quả kiểm toán checkpoint trong bản tác giả gửi, không phải kết quả vừa chạy lại ở lần sửa văn bản này.
- MRR@2 và nDCG@2 được suy ra từ cùng kết quả nhị phân; không phải ba bằng chứng độc lập về chất lượng truy xuất toàn corpus.
- B/C/D khác cả lượng mẫu, thành phần margin và có thể cả số bước tối ưu nếu giữ số epoch. Chưa tách riêng được tác dụng của “độ khó adversarial”. Những legacy run thay default margin hoặc detach trọng số không phải đối chứng loss như tên gọi ban đầu.
- Chẩn đoán RRF 80 câu validation cho thấy một vấn đề đáng nghiên cứu về chuyển giao từ pairwise sang candidate pool. Chỉ có một exact-gold proxy và qrels chưa đầy đủ; không suy ra mô hình làm câu trả lời tệ hơn theo các tỷ lệ đó.
- Giữ các pilot 10/30 câu ở trạng thái thăm dò, gồm cả kết quả bất lợi/không khác biệt; không dùng chúng để tuyên bố giảm hallucination.
- Corpus có 56,984 chunk, 4,122 URL và 6,200 doc ID nội bộ: đây là ba đơn vị đếm khác nhau.
- Hai người chấm độc lập đã hoàn thành bản 300 câu/492 cặp mỗi người. Giữ nguyên xác nhận của tác giả và số liệu đồng thuận; đồng thời ghi nhận rằng phần văn bản cũng trùng sau chuẩn hóa. Không tự suy đoán nguyên nhân hoặc coi κ=1 là chứng nhận tính đúng.
- 134 KEEP, 152 REVISE, 14 REJECT là quyết định cho câu gốc. Bản sau đó có 286 câu và 156 đề xuất viết lại; approval/qrels của câu thay đổi đã reset. Không gọi bản này là dữ liệu mới đã được human-validated.
- 14,300 ứng viên BM25 chưa có nhãn. Kiểm tra 104 anchor với ngưỡng lexical Jaccard 0.65 không chứng minh đã loại hết semantic leakage. Các câu từ chối còn cần kiểm tra phạm vi corpus và đa dạng hóa mẫu câu.

Không sửa dữ liệu nguồn, file ứng dụng, trọng số mô hình hoặc trạng thái phê duyệt. Bản gốc đúng được giữ tại `data/manuscript_working_001/tourism_original.tex` trong workspace. Không dùng bản n-gram retrieval gửi nhầm.

## 2. Tài liệu mới và mức kiểm chứng

`references.bib` có 22 mục, tất cả được trích trong bài, gồm 15 công trình năm 2026. Metadata được đối chiếu với Crossref/publisher hoặc arXiv; bảng `source_verification.csv` ghi rõ URL và mức truy cập. Đã lấy và kiểm tra các phần liên quan trong 12 PDF arXiv cùng toàn văn/PDF bài occupancy. Một số nguồn chỉ đọc được abstract/metadata; không coi việc đọc abstract là đã thẩm định toàn bộ phương pháp hay tái lập thí nghiệm.

Các đối chiếu ảnh hưởng trực tiếp đến claim:

| Công trình | Liên quan và giới hạn khác biệt |
|---|---|
| [ViTravelRAG (2026)](https://doi.org/10.1007/978-3-032-18316-3_7) | Chương sách về hỗ trợ du lịch Việt Nam bằng multimodal RAG; abstract báo cáo 1,000 query–document pairs có người gán nhãn. Không thể claim “dataset/RAG du lịch tiếng Việt đầu tiên”. Chỉ truy cập được abstract/metadata, chưa kiểm tra dữ liệu của họ. |
| [CausalNeg — KDD 2026](https://doi.org/10.1145/3770855.3818118) | Đã nghiên cứu tác hại/shortcut của synthesized hard negatives cho dense retrieval. Audit cross-encoder của chúng ta khác setting, nhưng không mới ở nhận xét chung “negative khó có thể gây hại”. |
| [CoRM-RAG — SIGIR 2026](https://doi.org/10.1145/3805712.3809631) | Evidence Critic, decision-supporting evidence và truy vấn có tiền đề sai. Phân biệt relevance với evidence utility đã có tiền lệ. |
| [BAR-RAG (preprint)](https://arxiv.org/abs/2602.03689) | Chọn evidence ở giai đoạn huấn luyện theo phản hồi generator; cần phân biệt với reranker chạy online của dự án. |
| [GRACE (preprint)](https://arxiv.org/abs/2601.04525) và [ARHN (preprint)](https://arxiv.org/abs/2604.11092) | Grounded response/abstention và xử lý false negatives đã là hướng nghiên cứu hiện hành; chỉ thêm nhãn từ chối hoặc làm sạch negative chưa đủ thành claim phương pháp mới. |
| [ViHoRec (preprint)](https://arxiv.org/abs/2607.12946), [Trip+ (preprint)](https://arxiv.org/abs/2606.21169), [TRACE — SIGIR 2026](https://doi.org/10.1145/3805712.3808370) | Các tác vụ recommendation/planning/sustainability khác passage QA; có thể so sánh định nghĩa tác vụ, không so trực tiếp các điểm số. |
| [Cultural Faithfulness (Research Square preprint)](https://doi.org/10.21203/rs.3.rs-8994226/v1) | Đã có đề xuất human adjudication và truy nguyên bằng chứng cho chatbot du lịch. Quy trình chấm của dự án không phải phương pháp thẩm định đầu tiên. |
| [Neubig et al., IT&T](https://doi.org/10.1007/s40558-026-00363-5) | Đã đọc toàn văn về truy cập dữ liệu occupancy với domain semantics; cho thấy đóng góp ứng dụng cần gắn vào nhu cầu thông tin cụ thể. |
| [Yoon et al., IT&T](https://doi.org/10.1007/s40558-026-00414-x) và [hotel-selection audit, IT&T](https://doi.org/10.1007/s40558-026-00417-8) | Abstract/metadata mô tả artifact hỗ trợ nhân viên và audit tín hiệu lựa chọn khách sạn. Đây là ví dụ đóng góp ứng dụng/audit trong tạp chí, không phải bằng chứng bài của chúng ta sẽ được nhận. |

Hai preprint tháng 10 về multilingual false-premise robustness và query clarification được dùng để làm rõ các năng lực riêng biệt; không nói hệ thống hiện tại đã đo hoặc giải quyết các năng lực đó. Không gán trạng thái đã phản biện cho các mục chỉ xác minh được là preprint.

Lần rà này có sàng lọc metadata 2026 của tạp chí và tìm các công trình gần với tác vụ, không phải systematic review đảm bảo bao phủ mọi công bố. Các tác vụ, corpus và nhãn khác nhau nên không lập bảng “SOTA” bằng cách ghép điểm số từ những bài này. Những báo cáo cũ lập lúc mạng bị giới hạn được giữ nguyên lịch sử; bảng nguồn đi kèm bản thảo này phản ánh đợt xác minh mới.

## 3. Claim hẹp hơn có đồng nghĩa không còn cửa Q1?

**Claim hiện hẹp hơn bản kể chuyện “hard adversarial training làm RAG đáng tin hơn”, vì kết quả hiện có chỉ kiểm chứng chắc hơn ở mức pairwise. Điều đó không tự loại một hướng dataset hoặc ứng dụng khỏi Q1. Tuy nhiên, bản hiện tại chưa đủ căn cứ để gọi là sẵn sàng nộp Q1.** Điểm yếu là đóng góp thực nghiệm cuối cùng chưa hoàn tất, chứ không có quy tắc “không phát minh thuật toán mới thì không thể Q1”.

[Aims and scope chính thức của Information Technology & Tourism](https://link.springer.com/journal/40558/aims-and-scope) chấp nhận empirical case studies và technical-theoretical papers, và yêu cầu đóng góp đáng kể cho hiểu biết về vai trò CNTT trong du lịch/lữ hành/khách sạn. Các công bố ứng dụng và audit ở trên phù hợp với phạm vi đó. Quartile phụ thuộc hệ thống phân hạng, năm và ngành; không phải thang chấm bài hoặc bảo đảm nhận bài. Lần sửa này không xác minh lại quartile của tạp chí.

Một định vị đáng theo đuổi là: **một tác vụ đánh giá có nguồn cho QA du lịch tiếng Việt, kiểm tra khi nào cải thiện phân biệt passage chuyển thành câu trả lời đủ bằng chứng, và khi nào hệ thống cần sửa tiền đề hoặc từ chối**. Đây là mục tiêu nghiên cứu tiếp theo, chưa phải đóng góp đã hoàn thành. Muốn claim nó, phải chứng minh khác biệt thực chất với ViTravelRAG và các nghiên cứu về evidence utility/abstention, thay vì chỉ đổi ngôn ngữ hoặc tên miền.

## 4. Ba ưu tiên tăng đóng góp

1. **Hoàn thiện một dataset có tác vụ rõ.** Chọn đối tượng sử dụng và quyết định cụ thể; bổ sung câu hỏi tự nhiên, đa dạng nhu cầu du lịch, tiền đề sai thực sự và trường hợp thiếu bằng chứng. Duyệt lại 156 câu sửa cùng evidence, xử lý 88 cờ relevance, 13 trường hợp xấp xỉ và 5 khoảng giá trị. Đa dạng hóa 100 câu từ chối. Tách “nguồn không nói” khỏi “nguồn bác bỏ”, giữ thời điểm/phạm vi tri thức và pool BM25+dense/RRF có nhiều passage đủ bằng chứng. Chốt version/split/protocol trước khi chạy test; cỡ mẫu cần theo mục tiêu sai số hoặc power, không theo một con số bảo đảm Q1.
2. **Đo chuyển giao bằng đối chứng công bằng.** Trên cùng candidate pool và dữ liệu đã duyệt, so sánh RRF, reranker gốc, B/C/D nếu phục hồi đủ checkpoint/provenance, một reranker đa ngôn ngữ hiện hành và chính sách evidence-sufficiency đơn giản. Thí nghiệm dữ liệu phải khớp lượng mẫu, margin, nguồn và số bước tối ưu; đối chứng ordinary/natural/synthetic negatives giúp tách nguyên nhân. Thí nghiệm loss phải tách riêng và ghi effective margin; các file hinge/validation lưu rời không tự tạo thành bằng chứng ablation hợp lệ nếu chưa có provenance tương ứng.
3. **Chứng minh ý nghĩa cho ứng dụng du lịch.** Giữ generator/prompt/context budget cố định; đo độ đúng, câu trả lời được bằng chứng hỗ trợ, correction/refusal đúng và từ chối sai. Tách lỗi candidate generation, reranking và generation; báo cáo bất định theo source group, cùng latency/cost phù hợp. Để tăng đóng góp cho IT&T, thêm một đánh giá phù hợp với người dùng mục tiêu, chẳng hạn độ đúng và thời gian hoàn thành nhiệm vụ tìm thông tin. Hai người gán nhãn dữ liệu không tự thay thế đánh giá kết quả sử dụng này.

Không cần mặc định làm cả dataset mới, loss mới và nhiều agent. Một đóng góp chính có khác biệt được chứng minh, dữ liệu hợp lệ và một kết quả ứng dụng rõ sẽ thuyết phục hơn việc thêm nhiều thành phần chưa được kiểm chứng. Nếu chỉ dừng ở các bảng pairwise hiện tại cộng một dataset còn đang sửa, nguy cơ bị đánh giá thiếu đóng góp hoặc chưa đủ xác thực vẫn đáng kể; không thể cam kết tránh desk reject bằng cách viết lại claim.

## 5. Việc biên tập còn lại

Khôi phục/đóng gói raw scores, manifests và cấu hình huấn luyện phù hợp với các bảng lưu trữ; không trộn log cũ với kết quả re-inference. Xác minh quyền dữ liệu scraped, điều kiện phát hành, DOI/version archive. Điền cơ quan, liên hệ, vai trò CRediT và declarations từ tác giả; chuyển template và ẩn danh theo hướng dẫn tạp chí. Các kết quả mới chỉ được đưa vào bài sau khi thực sự chạy theo giao thức đã chốt.
