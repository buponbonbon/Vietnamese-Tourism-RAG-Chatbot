# Đối chiếu hướng nghiên cứu với tài liệu năm 2026

Mốc tra cứu: **31/10/2026**. Kết luận hiện tại: **chưa nên nộp bản thảo đang có với
tuyên bố novelty phương pháp hoặc hiệu quả RAG đầu cuối**. Hướng nguồn lực và ứng
dụng vẫn có thể phát triển thành đóng góp, nhưng chưa có đủ bằng chứng để khẳng
định khoảng trống hẹp đó chưa được công bố hoặc để bảo đảm tránh desk reject.

## Phạm vi đã thực sự kiểm tra

- 12 truy vấn trên Hugging Face Papers: tourism/travel, chatbot, benchmark,
  planning, false premise, abstention, unanswerability, hard negatives, adversarial
  retrieval và retrieval robustness.
- 792 bản ghi khác nhau được thu về; 183 có ngày metadata thuộc năm 2026 và không
  sau mốc tra cứu. Đây là số bản ghi khám phá, **không phải 792 bài đã đọc**.
- Đọc và lập đối chiếu cho **18 abstract năm 2026**; tải 7 README từ kho do tác giả
  dẫn, trong đó 5 có mô tả phương pháp/dữ liệu đáng kể. Không chạy mã của tác giả.
- **Chưa đọc PDF gốc nào trong vòng tra cứu này.** Crossref, OpenAlex và Springer
  trả lỗi proxy CONNECT 403. Chưa kiểm chứng đầy đủ các bài tạp chí năm 2026,
  proceedings, DOI hoặc aims/scope hiện hành của Information Technology & Tourism.
- Nguồn tìm kiếm có giới hạn độ phủ và một số truy vấn bị giới hạn 100 kết quả.
  Đây là rà soát có lưu vết để kiểm tra chồng lấn, **không phải systematic review**.

Không đồng nhất ngày trên metadata với ngày chấp nhận, xuất bản chính thức hoặc
lần công bố đầu tiên. Các tài liệu dưới đây được xử lý như preprint/tài liệu tác
giả cho đến khi xác minh được nguồn xuất bản. README CausalNeg tự ghi KDD 2026;
proceedings chưa được đối chiếu độc lập. File BibTeX đi kèm cũng ghi giới hạn này.

Toàn bộ tiêu đề, tác giả, ngày được nguồn liệt kê, URL và mức kiểm chứng nằm trong
[literature_matrix_2026.csv](literature_matrix_2026.csv). Metadata và phản hồi mạng
được lưu cục bộ tại `data/research_v2/literature_2026_001/`.

## Các tài liệu gần nhất với những tuyên bố của chúng ta

| Tài liệu, ngày metadata | Nội dung đã đối chiếu | Ý nghĩa với bài của chúng ta |
|---|---|---|
| [CausalNeg — When Hard Negatives Hurt](https://huggingface.co/papers/2606.01304), 07/06/2026; [mã tác giả](https://github.com/mzhangzhicheng/CausalNeg) | Hard negatives tổng hợp có thể làm giảm truy xuất; tác giả mô tả dấu vết nguồn và generative–discriminative gap, dùng counterfactual perturbation và entropy regularization. README xác định đối tượng là dense retrievers. | Không nhận phát hiện tổng quát “hard negatives có thể gây hại” là mới. Cross-encoder pairwise→candidate-pool trong du lịch là đối tượng khác, nhưng phải chứng minh một cơ chế/kết quả bổ sung cụ thể. |
| [CoRM-RAG — Beyond Semantic Relevance](https://huggingface.co/papers/2605.01302), 02/05/2026; [mã tác giả](https://github.com/PeiYangLiu/CoRM-RAG) | Tiền đề sai/confirmation bias, Evidence Critic và risk-aware abstention. README mô tả critic DeBERTa, nhóm ứng viên và đánh giá retrieve→rerank→answer. | Chồng lấn cao với ý tưởng tổng quát chọn bằng chứng để sửa tiền đề và từ chối theo rủi ro. Đổi sang tiếng Việt/du lịch không tự tạo novelty phương pháp. |
| [BAR-RAG — Rethinking the Reranker](https://huggingface.co/papers/2602.03689), 03/02/2026; [mã tác giả](https://github.com/GasolSun36/BAR-RAG) | Chọn bằng chứng phù hợp khả năng generator và căn chỉnh phân phối huấn luyện. README ghi selector bị bỏ ở suy luận; generator đã fine-tune dùng kết quả retriever. | Không mô tả sai BAR-RAG thành một cross-encoder online thông thường. Sự khác biệt giữa relevance/reranking và tiện ích sinh đáp án đã có tiền lệ rõ. |
| [ViHoRec](https://huggingface.co/papers/2607.12946), 14/07/2026; [mã/dữ liệu tác giả](https://github.com/MinhNguyenDS/ViHoRec) | Tài nguyên khách sạn tiếng Việt có kiểm soát chất lượng và benchmark theo tương tác. | Không dùng tuyên bố rộng “dataset du lịch tiếng Việt đầu tiên”. ViHoRec không phải QA có bằng chứng; khoảng trống QA hẹp hơn vẫn cần tra cứu tiếp. |
| [TRACE](https://huggingface.co/papers/2604.14223), 14/04/2026 | Hội thoại khuyến nghị du lịch bền vững, giải thích counterfactual và nghiên cứu người dùng theo abstract. README tải được chỉ là trang giới thiệu ngắn. | Ứng dụng chatbot du lịch/giải thích đã có. “Counterfactual alternative” trong khuyến nghị khác “false factual premise”; không gộp hai nhiệm vụ khi so sánh. |
| [Trip+](https://huggingface.co/papers/2606.21169), 19/06/2026; [mã tác giả](https://github.com/junle-chen/trip-plus) | Lập hành trình tương tác, hồ sơ người đi và chất lượng trải nghiệm. README mô tả kiểm tra ràng buộc và LLM user simulation. | Đánh giá ứng dụng phải gắn với nhiệm vụ du lịch. User simulation bằng LLM không đồng nhất với thẩm định/nghiên cứu người thật. Không so điểm lập hành trình trực tiếp với pairwise accuracy. |
| [GroupTravelBench](https://huggingface.co/papers/2605.25200), 24/05/2026 | Abstract mô tả 650 tác vụ tổng hợp, dữ liệu POI/giá vé và sandbox cho lập hành trình nhóm. | Số lượng câu hỏi mới không đủ; nguồn lực cần nhiệm vụ, nhãn và protocol có ích riêng. Không bắt buộc bài QA của chúng ta phải làm multi-agent. |
| [Disentangling Robustness and Sycophancy in Multilingual LLMs](https://huggingface.co/papers/2610.30696), 22/10/2026 | Abstract phân biệt năng lực biết đáp án với khả năng giữ đáp án khi chịu áp lực tiền đề sai. Chưa kiểm tra toàn văn hoặc danh sách ngôn ngữ. | Rất gần mốc nộp hiện tại. Tách trả lời câu gốc và sửa tiền đề có ích, nhưng không nhận cách phân rã tổng quát đó là mới của mình. |

Các tài liệu bổ sung trong ma trận gồm GRACE, Over-Searching/OverSearchQA, ARHN,
Querit-Reranker, ClariEdge, ISLAMICFAITHQA, TourPlanner, WorldTravel, IntTravel và
HiMAP-Travel. Chúng cung cấp tiền lệ cho grounding/abstention, false negatives,
thích ứng cross-encoder đa ngôn ngữ, clarification và đối chứng tác vụ du lịch.

Một lưu ý phiên bản: abstract ViHoRec và README hiện tại ghi số lượng tương tác
khác nhau; README còn phân biệt short-history với strict cold-start. Vì vậy không
dùng một con số từ abstract như thống kê chắc chắn của bản dữ liệu mới nhất, cũng
không coi khác biệt phiên bản này là bằng chứng để hạ chất lượng công trình đó.

Ngoài năm 2026, [TourLLM/Cultour](https://huggingface.co/papers/2407.12791) và
[Sustainability Augmented Reranking cho du lịch](https://huggingface.co/papers/2409.18003)
đã có metadata/abstract từ năm 2024. RAG, fine-tuning hoặc reranking áp dụng vào du
lịch vì thế không nên được trình bày như một ý tưởng chưa có tiền lệ.

## Đánh giá từng loại đóng góp

**Phương pháp.** BM25 + dense + RRF + cross-encoder là cấu hình hệ thống. Hard
negative augmentation và weighted margin loss cũng chưa đủ tạo một đóng góp
phương pháp riêng nếu không có lý do kỹ thuật khác biệt và đối chứng đúng. Bản
thảo tự ghi nhận legacy loss ablations chưa hợp lệ; không thể dùng các lượt chạy
đó để chứng minh loss mới vượt standard hinge. Các công trình CoRM-RAG, CausalNeg
và BAR-RAG làm tăng mức chồng lấn cần giải thích, chứ không buộc chúng ta phải
chạy đua thêm RL/multi-agent để được coi là nghiên cứu.

**Nguồn lực.** Có khả năng bảo vệ một nguồn lực QA du lịch tiếng Việt có nguồn
bằng chứng, nhiều đoạn liên quan, nhãn phản ứng trả lời/sửa tiền đề/thiếu bằng
chứng và chia tập theo nguồn. Đây là **khoảng trống cần kiểm chứng**, chưa phải
tuyên bố “đầu tiên”. Giá trị nằm ở nhu cầu thực, nhãn đúng, tài liệu nguồn/protocol
và khả năng dùng lại, không nằm riêng ở tên tiếng Việt hoặc số chunk.

**Kết quả thực nghiệm.** Pairwise accuracy tăng từ khoảng 0,883 lên 0,968 là kết
quả cụ thể của bài, nhưng không tự chứng minh cải thiện corpus retrieval hoặc
đáp án. Một nghiên cứu có thể đáng công bố dù phát hiện augmentation gây hại,
nếu kiểm chứng được nơi xảy ra lỗi, nguyên nhân và ý nghĩa thực tế. Cần phân rã:
gold evidence không vào pool; gold vào pool nhưng bị rerank xuống; evidence đủ
nhưng generator trả lời sai; nguồn thực sự thiếu bằng chứng; nguồn đủ nhưng hệ
thống từ chối. Không gộp các lỗi này thành một chỉ số hallucination chung.

**Ứng dụng du lịch.** Với định vị nhắm Information Technology & Tourism, nên đặt
vấn đề quanh thông tin du lịch có thể dựa vào để quyết định: lịch tham quan, điều
kiện vé, đường đi, đặc điểm điểm đến hoặc thông tin theo thời điểm. Thống kê năm
trong tin tổng kết ngành có thể thuộc miền du lịch nhưng chưa chắc đại diện tốt
cho nhu cầu chatbot của du khách. Đây là khuyến nghị định vị nghiên cứu, không
phải trích một yêu cầu editorial mới đã kiểm chứng của tạp chí.

Audit và nhiều seed làm tăng độ tin cậy. Chúng hỗ trợ các đóng góp trên; chúng
không tự bảo đảm một khoảng trống mới hoặc một mức quartile/khả năng nhận bài.

## Rủi ro cụ thể của bộ dữ liệu hiện tại

1. Nhóm 100 câu từ chối ban đầu cùng hỏi ưu đãi sinh viên, thay chủ yếu tên bài.
   Mô hình có thể học mẫu câu thay vì khả năng nhận biết đủ bằng chứng. Phải đa
   dạng hóa nhu cầu và kiểm tra shortcut trước khi xem đó là benchmark hành vi.
2. Một số nhiễu số trên phát biểu “gần/trên/khoảng” hoặc khoảng giá trị chưa chắc
   tạo mệnh đề bị nguồn bác bỏ. Cần tách **contradicted** với **unsupported**.
3. Có nhiều câu lịch sử/thống kê/quản lý. Bản sửa gắn 88 cờ cần kiểm tra lại tính
   phù hợp tác vụ; đây là cờ rà soát, không phải nhãn người dùng đã xác nhận.
4. Đã thấy một trường hợp tiêu đề/nội dung Đà Nẵng nhưng metadata tỉnh là Bắc Kạn
   (`FAM_0ac8f76b76e03a6be438`). Không tự sửa tỉnh hoặc nhận geography holdout là
   đáng tin trước khi kiểm tra nguồn và quy tắc địa danh theo thời gian.
5. Nhãn từ chối corpus-wide chưa được xác lập; thẩm định một đoạn không tìm thấy
   thông tin chưa đủ. Các kết quả downstream cũ vẫn là exploratory.

Các điểm này là lý do sửa thiết kế và bằng chứng trước nộp, không phải lý do kết
luận tự động bài chỉ có thể đăng Q3.

## Hướng đóng góp nên theo đuổi

Định vị có cơ sở nhất để kiểm chứng tiếp là:

> Một nguồn lực đánh giá độ tin cậy bằng chứng cho QA du lịch tiếng Việt, kết hợp
> nghiên cứu có kiểm soát về việc cải thiện pairwise có chuyển thành chọn bằng
> chứng và hỗ trợ quyết định đúng trong chatbot hay không.

Cần hoàn thành ba phần để biến định vị này thành kết quả:

- Xác minh bài gần nhất trong tạp chí và nguồn xuất bản năm 2026, nhất là QA du
  lịch tiếng Việt/low-resource evidence-grounded tourism QA; ghi rõ phần đã có
  và phần khác. Không dùng “first” khi chưa đủ độ phủ.
- Hoàn thiện benchmark từ các nhu cầu du lịch đa dạng, kiểm tra bằng chứng và
  nghĩa của nhiễu số, công bố phiên bản/điều kiện truy cập và quy trình nhãn rõ.
  Duyệt lại các mục thay đổi; giữ kết quả thẩm định cũ đúng với phiên bản cũ.
- Chạy đối chứng cùng candidate pools, dữ liệu và update budget; báo cáo ranking
  thực và đáp án có bằng chứng, độ bao phủ/từ chối sai, thời gian/chi phí. Một
  baseline reranker đa ngôn ngữ hiện đại và evidence-policy baseline sẽ giúp
  phân biệt tác động của dữ liệu, mô hình và cách chọn/trả lời. Nếu claim chính
  là hỗ trợ quyết định du lịch, cần phép đánh giá quyết định phù hợp; chấm mức
  tin tưởng đơn thuần không thay cho độ đúng của quyết định.

Không cần thêm một loss mới chỉ để gắn chữ novelty. Nếu phần nguồn lực và thực
nghiệm giải quyết được khoảng trống hẹp, đó có thể là hướng bài phù hợp. Hiện
chưa có đủ căn cứ để nói bản thảo đã đạt mức đó hoặc tránh chắc chắn desk reject.

## Công việc đã thực hiện trong vòng này

- Tạo phiên bản nháp mới `data/research_v2/batch_002_revision_001/`, lưu riêng
  14 câu bị đề nghị loại; còn **286 câu**. Bản gốc không đổi.
- Đề xuất **156 câu viết lại**: 93 câu hỏi thông tin và 63 câu kiểm tra tiền đề,
  gồm 4 trường hợp phát hiện thêm khi mở rộng ngữ cảnh. 8 bản ghi đổi anchor
  trong cùng nguồn/component/partition; không đưa nguồn mới xuyên tập.
- Trạng thái duyệt, nhãn và qrels của câu đổi nội dung được xóa ở bản đề xuất;
  bản ghi người chấm cũ giữ nguyên trong lineage. Không tự ghi VERIFIED.
- Gắn cờ 13 counterfactual số gần đúng và 5 khoảng giá trị để kiểm tra nghĩa;
  chưa đổi nhãn tự động.
- Chuẩn bị lại truy xuất trên câu sửa và kiểm tra trùng nguồn. Không huấn luyện
  hoặc dùng kết quả TEST để chọn cách viết câu.

Các miền học thuật cần thiết đã lưu thêm vào bản nháp cấu hình môi trường; runtime
vẫn chưa cho truy cập Crossref/OpenAlex/Springer. Cần áp dụng cấu hình mạng trước
khi hoàn tất phần đối chiếu tạp chí và toàn văn mà yêu cầu “tra thật kỹ” đòi hỏi.
