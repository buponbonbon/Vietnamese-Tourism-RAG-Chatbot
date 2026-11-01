# Nguồn và thay đổi của bản phục hồi

Ngày: 02/11/2026 (Asia/Ho_Chi_Minh). Ghi chú lịch sử ngày 31/10 được giữ riêng
trong `history_20261031/`. Không ghi đè các kết luận lịch sử thành kết quả mới.

## Căn cứ phục hồi

- ZIP bản thảo còn lưu: `data/manuscript_working_001/deliverables/`;
  SHA-256 `abe7f2c5216863909166376028d55428cc907f62aed926c7d8d7294e5389e1bb`.
- Bundle câu hỏi: `batch_002_revision_001`, benchmark SHA-256
  `e79ca318b867d5d9c11cc6f14ccfec83736e2297c9bfefe5443834b6cb0ff8fa`.
- R01 vòng 2: `ab242188f977562fed86f9ed473439db8cd59ae2b62671db7efca46c2672aa2e`.
- R02 bản COMPLETED: `7aa4f4646c584308389136554d2a712dbe324d36d88f899e41b2ba4b5c7c182b`.

Workspace chứa các thay đổi và lượt chạy sau mốc này không khởi động được. Các
kết quả đó chưa thể đọc và không được chèn vào bản thảo phục hồi từ ghi nhớ hội
thoại. Ghi chú phiên trước chỉ dùng xác định phần cần tìm lại.

## Sửa nội dung và tài liệu

Giữ các bảng kiểm toán pairwise cũ và ghi rõ nguồn lưu trữ; không trình bày chúng
như kết quả vừa chạy lại. Bổ sung kết quả **audit danh tính văn bản** của hai file
vòng 2. Bản R02 mới nhất là bản trả lại sau phản hồi nên không dùng thống kê
đồng thuận hiện tại như một vòng đánh giá mù độc lập mới. Việc trích nguyên văn
đúng chỉ kiểm tra tính hợp lệ cơ học, không chứng minh đoạn văn đủ bằng chứng.

Điền thông tin tác giả đúng xác nhận trong hội thoại: hai tác giả thuộc FPT
University, Ho Chi Minh City, Vietnam; Tran An Khang liên hệ tại
`khangtran.mschcm@gmail.com`. Khang phụ trách các phần nghiên cứu và viết bài;
Hai Anh thực hiện training. Không nhận tài trợ và không có xung đột lợi ích.
Không suy ra cấu hình phần mềm hay thông số huấn luyện chỉ từ tên GPU người dùng kể.

Danh mục hiện có 13 công trình đã xuất bản; bảy công trình năm 2026 gồm CausalNeg,
CoRM-RAG, TRACE, ViTravelRAG và ba bài IT&T về occupancy, hotel reviews và hotel
selection audit. Giữ đúng giới hạn đọc toàn văn/abstract trong bảng nguồn.
RAG được dẫn theo NeurIPS 2020 và M3-Embedding theo Findings ACL 2024, thay các
bản arXiv. Sửa tên tác giả BGE-M3 thành Jianlyu Chen theo ACL Anthology.
Model card của reranker là nguồn phần mềm, không đưa vào danh mục bài đã phản biện.
Các tài liệu chỉ có preprint được giữ trong ghi chú lịch sử, bỏ khỏi trích dẫn
nghiên cứu hiện tại; không đổi nhãn thành bài tạp chí chỉ vì có DOI.

Trong lần phục hồi đã truy cập lại trang NeurIPS, ACL và hướng dẫn/scope IT&T;
đối chiếu được metadata Crossref của 10/11 DOI còn lại. DOI hotel-selection audit
trả HTTP 429 trong lần kiểm tra này nên giữ bằng chứng publisher/Crossref đã lưu
ngày 31/10, không ghi như đã tái kiểm chứng thành công hôm nay. Đây không phải
một rà soát hệ thống mới bao phủ mọi công bố đến ngày hiện tại.

## Giới hạn đóng góp và công việc còn thiếu

Dynamic Loss hiện là mô tả một objective đã triển khai, chưa có đối chứng đúng
để quy lợi ích cho trọng số sigmoid. Chưa tìm lại được công thức hoặc mã định
nghĩa C-TDQ; không thêm tên này thành thuật toán. Retrieval–Decision Gap chỉ là
giả thuyết cần đo bằng cùng câu hỏi/pool và nhãn hợp lệ. Không lấy accuracy TEST
trừ metric trên bộ VAL khác để tạo một “gap” định lượng.

Phần nguồn lực/ứng dụng vẫn có thể phát triển, nhưng chưa có cơ sở xác nhận bài
đã sẵn sàng Q1. Cần phục hồi kết quả mới, chốt nhãn đúng phiên bản, rồi chạy và
phân tích phép so sánh cùng pool trên phần held-out được duyệt. Không bảo đảm
tránh desk reject bằng sửa cách diễn đạt. Chi tiết đánh giá phương pháp nằm tại
[RECOVERY_STATUS_20261102.md](../research/RECOVERY_STATUS_20261102.md).
