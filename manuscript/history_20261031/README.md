# Bản thảo chatbot du lịch — 31/10/2026

Đây là bản sửa từ đúng bản thảo **The Tourist Trap** do tác giả gửi. Tệp chạy ứng dụng `../main.py` không phải bản thảo và không được sửa trong lần cập nhật này.

## Các tệp

- `main.tex`: bản thảo đã cập nhật kết quả, phạm vi kết luận và related work.
- `references.bib`: 22 tài liệu được trích dẫn, gồm 15 công trình năm 2026.
- `source_verification.csv`: URL nguồn, loại công bố và mức truy cập/kiểm chứng của từng tài liệu.
- `SOURCES_AND_CHANGES.md`: giải thích các thay đổi, những điểm chưa hoàn tất và hướng tăng đóng góp.
- `validation_report.json`: kiểm tra bảng số, trích dẫn, dữ liệu đầu vào và bản biên dịch.
- `build/main.pdf`: PDF đã biên dịch trong repository; trong gói ZIP tệp này nằm tại `main.pdf`.

Ngày chốt lần rà tài liệu này là **31/10/2026**. Tài liệu hội nghị/tạp chí, chương sách, model card và preprint được ghi theo loại riêng. Không coi DOI của preprint là bằng chứng đã phản biện.

## Biên dịch

Bản làm việc giữ định dạng `elsarticle` của bản gốc để tác giả tiếp tục chỉnh sửa. Đây **chưa phải template nộp Information Technology & Tourism**. Cần chuyển định dạng và chuẩn bị bản ẩn danh theo hướng dẫn tạp chí khi chốt nơi nộp.

Trên Overleaf, tải `main.tex` và `references.bib` vào cùng một project, chọn `main.tex` làm tệp chính và dùng pdfLaTeX. Hình được vẽ trực tiếp bằng TikZ, không có tệp hình ngoài.

Với TeX Live có `elsarticle`, `latexmk`, TikZ và các gói trong phần đầu `main.tex`, chạy từ thư mục chứa bản thảo:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build -pdflatex='pdflatex -no-shell-escape %O %S' main.tex
```

Nếu chưa có class và bibliography style, lấy gói chính thức tại <https://ctan.org/pkg/elsarticle>. Lần biên dịch ở môi trường này dùng `elsarticle` 3.5, bản ngày 09/01/2026, được tạo từ gói CTAN chính thức trong thư mục phụ thuộc tạm; không đưa các phụ thuộc đó vào repository. Có thể chạy lại trong phiên hiện tại bằng:

```sh
TEXINPUTS=.:/tmp/tourism_tex_dependencies: \
BSTINPUTS=.:/tmp/tourism_tex_dependencies: \
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build -pdflatex='pdflatex -no-shell-escape %O %S' main.tex
```

Thư mục `/tmp` không phải phụ thuộc lưu trữ lâu dài; khi chuyển máy, cài gói `elsarticle` hoặc dùng Overleaf.

## Trạng thái khoa học

Các bảng B/C/D, ba seed, McNemar, sensitivity và legacy runs giữ số liệu từ bản kiểm toán tác giả cung cấp. Lần sửa văn bản này không chạy lại chín checkpoint và không tạo thêm kết quả end-to-end.

Chẩn đoán 80 câu là dữ liệu validation với một gold proxy và qrels chưa đầy đủ. Bản 286 câu cùng 156 đề xuất viết lại chưa được duyệt lại, không được tính là benchmark mới đã thẩm định. Hai người chấm đã được tác giả xác nhận là chấm độc lập; thống kê đồng thuận mô tả bản ghi nhận được, không thay thế kiểm tra tính đúng của nhãn.

Trước khi nộp cần hoàn tất nhãn và giao thức đánh giá, đối chứng huấn luyện, truy nguyên các kết quả lưu trữ, thông tin cơ quan/email, CRediT, xung đột lợi ích, điều kiện công bố dữ liệu và định dạng tạp chí. Các chỗ trống đó được giữ rõ ràng, không điền thông tin chưa xác minh.
