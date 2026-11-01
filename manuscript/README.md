# Bản thảo du lịch — bản phục hồi 02/11/2026

Đây là bản làm việc để tác giả thẩm định, **chưa phải hồ sơ nộp bài hoàn tất**.
Mốc ngày dùng Asia/Ho_Chi_Minh. Bản thảo được phục hồi từ ZIP ngày 31/10 còn
truy cập được; không đại diện cho các thay đổi chưa lấy lại được từ workspace lỗi.

Các tệp chính:

- `main.tex`: bản có trích dẫn số để tác giả đọc và sao chép.
- `main_itt.tex`: cùng nội dung, trích dẫn tác giả–năm, bản làm việc có tên tác giả.
- `main_itt_anonymous.tex`: bản Springer `sn-jnl`, `sn-basic`, ẩn danh; được tạo
  bằng `python -m research.render_springer_manuscript` từ thư mục gốc repository.
- `title_page.tex`: thông tin tác giả, liên hệ, đóng góp, tài trợ và xung đột lợi ích.
- `references.bib`: 13 công trình đã xuất bản, gồm bảy công trình năm 2026.
- `source_verification.csv`: nguồn và phạm vi kiểm chứng từng tài liệu.
- `SOURCES_AND_CHANGES.md`: thay đổi và giới hạn của bản phục hồi.
- `history_20261031/`: ghi chú lịch sử nguyên vẹn; trạng thái cũ không thay cho
  trạng thái phục hồi hiện tại.

Hướng dẫn IT&T kiểm tra lại trong lần phục hồi yêu cầu trích dẫn tác giả–năm và
phản biện ẩn danh. Vì vậy bản trích dẫn số là bản đọc của tác giả. Khi nộp cần dùng
bản theo hướng dẫn tạp chí và điền các trường khai báo trong hệ thống SNAPP theo
yêu cầu lúc nộp. Hướng dẫn chính thức:
<https://link.springer.com/journal/40558/submission-guidelines>.

Các bảng B/C/D giữ kết quả của bản kiểm toán lưu trữ; lần phục hồi chỉ kiểm tra
nhất quán số học, dữ liệu đầu vào và bản review còn truy cập được. Chưa chạy lại
chín checkpoint trong workspace hiện tại. Chưa thêm kết quả full-corpus hoặc
generated-answer mới. Không gọi các nhãn do trợ lý đề xuất là thẩm định người thật.

Phục hồi đã xác định đúng danh tính văn bản của 268 câu/1.688 cặp mỗi người từ
hai file vòng 2. Chưa phục hồi được bundle sửa sau đó và kết quả chạy mới. Các
vấn đề trong bản review gốc được giữ trong ledger; đây **không phải yêu cầu tác
giả chấm lại toàn bộ**, cũng không phải kết luận các chỉnh sửa sau đó chưa từng làm.
Xem [trạng thái phục hồi](../research/RECOVERY_STATUS_20261102.md).

Biên dịch từ thư mục gốc repository:

```sh
python -m research.bootstrap_manuscript_tex --out data/manuscript_tex_dependencies
python -m research.render_springer_manuscript
cd manuscript
TEXINPUTS=.:../data/manuscript_tex_dependencies: \
BSTINPUTS=.:../data/manuscript_tex_dependencies: \
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build_review -pdflatex='pdflatex -no-shell-escape %O %S' \
  main.tex main_itt.tex main_itt_anonymous.tex title_page.tex
```

Cần TeX Live, `latexmk` và các gói được liệt kê trong phần đầu tệp TeX. Công cụ
bootstrap chỉ lấy các phụ thuộc chính thức có checksum cố định; không sửa dữ liệu
hay mô hình. Trên Overleaf, có thể tải các tệp TeX/BibTeX cùng các phụ thuộc đã
kiểm tra. Không đưa title page hoặc bản có tên tác giả vào gói phản biện ẩn danh.
