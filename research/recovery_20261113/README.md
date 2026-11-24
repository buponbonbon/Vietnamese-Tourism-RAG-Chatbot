# Kiểm chứng lại nghiên cứu Vietnamese Tourism RAG — 13/11/2026

Đây là snapshot mã và kết quả để tác giả xem xét. Branch không chứa bản thảo,
LaTeX/Bib/PDF, trọng số mô hình, Excel gốc của người chấm hoặc toàn bộ dữ liệu
huấn luyện. Các tệp được chép nguyên byte có checksum trong
`COPIED_FILE_IDENTITIES.json`.
`.gitattributes` giữ nguyên byte và CRLF của export gốc để checkout trên máy
khác không tự làm thay đổi checksum.

## Kết quả nào đã được kiểm chứng?

| Phân tích | Đầu vào/phạm vi | Kết luận được hỗ trợ |
|---|---|---|
| Native/oracle/K | 140 câu phát triển, 700 native rows, 81 nhóm nguồn; 13 checkpoint đã có | Bundle M1 chọn bằng chứng đủ kém hơn M0 trong thiết kế cùng ngân sách; lỗi có ở K=2 và sau oracle insertion. |
| Quality mask | Q/R, ba seed mỗi arm, cùng số cập nhật và số zero-weight presentations | Chưa chứng minh targeted mask cải thiện chọn bằng chứng so với random mask. |
| Objective controls | 9 score/log records R03/R05/R08, 1.136 cặp TEST | Không chứng minh weighting hoặc record-specific margin tốt hơn trong các đối chứng đã thực hiện. Đây là re-audit saved logits, không fresh inference 9 weights. |
| WikiQA | 237 native mixed-label pools, 2.341 câu, 234 title groups; BGE + M0/M1 | M1–M0 MRR CI chứa 0; không xác lập chiều khác biệt hoặc equivalence. |

Native grade-2 Hit@1: M0 **30,24%**, M1 **6,67%**; paired difference
−23,57 điểm phần trăm, 95% source-group interval [−32,40; −15,38].
Chỉ 45/140 native pools có grade-2 evidence. Với 95 pool ban đầu thiếu evidence,
thêm original anchor được hai reviewer DONE/grade2: M0 chọn đúng **91,23%**,
M1 **33,68%**. Đây là oracle diagnostic, không là retriever deploy được.

Tất cả số liệu hiện tại lấy từ `evidence_selection/results.json`. Không dùng
các con số làm tròn ở ghi chú/hội thoại thay cho score files. Đặc biệt oracle
Q là **34,04%**, R **34,39%** trong lần fresh inference này.

## Tái phân tích từ logits, không GPU/mạng

Chạy từ thư mục gốc repository, Python có các thư viện trong
`requirements-analysis.txt`:

```bash
python research/recovery_20261113/evidence_selection/analyze_results.py
python research/recovery_20261113/evidence_selection/verify_saved_scores.py
python research/recovery_20261113/objective_controls/verify_saved_scores.py
python research/recovery_20261113/wikiqa/analyze_wikiqa.py
python research/recovery_20261113/wikiqa/verify_metrics_independently.py
```

Native verifier dùng outranker counts, công thức tổ hợp cho exhaustive K,
và group-resampling frequencies để kiểm tra lại 1.820 model–query outcomes
và 7 contrast. Objective verifier kiểm tra 9 score/update logs, 6 McNemar/Holm
tests và group bootstrap. Các lệnh không train hoặc tái tạo logits.
Population SD mô tả ba fixed seeds; CI điều kiện theo checkpoint/cohort hiện có.
Các diagnostic CI không được diễn giải như nhiều confirmatory tests độc lập.

## Hai mức kiểm chứng nguồn

1. **Chạy được từ public bundle:** tính toán lại metric/CI từ frozen scoring
   rows và logits. Chúng có grade labels và source-group IDs cần cho phép tính;
   không có danh tính hoặc ghi chú tự do của reviewer.
2. **Cần đầu vào gốc do tác giả giữ:** kiểm tra lại nguyên bản Excel,
   eligibility và corpus/retrieval identities với
   `evidence_selection/verify_independently.py`. Cần đúng hai workbook,
   `benchmark_draft.csv`, `native_bm25_top50.csv` và corpus ở
   `evidence_selection/source_inputs/`, theo hash trong `protocol.json`.
   Objective source audit dùng `objective_controls/audit_with_original_test.py`
   với đúng TEST tại `data/processed/cross_encoder_TEST.json`.

Full source audit đã PASS ngày 13/11, report gốc được giữ nguyên. Hai public
saved-score verifiers ghi report riêng và **không** tuyên bố đã đọc lại workbook
hoặc TEST text. Workbook, raw human-review records và các pair files tiếp tục
được giữ riêng theo quy ước privacy của repository; không force-add chúng.

Đường dẫn model tuyệt đối trong protocol ghi lại máy khôi phục ban đầu. Full
inference trên máy khác cần một protocol mới ánh xạ tới đúng hashed weights,
không sửa protocol cũ rồi gọi đó là cùng một run. Scorer được lưu để kiểm tra
implementation; các lệnh mặc định ở trên chỉ tái phân tích.

## Giới hạn quan trọng

- 140 câu được dựng lại từ nguồn gốc, gồm recorded train/validation/calibration/
  test; đây là **development cohort**, không blind held-out test mới.
- R2 COMPLETED là post-feedback. Giữ 103 pending question records mỗi reviewer,
  118 pending passage records R2 và 26 grade disagreements ngoài phần đủ điều
  kiện; không xem chúng tự động là lỗi của R2 hoặc tự sửa cho đồng thuận.
- Các workbook/corpus/query/retrieval hashes gốc được lấy lại, nhưng old
  cohort export/logits trước lần gián đoạn chưa được tìm lại. Kết quả hiện tại
  là reconstructed cohort + fresh full inference, không byte-identical recovery
  của export bị mất hoặc independent replication bởi một nhóm khác.
- M0/M1 thay cả augmentation bundle tại matched budget; không cô lập mọi tính
  chất của hard negatives. Q/R còn khác repeated-row exposure patterns.
- Chỉ đánh giá BM25 top5, không full hybrid pipeline hoặc generator.
  Không có bằng chứng giảm hallucination, tăng lợi ích cho du khách, một loss
  mới vượt trội hoặc tác hại augmentation phổ quát.
- Original training-source script của matched/quality suites có hash trong
  plan nhưng chưa có trong archive nhận lại. Manifest/weights không thay cho
  việc kiểm tra source code đó. Không tự tạo định nghĩa C-TDQ hoặc thuật toán
  mới từ các ghi chú chưa khôi phục.

`DATA_TERMS.md` phân biệt các nguồn dữ liệu. Đây là analysis snapshot để review,
không phải thông báo phát hành một benchmark đã hoàn tất thẩm định hoặc DOI.
