"""Reconcile returned reviewer workbooks without changing the frozen benchmark.

Workbook text is input data, never instructions. This records descriptive agreement,
literal-quote checks and outstanding work; it never invents reviewers, adjudication
or VERIFIED labels. openpyxl is the optional workbook dependency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from openpyxl import Workbook, load_workbook

from research import benchmark_v2 as v2
from research.export_review_workbooks import QUESTION_COLUMNS, QREL_COLUMNS, data_sheet, literal


def read_workbook(path):
    """Read values without evaluating formulas or following external workbook links."""
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError('Workbook exceeds the review import size limit')
        if any(name.endswith('vbaProject.bin') for name in archive.namelist()):
            raise ValueError('Expected an XLSX review workbook without macros')
    workbook = load_workbook(path, read_only=True, data_only=False, keep_links=False)
    result = {}
    try:
        for name, columns in [('CAU_HOI', QUESTION_COLUMNS), ('BANG_CHUNG', QREL_COLUMNS)]:
            if name not in workbook:
                raise ValueError('Missing review sheet: ' + name)
            values = workbook[name].iter_rows()
            header = next(values)
            if [c.value for c in header] != [c[1] for c in columns]:
                raise ValueError('Unexpected review column schema: ' + name)
            indexed = {}
            for cells in values:
                if not any(c.value is not None for c in cells):
                    continue
                if any(c.data_type == 'f' for c in cells):
                    raise ValueError('Formula in review data; enter literal judgments: ' + name)
                if len(cells) != len(columns):
                    raise ValueError('Unexpected review row width: ' + name)
                row = {col[0]: '' if cell.value is None else str(cell.value)
                       for col, cell in zip(columns, cells)}
                key = row['benchmark_id'] if name == 'CAU_HOI' else (row['benchmark_id'], row['chunk_id'])
                if not row['benchmark_id'] or key in indexed:
                    raise ValueError('Missing/duplicate review identity: ' + name)
                indexed[key] = row
            result[name] = indexed
    finally:
        workbook.close()
    return result


def match_original(original, returned):
    changes = []
    for name, columns in [('CAU_HOI', QUESTION_COLUMNS), ('BANG_CHUNG', QREL_COLUMNS)]:
        if set(original[name]) != set(returned[name]):
            raise ValueError('Returned membership differs from the selected export: ' + name)
        for key, row in returned[name].items():
            for field, _, _, editable in columns:
                if not editable and v2.normalize(row[field]) != v2.normalize(original[name][key][field]):
                    changes.append(dict(sheet=name, key=str(key), field=field))
    if changes:
        raise ValueError('Immutable question/evidence fields changed: ' + json.dumps(changes[:3]))


def agreement(first, second):
    if len(first) != len(second):
        raise ValueError('Matched agreement lists must have the same size')
    n = len(first)
    if not n:
        return dict(n=0, equal=0, observed_agreement=None, expected_agreement=None, cohen_kappa=None)
    a, b = Counter(first), Counter(second)
    equal = sum(x == y for x, y in zip(first, second))
    observed = equal / n
    expected = sum(a[label] * b[label] for label in a.keys() | b.keys()) / (n * n)
    kappa = (observed-expected)/(1-expected) if expected < 1 else None
    return dict(n=n, equal=equal, observed_agreement=observed, expected_agreement=expected,
                cohen_kappa=kappa, confusion=[dict(first=x, second=y, count=count)
                                            for (x, y), count in sorted(Counter(zip(first, second)).items())])


def normalized_equal(a, b):
    return v2.normalize(a) == v2.normalize(b)


def analyze(reviews, mapping, questions, chunks, independence='unconfirmed'):
    """Analyze records only; human independence is separately user-attested."""
    issues, qrows, erows = [], [], []
    def issue(qid, code, detail, action, cid=''):
        issues.append(dict(review_id=qid, benchmark_id=mapping[qid], chunk_id=cid,
                           issue_type=code, detail=detail, required_action=action))

    reviewer_ids = []
    for number, review in enumerate(reviews, 1):
        ids = {v2.normalize(r['reviewer_id']) for sheet in review.values() for r in sheet.values()}
        if '' in ids or len(ids) != 1:
            raise ValueError(f'File {number} must have one consistent nonempty reviewer ID')
        reviewer_ids.append(next(iter(ids)))
    if reviewer_ids[0] == reviewer_ids[1]:
        raise ValueError('Two distinct reviewer IDs are required')
    if independence != 'user-confirmed':
        for qid in reviews[0]['CAU_HOI']:
            issue(qid, 'INDEPENDENCE_NOT_CONFIRMED', 'Review provenance is not yet confirmed.',
                  'Confirm whether labels were recorded independently before discussion.')

    ids = sorted(reviews[0]['CAU_HOI'])
    for qid in ids:
        a, b = (review['CAU_HOI'][qid] for review in reviews)
        q = questions[mapping[qid]]
        labels = [v2.normalize(r['behavior_label']) for r in (a, b)]
        decisions = [v2.normalize(r['question_decision']) for r in (a, b)]
        for row in (a, b):
            if row['review_status'] != 'DONE' or row['behavior_label'] not in (*v2.BEHAVIORS, 'unclear') or row['question_decision'] not in ('KEEP', 'REVISE', 'REJECT', 'UNSURE'):
                issue(qid, 'INCOMPLETE_OR_INVALID_REVIEW', 'Question review has an incomplete status or invalid label.',
                      'Return this record for completion; preserve both original judgments.')
            if not v2.normalize(row['reference_answer']) or not v2.normalize(row['task_type']):
                issue(qid, 'MISSING_REFERENCE_OR_TASK', 'Reference response or tourism task is blank.',
                      'Provide an evidence-grounded reference and task category.')
        label = labels[0] if labels[0] == labels[1] else ''
        decision = decisions[0] if decisions[0] == decisions[1] else ''
        if not label or not decision or 'unclear' in labels or 'UNSURE' in decisions:
            issue(qid, 'QUESTION_ADJUDICATION_REQUIRED', 'Categorical labels disagree or remain uncertain.',
                  'An independent third reviewer must adjudicate with a rationale.')
        if decision == 'REJECT':
            issue(qid, 'REJECT_NEW_VERSION_REQUIRED', 'Both reviewers propose rejecting this question.',
                  'Archive it and prepare new benchmark membership; do not delete rows from the frozen original.')
        if decision == 'REVISE':
            issue(qid, 'REVISION_REQUIRES_REVIEW', 'Both reviewers request a revised query; original-query qrels cannot silently transfer.',
                  'Finalize self-contained wording, confirm its intended fact and recheck affected qrels before freezing.')
            if not a['query_revision'] or not b['query_revision']:
                issue(qid, 'REVISION_MISSING', 'A requested revision is blank.', 'Supply a concrete revised question.')
            if any('vị trí cần xác minh trong đoạn trích' in v2.normalize(r['query_revision']).casefold()
                   and '[GIÁ TRỊ CẦN XÁC MINH]' not in r['query_revision'] for r in (a, b)):
                issue(qid, 'REVISION_MISSING_REFERENT', 'Proposed wording refers to an unspecified position/excerpt.',
                      'Ask directly about the named date, price, place or event; include enough information to identify it.')
            if any('...' in r['query_revision'] or '…' in r['query_revision'] for r in (a, b)):
                issue(qid, 'REVISION_STILL_TRUNCATED', 'The proposed assertion still contains an ellipsis.',
                      'Check whether the factual assertion is complete; replace truncation with a short, complete claim.')
        if label == 'refuse' and decision != 'REJECT':
            issue(qid, 'CORPUS_REFUSAL_NOT_ESTABLISHED', 'No reviewed corpus-wide search log was supplied with this import; source-level absence alone is insufficient.',
                  'Review BM25+dense/RRF and source-neighbor evidence, record the corpus snapshot/search log, then confirm or change the label.')
        if not normalized_equal(a['reference_answer'], b['reference_answer']) or not normalized_equal(a['query_revision'], b['query_revision']):
            issue(qid, 'TEXT_RECONCILIATION_REQUIRED', 'Reference responses or proposed rewrites differ.',
                  'Resolve wording explicitly without discarding either original review.')
        qrows.append(dict(review_id=qid, benchmark_id=mapping[qid], family_id=q['family_id'], split=q['split'],
                          original_query=q['query'], consensus_behavior=label, consensus_decision=decision,
                          reviewer_1_id=reviewer_ids[0], reviewer_2_id=reviewer_ids[1],
                          reviewer_1_label=labels[0], reviewer_2_label=labels[1],
                          revision_1=a['query_revision'], revision_2=b['query_revision'],
                          reference_1=a['reference_answer'], reference_2=b['reference_answer'],
                          task_type_1=a['task_type'], task_type_2=b['task_type'],
                          search_notes_1=a['evidence_search_notes'], search_notes_2=b['evidence_search_notes'],
                          notes_1=a['reviewer_notes'], notes_2=b['reviewer_notes'],
                          annotation_status='REVIEW_RECORDED_NOT_FROZEN'))

    sufficient = defaultdict(set)
    invalid_quotes = missing_quotes = 0
    pairs = sorted(reviews[0]['BANG_CHUNG'])
    for qid, cid in pairs:
        a, b = (review['BANG_CHUNG'][(qid, cid)] for review in reviews)
        q = questions[mapping[qid]]
        if cid not in chunks:
            raise ValueError('Unknown corpus chunk in review')
        grades = [v2.normalize(row['relevance_grade']) for row in (a, b)]
        for row in (a, b):
            if row['review_status'] != 'DONE' or row['relevance_grade'] not in ('0', '1', '2', 'unclear'):
                issue(qid, 'INCOMPLETE_QREL', 'Evidence review is incomplete or has an invalid grade.',
                      'Complete the independent evidence judgment.', cid)
            if row['relevance_grade'] in ('1', '2'):
                quote = v2.normalize(row['evidence_quote'])
                if not quote:
                    missing_quotes += 1
                    issue(qid, 'MISSING_QUOTE', 'Grade 1/2 has no literal quote.', 'Supply a quote from this passage.', cid)
                elif quote not in chunks[cid]['text_content']:
                    invalid_quotes += 1
                    issue(qid, 'QUOTE_NOT_IN_PASSAGE', 'Quote does not match the identified corpus passage.',
                          'Correct the quote or grade after checking the source.', cid)
        grade = grades[0] if grades[0] == grades[1] else ''
        if not grade or 'unclear' in grades:
            issue(qid, 'QREL_ADJUDICATION_REQUIRED', 'Evidence grades disagree or remain uncertain.',
                  'Obtain independent adjudication for this query/passage pair.', cid)
        if grade == '2':
            sufficient[qid].add(cid)
            if chunks[cid]['split'] != q['split']:
                issue(qid, 'SUFFICIENT_EVIDENCE_CROSSES_SPLITS', 'Grade-2 evidence comes from another source partition.',
                      'Resolve source grouping in a new benchmark version.', cid)
        erows.append(dict(review_id=qid, benchmark_id=mapping[qid], chunk_id=cid,
                          consensus_grade=grade, reviewer_1_id=reviewer_ids[0], reviewer_2_id=reviewer_ids[1],
                          reviewer_1_label=grades[0], reviewer_2_label=grades[1],
                          evidence_quote_1=a['evidence_quote'], evidence_quote_2=b['evidence_quote'],
                          notes_1=a['reviewer_notes'], notes_2=b['reviewer_notes'],
                          annotation_status='REVIEW_RECORDED_NOT_FROZEN'))
    for row in qrows:
        qid, label = row['review_id'], row['consensus_behavior']
        if label == 'refuse' and sufficient[qid]:
            issue(qid, 'REFUSAL_CONFLICTS_WITH_GRADE2', 'Refusal conflicts with sufficient evidence in the supplied pool.',
                  'Reconcile the question label and passage grades.')
        elif label in ('answer', 'correct') and not sufficient[qid]:
            issue(qid, 'NO_SUFFICIENT_EVIDENCE', 'Answer/correction has no consensus grade-2 passage.',
                  'Review additional evidence before accepting the question.')
    flagged = {r['review_id'] for r in issues}
    for row in qrows:
        row['issue_count'] = sum(i['review_id'] == row['review_id'] for i in issues)
    metrics = {}
    for key, sheet, field in [('behavior', 'CAU_HOI', 'behavior_label'),
                              ('question_quality', 'CAU_HOI', 'question_decision'),
                              ('evidence_grade', 'BANG_CHUNG', 'relevance_grade')]:
        keys = sorted(reviews[0][sheet])
        metrics[key] = agreement([v2.normalize(reviews[0][sheet][k][field]) for k in keys],
                                 [v2.normalize(reviews[1][sheet][k][field]) for k in keys])
    text_match = {}
    for name, fields in [('CAU_HOI', ('reference_answer', 'query_revision', 'task_type', 'evidence_search_notes', 'reviewer_notes')),
                         ('BANG_CHUNG', ('evidence_quote', 'reviewer_notes'))]:
        keys = sorted(reviews[0][name])
        text_match[name] = {field: dict(equal=sum(normalized_equal(reviews[0][name][k][field], reviews[1][name][k][field]) for k in keys), total=len(keys))
                            for field in fields}
    summary = dict(status='RECONCILED_NOT_FROZEN', benchmark_ready=False, training_started=False,
                   independent_review_provenance='User confirms two independent reviewers' if independence == 'user-confirmed' else 'Not confirmed',
                   human_independence_certified_by_code=False, questions_per_reviewer=len(ids), qrels_per_reviewer=len(pairs),
                   categorical_agreement=metrics, free_text_matching=text_match,
                   consensus_decisions=dict(Counter(r['consensus_decision'] for r in qrows)),
                   consensus_behaviors=dict(Counter(r['consensus_behavior'] for r in qrows)),
                   consensus_evidence_grades=dict(Counter(r['consensus_grade'] for r in erows)),
                   issues_by_type=dict(Counter(r['issue_type'] for r in issues)),
                   unique_questions_with_issues=len(flagged), questions_without_recorded_issues=len(ids)-len(flagged),
                   missing_quote_checks=missing_quotes, invalid_quote_checks=invalid_quotes,
                   decision_counts_by_split={split:dict(Counter(r['consensus_decision'] for r in qrows if r['split']==split)) for split in v2.SPLITS},
                   note='Agreement is descriptive before adjudication; high agreement does not establish label validity or exhaustive evidence coverage.')
    return summary, qrows, erows, issues


def audit_refusal_sources(qrows, questions, chunks):
    """Lexical review cues in same-URL neighbors; never infer an answer label."""
    by_url = defaultdict(list)
    for chunk in chunks.values():
        by_url[chunk['canonical_url']].append(chunk)
    student = re.compile(r'sinh viên|học sinh', re.I)
    benefit = re.compile(r'ưu đãi|giảm giá|miễn phí|miễn vé|giảm vé|giá vé', re.I)
    scopes, candidates = [], []
    for row in qrows:
        if row['consensus_behavior'] != 'refuse':
            continue
        q = questions[row['benchmark_id']]
        neighbors = by_url[q['source_url']]
        matches = [r for r in neighbors if student.search(r['text_content']) and benefit.search(r['text_content'])]
        scopes.append(dict(review_id=row['review_id'], benchmark_id=row['benchmark_id'],
                           source_url=q['source_url'], source_chunks=len(neighbors),
                           keyword_candidate_chunks=len(matches), status='LEXICAL_SCOPE_CUE_ONLY'))
        for chunk in matches:
            candidates.append(dict(review_id=row['review_id'], query=q['query'], chunk_id=chunk['chunk_id'],
                                   candidate_evidence=chunk['text_content'], source_url=chunk['canonical_url'],
                                   label_status='UNLABELED', relevance_grade='', evidence_quote='', review_status='TODO'))
    return scopes, candidates, dict(
        refusal_questions=len(scopes), source_neighbor_passages_checked=sum(r['source_chunks'] for r in scopes),
        questions_with_keyword_candidates=sum(r['keyword_candidate_chunks']>0 for r in scopes),
        candidate_pairs=len(candidates), automatic_relabeling=False,
        limitations=['Student/benefit keyword co-occurrence is a review cue, not sufficient evidence.',
                     'No keyword match does not establish corpus-wide unanswerability.',
                     'Only same-URL neighbors were searched; dense/RRF pooling remains outstanding.'])


def export_report(out, summary, qrows, erows, issues, scope_candidates=()):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'TONG_HOP'
    lines = [
        ('KẾT QUẢ ĐỐI CHIẾU', 'Đã ghi nhận hai bộ đánh giá; benchmark chưa chốt.'),
        ('Nguồn đánh giá', summary['independent_review_provenance']),
        ('Số câu mỗi người', summary['questions_per_reviewer']), ('Số cặp bằng chứng mỗi người', summary['qrels_per_reviewer']),
        ('Đồng thuận hành vi / Cohen kappa', str(summary['categorical_agreement']['behavior']['observed_agreement'])+' / '+str(summary['categorical_agreement']['behavior']['cohen_kappa'])),
        ('Quyết định giữ / sửa / loại', json.dumps(summary['consensus_decisions'], ensure_ascii=False)),
        ('Câu có mục cần xử lý', summary['unique_questions_with_issues']),
        ('Câu chưa có cờ lỗi trong lần kiểm tra này', summary['questions_without_recorded_issues']),
        ('Phạm vi kết luận', 'Không tự chuyển DONE thành VERIFIED; giữ bản gốc và cả hai lượt đánh giá.'),
        ('Nguồn từ chối', 'Thiếu thông tin ở một bài nguồn chưa chứng minh toàn corpus không có bằng chứng.'),
        ('Câu sửa', 'Kiểm tra lại câu đầy đủ, đúng thông tin cần hỏi và qrels liên quan trước khi chốt.'),
        ('Tính độc lập', 'Người dùng xác nhận hai người chấm độc lập; mã chỉ kiểm tra bản ghi và mức trùng khớp.'),
        ('Cách dùng', 'Xem CAN_XU_LY và DOI_CHIEU_CAU; không dùng bảng này như bộ nhãn TEST cuối cùng.'),
    ]
    sheet.column_dimensions['A'].width = 45
    sheet.column_dimensions['B'].width = 115
    from openpyxl.styles import Alignment
    for index, (key, value) in enumerate(lines, 1):
        literal(sheet.cell(index, 1), key)
        literal(sheet.cell(index, 2), value)
        for cell in sheet[index]:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        sheet.row_dimensions[index].height = 48
    qcols = [('review_id', 'Mã câu', 20, False), ('split', 'Tập', 18, False),
             ('consensus_behavior', 'Nhãn đồng thuận', 20, False), ('consensus_decision', 'Giữ / sửa / loại', 20, False),
             ('original_query', 'Câu gốc', 65, False), ('revision_1', 'Câu sửa người 1', 65, False),
             ('revision_2', 'Câu sửa người 2', 65, False), ('reference_1', 'Đáp án người 1', 65, False),
             ('reference_2', 'Đáp án người 2', 65, False), ('search_notes_1', 'Tìm bằng chứng người 1', 60, False),
             ('search_notes_2', 'Tìm bằng chứng người 2', 60, False), ('notes_1', 'Ghi chú người 1', 60, False),
             ('notes_2', 'Ghi chú người 2', 60, False), ('issue_count', 'Số mục cần xử lý', 20, False)]
    data_sheet(workbook, 'DOI_CHIEU_CAU', qcols, qrows, 'DoiChieuCau')
    if issues:
        translations = {
            'REVISION_REQUIRES_REVIEW': ('Chốt câu sửa và kiểm tra lại bằng chứng', 'Hai người cùng yêu cầu sửa câu; nhãn bằng chứng hiện gắn với câu gốc.', 'Chốt câu hỏi đầy đủ, xác định đúng thông tin cần hỏi và kiểm tra lại qrels bị ảnh hưởng.'),
            'REVISION_MISSING_REFERENT': ('Câu sửa thiếu thông tin để hiểu câu hỏi', 'Câu sửa nhắc vị trí cần xác minh nhưng không kèm đoạn trích để xác định vị trí.', 'Hỏi trực tiếp về ngày, giá, địa điểm hoặc sự kiện cụ thể; không để người đọc phải đoán đoạn trích.'),
            'REVISION_STILL_TRUNCATED': ('Câu sửa còn dấu cắt cụt', 'Nhận định được đề xuất vẫn có dấu ba chấm; cần kiểm tra tính đầy đủ.', 'Viết một nhận định ngắn, đầy đủ chủ ngữ và thông tin cần xác minh.'),
            'CORPUS_REFUSAL_NOT_ESTABLISHED': ('Chưa đủ phạm vi để chốt nhãn từ chối', 'Chưa có log thẩm định tìm bằng chứng trong toàn corpus; không thấy ở một nguồn chưa đủ.', 'Bổ sung BM25 + dense/RRF và các đoạn cùng nguồn, ghi phạm vi corpus và kết quả, rồi chốt nhãn.'),
            'REJECT_NEW_VERSION_REQUIRED': ('Đề xuất loại khỏi phiên bản tiếp theo', 'Hai người cùng đề xuất loại câu này.', 'Lưu vết lý do và loại ở phiên bản benchmark mới; giữ nguyên bản gốc.'),
        }
        display_issues = []
        for issue in issues:
            row = dict(issue)
            if issue['issue_type'] in translations:
                row['issue_type'], row['detail'], row['required_action'] = translations[issue['issue_type']]
            display_issues.append(row)
        cols = [('review_id', 'Mã câu', 20, False), ('chunk_id', 'Mã đoạn', 25, False),
                ('issue_type', 'Loại vấn đề', 42, False), ('detail', 'Chi tiết', 85, False),
                ('required_action', 'Việc cần xử lý', 85, False), ('resolution', 'Kết quả xử lý', 70, True)]
        data_sheet(workbook, 'CAN_XU_LY', cols, display_issues, 'CanXuLy')
    ecols = [('review_id', 'Mã câu', 20, False), ('chunk_id', 'Mã đoạn', 25, False),
             ('reviewer_1_label', 'Mức người 1', 20, False), ('reviewer_2_label', 'Mức người 2', 20, False),
             ('evidence_quote_1', 'Trích dẫn người 1', 80, False), ('evidence_quote_2', 'Trích dẫn người 2', 80, False),
             ('notes_1', 'Ghi chú người 1', 60, False), ('notes_2', 'Ghi chú người 2', 60, False)]
    data_sheet(workbook, 'DOI_CHIEU_BANG_CHUNG', ecols, erows, 'DoiChieuBangChung')
    if scope_candidates:
        cols = [('review_id','Mã câu',20,False), ('query','Câu hỏi',65,False), ('chunk_id','Mã đoạn',25,False),
                ('candidate_evidence','Đoạn cùng nguồn cần xem thêm',85,False), ('source_url','Link nguồn',38,False),
                ('relevance_grade','Mức bằng chứng cần thẩm định',25,True), ('evidence_quote','Trích dẫn',65,True),
                ('review_status','Trạng thái',20,True)]
        data_sheet(workbook, 'NGUON_CAN_XEM_THEM', cols, scope_candidates, 'NguonCanXemThem')
    path = out/'Doi_chieu_va_xu_ly_tham_dinh.xlsx'
    workbook.save(path)
    check = load_workbook(path, read_only=True, data_only=False, keep_links=False)
    assert check['DOI_CHIEU_CAU'].max_row == len(qrows)+1
    assert check['DOI_CHIEU_BANG_CHUNG'].max_row == len(erows)+1
    assert all(cell.data_type != 'f' for ws in check for row in ws for cell in row)
    check.close()
    counts = summary['consensus_decisions']
    report = f'''# Đối chiếu kết quả thẩm định

Hai file khớp bộ **{len(qrows)} câu / {len(erows)} cặp bằng chứng mỗi người**.
Dữ liệu định danh, câu hỏi gốc và đoạn nguồn khớp bản xuất. Người dùng xác nhận hai người chấm độc lập.

| Nội dung | Kết quả |
|---|---:|
| Đồng thuận nhãn hành vi | {summary['categorical_agreement']['behavior']['equal']}/{len(qrows)} |
| Đồng thuận quyết định chất lượng | {summary['categorical_agreement']['question_quality']['equal']}/{len(qrows)} |
| Đồng thuận mức bằng chứng | {summary['categorical_agreement']['evidence_grade']['equal']}/{len(erows)} |
| Giữ / sửa / loại | {counts.get('KEEP',0)} / {counts.get('REVISE',0)} / {counts.get('REJECT',0)} |
| Câu có ít nhất một mục cần xử lý | {summary['unique_questions_with_issues']} |
| Câu chưa có cờ lỗi trong lần kiểm tra này | {summary['questions_without_recorded_issues']} |

Cohen kappa danh nghĩa: hành vi **{summary['categorical_agreement']['behavior']['cohen_kappa']}**,
quyết định chất lượng **{summary['categorical_agreement']['question_quality']['cohen_kappa']}**,
mức bằng chứng **{summary['categorical_agreement']['evidence_grade']['cohen_kappa']}**.
Đây là thống kê mô tả trước phân xử, không chứng nhận độ đúng của nhãn hoặc tính độc lập bằng mã.
Toàn bộ phần văn bản đánh giá cũng trùng nhau sau chuẩn hóa; thông tin này được giữ trong audit.

## Việc cần xử lý trước khi chốt benchmark

'''
    for code, count in summary['issues_by_type'].items():
        report += f'- `{code}`: {count} bản ghi.\n'
    report += f'''
Các cờ có thể trùng trên cùng một câu. Bảng Excel đính kèm giữ cả hai đề xuất để đối chiếu.
{summary['issues_by_type'].get('REVISION_MISSING_REFERENT',0)} câu sửa có tham chiếu mơ hồ và
{summary['issues_by_type'].get('CORPUS_REFUSAL_NOT_ESTABLISHED',0)} nhãn từ chối cần mở rộng phạm vi bằng chứng.
Các trích dẫn mức 1/2 được đối chiếu nguyên văn với corpus; việc khớp chữ không tự chứng minh đủ bằng chứng.

Rà thêm các đoạn cùng URL nguồn cho nhóm từ chối: đã kiểm tra
{summary['refusal_source_scope_audit']['source_neighbor_passages_checked']} đoạn, tìm thấy
{summary['refusal_source_scope_audit']['candidate_pairs']} cặp ứng viên chứa từ khóa về học sinh/sinh viên
và ưu đãi/giá vé, thuộc {summary['refusal_source_scope_audit']['questions_with_keyword_candidates']} câu.
Đây là gợi ý xem thêm, không phải kết luận nhãn từ chối sai và không thay thế truy xuất dense/RRF.

Không ghi đè benchmark gốc, không tự áp dụng câu sửa, không xóa câu bị đề xuất loại,
không chuyển trạng thái thành VERIFIED và không chạy thí nghiệm chính từ các bản ghi này.
Hoàn tất câu sửa/loại trong phiên bản mới, bổ sung pool và log tìm bằng chứng, rồi chốt nhãn và giao thức.
'''
    (out/'REPORT.md').write_text(report, encoding='utf-8')
    return path


def run(args):
    if args.out.exists():
        raise FileExistsError('Choose a new reconciliation version directory')
    v2.validate_bundle(args.bundle, args.corpus, allow_draft=True)
    packet_hash = v2.digest(args.export_dir/'export_manifest.json')
    export_meta = json.loads((args.export_dir/'export_manifest.json').read_text())
    reviews, original_hashes = [], {}
    for n, path in enumerate((args.first, args.second), 1):
        original = args.export_dir/f'Nguoi_tham_dinh_{n:02d}.xlsx'
        expected = export_meta['files'][original.name]['sha256']
        if v2.digest(original) != expected:
            raise ValueError('Original workbook differs from export manifest')
        returned = read_workbook(path)
        match_original(read_workbook(original), returned)
        reviews.append(returned)
        original_hashes[f'reviewer_{n}'] = dict(filename=path.name, sha256=v2.digest(path))
    if set(reviews[0]['CAU_HOI']) != set(reviews[1]['CAU_HOI']) or set(reviews[0]['BANG_CHUNG']) != set(reviews[1]['BANG_CHUNG']):
        raise ValueError('The two reviewer sets differ')
    mapping = {r['review_id']:r['benchmark_id'] for r in v2.read_csv(args.export_dir/'DIEU_PHOI_doi_chieu_ma_GIU_RIENG.csv')}
    questions = {q['benchmark_id']:q for q in v2.read_csv(args.bundle/'benchmark_draft.csv')}
    if any(qid not in mapping or mapping[qid] not in questions for qid in reviews[0]['CAU_HOI']):
        raise ValueError('Opaque review identity has no frozen benchmark mapping')
    for qid, row in reviews[0]['CAU_HOI'].items():
        identity = mapping[qid]
        expected = 'Q-'+hashlib.sha256(('review-v2:'+identity).encode()).hexdigest()[:12].upper()
        if qid != expected or not normalized_equal(row['query'], questions[identity]['query']):
            raise ValueError('Coordinator mapping does not match the frozen question identity/text')
    chunks = {r['chunk_id']:r for r in v2.load_corpus(args.corpus)}
    partitions = {r['chunk_id']:r['split'] for r in v2.read_csv(args.bundle/'corpus_manifest.csv')}
    for cid, row in chunks.items():
        row['split'] = partitions[cid]
    summary, qrows, erows, issues = analyze(reviews, mapping, questions, chunks, args.independence)
    scopes, scope_candidates, scope_summary = audit_refusal_sources(qrows, questions, chunks)
    summary['refusal_source_scope_audit'] = scope_summary
    out = v2.immutable_output(args.out)
    saved_inputs = out/'inputs'
    saved_inputs.mkdir()
    for n, path in enumerate((args.first, args.second), 1):
        shutil.copyfile(path, saved_inputs/f'reviewer_{n}_returned.xlsx')
    v2.write_csv(out/'question_consensus.csv', qrows)
    v2.write_csv(out/'qrel_consensus.csv', erows)
    v2.write_csv(out/'issues.csv', issues, ['review_id','benchmark_id','chunk_id','issue_type','detail','required_action'])
    v2.write_csv(out/'refusal_source_scope.csv', scopes,
                 ['review_id','benchmark_id','source_url','source_chunks','keyword_candidate_chunks','status'])
    v2.write_csv(out/'additional_source_candidate_review.csv', scope_candidates,
                 ['review_id','query','chunk_id','candidate_evidence','source_url','label_status','relevance_grade','evidence_quote','review_status'])
    summary.update(input_hashes=original_hashes, export_manifest_sha256=packet_hash,
                   benchmark_sha256=v2.digest(args.bundle/'benchmark_draft.csv'),
                   corpus_sha256=v2.digest(args.corpus),
                   annotation_status='REVIEW_RECORDED_NOT_FROZEN')
    v2.write_json(out/'review_summary.json', summary)
    report = export_report(out, summary, qrows, erows, issues, scope_candidates)
    archive = out/'Ket_qua_doi_chieu_tham_dinh.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as stream:
        for path in (report, out/'REPORT.md', out/'review_summary.json', out/'issues.csv', out/'question_consensus.csv', out/'qrel_consensus.csv', out/'refusal_source_scope.csv', out/'additional_source_candidate_review.csv'):
            stream.write(path, path.name)
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
    v2.write_json(out/'output_manifest.json', {p.name: v2.digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(questions=len(qrows), qrels=len(erows), decisions=summary['consensus_decisions'],
                         categorical_agreement=summary['categorical_agreement'], issues=summary['issues_by_type'],
                         unique_questions_with_issues=summary['unique_questions_with_issues'],
                         status=summary['status'], out=str(out)), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=v2.ROOT/'data/processed/vietnam_tourism_rag_dataset.csv')
    parser.add_argument('--independence', choices=('unconfirmed', 'user-confirmed'), default='unconfirmed')
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
