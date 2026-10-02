"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re
import unicodedata

from harness.middleware import Middleware

#: Scorer bỏ qua trích dẫn ngắn hơn ngưỡng này (`arena.scorer.MIN_SUPPORT_CHARS`).
MIN_QUOTE_CHARS = 12

_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Cùng dạng chuẩn hoá scorer dùng để so: NFC, casefold, gộp khoảng trắng.

    Chỉ dùng để SO SÁNH — không bao giờ ghi kết quả này vào claim.
    """
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def quotes_a_line(body: str, text: str) -> bool:
    """`text` có phải trích nguyên văn MỘT DÒNG của `body` không."""
    if not isinstance(body, str) or not isinstance(text, str):
        return False
    needle = _norm(text)
    if len(needle) < MIN_QUOTE_CHARS:
        return False
    return any(needle in _norm(line) for line in body.splitlines())


def observed_sources(ctx, text: str) -> list:
    """Các tài liệu đã về NGUYÊN VẸN trong quan sát và có một dòng chứa `text`."""
    if ctx.corpus is None:
        return []
    observed = ctx.observed_text
    return [
        doc
        for doc in ctx.corpus.docs
        if doc.body and doc.body in observed and quotes_a_line(doc.body, text)
    ]


def cited_ids(claims: list) -> list:
    return sorted(
        {c["doc_id"] for c in claims if isinstance(c, dict) and isinstance(c.get("doc_id"), str)}
    )


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        fixed = []
        moved = 0
        for claim in claims:
            text = claim.get("text") if isinstance(claim, dict) else None
            if not isinstance(text, str):
                fixed.append(claim)
                continue
            doc_id = claim.get("doc_id")
            doc = ctx.corpus.get(doc_id) if isinstance(doc_id, str) else None
            if doc is not None and quotes_a_line(doc.body, text):
                fixed.append(claim)
                continue
            sources = observed_sources(ctx, text)
            if not sources:
                # Không có trong bằng chứng nào: đó là bịa, phần của `critic`.
                fixed.append(claim)
                continue
            # Chỉ đổi doc_id; text giữ nguyên từng ký tự.
            fixed.append({**claim, "doc_id": sources[0].doc_id})
            moved += 1

        ctx.state["citation_checker"] = {"reattributed": moved}
        report["claims"] = fixed
        report["citations"] = cited_ids(fixed)
        return report
