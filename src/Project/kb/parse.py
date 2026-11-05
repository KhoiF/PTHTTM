"""Tách trang Long Châu đã lưu thành các mục (tiêu đề + nội dung) và chia chunk.

Duyệt nội dung theo thứ tự tài liệu: mỗi thẻ h2/h3/h4 mở một mục mới, văn bản của p/li/td phía sau
thuộc về mục đó. Trước khi duyệt, các khối không phải nội dung bài bị gỡ khỏi cây HTML: thẻ sản phẩm gợi ý,
bài viết/bệnh liên quan, hỏi đáp, đánh giá, mục lục, khung tác giả, câu hỏi chung lặp lại ở mọi trang thuốc.
"""

from __future__ import annotations

import hashlib
import re

from bs4 import BeautifulSoup, NavigableString, Tag

from kb.schema import MAIN_DISEASE_BY_TOPIC
from kb.seed import Seed
from kb.textutil import display_name, plain, squash

DROP_TAGS = ["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav", "aside", "form", "button"]
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
TEXT_TAGS = {"p", "li", "td", "th", "dd", "blockquote", "figcaption"}
SKIP_SECTION = re.compile(r"bai viet lien quan|san pham lien quan|xem them|cau hoi thuong gap|hoi dap|binh luan|danh gia"
                          r"|tim hieu them|chuyen muc|tin tuc|nguon tham khao|tai lieu tham khao|dang ky|lien he"
                          r"|he thong nha thuoc|ve chung toi|tai ung dung|mien tru|thong tin san pham|xem tat ca")
# Tiêu đề mở đầu một khối không phải nội dung bài: gỡ cả khối chứa nó (xem _drop_boxes)
JUNK_BOX = re.compile(r"^(co the ban quan tam|(cac )?bai viet lien quan|(cac )?benh lien quan|san pham lien quan"
                      r"|hoi dap|danh gia san pham|infographic)")
# Câu hỏi thường gặp chung, giống hệt nhau ở mọi trang /thuoc/ (không nói về thuốc cụ thể nào)
GENERIC_FAQ = re.compile(r"^(duoc luc hoc la gi|duoc dong hoc la gi|tac dung phu cua thuoc la gi|su dung thuoc dung cach"
                         r"|cac dang bao che cua thuoc)")
# Khung tác giả/người duyệt, thanh mục lục bên cạnh bài /benh/, tab điều hướng của trang /thuoc/
DROP_SELECTORS = ["div.approver", "ul.lc-detail-tabs", "div.lc-comment", "div.lc-preview"]
# Dòng chữ của giao diện (ngày đăng, nút cỡ chữ, trình đọc bài...) lọt vào văn bản
NOISE_LINE = re.compile(r"^(\d{1,2}/\d{1,2}/\d{4}|mac dinh|lon hon|nghe bai viet|thoi luong [\d ]+|xem tat ca thong tin"
                        r"|so dang ky|quy cach|thanh phan|xem them|chia se:?)$")
MIN_SECTION_CHARS = 60
CHUNK_WORDS = 320


def _own_text(tag: Tag) -> str:
    """Văn bản của thẻ, bỏ phần nằm trong các thẻ khối con sẽ được duyệt riêng."""
    parts = []
    for child in tag.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name not in TEXT_TAGS | HEADING_TAGS | {"ul", "ol", "table"}:
            parts.append(child.get_text(" "))
    return squash(" ".join(parts))


def _is_card_heading(heading: Tag) -> bool:
    """Tiêu đề nằm trong link là tiêu đề của thẻ sản phẩm/bài viết gợi ý, không phải mục của bài."""
    return heading.find_parent("a") is not None


def _content_headings(box: Tag, exclude: Tag) -> list[Tag]:
    return [h for h in box.find_all(list(HEADING_TAGS)) if h is not exclude and not _is_card_heading(h)
            and not JUNK_BOX.search(plain(h.get_text(" ")))]


def _drop_boxes(body: Tag) -> None:
    """Gỡ các khối gợi ý/liên quan/hỏi đáp: từ tiêu đề rác, leo lên tổ tiên lớn nhất chưa chứa mục nội dung nào."""
    for selector in DROP_SELECTORS:
        for tag in body.select(selector):
            tag.decompose()
    # Thanh mục lục của bài /benh/ (cột trái, ẩn trên điện thoại): danh sách tên mục, không có tiêu đề
    for tag in body.select("div.hidden.h-full"):
        if not tag.find(list(HEADING_TAGS)):
            tag.decompose()
    for heading in body.find_all(list(HEADING_TAGS)):
        if heading.parent is None or not JUNK_BOX.search(plain(heading.get_text(" "))):
            continue  # parent None: đã bị gỡ cùng một khối trước đó
        box = heading
        while box.parent is not None and box.parent.name not in {"body", "[document]"} \
                and not _content_headings(box.parent, heading):
            box = box.parent
        box.decompose()
    for heading in body.find_all(list(HEADING_TAGS)):
        if heading.parent is not None and _is_card_heading(heading):
            heading.find_parent("a").decompose()
    for heading in body.find_all(list(HEADING_TAGS)):
        if heading.parent is not None and GENERIC_FAQ.search(plain(heading.get_text(" "))):
            item = heading.find_parent("li") or heading
            item.decompose()


def main_entities(seed: Seed, title: str) -> list[str]:
    if seed.type == "benh":
        return [MAIN_DISEASE_BY_TOPIC.get(seed.topic, display_name(title.split(":")[0].split(" là gì")[0]))]
    if seed.type in {"thuoc", "thanh-phan"}:
        return [display_name(part) for part in seed.topic.split("+")]
    return []


def parse_page(html: str, seed: Seed) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1")
    title = squash(title_tag.get_text(" ")) if title_tag else squash(soup.title.get_text(" ")) if soup.title else seed.topic
    for tag in soup(DROP_TAGS):
        tag.decompose()
    body = soup.body or soup
    _drop_boxes(body)

    sections: list[dict] = []
    current = {"heading": "Giới thiệu", "level": 1, "texts": []}
    started = title_tag is None
    for element in body.descendants:
        if not isinstance(element, Tag):
            continue
        if element is title_tag:
            started = True
            continue
        if not started:
            continue
        if element.name in HEADING_TAGS:
            sections.append(current)
            current = {"heading": squash(element.get_text(" ")), "level": int(element.name[1]), "texts": []}
        elif element.name in TEXT_TAGS:
            text = _own_text(element)
            if len(text) > 2 and not NOISE_LINE.match(plain(text)):
                current["texts"].append(("• " if element.name == "li" else "") + text)
    sections.append(current)

    cleaned, seen = [], set()
    for section in sections:
        text = "\n".join(dict.fromkeys(section["texts"]))  # bỏ đoạn lặp, giữ thứ tự
        if len(text) < MIN_SECTION_CHARS or SKIP_SECTION.search(plain(section["heading"])):
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append({"heading": section["heading"], "level": section["level"], "text": text})

    return {"url": seed.url, "type": seed.type, "topic": seed.topic, "title": title,
            "main_entities": main_entities(seed, title), "sections": cleaned}


def chunk_page(page: dict, max_words: int = CHUNK_WORDS) -> list[dict]:
    """Mỗi mục thành một hoặc nhiều chunk (cắt theo đoạn khi mục quá dài)."""
    chunks = []
    for s_index, section in enumerate(page["sections"]):
        buffer: list[str] = []
        parts: list[str] = []
        for paragraph in section["text"].split("\n"):
            if buffer and sum(len(p.split()) for p in buffer) + len(paragraph.split()) > max_words:
                parts.append("\n".join(buffer))
                buffer = []
            buffer.append(paragraph)
        if buffer:
            parts.append("\n".join(buffer))
        for c_index, text in enumerate(parts):
            chunk_id = hashlib.sha1(f"{page['url']}|{s_index}|{c_index}|{text}".encode()).hexdigest()[:16]
            chunks.append({"id": chunk_id, "url": page["url"], "page_type": page["type"], "title": page["title"],
                           "main_entities": page["main_entities"], "heading": section["heading"], "text": text})
    return chunks
