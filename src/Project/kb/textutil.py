"""Tiện ích xử lý văn bản tiếng Việt: khóa so khớp không dấu, chuẩn hóa đồng nghĩa, kiểm tra câu trích."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Đồng nghĩa áp dụng trên khóa không dấu (thứ tự quan trọng)
SYNONYMS = [
    (r"\b(benh )?tieu duong\b", "dai thao duong"),
    (r"\bdtd\b", "dai thao duong"),
    (r"\b(type|typ|tip|tuyp|loai)\s*(2|ii)\b", "tuyp 2"),
    (r"\b(type|typ|tip|tuyp|loai)\s*(1|i)\b", "tuyp 1"),
    (r"\bcao huyet ap\b", "tang huyet ap"),
    (r"\bhuyet ap cao\b", "tang huyet ap"),
    (r"\bdai thao duong khong phu thuoc insulin\b", "dai thao duong tuyp 2"),
    (r"\bdai thao duong phu thuoc insulin\b", "dai thao duong tuyp 1"),
    (r"\bhba1c\b|\bhb a1c\b|\bhemoglobin a1c\b", "hba1c"),
    (r"\bglucose (mau|huyet tuong)\b", "duong huyet"),
    (r"\bmo mau cao\b|\broi loan mo mau\b|\broi loan lipid mau\b", "roi loan lipid mau"),
    (r"\bhut thuoc( la)?\b", "hut thuoc la"),
]
PREFIXES = r"^(benh ly |benh |chung |hoi chung |tinh trang |xet nghiem |chi so |thuoc |hoat chat |nhom thuoc )"


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def plain(text: str) -> str:
    """Chữ thường, bỏ dấu, chỉ giữ chữ-số-khoảng trắng."""
    text = strip_accents(text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.,/ ]+", " ", text)).strip()


def name_key(name: str) -> str:
    key = re.sub(r"[.,/]", " ", plain(name))
    for pattern, replacement in SYNONYMS:
        key = re.sub(pattern, replacement, key)
    key = re.sub(PREFIXES, "", key)
    return re.sub(r"\s+", " ", key).strip()


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def evidence_ratio(evidence: str, source: str) -> float:
    """Tỉ lệ độ dài khối khớp liên tục dài nhất của câu trích trong văn bản gốc (1.0 = khớp nguyên văn)."""
    e, s = plain(evidence), plain(source)
    if not e:
        return 0.0
    if e in s:
        return 1.0
    match = SequenceMatcher(None, e, s, autojunk=False).find_longest_match(0, len(e), 0, len(s))
    return match.size / len(e)


def name_supported(name: str, evidence: str) -> bool:
    """Phần lớn từ trong tên thực thể phải xuất hiện trong câu trích (chống tên bịa)."""
    tokens = [t for t in plain(name).split() if len(t) > 1]
    if not tokens:
        return False
    text = plain(evidence)
    return sum(t in text for t in tokens) / len(tokens) >= 0.6


def display_name(name: str) -> str:
    name = squash(name).strip(" .,:;-")
    return name[:1].upper() + name[1:] if name else name


STRENGTH = re.compile(r"\s*\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|iu|ui)(?:\s*/\s*\d+(?:[.,]\d+)?\s*(?:mg|ml))?\b", re.I)
PACKAGING = re.compile(r"\s*\([^()]*\b(?:vỉ|viên|hộp|lọ|gói|ống|chai)\b[^()]*\)", re.I)


def source_title(title: str) -> str:
    """Tên trang nguồn để hiển thị: bỏ hàm lượng và quy cách đóng gói của trang thuốc
    ("Thuốc Metformin Stella 1000mg … (6 vỉ x 10 viên)" → "Thuốc Metformin Stella …"), để câu trả lời không có số liều."""
    return squash(PACKAGING.sub("", STRENGTH.sub("", title or "")))


def sentence_containing(phrase: str, text: str) -> str | None:
    """Câu (hoặc dòng) đầu tiên trong văn bản chứa nguyên cụm từ, so khớp không dấu."""
    target = plain(phrase)
    if not target:
        return None
    for sentence in re.split(r"(?<=[.!?;])\s+|\n", text):
        if target in plain(sentence):
            return squash(sentence.lstrip("• "))
    return None
