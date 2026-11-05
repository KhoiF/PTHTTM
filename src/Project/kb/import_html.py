"""Nhập các trang Long Châu đã lưu từ trình duyệt vào kb/raw_html.

Nhận diện URL gốc của từng file qua chú thích của SingleFile / Chrome, thẻ canonical
hoặc og:url, rồi khớp với kb/seed_urls.csv. Trang chặn của Cloudflare bị loại. Bài /benh/ lưu khi chưa bấm
"Xem thêm" chỉ có nửa đầu bài: vẫn được nhập nhưng bị báo [thu gọn], và bản đầy đủ lưu sau sẽ thay thế nó.

Chạy từ thư mục src/Project:
    python -m kb.import_html                       # quét ~/Downloads
    python -m kb.import_html --src ~/Desktop/lc    # quét thư mục khác
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from config import RAW_HTML_DIR
from kb.seed import load_seeds, normalize_url
from kb.textutil import plain

URL_PATTERNS = [
    re.compile(r"Page saved with SingleFile\s*\n?\s*url:\s*(\S+)", re.I),
    re.compile(r"<!--\s*saved from url=\(\d+\)(\S+?)\s*-->", re.I),
    re.compile(r"<link[^>]+rel=[\"']canonical[\"'][^>]+href=[\"']([^\"']+)", re.I),
    re.compile(r"<link[^>]+href=[\"']([^\"']+)[\"'][^>]+rel=[\"']canonical", re.I),
    re.compile(r"<meta[^>]+property=[\"']og:url[\"'][^>]+content=[\"']([^\"']+)", re.I),
]
BLOCK_MARKERS = ("Attention Required! | Cloudflare", "Just a moment...", "Sorry, you have been blocked")


def detect_url(html: str) -> str | None:
    head = html[:400_000]
    for pattern in URL_PATTERNS:
        match = pattern.search(head)
        if match and "nhathuoclongchau.com.vn" in match.group(1):
            return normalize_url(match.group(1))
    return None


def is_blocked(html: str) -> bool:
    return any(marker in html[:20_000] for marker in BLOCK_MARKERS)


def is_collapsed(html: str) -> bool:
    """Bài còn nút "Xem thêm" trong khung nội dung: phần sau nút chỉ được tải khi bấm, nên file lưu bị thiếu."""
    soup = BeautifulSoup(html, "lxml")
    return any("xem them" in plain(button.get_text(" ")) and button.find_parent("div", class_="overflow-hidden")
               for button in soup.find_all("button"))


def _keep_existing(target: Path, new_path: Path, new_collapsed: bool) -> bool:
    """Giữ file đã nhập nếu nó không kém file mới: bản đầy đủ thắng bản thu gọn, cùng loại thì file lớn hơn thắng."""
    old_collapsed = is_collapsed(target.read_text(encoding="utf-8", errors="ignore"))
    if old_collapsed != new_collapsed:
        return new_collapsed
    return target.stat().st_size >= new_path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = {normalize_url(s.url): s for s in load_seeds()}
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in args.src.expanduser().rglob("*") if p.suffix.lower() in {".html", ".htm"})

    imported, blocked, unmatched = [], [], []
    for path in files:
        html = path.read_text(encoding="utf-8", errors="ignore")
        url = detect_url(html)
        if url is None or url not in seeds:
            if url is not None or "nhathuoclongchau" in html[:400_000]:
                unmatched.append((path.name, url))
            continue
        if is_blocked(html):
            blocked.append(path.name)
            continue
        target = seeds[url].html_path
        if target.exists() and _keep_existing(target, path, is_collapsed(html)):
            imported.append((url, target, "đã có"))
            continue
        if not args.dry_run:
            shutil.copyfile(path, target)
        imported.append((url, target, f"từ {path.name}"))

    have = {normalize_url(s.url) for s in seeds.values() if s.html_path.exists()}
    missing = [u for u in seeds if u not in have]
    collapsed = [u for u in sorted(have)
                 if is_collapsed(seeds[u].html_path.read_text(encoding="utf-8", errors="ignore"))]

    if not args.dry_run:
        with open(RAW_HTML_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "type", "topic", "file", "size", "imported_at"])
            for url in sorted(have):
                seed = seeds[url]
                writer.writerow([url, seed.type, seed.topic, seed.html_path.name,
                                 seed.html_path.stat().st_size, datetime.now().isoformat(timespec="seconds")])

    print(f"Quét {len(files)} file HTML trong {args.src}")
    print(f"Đã nhập / đã có: {len(imported)}   Trang chặn Cloudflare: {len(blocked)}   Không khớp seed: {len(unmatched)}")
    for name in blocked:
        print(f"  [chặn] {name}")
    for name, url in unmatched:
        print(f"  [lạ]   {name} -> {url}")
    print(f"Còn thiếu {len(missing)}/{len(seeds)} trang" + (":" if missing else "."))
    for url in missing:
        print(f"  - {url}")
    if collapsed:
        print(f"Bị thu gọn (lưu khi chưa bấm \"Xem thêm\"): {len(collapsed)} trang")
        for url in collapsed:
            print(f"  [thu gọn] {url}")
    if missing or collapsed:
        print("Mở lại các trang thiếu / thu gọn: python -m kb.open_urls --missing")


if __name__ == "__main__":
    main()
