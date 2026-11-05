"""Mở các trang Long Châu trong Chrome theo từng lượt để bạn tự lưu (điểm chờ A).

Long Châu chặn request từ script (Cloudflare 403), nên trang được mở bằng trình duyệt
thường và lưu thủ công. Sau mỗi lượt, lưu tất cả tab rồi nhấn Enter để mở lượt tiếp.

Chạy từ thư mục src/Project:
    python -m kb.open_urls               # mở toàn bộ, mỗi lượt 10 trang
    python -m kb.open_urls --missing     # chỉ mở các trang chưa có hoặc bị thu gọn trong kb/raw_html
"""

from __future__ import annotations

import argparse
import subprocess

from kb.import_html import is_collapsed
from kb.seed import load_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--missing", action="store_true",
                        help="chỉ mở trang chưa nhập vào kb/raw_html hoặc đã nhập nhưng bị thu gọn")
    parser.add_argument("--browser", default="Google Chrome")
    args = parser.parse_args()

    seeds = load_seeds()
    if args.missing:
        seeds = [s for s in seeds if not s.html_path.exists()
                 or is_collapsed(s.html_path.read_text(encoding="utf-8", errors="ignore"))]
    if not seeds:
        print("Không còn trang nào cần mở.")
        return

    batches = [seeds[i:i + args.batch_size] for i in range(0, len(seeds), args.batch_size)]
    print(f"Sẽ mở {len(seeds)} trang trong {len(batches)} lượt.")
    print("Mỗi lượt: chờ trang tải xong → ở trang /benh/, kéo xuống bấm 'Xem thêm' để mở hết bài → "
          "SingleFile ▸ 'Save all tabs' (hoặc Cmd+S từng tab, chọn 'Webpage, Complete') vào ~/Downloads → "
          "đóng các tab đó → quay lại đây nhấn Enter.\n")
    for index, batch in enumerate(batches, start=1):
        print(f"Lượt {index}/{len(batches)}:")
        for seed in batch:
            print(f"  - [{seed.type}] {seed.topic}: {seed.url}")
        subprocess.run(["open", "-a", args.browser, *[s.url for s in batch]], check=True)
        if index < len(batches):
            input("Lưu xong lượt này thì nhấn Enter để mở lượt tiếp... ")
    print("\nĐã mở hết. Chạy tiếp: python -m kb.import_html")


if __name__ == "__main__":
    main()
