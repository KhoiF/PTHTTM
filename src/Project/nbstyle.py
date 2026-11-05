"""Hiển thị bảng và mã nguồn trong notebook với màu chữ/nền cố định.

Bảng HTML mặc định của pandas để màu chữ theo giao diện của trình soạn thảo, nên trên VS Code giao diện tối dễ ra
chữ trắng trên nền sáng. `install()` đăng ký bộ định dạng HTML cho DataFrame/Series/Styler: mọi bảng có nền trắng,
chữ tối, tiêu đề đậm, số căn phải với số chữ số thập phân vừa đủ, chữ căn trái và tự xuống dòng.
"""
from __future__ import annotations

import html
import inspect

import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import HTML
from pandas.io.formats.style import Styler

MAX_ROWS = 60
MAX_DECIMALS = 4
TEXT, MUTED, BORDER = "#1f2328", "#59636e", "#d1d9e0"
HEAD_BG, ZEBRA = "#24292f", "#f6f8fa"
BEST = "background-color: #dafbe1 !important; color: #116329 !important; font-weight: 700;"

# Mọi màu được ghi rõ (kể cả màu chữ của ô) để giao diện sáng/tối của trình soạn thảo không chen vào được.
TABLE_STYLES = [
    {"selector": "", "props": f"border-collapse: collapse; background-color: #ffffff; color: {TEXT}; "
                              "font-size: 13px; line-height: 1.45; margin: 4px 0 8px;"},
    {"selector": "caption", "props": f"caption-side: top; text-align: left; color: {MUTED}; background-color: #ffffff; "
                                     "padding: 8px 12px; font-size: 12px;"},
    {"selector": "thead th", "props": f"background-color: {HEAD_BG} !important; color: #ffffff !important; "
                                      "font-weight: 600; padding: 7px 12px; text-align: center; vertical-align: bottom;"},
    {"selector": "tbody tr", "props": "background-color: #ffffff;"},
    {"selector": "tbody tr:nth-child(even)", "props": f"background-color: {ZEBRA};"},
    {"selector": "tbody th", "props": f"color: {TEXT}; font-weight: 600; text-align: left; padding: 6px 12px; "
                                      f"vertical-align: top; border-bottom: 1px solid {BORDER} !important;"},
    {"selector": "tbody td", "props": f"color: {TEXT}; padding: 6px 12px; vertical-align: top; "
                                      f"border-bottom: 1px solid {BORDER} !important;"},
    {"selector": "a", "props": "color: #0969da !important;"},
]
NUMBER_CELL = "text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap;"
TEXT_CELL = "text-align: left; max-width: 36em; white-space: normal; overflow-wrap: anywhere;"
BOOL_CELL = "text-align: center;"


def _decimals(column: pd.Series) -> int:
    """Số chữ số thập phân nhỏ nhất (≤ MAX_DECIMALS) đủ biểu diễn mọi giá trị của cột."""
    values = column.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    for decimals in range(MAX_DECIMALS):
        if np.allclose(values, np.round(values, decimals), rtol=0, atol=10 ** -(MAX_DECIMALS + 2)):
            return decimals
    return MAX_DECIMALS


def _number_formatter(decimals: int):
    def fmt(value) -> str:
        if pd.isna(value):
            return "—"
        text = f"{value:,.{decimals}f}" if abs(value) >= 10_000 else f"{value:.{decimals}f}"
        return text.replace(",", " ")       # 96 128: khoảng trắng hẹp, không lẫn với dấu thập phân
    return fmt


def _text(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "✓" if value else "✗"
    if isinstance(value, (int, np.integer)):
        return _number_formatter(0)(value)
    if isinstance(value, (list, tuple, set)):
        return html.escape(", ".join(map(str, value)))
    if isinstance(value, str) and value.startswith(("http://", "https://")):
        label = value.rstrip("/").rsplit("/", 1)[-1].removesuffix(".html")
        return f'<a href="{html.escape(value)}" target="_blank">{html.escape(label)}</a>'
    if isinstance(value, (float, np.floating)):        # cột trộn kiểu: 62.0 → "62", 34.5 → "34.5"
        return _number_formatter(MAX_DECIMALS)(value).rstrip("0").rstrip(".")
    return html.escape(str(value))


def _column_kinds(data: pd.DataFrame) -> list[str]:
    kinds = []
    for _, column in data.items():
        if pd.api.types.is_bool_dtype(column):
            kinds.append("bool")
        elif pd.api.types.is_numeric_dtype(column):
            kinds.append("number")
        else:
            kinds.append("text")
    return kinds


def theme(styler: Styler) -> Styler:
    """Gắn màu và căn lề cho một Styler (không đổi định dạng số đã có)."""
    styles = list(TABLE_STYLES)
    for position, kind in enumerate(_column_kinds(styler.data)):
        props = {"number": NUMBER_CELL, "bool": BOOL_CELL, "text": TEXT_CELL}[kind]
        styles.append({"selector": f"td.col{position}", "props": props})
    styler.set_table_styles(styles, overwrite=False)
    if isinstance(styler.data.index, pd.RangeIndex):
        styler.hide(axis="index")
    styler._nbstyle = True
    return styler


def table(data: pd.DataFrame | pd.Series, caption: str | None = None) -> Styler:
    """DataFrame → Styler đã định dạng. Bảng dài chỉ hiện MAX_ROWS dòng đầu."""
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    total = len(frame)
    if total > MAX_ROWS:
        frame = frame.head(MAX_ROWS)
        caption = (caption + " · " if caption else "") + f"hiện {MAX_ROWS}/{total} dòng đầu"
    formatters = {}
    for (name, column), kind in zip(frame.items(), _column_kinds(frame)):
        if kind == "number" and not pd.api.types.is_integer_dtype(column):
            formatters[name] = _number_formatter(_decimals(column))
        elif kind == "number":
            formatters[name] = _number_formatter(0)
        else:
            formatters[name] = _text
    styler = frame.style.format(formatters).format_index(_text, axis=0).format_index(_text, axis=1)
    if caption:
        styler.set_caption(html.escape(caption))
    return theme(styler)


def highlight_best(styler: Styler, columns: list[str], by: str | None = None) -> Styler:
    """Tô xanh giá trị lớn nhất của từng cột trong `columns`, tính riêng trong từng nhóm `by` (cột hoặc mức index)."""
    data = styler.data

    def mark(column: pd.Series):
        if by is None:
            best = column.max()
        elif by in (data.index.names or []):
            best = data.groupby(level=by)[column.name].transform("max")
        else:
            best = data.groupby(by)[column.name].transform("max")
        return np.where(column == best, BEST, "")

    return styler.apply(mark, subset=columns)


def source(obj) -> HTML:
    """Mã nguồn của một hàm/lớp, tô màu cú pháp trên nền sáng cố định."""
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    code = highlight(inspect.getsource(obj), PythonLexer(), HtmlFormatter(style="friendly", noclasses=True, nobackground=True))
    return HTML(f'<div style="background-color: {ZEBRA}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; '
                f'padding: 4px 14px; font-size: 12.5px; overflow-x: auto;">{code}</div>')


def _render(obj) -> str | None:
    try:
        if isinstance(obj, Styler):
            return (obj if getattr(obj, "_nbstyle", False) else theme(obj)).to_html()
        return table(obj).to_html()
    except Exception:                       # bảng lạ (cột trùng tên, index không duy nhất…) → để pandas tự vẽ
        return obj._repr_html_() if hasattr(obj, "_repr_html_") else None


def install() -> None:
    """Đăng ký bộ định dạng HTML cho phiên IPython hiện tại."""
    shell = get_ipython()
    if shell is None:
        return
    formatter = shell.display_formatter.formatters["text/html"]
    for kind in (pd.DataFrame, pd.Series, Styler):
        formatter.for_type(kind, _render)
