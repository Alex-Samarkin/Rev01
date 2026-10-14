"""Конвертер Markdown -> DOCX для материалов статьи.

Поддерживаются: заголовки (#, ##, ###), абзацы, **жирный**, *курсив*,
<sub>/<sup>, markdown-таблицы, изображения ![alt](path), маркированные и
нумерованные списки, горизонтальные линейки.

Запуск:
    python code/python/md_to_docx.py article/article.md article/article.docx
    python code/python/md_to_docx.py article/references.md article/references.docx --no-tables
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INLINE_RE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|<sub>.+?</sub>|<sup>.+?</sup>|`[^`]+?`)")


def add_runs(par, text: str) -> None:
    """Разбор встроенной разметки в один абзац."""
    for part in INLINE_RE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9)
        elif part.startswith("<sub>"):
            r = par.add_run(part[5:-6])
            r.font.subscript = True
        elif part.startswith("<sup>"):
            r = par.add_run(part[5:-6])
            r.font.superscript = True
        else:
            par.add_run(part)


def is_table_row(line: str) -> bool:
    return line.strip().startswith("|") and line.strip().endswith("|")


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def add_table(doc: Document, rows: list[list[str]], total_width: float = 7.4) -> None:
    n_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=0, cols=n_cols)
    table.style = "Table Grid"
    table.autofit = False

    # ширины колонок пропорционально максимальной длине содержимого
    maxlen = []
    for j in range(n_cols):
        vals = [len(r[j]) if j < len(r) else 0 for r in rows]
        maxlen.append(max(2, min(max(vals), 40)))
    total = sum(maxlen)
    widths = [max(0.32, total_width * m / total) for m in maxlen]
    scale = total_width / sum(widths)
    widths = [w * scale for w in widths]

    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j in range(n_cols):
            txt = row[j] if j < len(row) else ""
            cells[j].width = Inches(widths[j])
            par = cells[j].paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.line_spacing = 1.0
            add_runs(par, txt)
            for r in par.runs:
                r.font.size = Pt(7.5)
                if i == 0:
                    r.bold = True

    # фиксированная раскладка таблицы
    tblPr = table._tbl.tblPr
    layout = tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
    tblPr.append(layout)
    doc.add_paragraph()


def convert(md_path: Path, docx_path: Path, keep_tables: bool = True,
            title: str | None = None) -> None:
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    doc = Document()
    # поля 2 см — стандарт для рукописей, подаваемых в российские журналы
    for section in doc.sections:
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(2.0)
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.line_spacing = 1.0
    for hname, before, after in (("Heading 1", 10, 4), ("Heading 2", 8, 3),
                                 ("Heading 3", 6, 2), ("Title", 0, 10)):
        if hname in [s.name for s in doc.styles]:
            hs = doc.styles[hname]
            hs.paragraph_format.space_before = Pt(before)
            hs.paragraph_format.space_after = Pt(after)
            hs.paragraph_format.line_spacing = 1.0

    i = 0
    in_code = False
    buf: list[str] = []
    last_list_par = None

    def flush() -> None:
        """Склеивает накопленные строки одного абзаца в один параграф Word."""
        if not buf:
            return
        text = " ".join(s.strip() for s in buf)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_runs(p, text)
        buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            in_code = not in_code
            i += 1
            continue
        if in_code:
            p = doc.add_paragraph()
            r = p.add_run(line)
            r.font.name = "Consolas"
            r.font.size = Pt(8.5)
            i += 1
            continue

        if not stripped:
            flush()
            i += 1
            continue

        if stripped.startswith("!["):
            flush()
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                alt, src = m.group(1), m.group(2)
                img = (md_path.parent / src).resolve()
                if img.exists():
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    p.add_run().add_picture(str(img), width=Inches(4.2))
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = cap.add_run(alt)
                    r.italic = True
                    r.font.size = Pt(9)
                else:
                    doc.add_paragraph(f"[изображение не найдено: {src}]")
            i += 1
            continue

        # таблица
        if is_table_row(stripped) and i + 1 < len(lines) and is_table_row(lines[i + 1].strip()):
            flush()
            if keep_tables:
                rows = [split_row(stripped)]
                i += 2  # пропускаем разделитель
                while i < len(lines) and is_table_row(lines[i].strip()):
                    rows.append(split_row(lines[i].strip()))
                    i += 1
                add_table(doc, rows)
            else:
                while i < len(lines) and is_table_row(lines[i].strip()):
                    i += 1
            continue

        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            content = stripped.lstrip("#").strip()
            if level == 1 and title is None:
                doc.add_heading(content, level=0)
            else:
                doc.add_heading(content, level=min(level, 4))
            i += 1
            continue

        if stripped.startswith("---") and set(stripped) <= {"-", " "}:
            flush()
            i += 1
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            flush()
            indent, marker, content = m.groups()
            style_name = "List Bullet" if marker in ("-", "*") else "List Number"
            if len(indent) >= 2:
                style_name += " 2"
            p = doc.add_paragraph(style=style_name)
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, content)
            last_list_par = p
            i += 1
            continue

        # продолжение предыдущего пункта списка (перенос строки внутри пункта);
        # проверяется раньше формулы, так как продолжение тоже имеет отступ
        if last_list_par is not None and not is_table_row(stripped) \
                and not stripped.startswith("#") and not stripped.startswith("!["):
            last_list_par.add_run(" ")
            add_runs(last_list_par, stripped)
            i += 1
            continue

        # выключная формула: строка с отступом вне списка
        if line.startswith("  ") and not is_table_row(stripped):
            flush()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, stripped)
            i += 1
            continue

        last_list_par = None
        buf.append(line)
        i += 1

    flush()
    doc.save(docx_path)
    print(f"Записано: {docx_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--no-tables", action="store_true")
    args = ap.parse_args()
    convert(Path(args.src), Path(args.dst), keep_tables=not args.no_tables)


if __name__ == "__main__":
    main()
