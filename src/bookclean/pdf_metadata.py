"""M4.3 PDF 文档元数据：DocumentInfo 与 XMP 双写并保持一致。

字段：Title/Author/Subject/Creator/Producer/CreationDate/ModDate/Keywords。
用户指定优先；不推断作者、不虚构出版信息、不把处理日期伪装成出版日期。
写入后必须实际读回验证（verify_metadata）。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pikepdf

from . import __version__
from .config import PDFExportConfig

CREATOR = "BookClean PDF"
DEFAULT_SUBJECT = "Image-based PDF optimized by BookClean PDF (no OCR text layer)"


def _pdf_date(dt: datetime) -> str:
    base = dt.strftime("D:%Y%m%d%H%M%S")
    off = dt.utcoffset()
    if off is None:
        return base + "Z"
    total = int(off.total_seconds() // 60)
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{base}{sign}{total // 60:02d}'{total % 60:02d}'"


def resolve_fields(config: PDFExportConfig, output_name: str, now: datetime | None = None) -> dict:
    now = now or datetime.now().astimezone()
    return {
        "title": config.title or Path(output_name).stem,
        "author": config.author or None,
        "subject": config.subject or DEFAULT_SUBJECT,
        "creator": CREATOR,
        "producer": f"{CREATOR} {__version__}",
        "keywords": config.keywords or None,
        "creation_date": now,
        "mod_date": now,
    }


def apply_metadata(pdf: pikepdf.Pdf, config: PDFExportConfig, output_name: str,
                   now: datetime | None = None) -> dict:
    f = resolve_fields(config, output_name, now)
    iso = f["creation_date"].isoformat()
    pdfdate = _pdf_date(f["creation_date"])

    di = pdf.docinfo
    di.Title = f["title"]
    di.Subject = f["subject"]
    di.Creator = f["creator"]
    di.Producer = f["producer"]
    di.CreationDate = pikepdf.String(pdfdate)
    di.ModDate = pikepdf.String(pdfdate)
    if f["author"]:
        di.Author = f["author"]
    if f["keywords"]:
        di.Keywords = f["keywords"]

    with pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False) as meta:
        meta["dc:title"] = f["title"]
        meta["dc:description"] = f["subject"]
        meta["pdf:Producer"] = f["producer"]
        meta["xmp:CreatorTool"] = f["creator"]
        meta["xmp:CreateDate"] = iso
        meta["xmp:ModifyDate"] = iso
        if f["author"]:
            meta["dc:creator"] = [f["author"]]
        if f["keywords"]:
            meta["pdf:Keywords"] = f["keywords"]
    return f


def read_metadata(pdf_path: str | Path) -> dict:
    with pikepdf.open(pdf_path) as pdf:
        di = pdf.docinfo
        out = {
            "docinfo": {
                "title": str(di.get("/Title", "")) or None,
                "author": str(di.get("/Author", "")) or None,
                "subject": str(di.get("/Subject", "")) or None,
                "creator": str(di.get("/Creator", "")) or None,
                "producer": str(di.get("/Producer", "")) or None,
                "keywords": str(di.get("/Keywords", "")) or None,
                "creation_date": str(di.get("/CreationDate", "")) or None,
                "mod_date": str(di.get("/ModDate", "")) or None,
            },
            "xmp": {},
        }
        try:
            with pdf.open_metadata() as meta:
                out["xmp"] = {
                    "title": meta.get("dc:title"),
                    "author": (meta.get("dc:creator") or [None])[0],
                    "subject": meta.get("dc:description"),
                    "creator": meta.get("xmp:CreatorTool"),
                    "producer": meta.get("pdf:Producer"),
                    "keywords": meta.get("pdf:Keywords"),
                    "creation_date": meta.get("xmp:CreateDate"),
                    "mod_date": meta.get("xmp:ModifyDate"),
                }
        except Exception:
            out["xmp"] = {}
    return out


def verify_metadata(pdf_path: str | Path, expected: dict) -> list[str]:
    """读回并核对 DocumentInfo 与 XMP 一致性。返回问题列表（空=通过）。"""
    got = read_metadata(pdf_path)
    di, xmp = got["docinfo"], got["xmp"]
    issues: list[str] = []
    pairs = [("title", "title"), ("author", "author"), ("subject", "subject"),
             ("creator", "creator"), ("producer", "producer"), ("keywords", "keywords")]
    for key, xkey in pairs:
        want = expected.get(key)
        if want is None:
            continue
        if di.get(key) != want:
            issues.append(f"DocumentInfo.{key}={di.get(key)!r} 期望 {want!r}")
        if xmp.get(xkey) != want:
            issues.append(f"XMP.{xkey}={xmp.get(xkey)!r} 期望 {want!r}")
    for key in ("creation_date", "mod_date"):
        if not di.get(key):
            issues.append(f"DocumentInfo.{key} 缺失")
        if not xmp.get(key):
            issues.append(f"XMP.{key} 缺失")
    return issues