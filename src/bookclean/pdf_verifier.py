"""M4.4 PDF 验证：重读最终 PDF 逐页检查，而非仅看文件存在/大小。

结构：存在/大小>0/可打开/页数/逐页宽高可读/尺寸与计划≤0.5pt/元数据可读。
内容：逐页低 DPI 渲染；空白判定同时参考源图（纯白源页不算错误）；异常大面积深色；
重复页（渲染相同但源不同）；顺序/内容（内容区粗签名比对，容忍缩放/留白/JPEG）。
视觉：全页低 DPI 联系表 + 首/中/末 + 回退页 + 彩色页高清预览。
所有指标为启发式像素口径，非 OCR。验证全部通过才允许原子替换为正式输出。
"""

from __future__ import annotations

import logging
from contextlib import closing
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium
from PIL import Image

logger = logging.getLogger("bookclean")

SIZE_TOL_PT = 0.5
CONTENT_DIFF_MAX = 45.0     # 内容区粗签名平均绝对差上限（容忍 JPEG/缩放/留白）
DUP_DIFF_MAX = 1.0          # 渲染签名几乎相同视为重复
BLANK_WHITE_THRESH = 250    # 灰度>=此值视为白
SIG_SIZE = (32, 48)


def _signature(arr: np.ndarray):
    """内容区粗签名：裁掉白边后缩放到固定尺寸的灰度数组；全白返回 None。"""
    gray = arr.astype(np.float64).mean(axis=2)
    mask = gray < BLANK_WHITE_THRESH
    if mask.sum() < 8:
        return None
    ys, xs = np.where(mask)
    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray(crop).resize(SIG_SIZE, Image.LANCZOS)
    return np.asarray(im, dtype=np.float64).mean(axis=2)


def _load_source_sig(path):
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        return _signature(arr)
    except Exception:
        return None


# ---------- M4.7.5 灰度结构验证 ----------

RGB_LIKE = ("/DeviceRGB", "/CalRGB", "/Lab")


def _classify_colorspace(cs):
    """返回 (label, 组件数, 是否单通道灰度)。ICCBased 读 /N。"""
    if cs is None:
        return ("None", 0, False)
    try:
        if isinstance(cs, pikepdf.Array):
            head = str(cs[0])
            if head == "/ICCBased":
                n = int(cs[1].get("/N", 0))
                return (f"ICCBased(N={n})", n, n == 1)
            if head == "/Indexed":
                base_label, base_n, _ = _classify_colorspace(cs[1])
                return (f"Indexed({base_label})", base_n, False)
            return (f"Array{head}", 0, False)
        name = str(cs)
        if name in ("/DeviceGray", "/CalGray"):
            return (name, 1, True)
        if name in RGB_LIKE:
            return (name, 3, False)
        if name == "/DeviceCMYK":
            return (name, 4, False)
        return (name, 0, False)
    except Exception:
        return (repr(cs), 0, False)


def _iter_image_xobjects(pdf):
    """递归遍历每页 Resources/XObject（含 Form XObject 嵌套）中的所有图像对象。"""
    for pno, page in enumerate(pdf.pages):
        stack = [(page.get("/Resources", {}), f"page{pno + 1}")]
        seen = set()
        while stack:
            res, path = stack.pop()
            try:
                xobjs = res.get("/XObject", {})
            except Exception:
                continue
            for name, obj in xobjs.items():
                key = (path, str(name), obj.objgen)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    subtype = obj.get("/Subtype")
                except Exception:
                    continue
                if subtype == pikepdf.Name("/Image"):
                    yield pno + 1, f"{path}/{name}", obj
                elif subtype == pikepdf.Name("/Form"):
                    stack.append((obj.get("/Resources", {}), f"{path}/{name}"))


class PDFVerifier:
    def __init__(self, check_dpi: int = 60, preview_dpi: int = 100):
        self.check_dpi = check_dpi
        self.preview_dpi = preview_dpi

    def verify(self, pdf_path: str | Path, info: dict, contact_dir: str | Path | None = None,
               progress_cb=None, text_pages=None) -> dict:
        """重读 PDF 逐页验证。text_pages 为 M6.2 可选 OCR 文字层检查：

        传入应含文字的页索引集合（0 基）时，追加"这些页提取字符数 > 0"检查；
        默认 None ⇒ 不做 OCR 检查，行为与 M5 完全一致。既有验证项一律保留。
        """
        pdf_path = Path(pdf_path)
        issues: list[str] = []
        per_page: list[dict] = []
        result = {"passed": False, "issues": issues, "page_count": 0,
                  "per_page": per_page, "contact_sheet": None, "previews": []}
        if not pdf_path.exists():
            issues.append(f"文件不存在: {pdf_path}")
            return result
        if pdf_path.stat().st_size <= 0:
            issues.append("文件大小为 0")
            return result
        try:
            doc = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            issues.append(f"PDF 无法打开: {exc}")
            return result

        # closing 确保 progress_cb 协作取消（或任何异常） unwind 时也关闭 PDFium 句柄，
        # 否则 Windows 下 .partial.pdf 被占用，导出清理 unlink 会抛 PermissionError。
        with closing(doc):
            n = len(doc)
            result["page_count"] = n
            expected = info.get("summary", {}).get("total_pages")
            if expected is not None and n != expected:
                issues.append(f"页数 {n} != 计划 {expected}")
            try:
                with pikepdf.open(pdf_path) as pk:
                    _ = str(pk.docinfo.get("/Title", ""))
            except Exception as exc:
                issues.append(f"元数据不可读: {exc}")

            plan = info.get("pages", [])
            contact_dir = Path(contact_dir) if contact_dir else pdf_path.parent
            contact_dir.mkdir(parents=True, exist_ok=True)
            cols = 6
            rows = (n + cols - 1) // cols
            tw = 120
            sheet = Image.new("RGB", (cols * (tw + 6) + 6, rows * (int(tw * 1.4) + 6) + 6), (240, 240, 240))

            render_sigs: list = []
            source_sigs: list = []
            for i in range(n):
                w, h = doc[i].get_size()
                exp = plan[i] if i < len(plan) else {}
                size_ok = (abs(w - exp.get("output_page_width_pt", w)) <= SIZE_TOL_PT
                           and abs(h - exp.get("output_page_height_pt", h)) <= SIZE_TOL_PT)
                if not size_ok:
                    issues.append(f"第 {i + 1} 页尺寸 {w:.1f}x{h:.1f} 与计划 "
                                  f"{exp.get('output_page_width_pt')}x{exp.get('output_page_height_pt')} 偏差>0.5pt")
                arr = np.asarray(doc[i].render(scale=self.check_dpi / 72.0).to_pil().convert("RGB"))
                rs = _signature(arr)
                ss = _load_source_sig(exp.get("source_image_path")) if exp.get("source_image_path") else None
                render_sigs.append(rs)
                source_sigs.append(ss)
                blank_ok = True
                content_diff = None
                if rs is None and ss is not None:
                    blank_ok = False
                    issues.append(f"第 {i + 1} 页渲染为空白但源图有内容")
                if rs is not None and ss is not None:
                    content_diff = float(np.abs(rs - ss).mean())
                    if content_diff > CONTENT_DIFF_MAX:
                        issues.append(f"第 {i + 1} 页内容与源图差异过大({content_diff:.1f})，疑似顺序/内容错误")
                    if rs.mean() < 40 and ss.mean() > 120:
                        issues.append(f"第 {i + 1} 页大面积异常深色")
                if i > 0 and rs is not None and render_sigs[i - 1] is not None:
                    if np.abs(rs - render_sigs[i - 1]).mean() < DUP_DIFF_MAX and \
                            ss is not None and source_sigs[i - 1] is not None and \
                            np.abs(ss - source_sigs[i - 1]).mean() > 5:
                        issues.append(f"第 {i + 1} 页与上一页渲染重复但源不同")
                per_page.append({
                    "page_index": exp.get("page_index", i),
                    "display_index": exp.get("display_index", i + 1),
                    "width_pt": round(w, 2), "height_pt": round(h, 2),
                    "size_ok": size_ok, "blank_ok": blank_ok,
                    "content_diff": round(content_diff, 2) if content_diff is not None else None,
                    "fallback": exp.get("fallback", False),
                })
                if text_pages is not None:
                    ocr_chars = 0
                    if i in text_pages:
                        try:
                            ocr_chars = doc[i].get_textpage().count_chars()
                        except Exception:
                            ocr_chars = 0
                        if ocr_chars <= 0:
                            issues.append(f"第 {i + 1} 页 OCR 文字层缺失（提取字符数 0）")
                    per_page[-1]["ocr_text_chars"] = ocr_chars
                th = int(tw * 1.4)
                thumb = Image.fromarray(arr).resize((tw, th), Image.LANCZOS)
                sheet.paste(thumb, (6 + (i % cols) * (tw + 6), 6 + (i // cols) * (th + 6)))
                if progress_cb is not None:
                    progress_cb(i + 1, n)

            sheet_path = contact_dir / "verify_contact_sheet.png"
            sheet.save(sheet_path)
            result["contact_sheet"] = str(sheet_path)

            focus = {0, n // 2, n - 1}
            focus.update(i for i, p in enumerate(per_page) if p["fallback"])
            focus.update(i for i, p in enumerate(plan) if p.get("color_mode") == "RGB")
            for i in sorted(focus):
                if i >= n:
                    continue
                pv = contact_dir / f"verify_preview_p{i + 1:03d}.jpg"
                doc[i].render(scale=self.preview_dpi / 72.0).to_pil().convert("RGB").save(pv, quality=90)
                result["previews"].append(str(pv))
            if text_pages is not None:
                result["ocr_text_check"] = {
                    "enabled": True,
                    "pages_expected": len(text_pages),
                    "pages_with_text": sum(1 for p in per_page if (p.get("ocr_text_chars") or 0) > 0),
                }
            result["passed"] = not issues
            return result

    def verify_grayscale(self, pdf_path: str | Path, progress_cb=None) -> dict:
        """M4.7.5：检查 PDF 内部所有图像 XObject 色彩空间 + PDFium 逐页渲染通道一致性。

        接受 /DeviceGray 与单通道 ICCBased(N=1)/CalGray；要求 8-bit。
        意外 RGB/多通道对象 -> 失败并记录页码与对象名，不宣称成功。
        """
        pdf_path = Path(pdf_path)
        result = {
            "enabled": True, "status": "passed",
            "device_gray_image_count": 0, "unexpected_rgb_image_count": 0,
            "other_unexpected_image_count": 0, "render_channel_check": "passed",
            "max_render_channel_diff": 0, "failed_pages": [], "images": [],
        }
        failed = result["failed_pages"]
        try:
            with pikepdf.open(pdf_path) as pdf:
                for pno, name, obj in _iter_image_xobjects(pdf):
                    cs = obj.get("/ColorSpace")
                    label, ncomp, is_gray = _classify_colorspace(cs)
                    bits = int(obj.get("/BitsPerComponent", 0))
                    w = int(obj.get("/Width", 0)); h = int(obj.get("/Height", 0))
                    rec = {"page": pno, "object": name, "colorspace": label,
                           "components": ncomp, "bits": bits}
                    # ICCBased 灰度：核对组件数与实际数据通道数
                    if label.startswith("ICCBased") and is_gray and w and h and bits:
                        try:
                            data_len = len(obj.read_raw_bytes())
                            per_channel = w * h * bits // 8
                            if per_channel:
                                rec["data_channels"] = round(data_len / per_channel, 3)
                        except Exception:
                            rec["data_channels"] = None
                    result["images"].append(rec)
                    if is_gray:
                        result["device_gray_image_count"] += 1
                        if bits != 8:
                            failed.append({"page": pno, "object": name,
                                           "reason": f"灰度但位深 {bits} != 8"})
                    else:
                        if label in RGB_LIKE or (label.startswith("ICCBased") and ncomp == 3):
                            result["unexpected_rgb_image_count"] += 1
                        else:
                            result["other_unexpected_image_count"] += 1
                        failed.append({"page": pno, "object": name,
                                       "reason": f"非单通道灰度色彩空间 {label}"})
        except Exception as exc:
            result["status"] = "failed"
            failed.append({"page": 0, "object": "-", "reason": f"无法读取 PDF 结构: {exc}"})
            return result

        try:
            doc = pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:
            result["render_channel_check"] = "failed"
            failed.append({"page": 0, "object": "-", "reason": f"渲染失败: {exc}"})
            result["status"] = "passed" if not failed else "failed"
            return result
        # closing 保证句柄关闭；progress_cb 置于渲染 try 之外，使协作取消异常能传播而非被吞没。
        with closing(doc):
            for i in range(len(doc)):
                try:
                    rgb = np.asarray(doc[i].render(scale=self.check_dpi / 72.0).to_pil().convert("RGB")).astype(int)
                    d = int(max(np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max(),
                                np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max()))
                    result["max_render_channel_diff"] = max(result["max_render_channel_diff"], d)
                    if d > 1:
                        result["render_channel_check"] = "failed"
                        failed.append({"page": i + 1, "object": "-",
                                       "reason": f"渲染 RGB 通道差 {d} > 1"})
                except Exception as exc:
                    result["render_channel_check"] = "failed"
                    failed.append({"page": 0, "object": "-", "reason": f"渲染失败: {exc}"})
                    break
                if progress_cb is not None:
                    progress_cb(i + 1, len(doc))

        result["status"] = "passed" if not failed else "failed"
        return result