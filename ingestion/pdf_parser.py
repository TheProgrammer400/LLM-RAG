import os
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class LayoutBlock:
    def __init__(self, page_num: int, bbox: Tuple[float, float, float, float], text: str, block_type: str = "text"):
        self.page_num = page_num
        self.bbox = bbox
        self.text = text
        self.block_type = block_type

class PDFParser:
    def __init__(self, scanned_threshold: float = 0.20):
        self.scanned_threshold = scanned_threshold

    def parse_pdf(self, file_path: str) -> Tuple[List[LayoutBlock], Dict[str, Any]]:
        """
        Parses PDF into layout blocks using PyMuPDF.
        Detects scanned pages and flags quality.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        doc = fitz.open(file_path)
        total_pages = len(doc)
        empty_or_scanned_pages = 0
        layout_blocks: List[LayoutBlock] = []

        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            blocks = page.get_text("blocks")
            
            page_text_len = sum(len(b[4].strip()) for b in blocks if len(b) >= 5)
            if page_text_len < 50:
                empty_or_scanned_pages += 1
                ocr_text = self._fallback_ocr(page)
                if ocr_text.strip():
                    layout_blocks.append(
                        LayoutBlock(page_num=page_num, bbox=(0, 0, page.rect.width, page.rect.height), text=ocr_text, block_type="ocr_text")
                    )
                continue

            for b in blocks:
                if len(b) >= 5:
                    bbox = (b[0], b[1], b[2], b[3])
                    text = b[4].strip()
                    if text:
                        # Determine rough block type by size/formatting
                        block_type = "heading" if len(text) < 120 and "\n" not in text else "paragraph"
                        layout_blocks.append(LayoutBlock(page_num=page_num, bbox=bbox, text=text, block_type=block_type))

        scanned_ratio = empty_or_scanned_pages / max(total_pages, 1)
        ocr_derived = scanned_ratio > self.scanned_threshold

        metadata = {
            "total_pages": total_pages,
            "scanned_ratio": scanned_ratio,
            "ocr_derived": ocr_derived,
            "total_blocks": len(layout_blocks)
        }

        logger.info(f"Parsed {file_path}: {total_pages} pages, {len(layout_blocks)} blocks, OCR ratio: {scanned_ratio:.2f}")
        return layout_blocks, metadata

    def _fallback_ocr(self, page: fitz.Page) -> str:
        """Fallback pytesseract OCR for scanned pages."""
        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes()))
            return pytesseract.image_to_string(img)
        except Exception as e:
            logger.warning(f"OCR fallback failed on page {page.number}: {e}")
            return ""
