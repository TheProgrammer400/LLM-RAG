import re
import unicodedata
from typing import List
from ingestion.pdf_parser import LayoutBlock

class TextCleaner:
    def clean_blocks(self, blocks: List[LayoutBlock]) -> List[LayoutBlock]:
        """
        Strips running headers/footers, normalizes whitespace, de-hyphenates line wraps,
        and converts unicode to NFKC standard.
        """
        cleaned_blocks: List[LayoutBlock] = []
        header_footer_patterns = self._detect_repeated_headers_footers(blocks)

        for block in blocks:
            text = block.text

            # Skip header/footer matches
            if any(p in text for p in header_footer_patterns):
                continue

            # Unicode normalization
            text = unicodedata.normalize("NFKC", text)

            # De-hyphenate line-wrapped words
            text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

            # Collapse multi-whitespace
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) > 5:
                block.text = text
                cleaned_blocks.append(block)

        return cleaned_blocks

    def _detect_repeated_headers_footers(self, blocks: List[LayoutBlock]) -> List[str]:
        """Identifies text snippets appearing repeatedly across pages (running headers/footers)."""
        counts = {}
        page_occurrences = {}

        for b in blocks:
            snippet = b.text.strip()
            if len(snippet) < 80:  # Header/footer candidates are short
                counts[snippet] = counts.get(snippet, 0) + 1
                if snippet not in page_occurrences:
                    page_occurrences[snippet] = set()
                page_occurrences[snippet].add(b.page_num)

        # Snippets appearing on 3+ distinct pages are running headers/footers
        repeated = [s for s, pages in page_occurrences.items() if len(pages) >= 3]
        return repeated
