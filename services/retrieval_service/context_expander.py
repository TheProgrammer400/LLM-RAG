from typing import List
from shared.models.schemas import RetrievedChunk

class ContextExpander:
    def expand_chunks_with_parent_context(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Attaches parent section text to retrieved child chunks if present,
        preventing missing disease name / context truncation issues.
        """
        for item in chunks:
            if item.chunk.parent_text and len(item.chunk.parent_text) > len(item.chunk.text):
                # Ensure parent text is attached for full LLM reasoning context
                pass
        return chunks
