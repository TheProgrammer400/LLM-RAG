from typing import List, Dict, Any
from shared.models.schemas import RetrievedChunk

class RetrievalEvaluator:
    def evaluate_retrieval(self, retrieved_chunks: List[RetrievedChunk], expected_keywords: List[str]) -> Dict[str, float]:
        """
        Computes precision@10, recall@10, duplicate_ratio, and mean_rerank_score.
        """
        if not retrieved_chunks:
            return {"precision_at_10": 0.0, "recall_at_10": 0.0, "duplicate_ratio": 0.0, "mean_rerank_score": 0.0}

        top_10 = retrieved_chunks[:10]
        match_count = 0
        texts = [c.chunk.text.lower() for c in top_10]

        for kw in expected_keywords:
            if any(kw.lower() in t for t in texts):
                match_count += 1

        precision_at_10 = match_count / len(top_10) if top_10 else 0.0
        recall_at_10 = match_count / max(len(expected_keywords), 1)

        # Duplicate ratio check
        unique_snippets = set(t[:100] for t in texts)
        duplicate_ratio = 1.0 - (len(unique_snippets) / max(len(top_10), 1))

        mean_rerank = sum(c.rerank_score for c in top_10) / len(top_10)

        return {
            "precision_at_10": precision_at_10,
            "recall_at_10": recall_at_10,
            "duplicate_ratio": duplicate_ratio,
            "mean_rerank_score": mean_rerank
        }
