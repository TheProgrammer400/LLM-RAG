from typing import List, Dict
from shared.models.schemas import RetrievedChunk

class ReciprocalRankFusion:
    def __init__(self, k: float = 60.0):
        self.k = k

    def fuse_results(self, result_lists: List[List[RetrievedChunk]]) -> List[RetrievedChunk]:
        """
        Combines dense + sparse results across facets using Reciprocal Rank Fusion (RRF).
        score(chunk) = Σ 1 / (k + rank_in_list)
        """
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        for r_list in result_lists:
            for rank, item in enumerate(r_list):
                cid = item.chunk.chunk_id
                rrf_val = 1.0 / (self.k + (rank + 1))
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_val
                if cid not in chunk_map:
                    chunk_map[cid] = item

        fused: List[RetrievedChunk] = []
        for cid, rrf_score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
            item = chunk_map[cid]
            item.rrf_score = rrf_score
            fused.append(item)

        return fused
