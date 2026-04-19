"""Dense retrieval over persisted paper-card embeddings.

This is intentionally simple:
- persisted embeddings are stored in .npy
- metadata/text rows are stored in JSONL
- retrieval is cosine similarity over normalized vectors

This avoids introducing a larger indexing framework in the first version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return x / norms


class PaperCardRAG:
    def __init__(self, index_dir: str, top_k: int = 5) -> None:
        self.index_dir = Path(index_dir)
        self.top_k = top_k
        self.embeddings = self._load_embeddings()
        self.records = self._load_records()
        if len(self.records) != self.embeddings.shape[0]:
            raise ValueError(
                f"Record count {len(self.records)} does not match embedding rows {self.embeddings.shape[0]}"
            )

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for query encoding. Install requirements.txt first."
            ) from exc

        manifest = self._load_manifest()
        self.embedding_model_name = manifest["embedding_model_name"]
        self.encoder = SentenceTransformer(self.embedding_model_name)

    def _load_embeddings(self) -> np.ndarray:
        arr = np.load(self.index_dir / "embeddings.npy")
        return normalize_rows(arr.astype(np.float32))

    def _load_records(self) -> List[Dict]:
        path = self.index_dir / "records.jsonl"
        rows: List[Dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    def _load_manifest(self) -> Dict:
        with (self.index_dir / "manifest.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict]:
        k = top_k or self.top_k
        query_vec = self.encoder.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype=np.float32)

        sims = (self.embeddings @ query_vec.T).squeeze(-1)
        order = np.argsort(-sims)[:k]

        out: List[Dict] = []
        for idx in order.tolist():
            rec = dict(self.records[idx])
            rec["score"] = float(sims[idx])
            out.append(rec)
        return out
