"""
Retriever adapter for Jupyter notebooks and UnifiedEntity chunks.
Plugs into LangChain / LangGraph as a tool once supervisor agent is ready.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Fix sys.path so imports work ===
repo_root = Path(__file__).resolve().parents[3]  # go up: Onto_wa_rag → agent → OntoFlow → aiengine2
sys.path.insert(0, str(repo_root))


# === Imports now work ===
from OntoFlow.agent.Onto_wa_rag.jupyter_analysis.jupyter_notebook_parser import (
    get_jupyter_analyzer,
    chunk_notebook_entities,
)

from OntoFlow.agent.Onto_wa_rag.fortran_analysis.core.entity_manager import UnifiedEntity


class SimpleRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

    def build_index_from_notebook(self, notebook_path: str):
        """Parse notebook → entities → chunks → embeddings"""
        analyzer = get_jupyter_analyzer()
        entities, _ = analyzer.analyze_file(notebook_path)
        chunks = chunk_notebook_entities(entities)

        # Save both raw chunks + embeddings
        self.chunks = chunks
        texts = [c["content"] for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        print(f"Indexed {len(chunks)} chunks from {notebook_path}")

    def query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k most relevant chunks as dicts"""
        if self.embeddings is None:
            raise ValueError("Retriever index is empty. Run build_index_from_notebook() first.")

        q_emb = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.embeddings)[0]
        top_idx = sims.argsort()[::-1][:k]

        results = []
        for idx in top_idx:
            res = self.chunks[idx].copy()
            res["score"] = float(sims[idx])
            results.append(res)
        return results


# === LangChain tool wrapper (optional now, needed later) ===
try:
    from langchain.tools import tool

    @tool("notebook_retriever", return_direct=False)
    def retriever_tool(query: str) -> str:
        """
        Retrieve relevant notebook chunks for a natural language query.
        Returns JSON list of dicts with content + token count + score.
        """
        global retriever_instance
        if retriever_instance is None:
            raise RuntimeError("Retriever not initialized. Call build_index_from_notebook() first.")
        results = retriever_instance.query(query, k=5)
        return json.dumps(results, indent=2)

except ImportError:
    retriever_tool = None
    print("LangChain not installed: retriever_tool will be None")

# Singleton retriever (for use across modules)
retriever_instance: SimpleRetriever | None = None

def init_retriever(notebook_path: str):
    global retriever_instance
    retriever_instance = SimpleRetriever()
    retriever_instance.build_index_from_notebook(notebook_path)
    return retriever_instance


if __name__ == "__main__":
    # Path to your test notebook
    nb_path = "2-aiengine/OntoFlow/agent/Onto_wa_rag/jupyter_analysis/test.ipynb"
    retriever = init_retriever(nb_path)

    # Better query for this notebook
    query = "How to modify input parameters in BigDFT?"
    results = retriever.query(query, k=3)

    print("\n=== Query Results ===")
    for r in results:
        print(f"\n--- Result (Score: {r['score']:.3f}) ---")
        print(r["content"])  # show full chunk text
