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
        self.indexed_files = set()

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
            res["similarity_score"] = float(sims[idx])
            results.append(res)
        return results

    def build_index_from_existing_chunks(self, rag_instance):
        """Construit l'index à partir des chunks déjà créés dans le document_store."""
        print("🔄 Construction de l'index sémantique à partir des chunks existants...")

        all_chunks = []
        notebook_count = 0
        self.indexed_files = set()

        # Parcourir tous les documents dans le document_store
        for doc_id, chunks in rag_instance.rag_engine.document_store.document_chunks.items():
            doc_info = rag_instance.rag_engine.document_store.documents.get(doc_id, {})
            original_path = doc_info.get('original_path', '')

            # Ne traiter que les notebooks Jupyter
            if original_path.endswith('.ipynb'):
                notebook_count += 1
                filename = Path(original_path).name
                self.indexed_files.add(original_path)
                #print(f"  📓 Récupération des chunks de {filename}...")

                for chunk in chunks:
                    # Enrichir le chunk avec les infos manquantes pour le retriever
                    enriched_chunk = {
                        "content": chunk.get('text', ''),
                        "tokens": self._estimate_tokens(chunk.get('text', '')),
                        "source_file": original_path,
                        "source_filename": filename,
                        "source_type": "notebook",
                        "chunk_id": chunk.get('id'),
                        "metadata": chunk.get('metadata', {})
                    }
                    all_chunks.append(enriched_chunk)

        if not all_chunks:
            print("  ⚠️ Aucun chunk de notebook trouvé")
            return 0

        # Construire l'index d'embeddings
        self.chunks = all_chunks
        texts = [chunk["content"] for chunk in all_chunks]

        print(f"  🔧 Génération des embeddings pour {len(texts)} chunks...")
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

        print(f"✅ Index sémantique prêt: {len(all_chunks)} chunks de {notebook_count} notebooks")
        return notebook_count

    def _estimate_tokens(self, text: str, chars_per_token: int = 4) -> int:
        """Approxime le nombre de tokens."""
        return max(1, len(text) // chars_per_token)

    def get_index_status(self):
        """Retourne le statut de l'index sémantique."""
        notebook_files = set()
        for chunk in self.chunks:
            notebook_files.add(chunk.get("source_filename", "unknown"))

        return {
            "chunks_indexed": len(self.chunks),
            "notebooks_indexed": len(notebook_files),
            "notebook_files": list(notebook_files),
            "embedding_model": getattr(self.model, 'model_name', "sentence-transformers model"),
            "index_ready": len(self.chunks) > 0
        }

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
    nb_path = "/home/yopla/PycharmProjects/llm-hackathon-2025/2-aiengine/OntoFlow/downloaded_docs/lessons/ComplexityReduction.ipynb"
    retriever = init_retriever(nb_path)

    # Better query for this notebook
    query = "How to modify input parameters in BigDFT?"
    results = retriever.query(query, k=3)

    print("\n=== Query Results ===")
    for r in results:
        print(f"\n--- Result (Score: {r['similarity_score']:.3f}) ---")
        print(r["content"])  # show full chunk text
