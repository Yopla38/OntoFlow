import json
from pathlib import Path
import asyncio
from typing import List, Dict, Any

from .CONSTANT import ONTOLOGY_PATH_TTL, API_KEY_PATH
from .Integration_fortran_RAG import OntoRAG
from .provider.get_key import get_openai_key


class RagContextProvider:
    """
    Service qui indexe un dossier et fournit des passages pertinents
    (code ou texte) à partir d’une question.
    """

    def __init__(self,
                 storage_dir: str = ".rag_storage",
                 ontology_path: str | None = None):
        self.rag = OntoRAG(storage_dir=storage_dir,
                           ontology_path=ontology_path,
                           chunk_size=2000,
                           chunk_overlap=0)

    async def initialize(self):
        await self.rag.initialize()

    # ----------- indexation d’un projet -----------------
    async def index_directory(self,
                              root: Path,
                              exclude_dirs: set[Path] | None = None,
                              include_exts: set[str] | None = None):
        exclude_dirs = exclude_dirs or set()
        include_exts = include_exts or set()

        files = [
            p for p in root.rglob("*")
            if p.is_file()
               and (not include_exts or p.suffix.lstrip('.').lower() in include_exts)
               and not any(ex in p.parents for ex in exclude_dirs)
        ]

        todo = [{"filepath": str(p)} for p in files]

        await self.rag.add_documents_batch(todo, max_concurrent=5)

    # ----------- recherche ------------------------------
    async def query(self,
                    question: str,
                    max_chunks: int = 8) -> List[Dict[str, Any]]:
        result = await self.rag.query(question,
                                      max_results=max_chunks,
                                      use_ontology=True)
        return result.get("passages", [])

    # ----------- formatage simple -----------------------
    @staticmethod
    def passages_to_prompt(passages: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, p in enumerate(passages, 1):
            md = p["metadata"]
            fname = md.get("filename", "unknown")
            blocks.append(
                f"--SOURCE {i} ({fname}:{md.get('start_pos')}-{md.get('end_pos')})\n{p['text']}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    async def passages_to_function_json(passages, join_with="\n") -> str:
        """
        Reconstitue la fonction ou subroutine complète demandée
        à partir des passages RAG et renvoie un JSON (string).

        • Si plusieurs entités différentes sont dans `passages`,
          seule la plus pertinente (score le plus élevé) est gardée.
        • On utilise les métadonnées générées par le chunker
          (parent_entity_id, part_sequence, entity_bounds, etc.).
        """

        if not passages:
            return json.dumps({"error": "no passages"}, indent=2)

        # ------ 1. choisir l’entité cible (meilleur passage) -------------
        best = max(passages, key=lambda p: p.get("similarity", 0))
        md = best["metadata"]

        # identifiant d’entité (parent pour les part_X)
        ent_id = (md.get("parent_entity_id")
                  if md.get("is_partial") else md.get("entity_id"))

        if not ent_id:
            # Chunk non splitté : on renvoie directement
            return json.dumps({
                "file": md.get("filepath"),
                "entity": md.get("entity_name"),
                "entity_type": md.get("entity_type"),
                "start_line": md.get("start_pos"),
                "end_line": md.get("end_pos"),
                "code": best["text"]
            }, indent=2, ensure_ascii=False)

        # ------ 2. collecter toutes les parties de la même entité --------
        parts = [p for p in passages
                 if (p["metadata"].get("parent_entity_id") or
                     p["metadata"].get("entity_id")) == ent_id]

        # Si RAG n’a ramené qu’une partie, on va chercher le code complet
        # sur disque grâce à entity_bounds
        entity_bounds = md.get("entity_bounds", {})
        if len(parts) < md.get("total_parts", 1) and entity_bounds:
            start_line = entity_bounds.get("start_line")
            end_line = entity_bounds.get("end_line")
            file_path = md.get("filepath")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()[start_line - 1:end_line]
                code = "".join(lines)
                return json.dumps({
                    "file": file_path,
                    "entity": md.get("base_entity_name", md.get("entity_name")),
                    "entity_type": md.get("entity_type"),
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": code
                }, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Impossible de lire {file_path}: {e}")

        # ------ 3. concaténer les parties ramenées -----------------------
        # On trie par part_sequence ou start_pos
        parts.sort(key=lambda p: p["metadata"].get("part_sequence",
                                                   p["metadata"].get("start_pos", 0)))

        full_code = join_with.join([p["text"] for p in parts])
        start_line = parts[0]["metadata"].get("start_pos")
        end_line = parts[-1]["metadata"].get("end_pos")
        file_path = parts[0]["metadata"].get("filepath")

        return json.dumps({
            "file": file_path,
            "entity": parts[0]["metadata"].get("base_entity_name",
                                               parts[0]["metadata"].get("entity_name")),
            "entity_type": parts[0]["metadata"].get("entity_type"),
            "start_line": start_line,
            "end_line": end_line,
            "code": full_code
        }, indent=2, ensure_ascii=False)


class RagTools:
    """Expose chaque tool sous forme de coroutine."""

    def __init__(self, storage_dir: Path, ontology_ttl_path: str = None):
        self.storage_dir = storage_dir
        self.rag = RagContextProvider(
            storage_dir=str(storage_dir / "rag_storage"),
            ontology_path=ontology_ttl_path
        )
        self._initialized = False
        self.my_list_of_response = []

    async def _ensure_init(self):
        if not self._initialized:
            await self.rag.initialize()
            self._initialized = True

    # ------------- tools -----------------

    async def add_to_list(self, question: str = None) -> List[str]:
        response_dict = await self.rag_function_json(question)
        self.my_list_of_response.append(response_dict)
        return self.my_list_of_response

    async def rag_index_directory(self, directory: str) -> str:
        await self._ensure_init()
        await self.rag.index_directory(Path(directory))
        return f"Dossier {directory} indexé avec succès."

    async def rag_add_file(self, filepath: str) -> str:
        await self._ensure_init()
        await self.rag.index_file(Path(filepath))
        return f"Fichier {filepath} ajouté."

    async def rag_remove_file(self, filepath: str) -> str:
        await self._ensure_init()
        await self.rag.remove_file(Path(filepath))
        return f"Fichier {filepath} supprimé."

    async def rag_query(self, question: str, max_chunks: int = 8) -> Dict[str, Any]:
        await self._ensure_init()
        passages = await self.rag.query(question, max_chunks=max_chunks)
        return passages   # serialisable

    async def rag_function_json(self, question: str) -> Dict[str, Any]:
        await self._ensure_init()
        passages = await self.rag.query(question, max_chunks=15)
        if passages:
            return json.loads(await self.rag.passages_to_function_json(passages))
        return {"error": "aucun passage trouvé"}


def rag_fc_jarvis():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_to_list",
                "description": "Ajoute une unique question à une liste",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                    },
                    "required": ["question"]
                }
            }
        }
    ]
    """
    {
            "type": "function",
            "function": {
                "name": "rag_query",
                "description": "Interroge l’index avec une question technique.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "max_chunks": {
                            "type": "integer",
                            "description": "Nombre maximum de chunks à retourner",
                            "default": 8
                        }
                    },
                    "required": ["question"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "rag_function_json",
                "description": "Retourne le code complet d’une fonction/subroutine détectée dans la question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"}
                    },
                    "required": ["question"]
                }
            }
        },
    """
    return tools



"""
async def example_usage():
    directory_to_look = Path("/home/yopla/test_agent")
    directory_to_save = directory_to_look / "rag_storage"
    idea = "Veuillez ecrire le docstring de la fonction inspect_rototranslation et faire un readme du module reformatting"
    rag_provider = RagContextProvider(
        storage_dir=str(directory_to_save),
        ontology_path=ONTOLOGY_PATH_TTL)  # TODO relier le ttl fortran
    await rag_provider.initialize()
    await rag_provider.index_directory(directory_to_look,
                                        include_exts={"f90", "py", "txt"},
                                        exclude_dirs={directory_to_save})  # dans le cas ou le dossier
                                                                           # du rag serait dans la zone a regarder
    question = f"Localise la fonction citée dans '{idea}'"
    passages = await rag_provider.query(question, max_chunks=8)

    if passages:
        func_json = await rag_provider.passages_to_function_json(passages)
        print(func_json)



if __name__ == "__main__":
    asyncio.run(example_usage())
"""
