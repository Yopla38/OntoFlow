"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# context_providers/local_context_provider.py
import re
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class LocalContextProvider:
    """Fournit le contexte local : dépendances immédiates d'une entité"""

    def __init__(self, document_store, rag_engine, entity_index):
        self.document_store = document_store
        self.rag_engine = rag_engine
        self.entity_index = entity_index

        # Patterns pour détecter les appels de fonctions/subroutines
        self.call_patterns = [
            re.compile(r'\bcall\s+(\w+)', re.IGNORECASE),  # call subroutine_name
            re.compile(r'(\w+)\s*\(', re.IGNORECASE),  # function_name(
            re.compile(r'=\s*(\w+)\s*\(', re.IGNORECASE),  # var = function_name(
        ]

        # Patterns pour les variables
        self.variable_patterns = [
            re.compile(r'(\w+)\s*=', re.IGNORECASE),  # variable =
            re.compile(r'(\w+)\s*\(', re.IGNORECASE),  # array(index)
        ]

    async def _get_complete_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Récupère les infos complètes d'une entité en regroupant ses parties"""

        # Chercher l'entité
        chunks = await self.entity_index.find_entity(entity_name)
        if not chunks:
            return None

        # Prendre le premier chunk et voir s'il a des métadonnées de regroupement
        chunk = await self._get_chunk_by_id(chunks[0])
        if not chunk:
            return None

        metadata = chunk.get('metadata', {})

        # Si c'est une partie, récupérer toutes les parties
        if metadata.get('is_partial') and metadata.get('all_chunks'):
            all_chunk_ids = metadata['all_chunks']

            # Récupérer toutes les parties
            all_parts = []
            for chunk_id in all_chunk_ids:
                part_chunk = await self._get_chunk_by_id(chunk_id)
                if part_chunk:
                    all_parts.append(part_chunk)

            # Combiner le texte de toutes les parties
            combined_text = '\n\n'.join([part['text'] for part in all_parts])

            # Créer une entité complète
            base_name = metadata.get('base_entity_name', entity_name.split('_part_')[0])

            return {
                'name': base_name,
                'type': metadata.get('entity_type', 'unknown'),
                'signature': await self._extract_signature(combined_text),
                'summary': f"Complete {metadata.get('entity_type', 'entity')} with {len(all_parts)} parts",
                'file': metadata.get('filename', ''),
                'combined_text': combined_text,
                'parts_count': len(all_parts),
                'is_complete': True
            }

        else:
            # Entité normale, pas de regroupement nécessaire
            return {
                'name': entity_name,
                'type': metadata.get('entity_type', 'unknown'),
                'signature': await self._extract_signature(chunk['text']),
                'summary': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'file': metadata.get('filename', ''),
                'is_complete': False
            }

    async def get_local_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Récupère le contexte local d'une entité"""

        # 1. Trouver l'entité principale
        main_chunks = await self.entity_index.find_entity(entity_name)

        if not main_chunks:
            return {
                "error": f"Entity '{entity_name}' not found",
                "suggestions": await self._get_similar_entity_names(entity_name)
            }

        main_chunk_id = main_chunks[0]  # Prendre le premier chunk trouvé
        main_chunk = await self._get_chunk_by_id(main_chunk_id)

        if not main_chunk:
            return {"error": f"Chunk {main_chunk_id} not found"}

        entity_info = await self.entity_index.get_entity_info(main_chunk_id)

        context = {
            "entity": entity_name,
            "type": "local",
            "main_definition": await self._get_entity_definition(main_chunk, entity_info),
            "immediate_dependencies": [],
            "called_functions": [],
            "used_variables": [],
            "parent_context": None,
            "children_context": [],
            "file_context": {},
            "tokens_used": 0
        }

        tokens_budget = max_tokens

        # 2. Dépendances immédiates (USE statements - métadonnées)
        if entity_info and entity_info.get('dependencies'):
            context["immediate_dependencies"] = await self._get_dependency_contexts(
                entity_info['dependencies'], tokens_budget * 0.2
            )
            tokens_budget -= len(str(context["immediate_dependencies"])) // 4

        # 3. Fonctions appelées (analyse du texte + cache)
        called_functions = await self._extract_and_resolve_function_calls(
            main_chunk, tokens_budget * 0.4
        )
        context["called_functions"] = called_functions
        tokens_budget -= len(str(called_functions)) // 4

        # 4. Contexte parent si fonction interne
        if entity_info and entity_info.get('is_internal'):
            parent_name = entity_info.get('parent')
            if parent_name:
                context["parent_context"] = await self._get_parent_context(
                    parent_name, tokens_budget * 0.2
                )
                tokens_budget -= len(str(context["parent_context"])) // 4

        # 5. Contexte des enfants (fonctions internes)
        children = await self.entity_index.get_children(entity_name)
        if children and tokens_budget > 100:
            regrouped_children = []
            for child_name in children:
                complete_info = await self._get_complete_entity_info(child_name)
                if complete_info:
                    regrouped_children.append(complete_info)

            context["children_context"] = regrouped_children[:5]  # Top 5

        # 6. Contexte du fichier (autres entités du même fichier)
        if entity_info and tokens_budget > 100:
            context["file_context"] = await self._get_file_context(
                entity_info.get('filepath', ''), entity_name, tokens_budget * 0.1
            )

        context["tokens_used"] = max_tokens - tokens_budget

        return context

    async def _get_entity_definition(self, chunk: Dict[str, Any], entity_info: Optional[Dict[str, Any]]) -> Dict[
        str, Any]:
        """Récupère la définition complète de l'entité"""

        definition = {
            "name": entity_info.get('name', '') if entity_info else '',
            "type": entity_info.get('type', '') if entity_info else '',
            "code": chunk.get('text', ''),
            "location": {
                "file": entity_info.get('filepath', '') if entity_info else '',
                "lines": f"{entity_info.get('start_line', '')}-{entity_info.get('end_line', '')}" if entity_info else ''
            },
            "signature": await self._extract_signature(chunk.get('text', '')),
            "concepts": entity_info.get('concepts', [])[:3] if entity_info else []  # Top 3 concepts
        }

        return definition

    async def _extract_signature(self, code: str) -> str:
        """Extrait la signature d'une fonction/subroutine"""
        lines = code.split('\n')

        for line in lines[:5]:  # Chercher dans les 5 premières lignes
            line = line.strip()

            # Patterns pour les signatures Fortran
            signature_patterns = [
                re.compile(r'(subroutine\s+\w+\s*\([^)]*\))', re.IGNORECASE),
                re.compile(r'(function\s+\w+\s*\([^)]*\))', re.IGNORECASE),
                re.compile(r'(.*function\s+\w+\s*\([^)]*\))', re.IGNORECASE),  # Avec type de retour
            ]

            for pattern in signature_patterns:
                match = pattern.search(line)
                if match:
                    return match.group(1).strip()

        return "Signature not found"

    async def _get_dependency_contexts(self, dependencies: List[str], max_tokens: int) -> List[Dict[str, Any]]:
        """Récupère le contexte des dépendances (modules USE)"""
        contexts = []
        tokens_per_dep = max_tokens / max(len(dependencies), 1)

        for dep in dependencies[:5]:  # Limiter à 5 dépendances
            dep_chunks = await self.entity_index.find_entity(dep)

            if dep_chunks:
                dep_chunk = await self._get_chunk_by_id(dep_chunks[0])
                dep_info = await self.entity_index.get_entity_info(dep_chunks[0])

                if dep_chunk and dep_info:
                    # Résumé de la dépendance
                    dep_context = {
                        "name": dep,
                        "type": dep_info.get('type', ''),
                        "file": dep_info.get('filepath', ''),
                        "summary": self._create_summary(dep_chunk['text'], int(tokens_per_dep)),
                        "public_interface": await self._extract_public_interface(dep_chunk['text'])
                    }
                    contexts.append(dep_context)

        return contexts

    async def _extract_and_resolve_function_calls(self, chunk: Dict[str, Any], max_tokens: int) -> List[Dict[str, Any]]:
        """Extrait et résout les appels de fonctions dans le code"""
        chunk_id = chunk['id']

        # Vérifier le cache d'abord
        cached_calls = self.entity_index.get_cached_call_patterns(chunk_id)

        if cached_calls is None:
            # Extraire les appels depuis le texte
            calls = self._extract_function_calls(chunk['text'])
            self.entity_index.cache_call_patterns(chunk_id, calls)
        else:
            calls = cached_calls

        # Résoudre les définitions des fonctions appelées
        resolved_calls = []
        tokens_per_call = max_tokens / max(len(calls), 1)

        for call_name in calls[:8]:  # Limiter à 8 appels
            call_definition = await self._find_function_definition(call_name, int(tokens_per_call))
            if call_definition:
                resolved_calls.append(call_definition)

        return resolved_calls

    def _extract_function_calls(self, code: str) -> List[str]:
        """Extrait les noms des fonctions appelées depuis le code"""
        calls = set()

        # Nettoyer le code (retirer les commentaires)
        cleaned_code = self._remove_comments(code)

        for pattern in self.call_patterns:
            matches = pattern.findall(cleaned_code)
            calls.update(matches)

        # Filtrer les mots-clés Fortran et les noms trop courts
        fortran_keywords = {
            'if', 'then', 'else', 'end', 'do', 'while', 'select', 'case',
            'where', 'forall', 'real', 'integer', 'logical', 'character',
            'type', 'class', 'procedure', 'interface', 'module', 'program',
            'subroutine', 'function', 'contains', 'use', 'implicit', 'none'
        }

        filtered_calls = []
        for call in calls:
            if (len(call) > 2 and
                    call.lower() not in fortran_keywords and
                    not call.isdigit()):
                filtered_calls.append(call)

        return list(set(filtered_calls))  # Retirer les doublons

    def _remove_comments(self, code: str) -> str:
        """Retire les commentaires Fortran du code"""
        lines = code.split('\n')
        cleaned_lines = []

        for line in lines:
            # Retirer les commentaires qui commencent par !
            comment_pos = line.find('!')
            if comment_pos != -1:
                line = line[:comment_pos]

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    async def _find_function_definition(self, function_name: str, max_tokens: int) -> Optional[Dict[str, Any]]:
        """Trouve la définition d'une fonction via l'index ou le RAG"""

        # 1. Chercher dans l'index d'abord
        chunks = await self.entity_index.find_entity(function_name)

        if chunks:
            chunk = await self._get_chunk_by_id(chunks[0])
            entity_info = await self.entity_index.get_entity_info(chunks[0])

            if chunk and entity_info:
                return {
                    "name": function_name,
                    "type": entity_info.get('type', ''),
                    "file": entity_info.get('filepath', ''),
                    "signature": await self._extract_signature(chunk['text']),
                    "summary": self._create_summary(chunk['text'], max_tokens),
                    "source": "index"
                }

        # 2. Utiliser le RAG en fallback
        try:
            search_query = f"function {function_name} subroutine {function_name}"
            similar_chunks = await self.rag_engine.find_similar(
                search_query, max_results=3, min_similarity=0.6
            )

            for chunk_id, similarity in similar_chunks:
                chunk = await self._get_chunk_by_id(chunk_id)
                if chunk and function_name.lower() in chunk['text'].lower():
                    entity_info = await self.entity_index.get_entity_info(chunk_id)

                    return {
                        "name": function_name,
                        "type": "found_via_rag",
                        "file": entity_info.get('filepath', '') if entity_info else '',
                        "signature": await self._extract_signature(chunk['text']),
                        "summary": self._create_summary(chunk['text'], max_tokens),
                        "similarity": similarity,
                        "source": "rag"
                    }

        except Exception as e:
            logger.debug(f"RAG search failed for {function_name}: {e}")

        return None

    async def _get_parent_context(self, parent_name: str, max_tokens: int) -> Optional[Dict[str, Any]]:
        """Récupère le contexte du parent (pour les fonctions internes)"""
        parent_chunks = await self.entity_index.find_entity(parent_name)

        if not parent_chunks:
            return None

        parent_chunk = await self._get_chunk_by_id(parent_chunks[0])
        parent_info = await self.entity_index.get_entity_info(parent_chunks[0])

        if parent_chunk and parent_info:
            return {
                "name": parent_name,
                "type": parent_info.get('type', ''),
                "summary": self._create_summary(parent_chunk['text'], max_tokens),
                "signature": await self._extract_signature(parent_chunk['text']),
                "children": await self.entity_index.get_children(parent_name)
            }

        return None

    async def _get_children_contexts(self, children: List[str], max_tokens: int) -> List[Dict[str, Any]]:
        """Récupère le contexte des enfants (fonctions internes)"""
        contexts = []
        tokens_per_child = max_tokens / max(len(children), 1)

        for child_name in children:
            child_chunks = await self.entity_index.find_entity(child_name)

            if child_chunks:
                child_chunk = await self._get_chunk_by_id(child_chunks[0])
                child_info = await self.entity_index.get_entity_info(child_chunks[0])

                if child_chunk and child_info:
                    contexts.append({
                        "name": child_name,
                        "type": child_info.get('type', ''),
                        "signature": await self._extract_signature(child_chunk['text']),
                        "summary": self._create_summary(child_chunk['text'], int(tokens_per_child))
                    })

        return contexts

    async def _get_file_context(self, filepath: str, current_entity: str, max_tokens: int) -> Dict[str, Any]:
        """Récupère le contexte des autres entités du même fichier"""
        if not filepath:
            return {}

        file_entities = await self.entity_index.find_entities_in_file(filepath)

        other_entities = []
        for chunk_id in file_entities:
            entity_info = await self.entity_index.get_entity_info(chunk_id)

            if entity_info and entity_info['name'] != current_entity:
                other_entities.append({
                    "name": entity_info['name'],
                    "type": entity_info['type'],
                    "lines": f"{entity_info.get('start_line', '')}-{entity_info.get('end_line', '')}"
                })

        return {
            "filepath": filepath,
            "other_entities": other_entities[:10],  # Limiter à 10
            "total_entities": len(file_entities)
        }

    async def _extract_public_interface(self, module_code: str) -> List[str]:
        """Extrait l'interface publique d'un module (fonctions/types exportés)"""
        interfaces = []

        # Chercher les déclarations public explicites
        public_pattern = re.compile(r'public\s*::\s*([^!\n]+)', re.IGNORECASE)
        matches = public_pattern.findall(module_code)

        for match in matches:
            # Séparer les noms multiples
            names = [name.strip() for name in match.split(',')]
            interfaces.extend(names)

        # Si pas de public explicite, chercher les fonctions/subroutines
        if not interfaces:
            func_patterns = [
                re.compile(r'subroutine\s+(\w+)', re.IGNORECASE),
                re.compile(r'function\s+(\w+)', re.IGNORECASE)
            ]

            for pattern in func_patterns:
                matches = pattern.findall(module_code)
                interfaces.extend(matches)

        return interfaces[:10]  # Limiter à 10 éléments

    def _create_summary(self, text: str, max_tokens: int) -> str:
        """Crée un résumé du texte en respectant la limite de tokens"""
        words = text.split()
        max_words = max_tokens * 3  # Approximation : 1 token ≈ 3/4 mots

        if len(words) <= max_words:
            return text

        # Prendre le début et essayer de finir à une phrase complète
        truncated = ' '.join(words[:max_words])

        # Chercher le dernier point pour finir proprement
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.8:  # Si le point est vers la fin
            truncated = truncated[:last_period + 1]

        return truncated + "..."

    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk par son ID depuis le document store"""
        # Parser le chunk_id pour extraire le document_id
        # Format: document_id-chunk-N
        parts = chunk_id.split('-chunk-')
        if len(parts) != 2:
            return None

        document_id = parts[0]

        # Charger les chunks du document
        await self.document_store.load_document_chunks(document_id)
        chunks = await self.document_store.get_document_chunks(document_id)

        if chunks:
            for chunk in chunks:
                if chunk['id'] == chunk_id:
                    return chunk

        return None

    async def _get_similar_entity_names(self, entity_name: str, limit: int = 5) -> List[str]:
        """Trouve des noms d'entités similaires pour les suggestions"""
        all_names = list(self.entity_index.name_to_chunks.keys())

        # Recherche par sous-chaîne
        similar = [name for name in all_names
                   if entity_name.lower() in name.lower() or name.lower() in entity_name.lower()]

        return similar[:limit]