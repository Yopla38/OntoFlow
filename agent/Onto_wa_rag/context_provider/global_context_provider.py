"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# context_providers/global_context_provider.py
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class GlobalContextProvider:
    """Fournit le contexte global : vue d'ensemble du projet et graphe de dépendances"""

    def __init__(self, document_store, rag_engine, entity_index):
        self.document_store = document_store
        self.rag_engine = rag_engine
        self.entity_index = entity_index

        # Cache pour les graphes de dépendances
        self._dependency_graph_cache: Dict[str, Dict[str, Any]] = {}
        self._module_hierarchy_cache: Optional[Dict[str, Any]] = None

    async def get_global_context(self, entity_name: str, max_tokens: int = 3000) -> Dict[str, Any]:
        """Récupère le contexte global d'une entité"""

        # Vérifier que l'entité existe
        entity_chunks = await self.entity_index.find_entity(entity_name)
        if not entity_chunks:
            return {
                "error": f"Entity '{entity_name}' not found",
                "suggestions": await self._get_similar_entity_names(entity_name)
            }

        main_chunk_id = entity_chunks[0]
        entity_info = await self.entity_index.get_entity_info(main_chunk_id)

        context = {
            "entity": entity_name,
            "type": "global",
            "project_overview": {},
            "module_hierarchy": {},
            "dependency_graph": {},
            "architectural_patterns": [],
            "related_modules": [],
            "impact_analysis": {},
            "tokens_used": 0
        }

        tokens_budget = max_tokens

        # 1. Vue d'ensemble du projet
        context["project_overview"] = await self._get_project_overview(tokens_budget * 0.15)
        tokens_budget -= len(str(context["project_overview"])) // 4

        # 2. Hiérarchie des modules
        context["module_hierarchy"] = await self._get_module_hierarchy(tokens_budget * 0.2)
        tokens_budget -= len(str(context["module_hierarchy"])) // 4

        # 3. Graphe de dépendances centré sur l'entité
        context["dependency_graph"] = await self._build_dependency_subgraph(
            main_chunk_id, depth=2, max_tokens=tokens_budget * 0.3
        )
        tokens_budget -= len(str(context["dependency_graph"])) // 4

        # 4. Analyse d'impact
        if entity_info and tokens_budget > 200:
            context["impact_analysis"] = await self._analyze_impact(
                entity_name, entity_info, tokens_budget * 0.2
            )
            tokens_budget -= len(str(context["impact_analysis"])) // 4

        # 5. Modules liés sémantiquement
        if tokens_budget > 100:
            context["related_modules"] = await self._find_semantically_related_modules(
                entity_name, tokens_budget * 0.15
            )

        context["tokens_used"] = max_tokens - tokens_budget

        return context

    async def _get_project_overview(self, max_tokens: int) -> Dict[str, Any]:
        """Génère une vue d'ensemble du projet"""
        stats = self.entity_index.get_stats()
        all_modules = await self.entity_index.get_all_modules()

        # Analyser la structure du projet
        file_analysis = await self._analyze_file_structure()

        overview = {
            "statistics": {
                "total_entities": stats['total_entities'],
                "modules": stats['modules'],
                "functions": stats['functions'],
                "subroutines": stats['subroutines'],
                "internal_functions": stats['internal_functions'],
                "files": len(file_analysis['files'])
            },
            "main_modules": await self._identify_main_modules(all_modules),
            "file_structure": file_analysis,
            "architectural_style": await self._identify_architectural_style(all_modules)
        }

        return overview

    async def _get_module_hierarchy(self, max_tokens: int) -> Dict[str, Any]:
        """Construit la hiérarchie complète des modules"""
        if self._module_hierarchy_cache:
            return self._module_hierarchy_cache

        all_modules = await self.entity_index.get_all_modules()

        hierarchy = {
            "modules": {},
            "dependency_tree": {},
            "circular_dependencies": [],
            "independent_modules": []
        }

        # Construire l'arbre de dépendances
        for module_name, module_info in all_modules.items():
            hierarchy["modules"][module_name] = {
                "dependencies": module_info['dependencies'],
                "children": module_info['children'],
                "filepath": module_info['filepath'],
                "concepts": [c.get('label', '') for c in module_info.get('concepts', [])][:3]
            }

        # Détecter les dépendances circulaires
        hierarchy["circular_dependencies"] = await self._detect_circular_dependencies(all_modules)

        # Identifier les modules indépendants (pas de dépendances)
        hierarchy["independent_modules"] = [
            name for name, info in all_modules.items()
            if not info['dependencies']
        ]

        # Construire l'arbre de dépendances
        hierarchy["dependency_tree"] = await self._build_dependency_tree(all_modules)

        self._module_hierarchy_cache = hierarchy
        return hierarchy

    async def _build_dependency_subgraph(self, root_chunk_id: str, depth: int = 2, max_tokens: int = 1000) -> Dict[
        str, Any]:
        """Construit un sous-graphe de dépendances centré sur une entité"""

        # Vérifier le cache
        cache_key = f"{root_chunk_id}_{depth}"
        if cache_key in self._dependency_graph_cache:
            return self._dependency_graph_cache[cache_key]

        root_entity_info = await self.entity_index.get_entity_info(root_chunk_id)
        if not root_entity_info:
            return {}

        subgraph = {
            "root_entity": root_entity_info['name'],
            "nodes": {},
            "edges": [],
            "levels": {},
            "summary": {}
        }

        # BFS pour explorer les dépendances
        visited = set()
        queue = deque([(root_chunk_id, 0)])  # (chunk_id, level)

        while queue and len(subgraph["nodes"]) < 20:  # Limiter la taille
            chunk_id, level = queue.popleft()

            if chunk_id in visited or level > depth:
                continue

            visited.add(chunk_id)
            entity_info = await self.entity_index.get_entity_info(chunk_id)

            if not entity_info:
                continue

            entity_name = entity_info['name']

            # Ajouter le noeud
            subgraph["nodes"][entity_name] = {
                "type": entity_info['type'],
                "file": entity_info['filepath'],
                "level": level,
                "chunk_id": chunk_id
            }

            # Ajouter au niveau approprié
            if level not in subgraph["levels"]:
                subgraph["levels"][level] = []
            subgraph["levels"][level].append(entity_name)

            # Explorer les dépendances directes
            dependencies = entity_info.get('dependencies', [])
            for dep in dependencies:
                dep_chunks = await self.entity_index.find_entity(dep)
                if dep_chunks:
                    dep_chunk_id = dep_chunks[0]

                    # Ajouter l'arête
                    subgraph["edges"].append({
                        "from": entity_name,
                        "to": dep,
                        "type": "uses"
                    })

                    # Ajouter à la queue pour exploration
                    if dep_chunk_id not in visited:
                        queue.append((dep_chunk_id, level + 1))

            # Explorer les relations parent-enfant
            children = await self.entity_index.get_children(entity_name)
            for child in children:
                child_chunks = await self.entity_index.find_entity(child)
                if child_chunks:
                    child_chunk_id = child_chunks[0]

                    subgraph["edges"].append({
                        "from": entity_name,
                        "to": child,
                        "type": "contains"
                    })

                    if child_chunk_id not in visited:
                        queue.append((child_chunk_id, level))  # Même niveau pour les enfants

        # Générer le résumé
        subgraph["summary"] = {
            "total_nodes": len(subgraph["nodes"]),
            "total_edges": len(subgraph["edges"]),
            "max_depth": max(subgraph["levels"].keys()) if subgraph["levels"] else 0,
            "node_types": self._count_node_types(subgraph["nodes"])
        }

        # Mettre en cache
        self._dependency_graph_cache[cache_key] = subgraph

        return subgraph

    async def _analyze_impact(self, entity_name: str, entity_info: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Analyse l'impact potentiel de modifications de l'entité"""

        impact = {
            "direct_dependents": [],
            "indirect_dependents": [],
            "affected_modules": set(),
            "risk_level": "low",
            "recommendations": []
        }

        # Trouver qui dépend directement de cette entité
        direct_dependents = await self._find_dependents(entity_name)
        impact["direct_dependents"] = direct_dependents[:10]  # Limiter

        # Analyser les dépendants indirects (niveau 2)
        indirect_dependents = set()
        for dependent in direct_dependents:
            sub_dependents = await self._find_dependents(dependent)
            indirect_dependents.update(sub_dependents)

        impact["indirect_dependents"] = list(indirect_dependents)[:15]

        # Identifier les modules affectés
        all_dependents = direct_dependents + list(indirect_dependents)
        for dependent in all_dependents:
            dep_chunks = await self.entity_index.find_entity(dependent)
            if dep_chunks:
                dep_info = await self.entity_index.get_entity_info(dep_chunks[0])
                if dep_info and dep_info.get('filepath'):
                    # Extraire le nom du module depuis le chemin
                    module = self._extract_module_name(dep_info['filepath'])
                    if module:
                        impact["affected_modules"].add(module)

        impact["affected_modules"] = list(impact["affected_modules"])

        # Évaluer le niveau de risque
        total_dependents = len(direct_dependents) + len(indirect_dependents)
        if total_dependents == 0:
            impact["risk_level"] = "low"
        elif total_dependents < 5:
            impact["risk_level"] = "medium"
        else:
            impact["risk_level"] = "high"

        # Générer des recommandations
        impact["recommendations"] = self._generate_impact_recommendations(
            entity_info, total_dependents, impact["affected_modules"]
        )

        return impact

    async def _find_semantically_related_modules(self, entity_name: str, max_tokens: int) -> List[Dict[str, Any]]:
        """Trouve les modules sémantiquement liés via embeddings"""
        try:
            # Récupérer l'entité principale pour construire une requête
            entity_chunks = await self.entity_index.find_entity(entity_name)
            if not entity_chunks:
                return []

            main_chunk = await self._get_chunk_by_id(entity_chunks[0])
            if not main_chunk:
                return []

            # Utiliser le RAG pour trouver des chunks similaires
            similar_chunks = await self.rag_engine.find_similar(
                main_chunk['text'],
                max_results=15,
                min_similarity=0.6
            )

            related_modules = []
            seen_modules = set()

            for chunk_id, similarity in similar_chunks:
                if chunk_id == main_chunk['id']:  # Skip l'entité elle-même
                    continue

                entity_info = await self.entity_index.get_entity_info(chunk_id)
                if not entity_info:
                    continue

                module_name = self._extract_module_name(entity_info.get('filepath', ''))
                if module_name and module_name not in seen_modules:
                    seen_modules.add(module_name)

                    related_modules.append({
                        "module": module_name,
                        "entity": entity_info['name'],
                        "similarity": similarity,
                        "type": entity_info['type'],
                        "concepts": [c.get('label', '') for c in entity_info.get('concepts', [])][:2]
                    })

            return related_modules[:8]  # Limiter à 8 modules

        except Exception as e:
            logger.debug(f"Semantic search failed for {entity_name}: {e}")
            return []

    async def _analyze_file_structure(self) -> Dict[str, Any]:
        """Analyse la structure des fichiers du projet"""
        stats = self.entity_index.get_stats()

        # Collecter tous les fichiers
        files = set()
        for chunk_id, entity_info in self.entity_index.chunk_to_entity.items():
            filepath = entity_info.get('filepath')
            if filepath:
                files.add(filepath)

        file_analysis = {
            "total_files": len(files),
            "files": list(files),
            "file_types": self._analyze_file_types(files),
            "directory_structure": self._analyze_directory_structure(files)
        }

        return file_analysis

    def _analyze_file_types(self, files: Set[str]) -> Dict[str, int]:
        """Analyse les types de fichiers"""
        types = defaultdict(int)

        for filepath in files:
            ext = filepath.split('.')[-1].lower() if '.' in filepath else 'no_extension'
            types[ext] += 1

        return dict(types)

    def _analyze_directory_structure(self, files: Set[str]) -> Dict[str, List[str]]:
        """Analyse la structure des répertoires"""
        dirs = defaultdict(list)

        for filepath in files:
            if '/' in filepath:
                directory = '/'.join(filepath.split('/')[:-1])
                filename = filepath.split('/')[-1]
                dirs[directory].append(filename)
            else:
                dirs['.'].append(filepath)

        return dict(dirs)

    async def _identify_main_modules(self, all_modules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifie les modules principaux du projet"""
        main_modules = []

        for module_name, module_info in all_modules.items():
            # Critères pour identifier un module principal :
            # 1. Beaucoup d'enfants (fonctions/subroutines)
            # 2. Peu ou pas de dépendances
            # 3. Nom suggestif (main, core, etc.)

            score = 0

            # Nombre d'enfants
            children_count = len(module_info.get('children', []))
            score += children_count * 2

            # Peu de dépendances
            deps_count = len(module_info.get('dependencies', []))
            score += max(0, 5 - deps_count)

            # Nom suggestif
            name_lower = module_name.lower()
            if any(keyword in name_lower for keyword in ['main', 'core', 'base', 'util']):
                score += 5

            main_modules.append({
                "name": module_name,
                "score": score,
                "children_count": children_count,
                "dependencies_count": deps_count,
                "filepath": module_info.get('filepath', '')
            })

        # Trier par score et retourner le top 5
        main_modules.sort(key=lambda x: x['score'], reverse=True)
        return main_modules[:5]

    async def _identify_architectural_style(self, all_modules: Dict[str, Any]) -> str:
        """Identifie le style architectural du projet"""
        total_modules = len(all_modules)

        if total_modules == 1:
            return "monolithic"

        # Analyser les patterns de dépendances
        total_dependencies = sum(len(info['dependencies']) for info in all_modules.values())
        avg_dependencies = total_dependencies / total_modules if total_modules > 0 else 0

        # Identifier les modules sans dépendances (utilities)
        independent_modules = sum(1 for info in all_modules.values() if not info['dependencies'])

        # Heuristiques pour le style architectural
        if avg_dependencies < 1:
            return "loosely_coupled"
        elif avg_dependencies > 3:
            return "highly_coupled"
        elif independent_modules > total_modules * 0.3:
            return "layered_with_utilities"
        else:
            return "modular"

    async def _detect_circular_dependencies(self, all_modules: Dict[str, Any]) -> List[List[str]]:
        """Détecte les dépendances circulaires entre modules"""
        circular_deps = []

        # Construire le graphe de dépendances
        graph = {}
        for module_name, module_info in all_modules.items():
            graph[module_name] = module_info['dependencies']

        # DFS pour détecter les cycles
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node):
            if node in rec_stack:
                # Cycle détecté
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                circular_deps.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor in graph:  # Le module existe
                    dfs(neighbor)

            rec_stack.remove(node)
            path.pop()

        for module in graph:
            if module not in visited:
                dfs(module)

        return circular_deps

    async def _build_dependency_tree(self, all_modules: Dict[str, Any]) -> Dict[str, Any]:
        """Construit l'arbre de dépendances des modules"""
        # Trouver les modules racines (sans dépendances)
        root_modules = [name for name, info in all_modules.items() if not info['dependencies']]

        tree = {
            "roots": root_modules,
            "levels": {},
            "orphans": []
        }

        # BFS pour construire les niveaux
        visited = set()
        queue = deque([(module, 0) for module in root_modules])

        while queue:
            module, level = queue.popleft()

            if module in visited:
                continue

            visited.add(module)

            if level not in tree["levels"]:
                tree["levels"][level] = []
            tree["levels"][level].append(module)

            # Trouver les modules qui dépendent de celui-ci
            dependents = [name for name, info in all_modules.items()
                          if module in info['dependencies']]

            for dependent in dependents:
                if dependent not in visited:
                    queue.append((dependent, level + 1))

        # Identifier les modules orphelins (non atteignables)
        tree["orphans"] = [name for name in all_modules.keys() if name not in visited]

        return tree

    async def _find_dependents(self, entity_name: str) -> List[str]:
        """Trouve tous les entités qui dépendent de l'entité donnée"""
        dependents = []

        # Chercher dans tous les chunks
        for chunk_id, entity_info in self.entity_index.chunk_to_entity.items():
            # 1. Dépendances USE existantes
            dependencies = entity_info.get('dependencies', [])
            if entity_name in dependencies:
                dependents.append(entity_info['name'])

            # 2. Relations parent-enfant existantes
            if entity_info.get('parent') == entity_name:
                dependents.append(entity_info['name'])

            # 3. NOUVEAU : Appels de fonctions depuis le cache
            if hasattr(self.entity_index, 'call_patterns_cache'):
                calls = self.entity_index.call_patterns_cache.get(chunk_id, [])
                if entity_name in calls:
                    dependents.append(entity_info['name'])

        return list(set(dependents))

    async def old_find_dependents(self, entity_name: str) -> List[str]:
        """Trouve tous les entités qui dépendent de l'entité donnée"""
        dependents = []

        # Chercher dans tous les chunks
        for chunk_id, entity_info in self.entity_index.chunk_to_entity.items():
            dependencies = entity_info.get('dependencies', [])

            if entity_name in dependencies:
                dependents.append(entity_info['name'])

            # Chercher aussi dans le parent (pour les fonctions internes)
            if entity_info.get('parent') == entity_name:
                dependents.append(entity_info['name'])

        return list(set(dependents))  # Retirer les doublons

    def _extract_module_name(self, filepath: str) -> Optional[str]:
        """Extrait le nom du module depuis le chemin du fichier"""
        if not filepath:
            return None

        filename = filepath.split('/')[-1]

        # Retirer l'extension
        if '.' in filename:
            module_name = '.'.join(filename.split('.')[:-1])
        else:
            module_name = filename

        return module_name

    def _count_node_types(self, nodes: Dict[str, Any]) -> Dict[str, int]:
        """Compte les types de noeuds dans un graphe"""
        type_counts = defaultdict(int)

        for node_info in nodes.values():
            node_type = node_info.get('type', 'unknown')
            type_counts[node_type] += 1

        return dict(type_counts)

    def _generate_impact_recommendations(self, entity_info: Dict[str, Any],
                                         total_dependents: int,
                                         affected_modules: List[str]) -> List[str]:
        """Génère des recommandations basées sur l'analyse d'impact"""
        recommendations = []

        entity_type = entity_info.get('type', '')

        if total_dependents == 0:
            recommendations.append("Cette entité semble isolée - modification à faible risque")
        elif total_dependents < 3:
            recommendations.append("Impact modéré - tester les dépendants directs")
        else:
            recommendations.append("Impact élevé - tests complets recommandés")

        if len(affected_modules) > 1:
            recommendations.append(f"Modification affectera {len(affected_modules)} modules - coordination nécessaire")

        if entity_type == 'module':
            recommendations.append("Modification d'un module - vérifier l'interface publique")
        elif entity_type in ['function', 'subroutine']:
            recommendations.append("Modification d'une fonction - vérifier la signature et les contrats")

        if entity_info.get('is_internal'):
            recommendations.append("Fonction interne - impact limité au module parent")

        return recommendations

    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk par son ID depuis le document store"""
        parts = chunk_id.split('-chunk-')
        if len(parts) != 2:
            return None

        document_id = parts[0]

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

        similar = [name for name in all_names
                   if entity_name.lower() in name.lower() or name.lower() in entity_name.lower()]

        return similar[:limit]