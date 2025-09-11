"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue - Jupyter Explorer
    ------------------------------------------
    """

# jupyter_analysis/core/entity_explorer.py
"""
Classe d'interface pour l'exploration et la consultation des entités Jupyter.
Fournit une API de haut niveau au-dessus de l'EntityManager pour les notebooks.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from rapidfuzz import fuzz

from ..fortran_analysis.core.entity_manager import EntityManager, UnifiedEntity
from ..ontology.ontology_manager import OntologyManager

logger = logging.getLogger(__name__)


class JupyterEntityExplorer:
    """
    Fournit des méthodes simples pour consulter les informations
    détaillées d'une entité Jupyter et de ses relations.
    """

    def __init__(self, entity_manager: EntityManager, ontology_manager: OntologyManager):
        """
        Initialise l'explorateur avec une instance de EntityManager.

        Args:
            entity_manager: Une instance de EntityManager déjà initialisée.
            ontology_manager: Gestionnaire d'ontologie pour la classification.
        """
        self.ontology_manager = ontology_manager
        self.em = entity_manager
        logger.info("✅ JupyterEntityExplorer initialisé.")

    def __repr__(self) -> str:
        return f"<JupyterEntityExplorer with {len(self.em.entities)} entities>"

    async def find_entity_by_name(self, entity_name: str) -> Dict[str, Any]:
        """Trouve une entité par nom et retourne ses informations."""
        entity = await self.em.find_entity(entity_name)
        return entity.to_dict() if entity else {}

    async def get_full_report(self, entity_name: str, include_source_code: bool = True) -> Dict[str, Any]:
        """
        Génère un rapport complet pour une entité Jupyter donnée.

        Args:
            entity_name: Le nom de l'entité à rechercher.
            include_source_code: Si True, inclut le code source de l'entité dans le rapport.

        Returns:
            Un dictionnaire contenant un rapport détaillé ou un message d'erreur.
        """
        logger.info(f"🔍 Génération du rapport complet pour l'entité Jupyter '{entity_name}'...")

        entity = await self.em.find_entity(entity_name)
        if not entity:
            logger.warning(f"Entité Jupyter '{entity_name}' non trouvée.")
            return {"error": f"Entity '{entity_name}' not found."}

        # Lancer toutes les requêtes en parallèle
        tasks = {
            "references": self.get_references(entity.entity_name),
            "local_context": self.get_local_context(entity, include_source_code),
            "global_context": self.get_global_context(entity)
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Vérifier si des tâches ont échoué
        for task_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Erreur lors de la récupération de '{task_name}': {result}")
                tasks[task_name] = {"error": str(result)}
            else:
                tasks[task_name] = result

        # Assembler le rapport final
        report = {
            "entity_name": entity.entity_name,
            "entity_id": entity.entity_id,
            "summary": self._get_basic_info(entity),
            "rich_signature": self._format_rich_signature(entity),
            "outgoing_relations": self.get_imports_and_calls(entity),
            "incoming_relations": tasks["references"],
            "local_context": tasks["local_context"],
            "global_context": tasks["global_context"],
            "detected_concepts": entity.detected_concepts,
            "notebook_summary": getattr(entity, 'notebook_summary', ''),
            "entity_role": getattr(entity, 'entity_role', 'default')
        }

        logger.info(f"✅ Rapport Jupyter généré pour '{entity_name}'.")
        return report

    def _format_rich_signature(self, entity: UnifiedEntity) -> str:
        """
        Formate une signature lisible pour les entités Python/Jupyter.
        """
        if entity.entity_type == 'notebook':
            return f"📓 Notebook: {entity.entity_name}"

        elif entity.entity_type in ['code_cell', 'markdown_cell']:
            return f"📄 {entity.entity_type.replace('_', ' ').title()}: {entity.entity_name}"

        elif entity.entity_type in ['function', 'async function']:
            signature = entity.signature or f"{entity.entity_type} {entity.entity_name}()"

            # Ajouter des informations sur les arguments si disponibles
            if hasattr(entity, 'arguments') and entity.arguments:
                args_info = []
                for arg in entity.arguments:
                    arg_name = arg.get('name', 'unknown')
                    arg_type = arg.get('type', '')
                    if arg_type:
                        args_info.append(f"{arg_name}: {arg_type}")
                    else:
                        args_info.append(arg_name)

                if args_info:
                    signature += f"\n  # Arguments: {', '.join(args_info)}"

            # Ajouter le type de retour si disponible
            if hasattr(entity, 'return_type') and entity.return_type:
                signature += f"\n  # Returns: {entity.return_type}"

            return signature

        elif entity.entity_type == 'class':
            signature = entity.signature or f"class {entity.entity_name}:"

            # Ajouter les méthodes si c'est une classe
            children = []
            try:
                # Essayer de récupérer les enfants (méthodes)
                loop = asyncio.get_event_loop()
                children = loop.run_until_complete(self.em.get_children(entity.entity_id))
            except:
                pass

            if children:
                methods = [child.entity_name for child in children if
                           child.entity_type in ['function', 'async function']]
                if methods:
                    signature += f"\n  # Methods: {', '.join(methods[:5])}"
                    if len(methods) > 5:
                        signature += f" (and {len(methods) - 5} more...)"

            return signature

        else:
            return entity.signature or f"{entity.entity_type}: {entity.entity_name}"

    def _get_basic_info(self, entity: UnifiedEntity) -> Dict[str, Any]:
        """Retourne les informations de base d'une entité Jupyter."""
        return {
            "type": entity.entity_type,
            "filepath": entity.filepath,
            "start_line": entity.start_line,
            "end_line": entity.end_line,
            "parent_entity": entity.parent_entity,
            "entity_role": getattr(entity, 'entity_role', 'default'),
            "signature": entity.signature or "Non disponible",
            "has_notebook_summary": bool(getattr(entity, 'notebook_summary', '')),
        }

    def get_imports_and_calls(self, entity: UnifiedEntity) -> Dict[str, List[Any]]:
        """
        Retourne les imports et appels de fonction de l'entité (relations sortantes).
        """
        # Séparer les imports des appels de fonction
        imports = []
        function_calls = []

        # Dependencies contient généralement les imports
        for dep in entity.dependencies:
            if isinstance(dep, dict):
                dep_name = dep.get('name', '')
                if dep_name.startswith(('.', 'from ', 'import ')):
                    imports.append(dep)
                else:
                    imports.append(dep)
            else:
                imports.append({'name': str(dep), 'line': 0})

        # Called_functions contient les appels de fonction
        for call in entity.called_functions:
            if isinstance(call, dict):
                function_calls.append(call)
            else:
                function_calls.append({'name': str(call), 'line': 0})

        # Trier par numéro de ligne
        imports.sort(key=lambda x: x.get('line', 0))
        function_calls.sort(key=lambda x: x.get('line', 0))

        return {
            "imports": imports,
            "function_calls": function_calls,
        }

    async def get_references(self, entity_name: str) -> List[Dict[str, str]]:
        """
        Trouve où cette entité est référencée (équivalent de get_callers pour Jupyter).
        Pour Jupyter, cela peut être des références dans d'autres cellules.
        """
        references = []
        all_entities = self.em.get_all_entities()

        entity_name_lower = entity_name.lower()

        for entity in all_entities:
            if entity.entity_name.lower() == entity_name_lower:
                continue

            # Vérifier dans les appels de fonction
            for call in entity.called_functions:
                call_name = ''
                if isinstance(call, dict):
                    call_name = call.get('name', '').lower()
                else:
                    call_name = str(call).lower()

                if entity_name_lower in call_name:
                    references.append({
                        'name': entity.entity_name,
                        'type': entity.entity_type,
                        'file': entity.filename if hasattr(entity, 'filename') else 'unknown',
                        'cell_type': entity.entity_type,
                        'parent': entity.parent_entity or 'N/A'
                    })
                    break

            # Vérifier dans le code source si disponible
            if hasattr(entity, 'source_code') and entity.source_code:
                if entity_name in entity.source_code:
                    references.append({
                        'name': entity.entity_name,
                        'type': entity.entity_type,
                        'file': entity.filename if hasattr(entity, 'filename') else 'unknown',
                        'cell_type': entity.entity_type,
                        'parent': entity.parent_entity or 'N/A'
                    })

        return sorted(references, key=lambda x: x['name'])

    async def get_local_context(self, entity: UnifiedEntity, include_source: bool) -> Dict[str, Any]:
        """
        Fournit le contexte local : ce qui est DANS l'entité (contenu de la cellule).
        """
        # Obtenir les enfants (pour les cellules, ce sont les fonctions/classes définies)
        children_entities = await self.em.get_children(entity.entity_id)
        children_summary = [
            {
                "name": child.entity_name,
                "type": child.entity_type,
                "role": getattr(child, 'entity_role', 'default')
            }
            for child in children_entities
        ]

        # Pour Jupyter, le code source est déjà disponible dans l'entité
        source_code = "Non demandé."
        if include_source:
            if hasattr(entity, 'source_code') and entity.source_code:
                source_code = entity.source_code
            else:
                source_code = "Code source non disponible pour cette entité."

        # Information spécifique aux notebooks
        notebook_info = {}
        if entity.entity_type == 'notebook':
            notebook_info = {
                "notebook_summary": getattr(entity, 'notebook_summary', ''),
                "total_cells": len(children_entities)
            }
        elif entity.entity_type in ['code_cell', 'markdown_cell']:
            notebook_info = {
                "cell_role": getattr(entity, 'entity_role', 'default'),
                "notebook_summary": getattr(entity, 'notebook_summary', ''),
                "defined_elements": len(children_entities)
            }

        return {
            "children_entities": sorted(children_summary, key=lambda x: x['name']),
            "source_code": source_code,
            "notebook_info": notebook_info
        }

    async def get_global_context(self, entity: UnifiedEntity) -> Dict[str, Any]:
        """
        Fournit le contexte global : où se situe l'entité (notebook, autres cellules).
        """
        # Obtenir le parent
        parent_entity = await self.em.get_parent(entity.entity_id)
        parent_summary = "Aucun (entité de haut niveau)."
        if parent_entity:
            parent_summary = {
                "name": parent_entity.entity_name,
                "type": parent_entity.entity_type,
                "role": getattr(parent_entity, 'entity_role', 'default')
            }

        # Obtenir les autres entités du même notebook
        siblings = []
        if entity.filepath:
            all_in_file = await self.em.get_entities_in_file(entity.filepath)
            siblings = [
                {
                    "name": e.entity_name,
                    "type": e.entity_type,
                    "role": getattr(e, 'entity_role', 'default')
                }
                for e in all_in_file
                if e.entity_id != entity.entity_id and e.parent_entity == entity.parent_entity
            ]

        # Information sur le notebook parent
        notebook_context = {}
        if entity.entity_type != 'notebook':
            # Chercher le notebook parent
            current_entity = entity
            while current_entity.parent_entity:
                parent = await self.em.find_entity(current_entity.parent_entity)
                if parent and parent.entity_type == 'notebook':
                    notebook_context = {
                        "notebook_name": parent.entity_name,
                        "notebook_summary": getattr(parent, 'notebook_summary', ''),
                    }
                    break
                current_entity = parent if parent else current_entity
                if not parent:
                    break

        return {
            "parent_entity": parent_summary,
            "sibling_entities": sorted(siblings, key=lambda x: x['name']),
            "notebook_context": notebook_context
        }

    async def find_entities_by_criteria(
            self,
            fuzzy_threshold: int = 85,
            **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Recherche des entités Jupyter en fonction de critères dynamiques.
        Adapté pour les spécificités des notebooks.
        """
        logger.info(f"🚀 Lancement de la recherche par critères Jupyter: {kwargs}")

        # Traduction des concepts comme pour Fortran
        target_concept_labels = []
        all_entities = self.em.get_all_entities()

        if 'detected_concept' in kwargs and kwargs['detected_concept'] is not None:
            user_query = kwargs['detected_concept']
            logger.info(f"Traduction de la requête de concept '{user_query}' en concept officiel...")

            classifier = getattr(self.ontology_manager, 'classifier', None)
            if not classifier:
                logger.error(
                    "Ontology classifier non disponible. Impossible de faire une recherche sémantique de concept.")
                kwargs.pop('detected_concept')
            else:
                try:
                    embedding = await classifier.rag_engine.embedding_manager.provider.generate_embeddings([user_query])
                    detected_concepts = await classifier.concept_classifier.auto_detect_concepts(
                        query_embedding=embedding[0],
                        min_confidence=0.6,
                        max_concepts=10
                    )

                    if detected_concepts:
                        target_concept_labels = [
                            concept.get('label')
                            for concept in detected_concepts[:1]
                            if concept.get('label')
                        ]
                        logger.info(f"Concepts cibles identifiés: {target_concept_labels}")

                        # Filtrer les entités par concept
                        if target_concept_labels:
                            target_set = set(target_concept_labels)
                            all_entities = [
                                e for e in all_entities
                                if
                                not target_set.isdisjoint({c.get('label') for c in getattr(e, 'detected_concepts', [])})
                            ]
                            logger.info(f"Après filtrage par concept: {len(all_entities)} entités restantes")

                    if not target_concept_labels:
                        logger.warning(f"Aucun concept officiel trouvé pour '{user_query}'.")
                        return []

                except Exception as e:
                    logger.error(f"Erreur lors de la traduction du concept '{user_query}': {e}")
                    return []

            kwargs.pop('detected_concept', None)

        # Si aucun critère restant, retourner toutes les entités
        if not kwargs:
            return [{"entity": e, "score": 100.0} for e in all_entities]

        # Définir les champs de recherche pour Jupyter
        exact_match_fields = ['entity_type', 'parent_entity', 'entity_role']
        fuzzy_match_fields = ['entity_name']
        substring_match_fields = ['filename', 'filepath', 'source_code']
        set_containment_fields = ['dependencies', 'called_functions']
        boolean_fields = []  # Pas de booléens spécifiques pour Jupyter pour l'instant

        scored_matches = []

        for entity in all_entities:
            is_match = True
            relevance_scores = []

            for key, value in kwargs.items():
                if value is None:
                    continue

                if not hasattr(entity, key):
                    is_match = False
                    break

                entity_value = getattr(entity, key)

                # Logique de correspondance par type de champ
                if key in exact_match_fields:
                    if str(entity_value).lower() != str(value).lower():
                        is_match = False
                        break
                    relevance_scores.append(100)

                elif key in fuzzy_match_fields:
                    score = fuzz.ratio(str(value).lower(), str(entity_value).lower())
                    if score < fuzzy_threshold:
                        is_match = False
                        break
                    relevance_scores.append(score)

                elif key in substring_match_fields:
                    if str(value).lower() not in str(entity_value).lower():
                        is_match = False
                        break
                    relevance_scores.append((len(str(value)) / len(str(entity_value))) * 100)

                elif key in boolean_fields:
                    search_bool = str(value).lower() in ('true', '1', 't', 'y', 'yes')
                    if entity_value != search_bool:
                        is_match = False
                        break
                    relevance_scores.append(100)

                elif key in set_containment_fields:
                    if isinstance(entity_value, (list, set)):
                        # Pour Jupyter, vérifier les noms dans les dictionnaires
                        found = False
                        search_value = str(value).lower()
                        for item in entity_value:
                            if isinstance(item, dict):
                                item_name = str(item.get('name', '')).lower()
                                if search_value in item_name:
                                    found = True
                                    break
                            elif search_value in str(item).lower():
                                found = True
                                break

                        if not found:
                            is_match = False
                            break
                    else:
                        is_match = False
                        break
                    relevance_scores.append(100)

                else:
                    if str(entity_value) != str(value):
                        is_match = False
                        break
                    relevance_scores.append(100)

            if is_match:
                final_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 100
                scored_matches.append({"entity": entity, "score": final_score})

        sorted_results = sorted(scored_matches, key=lambda x: x['score'], reverse=True)
        logger.info(f"✅ Recherche Jupyter terminée. {len(sorted_results)} entité(s) trouvée(s).")
        return sorted_results

    async def get_notebook_overview(self, notebook_name: str) -> Dict[str, Any]:
        """
        Génère une vue d'ensemble d'un notebook complet.
        """
        notebook_entity = await self.em.find_entity(notebook_name)
        if not notebook_entity or notebook_entity.entity_type != 'notebook':
            return {"error": f"Notebook '{notebook_name}' non trouvé."}

        # Obtenir toutes les cellules du notebook
        cells = await self.em.get_children(notebook_entity.entity_id)

        # Classifier les cellules
        code_cells = [c for c in cells if c.entity_type == 'code_cell']
        markdown_cells = [c for c in cells if c.entity_type == 'markdown_cell']

        # Obtenir les fonctions et classes définies
        functions = []
        classes = []
        imports = set()

        for cell in code_cells:
            cell_children = await self.em.get_children(cell.entity_id)
            for child in cell_children:
                if child.entity_type in ['function', 'async function']:
                    functions.append(child.entity_name)
                elif child.entity_type == 'class':
                    classes.append(child.entity_name)

            # Collecter les imports
            for dep in cell.dependencies:
                if isinstance(dep, dict):
                    imports.add(dep.get('name', ''))
                else:
                    imports.add(str(dep))

        return {
            "notebook_name": notebook_entity.entity_name,
            "notebook_summary": getattr(notebook_entity, 'notebook_summary', ''),
            "statistics": {
                "total_cells": len(cells),
                "code_cells": len(code_cells),
                "markdown_cells": len(markdown_cells),
                "functions_defined": len(functions),
                "classes_defined": len(classes),
                "unique_imports": len(imports)
            },
            "defined_functions": functions,
            "defined_classes": classes,
            "imports": list(imports),
            "cell_structure": [
                {
                    "name": cell.entity_name,
                    "type": cell.entity_type,
                    "role": getattr(cell, 'entity_role', 'default'),
                    "line_range": f"{cell.start_line}-{cell.end_line}"
                }
                for cell in cells
            ]
        }