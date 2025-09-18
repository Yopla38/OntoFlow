"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# fortran_analysis/core/entity_explorer.py
"""
Classe d'interface pour l'exploration et la consultation des entités Fortran.
Fournit une API de haut niveau au-dessus de l'EntityManager.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from rapidfuzz import fuzz

from ..core.entity_manager import EntityManager, UnifiedEntity
from ...ontology.ontology_manager import OntologyManager

logger = logging.getLogger(__name__)


class FortranEntityExplorer:
    """
    Fournit des méthodes simples pour consulter les informations
    détaillées d'une entité Fortran et de ses relations.
    """

    def __init__(self, entity_manager: EntityManager, ontology_manager: OntologyManager):
        """
        Initialise l'explorateur avec une instance de EntityManager.

        Args:
            entity_manager: Une instance de EntityManager déjà initialisée.
        """
        """
        if not isinstance(entity_manager, EntityManager) or not entity_manager._initialized:
            raise ValueError("L'EntityManager doit être fourni et initialisé.")
        """
        self.ontology_manager = ontology_manager
        self.em = entity_manager
        logger.info("✅ FortranEntityExplorer initialisé.")

    def __repr__(self) -> str:
        return f"<FortranEntityExplorer with {len(self.em.entities)} entities>"

    async def find_entity_by_name(self, entity_name: str) -> Dict[str, Any]:
        entity = await self.em.find_entity(entity_name)
        return entity.to_dict()

    async def get_full_report(self, entity_name: str, include_source_code: bool = True) -> Dict[str, Any]:
        """
        Génère un rapport complet pour une entité donnée.
        C'est la méthode principale à utiliser pour obtenir toutes les informations.

        Args:
            entity_name: Le nom de l'entité à rechercher.
            include_source_code: Si True, inclut le code source de l'entité dans le rapport.

        Returns:
            Un dictionnaire contenant un rapport détaillé ou un message d'erreur.
        """
        logger.info(f"🔍 Génération du rapport complet pour l'entité '{entity_name}'...")

        entity = await self.em.find_entity(entity_name)
        if not entity:
            logger.warning(f"Entité '{entity_name}' non trouvée.")
            return {"error": f"Entity '{entity_name}' not found."}

        # Lancer toutes les requêtes en parallèle
        tasks = {
            "callers": self.get_callers(entity.entity_name),
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
            "outgoing_relations": self.get_callees_and_dependencies(entity),
            "incoming_relations": tasks["callers"],
            "local_context": tasks["local_context"],
            "global_context": tasks["global_context"],
            "detected_concepts": entity.detected_concepts
        }

        logger.info(f"✅ Rapport généré pour '{entity_name}'.")
        return report

    def _format_rich_signature(self, entity: UnifiedEntity) -> str:
        """
        Formate une signature multi-lignes lisible à partir des
        données structurées de l'entité.
        """
        # Si pas d'arguments, retourner la signature simple
        if not entity.arguments and not entity.return_type:

            return entity.signature or "Signature non disponible."

        output = [entity.signature]  # Commence avec la ligne de déclaration

        if entity.arguments:
            output.append("  ! Arguments:")
            for arg in entity.arguments:
                arg_name = arg['name']
                arg_type = arg.get('type', 'unknown')

                # Formater les attributs (intent, dimension, etc.)
                attrs = arg.get('attributes', [])
                intent = next((a for a in attrs if "INTENT" in a), None)
                other_attrs = [a for a in attrs if "INTENT" not in a]

                attr_str = ""
                if intent:
                    attr_str += f"{intent.upper():<15}"  # ex: INTENT(IN)
                if other_attrs:
                    attr_str += ", ".join(other_attrs)

                line = f"  !   - {arg_type:<25} :: {attr_str:<25} :: {arg_name}"
                output.append(line)

        if entity.return_type:
            output.append(f"  ! Returns: {entity.return_type}")

        return "\n".join(output)

    def _get_basic_info(self, entity: UnifiedEntity) -> Dict[str, Any]:
        """Retourne les informations de base d'une entité."""
        return {
            "type": entity.entity_type,
            "filepath": entity.filepath,
            "start_line": entity.start_line,
            "end_line": entity.end_line,
            "access_level": entity.access_level,
            "is_grouped_part": entity.is_grouped,
            "signature": entity.signature or "Non disponible",
        }

    def get_callees_and_dependencies(self, entity: UnifiedEntity) -> Dict[str, List[Any]]: # Le type de retour change
        """
        Retourne ce que l'entité appelle (ses relations sortantes).
        MAINTENANT AVEC NUMÉROS DE LIGNE.
        """
        # On trie les appels par numéro de ligne pour un affichage logique
        sorted_calls = sorted(entity.called_functions, key=lambda x: x.get('line', 0))


        return {
            "called_functions_or_subroutines": sorted_calls, # Renvoie la liste de dicts
            "module_dependencies (USE)": entity.dependencies,
        }

    async def get_callers(self, entity_name: str) -> List[Dict[str, str]]:
        """
        Trouve qui appelle cette entité (relations entrantes).

        Args:
            entity_name: Le nom de l'entité.

        Returns:
            Une liste d'entités qui appellent celle-ci.
        """
        # Utilise directement la méthode optimisée et mise en cache de l'EntityManager
        callers = await self.em.find_entity_callers(entity_name)
        return sorted(callers, key=lambda x: x['name'])

    async def get_local_context(self, entity: UnifiedEntity, include_source: bool) -> Dict[str, Any]:
        """
        Fournit le contexte local : ce qui est DANS l'entité (enfants, code source).

        Args:
            entity: L'objet UnifiedEntity.
            include_source: Si True, lit le fichier pour extraire le code source.

        Returns:
            Un dictionnaire décrivant le contexte local.
        """
        # Obtenir les enfants
        children_entities = await self.em.get_children(entity.entity_id)
        children_summary = [
            {"name": child.entity_name, "type": child.entity_type}
            for child in children_entities
        ]

        # Obtenir le code source
        source_code = "Non demandé."
        if include_source:
            try:
                source_code = self._read_source_code(entity.filepath, entity.start_line, entity.end_line)
            except FileNotFoundError:
                source_code = f"Erreur : Fichier source non trouvé à l'emplacement '{entity.filepath}'."
            except Exception as e:
                source_code = f"Erreur lors de la lecture du code source : {e}"

        return {
            "children_entities": sorted(children_summary, key=lambda x: x['name']),
            "source_code": source_code
        }

    async def get_global_context(self, entity: UnifiedEntity) -> Dict[str, Any]:
        """
        Fournit le contexte global : où se situe l'entité (parent, autres entités du fichier).

        Args:
            entity: L'objet UnifiedEntity.

        Returns:
            Un dictionnaire décrivant le contexte global.
        """
        # Obtenir le parent
        parent_entity = await self.em.get_parent(entity.entity_id)
        parent_summary = "Aucun (entité de haut niveau)."
        if parent_entity:
            parent_summary = {"name": parent_entity.entity_name, "type": parent_entity.entity_type}

        # Obtenir les autres entités du même fichier
        siblings = []
        if entity.filepath:
            all_in_file = await self.em.get_entities_in_file(entity.filepath)
            siblings = [
                {"name": e.entity_name, "type": e.entity_type}
                for e in all_in_file
                if e.entity_id != entity.entity_id and not e.parent_entity  # Uniquement les "frères" de haut niveau
            ]

        return {
            "parent_entity": parent_summary,
            "other_top_level_entities_in_file": sorted(siblings, key=lambda x: x['name'])
        }

    def _read_source_code(self, filepath: str, start_line: int, end_line: int) -> str:
        """Helper pour lire une section spécifique d'un fichier source."""
        if not filepath:
            return "Chemin de fichier non disponible."

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Les lignes sont 1-based, les indices de liste sont 0-based
        start_index = max(0, start_line - 1)
        end_index = min(len(lines), end_line)

        return "".join(lines[start_index:end_index])

    async def find_entities_by_criteria(
            self,
            fuzzy_threshold: int = 85,
            **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Recherche des entités en fonction de critères dynamiques, avec une gestion
        de la similarité, des booléens et de la contenance dans des listes/sets.
        """
        logger.info(f"🚀 Lancement de la recherche par critères: {kwargs}")

        # ÉTAPE A : Traduire la requête utilisateur en concept officiel ET filtrer
        target_concept_labels = []
        all_entities = self.em.get_all_entities()

        if 'detected_concept' in kwargs and kwargs['detected_concept'] is not None:
            user_query = kwargs['detected_concept']
            logger.info(f"Traduction de la requête de concept '{user_query}' en concept officiel...")

            classifier = getattr(self.ontology_manager, 'classifier', None)
            if not classifier:
                logger.error(
                    "Ontology classifier non disponible. Impossible de faire une recherche sémantique de concept.")
                # Retirer le critère de concept et continuer avec les autres critères
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
                        #print(detected_concepts)
                        target_concept_labels = [
                            concept.get('label')
                            for concept in detected_concepts[:1]  # on choisit que le premier candidat
                            if concept.get('label')
                        ]
                        logger.info(f"Concepts cibles identifiés: {target_concept_labels}")

                        # FILTRER LES ENTITÉS PAR CONCEPT ICI, UNE SEULE FOIS
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

            # Retirer 'detected_concept' des kwargs pour éviter qu'il soit traité à nouveau
            kwargs.pop('detected_concept', None)

        # Si aucun critère restant, retourner toutes les entités (éventuellement filtrées par concept)
        if not kwargs:
            return [{"entity": e, "score": 100.0} for e in all_entities]

        # ÉTAPE B : Appliquer les autres filtres
        exact_match_fields = ['entity_type', 'access_level', 'parent_entity']
        fuzzy_match_fields = ['entity_name']
        substring_match_fields = ['filename', 'filepath']
        set_containment_fields = ['dependencies', 'called_functions']
        boolean_fields = ['is_grouped']

        scored_matches = []

        for entity in all_entities:  # all_entities est maintenant déjà filtré par concept si nécessaire
            is_match = True
            relevance_scores = []

            for key, value in kwargs.items():
                if value is None:
                    continue

                if not hasattr(entity, key):
                    is_match = False
                    break

                entity_value = getattr(entity, key)

                # Logique de correspondance par type de champ (inchangée)
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
                    if not isinstance(entity_value, (list, set)) or str(value).lower() not in {str(item).lower() for
                                                                                               item in entity_value}:
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
        logger.info(f"✅ Recherche terminée. {len(sorted_results)} entité(s) trouvée(s).")
        return sorted_results