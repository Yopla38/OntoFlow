"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import re

import networkx as nx
from pyvis.network import Network
import os
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging


class FortranDependencyVisualizer:
    """
    Visualiseur de dépendances Fortran utilisant les données du context_provider.
    Regroupe automatiquement les chunks splittés en entités complètes.
    """

    def __init__(self, smart_context_provider):
        """
        Initialise le visualiseur avec un SmartContextProvider.

        Args:
            smart_context_provider: Instance de SmartContextProvider
        """
        self.context_provider = smart_context_provider
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger(__name__)

        # Mapping chunk_id -> entity_id pour le regroupement
        self.chunk_to_entity_mapping = {}
        self.entity_groups = {}  # entity_id -> informations agrégées

        # Configuration des couleurs par type d'entité
        self.entity_colors = {
            'module': '#4285F4',  # Bleu - modules principaux
            'program': '#EA4335',  # Rouge - programmes
            'subroutine': '#34A853',  # Vert - subroutines
            'function': '#FBBC05',  # Jaune/Orange - fonctions
            'internal_function': '#FF9800',  # Orange - fonctions internes
            'type_definition': '#9C27B0',  # Violet - types
            'interface': '#607D8B',  # Gris bleu - interfaces
            'variable_declaration': '#795548',  # Marron - variables
            'parameter': '#009688',  # Teal - paramètres
            'default': '#9AA0A6'  # Gris par défaut
        }

        # Configuration des formes par type
        self.entity_shapes = {
            'module': 'box',
            'program': 'diamond',
            'subroutine': 'circle',
            'function': 'ellipse',
            'internal_function': 'dot',
            'type_definition': 'triangle',
            'interface': 'star',
            'variable_declaration': 'square',
            'parameter': 'triangleDown',
            'default': 'circle'
        }

        # Configuration des couleurs par type de relation
        self.edge_colors = {
            'contains': '#1976D2',  # Bleu foncé - relations de contenement
            'uses': '#388E3C',  # Vert foncé - USE statements
            'calls': '#F57C00',  # Orange foncé - appels de fonctions
            'depends_on': '#7B1FA2',  # Violet foncé - autres dépendances
            'internal': '#D32F2F',  # Rouge foncé - relations internes
            'cross_file': '#455A64'  # Gris foncé - relations inter-fichiers
        }

    async def build_dependency_graph(self,
                                     max_entities: Optional[int] = None,
                                     include_internal_functions: bool = True,
                                     include_variables: bool = False,
                                     focus_entity: Optional[str] = None,
                                     max_depth: int = 3):
        """
        Construit le graphe de dépendances en regroupant automatiquement les chunks splittés.
        """
        self.logger.info("🔍 Construction du graphe de dépendances Fortran avec regroupement...")
        await self.context_provider._ensure_context_provider()

        # Réinitialiser
        self.graph = nx.DiGraph()
        self.chunk_to_entity_mapping = {}
        self.entity_groups = {}

        entity_index = self.context_provider.context_provider.entity_index

        # 1. Regrouper tous les chunks en entités logiques (ça, c'est bon)
        await self._group_split_chunks(entity_index)

        # 2. SÉLECTIONNER LES ENTITÉS À VISUALISER (logique modifiée)
        # On ne sélectionne plus des chunks, mais des clés d'entités complètes.
        selected_entity_keys = self._select_entities_to_visualize_v3(
            max_entities, include_internal_functions, include_variables
        )

        # Si focus_entity, on filtre autour de lui
        if focus_entity:
            selected_entity_keys = await self._explore_from_focus_grouped(
                focus_entity, selected_entity_keys, max_depth
            )

        self.logger.info(f"📋 Entités complètes sélectionnées pour visualisation: {len(selected_entity_keys)}")

        # 3. Ajouter les NŒUDS (une seule fois par entité)
        await self._add_grouped_entity_nodes_v2(selected_entity_keys)

        # 4. Ajouter les ARÊTES
        await self._add_grouped_entity_relationships_v2(selected_entity_keys)

        self.logger.info(f"✅ Graphe construit: {len(self.graph.nodes)} nœuds, {len(self.graph.edges)} arêtes")
        return self.graph

    async def _build_comprehensive_function_call_cache(self):
        """
        Construit un cache complet des appels de fonctions avec patterns Fortran améliorés
        """
        print("🔗 Construction du cache des appels de fonctions...")

        entity_index = self.context_provider.context_provider.entity_index

        # Patterns Fortran améliorés
        call_patterns = [
            # Appels de subroutines
            re.compile(r'\bcall\s+(\w+)', re.IGNORECASE),

            # Appels de fonctions (plus restrictifs)
            re.compile(r'=\s*(\w+)\s*\(', re.IGNORECASE),  # var = function(
            re.compile(r'\+\s*(\w+)\s*\(', re.IGNORECASE),  # ... + function(
            re.compile(r'-\s*(\w+)\s*\(', re.IGNORECASE),  # ... - function(
            re.compile(r'\*\s*(\w+)\s*\(', re.IGNORECASE),  # ... * function(
            re.compile(r'/\s*(\w+)\s*\(', re.IGNORECASE),  # ... / function(
            re.compile(r'\(\s*(\w+)\s*\(', re.IGNORECASE),  # (function(
            re.compile(r',\s*(\w+)\s*\(', re.IGNORECASE),  # , function(

            # Fonctions dans expressions
            re.compile(r'sqrt\s*\(\s*.*?(\w+)\s*\(', re.IGNORECASE),  # sqrt(...function(...)
            re.compile(r'if\s*\(\s*(\w+)\s*\(', re.IGNORECASE),  # if (function(

            # USE statements
            re.compile(r'use\s+(\w+)', re.IGNORECASE),
        ]

        # Mots-clés Fortran à ignorer (étendus)
        fortran_keywords = {
            'if', 'then', 'else', 'endif', 'elseif', 'do', 'while', 'enddo', 'select',
            'case', 'where', 'forall', 'real', 'integer', 'logical', 'character', 'complex',
            'allocate', 'deallocate', 'nullify', 'write', 'read', 'print', 'open', 'close',
            'sqrt', 'sin', 'cos', 'exp', 'log', 'abs', 'max', 'min', 'sum', 'size', 'len',
            'trim', 'adjustl', 'adjustr', 'present', 'associated', 'allocated',
            'huge', 'tiny', 'epsilon', 'precision', 'range', 'digits',
            'modulo', 'mod', 'int', 'nint', 'floor', 'ceiling', 'aint', 'anint'
        }

        # Construire un index de toutes les entités disponibles
        all_entity_names = {}  # nom_lower -> (nom_original, type, fichier)

        for chunk_id, entity_info in entity_index.chunk_to_entity.items():
            name = entity_info.get('name', '')
            base_name = entity_info.get('base_name', '')
            entity_type = entity_info.get('type', '')
            filepath = entity_info.get('filepath', '')

            # Supprimer les suffixes _part_X pour la recherche
            clean_name = re.sub(r'_part_\d+$', '', name) if name else ''
            clean_base_name = re.sub(r'_part_\d+$', '', base_name) if base_name else ''

            if clean_name:
                all_entity_names[clean_name.lower()] = (clean_name, entity_type, filepath)
            if clean_base_name and clean_base_name != clean_name:
                all_entity_names[clean_base_name.lower()] = (clean_base_name, entity_type, filepath)

        print(f"📋 {len(all_entity_names)} entités disponibles pour la résolution des appels")

        # Analyser chaque chunk pour les appels
        total_calls_found = 0

        for chunk_id, entity_info in entity_index.chunk_to_entity.items():
            # Récupérer le texte du chunk
            try:
                chunk_text = await self._get_chunk_text(chunk_id)
                if not chunk_text:
                    continue
            except Exception as e:
                continue

            # Nettoyer le texte
            cleaned_text = self._remove_fortran_comments(chunk_text)

            # Détecter les appels
            detected_calls = set()

            for pattern in call_patterns:
                matches = pattern.findall(cleaned_text)
                for match in matches:
                    match_lower = match.lower()

                    # Filtrer les mots-clés Fortran
                    if match_lower in fortran_keywords:
                        continue

                    # Vérifier si c'est un nom d'entité connu
                    if match_lower in all_entity_names:
                        original_name, entity_type, filepath = all_entity_names[match_lower]
                        detected_calls.add(original_name)

                        # Debug pour les appels importants
                        if match_lower in ['random_gaussian', 'distance', 'lennard_jones_force',
                                           'apply_periodic_boundary']:
                            entity_name = entity_info.get('name', 'unknown')
                            print(f"✅ Appel détecté: {entity_name} → {original_name}")

            # Mettre à jour le cache
            if detected_calls:
                entity_index.call_patterns_cache[chunk_id] = list(detected_calls)
                total_calls_found += len(detected_calls)
            else:
                entity_index.call_patterns_cache[chunk_id] = []

        print(
            f"✅ Cache des appels construit: {total_calls_found} appels détectés dans {len(entity_index.call_patterns_cache)} chunks")

    async def _get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Récupère le texte d'un chunk (méthode async corrigée)"""
        try:
            # Parser le chunk_id pour extraire le document_id
            parts = chunk_id.split('-chunk-')
            if len(parts) != 2:
                return None

            document_id = parts[0]

            # Charger les chunks du document si nécessaire
            document_store = self.context_provider.rag_engine.document_store
            await document_store.load_document_chunks(document_id)
            chunks = await document_store.get_document_chunks(document_id)

            if chunks:
                for chunk in chunks:
                    if chunk['id'] == chunk_id:
                        return chunk['text']

            return None

        except Exception as e:
            return None

    def _remove_fortran_comments(self, text: str) -> str:
        """Supprime les commentaires Fortran de manière robuste"""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Trouver le premier '!' qui n'est pas dans une chaîne
            in_string = False
            quote_char = None
            comment_pos = -1

            i = 0
            while i < len(line):
                char = line[i]

                if not in_string:
                    if char in ['"', "'"]:
                        in_string = True
                        quote_char = char
                    elif char == '!':
                        comment_pos = i
                        break
                else:
                    if char == quote_char:
                        # Vérifier si ce n'est pas échappé
                        if i == 0 or line[i - 1] != '\\':
                            in_string = False
                            quote_char = None

                i += 1

            if comment_pos >= 0:
                line = line[:comment_pos]

            cleaned_lines.append(line.rstrip())

        return '\n'.join(cleaned_lines)

    async def _add_grouped_entity_relationships_v2(self, selected_entity_keys: Set[str]):
        """Version améliorée pour créer les relations avec debug détaillé"""
        self.logger.info(f"🔗 Ajout des relations pour {len(selected_entity_keys)} entités...")

        # Construire le cache des appels AVANT de créer les relations
        await self._build_comprehensive_function_call_cache()

        # Créer un index rapide nom → clé d'entité
        name_to_key_map = {}
        for key, group in self.entity_groups.items():
            if key in selected_entity_keys:
                entity_name = group['entity_name'].lower()
                name_to_key_map[entity_name] = key

        self.logger.info(f"📇 Index des noms: {len(name_to_key_map)} entités mappées")

        edge_count = 0
        call_edge_count = 0

        for source_key in selected_entity_keys:
            if source_key not in self.entity_groups:
                continue

            source_group = self.entity_groups[source_key]
            source_name = source_group['entity_name']

            # 1. Relations de dépendance ('uses')
            for dep_name in source_group.get('all_dependencies', set()):
                target_key = name_to_key_map.get(dep_name.lower())
                if target_key and target_key != source_key and self.graph.has_node(target_key):
                    self.graph.add_edge(
                        source_key, target_key,
                        label="uses",
                        color=self.edge_colors['uses'],
                        title=f"{source_name} uses {dep_name}",
                        width=2.0,
                        dashes=False
                    )
                    edge_count += 1

            # 2. Relations d'appels (AMÉLIORÉES avec debug)
            all_called_functions = set()

            # Collecter tous les appels depuis les chunks de cette entité
            entity_index = self.context_provider.context_provider.entity_index

            for chunk_info in source_group.get('chunks', []):
                chunk_id = chunk_info.get('chunk_id', '')
                if chunk_id and hasattr(entity_index, 'call_patterns_cache'):
                    calls = entity_index.call_patterns_cache.get(chunk_id, [])
                    all_called_functions.update(calls)

            # Debug pour les entités importantes
            if source_name.lower() in ['simulation_main', 'force_calculation', 'velocity_verlet_step']:
                print(f"🔍 Debug appels pour {source_name}: {list(all_called_functions)}")

            # Créer les arêtes d'appel
            for called_function in all_called_functions:
                target_key = name_to_key_map.get(called_function.lower())

                if target_key and target_key != source_key and self.graph.has_node(target_key):
                    self.graph.add_edge(
                        source_key, target_key,
                        label="calls",
                        color=self.edge_colors['calls'],
                        title=f"{source_name} calls {called_function}",
                        dashes=True,
                        width=1.5
                    )
                    edge_count += 1
                    call_edge_count += 1

                    # Debug pour les appels importants
                    if called_function.lower() in ['random_gaussian', 'distance', 'lennard_jones_force']:
                        print(f"✅ Relation créée: {source_name} → {called_function}")
                else:
                    # Debug pour les appels non résolus
                    if called_function.lower() in ['random_gaussian', 'distance', 'lennard_jones_force']:
                        print(f"❌ Appel non résolu: {source_name} → {called_function}")
                        print(f"   target_key: {target_key}")
                        print(f"   graph.has_node: {self.graph.has_node(target_key) if target_key else False}")

        self.logger.info(f"✅ {edge_count} relations créées (dont {call_edge_count} appels)")

    def _clean_entity_groups(self) -> Dict[str, Dict]:
        """
        Nettoie les groupes d'entités en supprimant les redondances et les fragments
        """
        cleaned_groups = {}

        # 1. Identifier les entités de base (sans _part_)
        base_entities = {}
        for entity_key, group in self.entity_groups.items():
            entity_name = group['entity_name']

            # Extraire le nom de base (sans _part_X)
            base_name = re.sub(r'_part_\d+$', '', entity_name)

            # Créer une clé unique basée sur le nom de base + type + fichier
            unique_key = f"{group['filepath']}#{group['entity_type']}#{base_name}"

            if unique_key not in base_entities:
                base_entities[unique_key] = {
                    'base_name': base_name,
                    'entity_type': group['entity_type'],
                    'filepath': group['filepath'],
                    'filename': group['filename'],
                    'all_groups': [],
                    'total_chunks': 0,
                    'all_dependencies': set(),
                    'all_concepts': set(),
                    'best_score': 0,
                    'entity_start': float('inf'),
                    'entity_end': 0
                }

            # Ajouter ce groupe à l'entité de base
            base_entities[unique_key]['all_groups'].append(group)
            base_entities[unique_key]['total_chunks'] += len(group['chunks'])
            base_entities[unique_key]['all_dependencies'].update(group['all_dependencies'])
            base_entities[unique_key]['all_concepts'].update(group['all_concepts'])
            base_entities[unique_key]['best_score'] = max(
                base_entities[unique_key]['best_score'],
                group['best_score']
            )

            # Mettre à jour les bounds de l'entité complète
            if group.get('entity_start'):
                base_entities[unique_key]['entity_start'] = min(
                    base_entities[unique_key]['entity_start'],
                    group['entity_start']
                )
            if group.get('entity_end'):
                base_entities[unique_key]['entity_end'] = max(
                    base_entities[unique_key]['entity_end'],
                    group['entity_end']
                )

        # 2. Créer les groupes nettoyés
        for unique_key, base_entity in base_entities.items():
            # Fusionner tous les chunks des groupes
            all_chunks = []
            for group in base_entity['all_groups']:
                all_chunks.extend(group['chunks'])

            cleaned_groups[unique_key] = {
                'entity_name': base_entity['base_name'],
                'entity_type': base_entity['entity_type'],
                'file': base_entity['filename'],
                'filepath': base_entity['filepath'],
                'chunks': all_chunks,
                'all_concepts': base_entity['all_concepts'],
                'all_matched_concepts': set(),  # À recalculer si nécessaire
                'all_dependencies': base_entity['all_dependencies'],
                'best_score': base_entity['best_score'],
                'total_score': sum(group['total_score'] for group in base_entity['all_groups']),
                'entity_start': base_entity['entity_start'] if base_entity['entity_start'] != float('inf') else None,
                'entity_end': base_entity['entity_end'] if base_entity['entity_end'] > 0 else None,
                'expected_parts': base_entity['total_chunks'],
                'is_merged': len(base_entity['all_groups']) > 1
            }

        return cleaned_groups

    def _select_entities_to_visualize_v3(self,
                                         max_entities: Optional[int],
                                         include_internal_functions: bool,
                                         include_variables: bool) -> Set[str]:
        """
        Version améliorée qui évite les fragments et redondances
        """

        # 1. Nettoyer d'abord les groupes
        cleaned_groups = self._clean_entity_groups()

        # 2. Filtrer par type et importance
        selected_keys = set()

        # Priorités par type d'entité
        type_priorities = {
            'module': 1,
            'program': 1,
            'type_definition': 2,
            'interface': 3,
            'subroutine': 4,
            'function': 4,
            'internal_function': 5 if include_internal_functions else 999,
            'variable_declaration': 6 if include_variables else 999,
            'parameter': 6 if include_variables else 999
        }

        # Trier les entités par priorité et score
        sorted_entities = sorted(
            cleaned_groups.items(),
            key=lambda x: (
                type_priorities.get(x[1]['entity_type'], 999),
                -x[1]['best_score'],  # Score décroissant
                x[1]['entity_name']  # Nom alphabétique
            )
        )

        # Sélectionner selon les critères
        for entity_key, entity_info in sorted_entities:
            entity_type = entity_info['entity_type']
            priority = type_priorities.get(entity_type, 999)

            if priority < 999:  # Type autorisé
                selected_keys.add(entity_key)

            # Limiter si nécessaire
            if max_entities and len(selected_keys) >= max_entities:
                break

        print(f"📋 Entités nettoyées sélectionnées: {len(selected_keys)}")

        # Remplacer l'ancien dictionnaire par le nouveau
        self.entity_groups = cleaned_groups

        return selected_keys

    async def _group_split_chunks(self, entity_index):
        """
        Regroupe les chunks splittés en entités complètes.
        Utilise la même logique que _create_entity_groups dans smart_context_provider.
        """
        self.logger.info("📦 Regroupement des chunks splittés...")

        # Parcourir tous les chunks
        for chunk_id, entity_info in entity_index.chunk_to_entity.items():

            # Déterminer l'ID de l'entité complète
            if entity_info.get('is_partial', False):
                # Chunk partiel - utiliser parent_entity_id
                entity_key = entity_info.get('parent_entity_id')
                if not entity_key:
                    # Fallback sur base_entity_name
                    entity_key = entity_info.get('base_entity_name') or entity_info.get('entity_name', 'unknown')
            else:
                # Chunk complet - utiliser entity_id ou créer une clé
                entity_key = entity_info.get('entity_id')
                if not entity_key:
                    # Créer une clé basée sur le nom et la position
                    base_name = entity_info.get('base_entity_name') or entity_info.get('entity_name', 'unknown')
                    filepath = entity_info.get('filepath', '')
                    start_line = entity_info.get('start_line', 0)
                    entity_key = f"{filepath}#{base_name}#{start_line}"

            # Stocker le mapping chunk -> entité
            self.chunk_to_entity_mapping[chunk_id] = entity_key

            # Initialiser le groupe d'entité s'il n'existe pas
            if entity_key not in self.entity_groups:
                # Récupérer les bounds de l'entité complète si disponibles
                entity_bounds = entity_info.get('entity_bounds', {})

                self.entity_groups[entity_key] = {
                    'entity_id': entity_key,
                    'entity_name': entity_info.get('base_name') or entity_info.get('name', 'unknown'),                    'entity_type': entity_info.get('type', 'code'),                    'filepath': entity_info.get('filepath', ''),
                    'filename': entity_info.get('filename', 'Unknown'),
                    'chunks': [],
                    'chunk_ids': set(),
                    'all_dependencies': set(),
                    'all_concepts': set(),
                    'all_matched_concepts': set(),
                    'best_score': 0,
                    'total_score': 0,
                    'entity_start': entity_bounds.get('start_line') or entity_info.get('start_line'),
                    'entity_end': entity_bounds.get('end_line') or entity_info.get('end_line'),
                    'expected_parts': entity_info.get('total_parts', 1),
                    'is_internal': entity_info.get('is_internal_function', False),
                    'parent_entity': entity_info.get('parent_entity_name', ''),
                    'parent_entity_type': entity_info.get('parent_entity_type', ''),
                    'qualified_name': entity_info.get('full_qualified_name', ''),
                }

            # Ajouter le chunk au groupe
            group = self.entity_groups[entity_key]
            group['chunks'].append({
                'chunk_id': chunk_id,
                'entity_info': entity_info,
                'part_index': entity_info.get('part_index', 0),
                'part_sequence': entity_info.get('part_sequence', 0)
            })
            group['chunk_ids'].add(chunk_id)

            # Agréger les dépendances
            dependencies = entity_info.get('dependencies', [])
            if isinstance(dependencies, list):
                group['all_dependencies'].update(dependencies)

            # Agréger les concepts
            concepts = entity_info.get('concepts', [])
            if isinstance(concepts, list):
                for concept in concepts:
                    if isinstance(concept, dict):
                        group['all_concepts'].add(concept.get('label', str(concept)))
                    else:
                        group['all_concepts'].add(str(concept))

            # Mettre à jour les bounds si on trouve des valeurs plus précises
            if entity_info.get('start_line'):
                if not group['entity_start'] or entity_info['start_line'] < group['entity_start']:
                    group['entity_start'] = entity_info['start_line']

            if entity_info.get('end_line'):
                if not group['entity_end'] or entity_info['end_line'] > group['entity_end']:
                    group['entity_end'] = entity_info['end_line']

        # Post-traitement : vérifier l'intégrité des groupes et trier les chunks
        for entity_key, group in self.entity_groups.items():
            actual_parts = len(group['chunks'])
            expected_parts = group['expected_parts']

            if actual_parts != expected_parts and expected_parts > 1:
                self.logger.debug(f"⚠️ Entité {group['entity_name']}: {actual_parts}/{expected_parts} parties trouvées")

            # Trier les chunks par part_sequence ou start_pos
            group['chunks'].sort(
                key=lambda x: (
                    x['entity_info'].get('part_sequence', 0),
                    x['entity_info'].get('start_line', 0)
                )
            )

            # Déterminer si l'entité est complète
            group['is_complete'] = actual_parts == expected_parts or expected_parts == 1

        self.logger.info(f"✅ Regroupement terminé: {len(self.entity_groups)} entités complètes créées")

    async def _explore_from_focus_grouped(self,
                                          focus_entity: str,
                                          all_entities: Set[str],
                                          max_depth: int) -> Set[str]:
        """Explore les entités connectées à partir d'une entité focus (version groupée)."""

        # Trouver l'entité focus
        focus_entity_key = None
        for entity_key, group in self.entity_groups.items():
            if group['entity_name'].lower() == focus_entity.lower():
                focus_entity_key = entity_key
                break

        if not focus_entity_key:
            self.logger.warning(f"⚠️ Entité focus '{focus_entity}' non trouvée")
            return all_entities

        explored = {focus_entity_key}
        current_layer = {focus_entity_key}

        for depth in range(max_depth):
            next_layer = set()

            for entity_key in current_layer:
                if entity_key not in all_entities:
                    continue

                group = self.entity_groups[entity_key]

                # Explorer les dépendances
                dependencies = group.get('all_dependencies', set())
                for dep in dependencies:
                    # Trouver l'entité correspondante
                    for other_key, other_group in self.entity_groups.items():
                        if other_group['entity_name'] == dep:
                            next_layer.add(other_key)

                # Explorer les relations parent-enfant via le nom
                entity_name = group['entity_name']
                parent_name = group.get('parent_entity', '')

                # Chercher les enfants et parents
                for other_key, other_group in self.entity_groups.items():
                    # Si l'autre entité est un enfant
                    if other_group.get('parent_entity', '') == entity_name:
                        next_layer.add(other_key)

                    # Si l'autre entité est le parent
                    if parent_name and other_group['entity_name'] == parent_name:
                        next_layer.add(other_key)

            # Filtrer pour ne garder que les entités valides
            next_layer = next_layer.intersection(all_entities)
            explored.update(next_layer)
            current_layer = next_layer

            if not next_layer:  # Plus rien à explorer
                break

        return explored

    async def _add_grouped_entity_nodes_v2(self, selected_entity_keys: Set[str]):
        """
        Ajoute les nœuds au graphe à partir d'un set de clés d'entités uniques.
        Chaque clé correspond à une entité logique (ex: un module, une fonction).
        """
        self.logger.info(f"🎨 Ajout de {len(selected_entity_keys)} nœuds uniques au graphe...")

        for entity_key in selected_entity_keys:
            if entity_key not in self.entity_groups:
                self.logger.warning(f"Clé d'entité {entity_key} sélectionnée mais non trouvée dans les groupes.")
                continue

            group = self.entity_groups[entity_key]

            # --- Récupération des informations de l'entité regroupée ---
            entity_name = group.get('entity_name', 'unknown')
            entity_type = group.get('entity_type', 'default')
            filepath = group.get('filepath', '')
            filename = group.get('filename', 'unknown')
            is_grouped = len(group['chunks']) > 1

            # --- Création du label affiché ---
            display_label = entity_name
            if is_grouped:
                display_label += f" ({len(group['chunks'])} parts) [📦]"

            # --- Création de la description pour le tooltip ---
            description = self._create_grouped_entity_description(group)

            # --- Calcul de la taille du nœud ---
            size = self._calculate_grouped_node_size(group)

            # --- Ajout du nœud au graphe ---
            # On utilise entity_key comme ID unique pour le nœud
            self.graph.add_node(
                entity_key,
                label=display_label,
                title=description,
                # Attributs pour le style
                size=size,
                color=self.entity_colors.get(entity_type, self.entity_colors['default']),
                shape=self.entity_shapes.get(entity_type, self.entity_shapes['default']),
                # Attributs pour l'analyse
                entity_name=entity_name,
                entity_type=entity_type,
                filepath=filepath,
                filename=filename,
                is_grouped=is_grouped,
                is_complete=group.get('is_complete', True)
            )

    def _create_grouped_entity_description(self, group: Dict[str, Any]) -> str:
        """Crée une description HTML pour le tooltip d'une entité regroupée (version corrigée)."""

        entity_name = group.get('entity_name', 'Unknown')
        entity_type = group.get('entity_type', 'unknown')
        filename = group.get('filename', group.get('file', 'unknown'))  # Essayer les deux clés
        actual_parts = len(group.get('chunks', []))
        expected_parts = group.get('expected_parts', 1)
        is_complete = group.get('is_complete', True)

        html = f"<div style='max-width:450px;'>"
        html += f"<h3>{entity_name}</h3>"
        html += f"<p><strong>Type:</strong> {entity_type}</p>"
        html += f"<p><strong>Fichier:</strong> {filename}</p>"

        # Informations sur le regroupement
        if actual_parts > 1:
            status = "✅ Complète" if is_complete else f"⚠️ Partielle ({actual_parts}/{expected_parts})"
            html += f"<p><strong>Parties:</strong> {actual_parts} chunks - {status}</p>"

        # Positions avec vérification sécurisée
        entity_start = group.get('entity_start')
        entity_end = group.get('entity_end')
        if entity_start and entity_end:
            html += f"<p><strong>Lignes:</strong> {entity_start}-{entity_end}</p>"

        # Ajouter les dépendances avec vérification
        dependencies = group.get('all_dependencies', set())
        if dependencies:
            deps_list = list(dependencies)
            html += f"<p><strong>Dépendances:</strong> {', '.join(deps_list[:5])}"
            if len(deps_list) > 5:
                html += f" (+{len(deps_list) - 5} autres)"
            html += "</p>"

        # Ajouter les concepts détectés si disponibles
        concepts = group.get('all_concepts', set())
        if concepts:
            concepts_list = list(concepts)
            html += f"<p><strong>Concepts:</strong> {', '.join(concepts_list[:3])}"
            if len(concepts_list) > 3:
                html += f" (+{len(concepts_list) - 3} autres)"
            html += "</p>"

        # Information sur le parent pour les fonctions internes
        is_internal = group.get('is_internal', False)
        if is_internal:
            parent = group.get('parent_entity', '')
            if parent:
                html += f"<p><strong>Parent:</strong> {parent}</p>"

        # Informations détaillées sur les chunks
        if actual_parts > 1:
            html += "<hr><p><strong>Détails des parties:</strong></p><ul>"
            for i, chunk_info in enumerate(group['chunks'][:3]):  # Montrer max 3 parties
                # Gérer différents formats de chunk_info de manière robuste
                if isinstance(chunk_info, dict):
                    if 'entity_info' in chunk_info:
                        # Format avec entity_info
                        entity_info = chunk_info['entity_info']
                        start_line = entity_info.get('start_line', entity_info.get('start_pos', '?'))
                        end_line = entity_info.get('end_line', entity_info.get('end_pos', '?'))
                        part_idx = entity_info.get('part_index', i + 1)
                    else:
                        # Format direct
                        start_line = chunk_info.get('start_line', chunk_info.get('start_pos', '?'))
                        end_line = chunk_info.get('end_line', chunk_info.get('end_pos', '?'))
                        part_idx = chunk_info.get('part_index', i + 1)
                else:
                    # Fallback si ce n'est pas un dict
                    start_line = '?'
                    end_line = '?'
                    part_idx = i + 1

                html += f"<li>Partie {part_idx}: lignes {start_line}-{end_line}</li>"

            if actual_parts > 3:
                html += f"<li>... et {actual_parts - 3} autres parties</li>"
            html += "</ul>"

        # ID de l'entité avec gestion robuste des clés manquantes
        entity_id = (group.get('entity_id') or
                     group.get('parent_entity_id') or
                     group.get('qualified_name') or
                     f"{entity_name}_{entity_type}")

        html += f"<hr><p><small>Entity ID: {entity_id}</small></p>"
        html += "</div>"

        return html

    def _calculate_grouped_node_size(self, group: Dict[str, Any]) -> int:
        """Calcule la taille d'un nœud pour une entité regroupée."""

        base_size = 20
        entity_type = group.get('entity_type', '')

        # Taille de base par type
        type_sizes = {
            'module': 40,
            'program': 35,
            'subroutine': 25,
            'function': 25,
            'internal_function': 15,
            'type_definition': 20,
            'interface': 20,
            'variable_declaration': 10,
            'parameter': 10
        }

        size = type_sizes.get(entity_type, base_size)

        # Bonus pour les entités avec beaucoup de dépendances
        dependencies_count = len(group.get('all_dependencies', set()))
        size += min(dependencies_count * 2, 15)

        # Bonus pour les entités avec plusieurs parties (plus complexes)
        parts_count = len(group['chunks'])
        if parts_count > 1:
            size += min(parts_count * 3, 20)

        # Bonus pour les entités avec beaucoup de concepts
        concepts_count = len(group.get('all_concepts', set()))
        size += min(concepts_count * 1, 10)

        return min(size, 70)  # Limiter la taille maximale

    def visualize_with_pyvis(self,
                             output_file: str = "fortran_dependencies.html",
                             hierarchical: bool = True,
                             height: str = "800px",
                             width: str = "100%",
                             physics_enabled: bool = True):
        """
        Génère une visualisation interactive avec PyVis.
        Version mise à jour pour les entités regroupées.
        """

        if not self.graph.nodes:
            self.logger.error("❌ Aucun graphe à visualiser. Appelez build_dependency_graph() d'abord.")
            return

        self.logger.info(f"🎨 Génération de la visualisation interactive...")

        # Créer le réseau PyVis
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#000000"
        )

        # Configuration des options (identique à avant)
        if hierarchical:
            net.set_options("""
            var options = {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed",
                        "levelSeparation": 200,
                        "nodeSpacing": 150,
                        "treeSpacing": 200,
                        "blockShifting": true,
                        "edgeMinimization": true,
                        "parentCentralization": true
                    }
                },
                "physics": {
                    "enabled": """ + str(physics_enabled).lower() + """,
                    "hierarchicalRepulsion": {
                        "centralGravity": 0.0,
                        "springLength": 150,
                        "springConstant": 0.01,
                        "nodeDistance": 150,
                        "damping": 0.09
                    },
                    "solver": "hierarchicalRepulsion",
                    "stabilization": {
                        "enabled": true,
                        "iterations": 1000
                    }
                },
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true,
                    "hideEdgesOnDrag": true,
                    "hover": true,
                    "multiselect": true,
                    "selectConnectedEdges": true
                },
                "edges": {
                    "smooth": {
                        "enabled": true,
                        "type": "dynamic",
                        "roundness": 0.5
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "scaleFactor": 0.8
                        }
                    },
                    "font": {
                        "size": 12,
                        "strokeWidth": 2,
                        "strokeColor": "#ffffff"
                    }
                },
                "nodes": {
                    "font": {
                        "size": 14,
                        "face": "arial"
                    },
                    "borderWidth": 2,
                    "shadow": true
                }
            }
            """)
        else:
            net.set_options("""
            var options = {
                "physics": {
                    "enabled": """ + str(physics_enabled).lower() + """,
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08,
                        "damping": 0.4,
                        "avoidOverlap": 0.5
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
                    "stabilization": {"enabled": true, "iterations": 1000}
                },
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true,
                    "hideEdgesOnDrag": true,
                    "hover": true,
                    "multiselect": true
                }
            }
            """)

        # Ajouter les nœuds
        for node_id, attr in self.graph.nodes(data=True):
            # Modifier le label pour indiquer les entités regroupées
            label = attr.get('label', 'Unknown')
            if attr.get('is_grouped', False):
                label += f" [📦]"  # Icône pour indiquer le regroupement

            # Modifier la couleur de bordure pour les entités incomplètes
            color = attr.get('color', self.entity_colors['default'])
            if not attr.get('is_complete', True):
                # Ajouter une bordure rouge pour les entités incomplètes
                net.add_node(
                    node_id,
                    label=label,
                    title=attr.get('description', ''),
                    color={'background': color, 'border': '#dc3545',
                           'highlight': {'background': color, 'border': '#dc3545'}},
                    shape=attr.get('shape', 'circle'),
                    size=attr.get('size', 20),
                    font={'size': 14},
                    borderWidth=3  # Bordure plus épaisse pour les incomplètes
                )
            else:
                net.add_node(
                    node_id,
                    label=label,
                    title=attr.get('description', ''),
                    color=color,
                    shape=attr.get('shape', 'circle'),
                    size=attr.get('size', 20),
                    font={'size': 14},
                    borderWidth=2
                )

        # Ajouter les arêtes (identique à avant)
        for source, target, attr in self.graph.edges(data=True):
            edge_label = attr.get('label', '')
            edge_type = attr.get('edge_type', 'default')

            # Déterminer le style selon le type
            width = 1.0
            dashes = False

            if edge_type == 'contains':
                width = 3.0
            elif edge_type == 'uses':
                width = 2.0
                dashes = False
            elif edge_type == 'calls':
                width = 1.5
                dashes = True
            elif edge_type == 'cross_file':
                width = 1.0
                dashes = True

            net.add_edge(
                source, target,
                label=edge_label,
                title=attr.get('title', edge_label),
                color=attr.get('color', self.edge_colors.get(edge_type, '#666666')),
                width=width,
                dashes=dashes,
                arrows="to",
                smooth={"enabled": True, "type": "curvedCW" if edge_type == "calls" else "dynamic"}
            )

        # Ajouter des informations de navigation
        net.add_node(
            "legend",
            label="LÉGENDE",
            title=self._create_enhanced_legend_html(),
            color="#f0f0f0",
            shape="box",
            size=30,
            fixed=True,
            x=-1000,
            y=-1000,
            font={'size': 16, 'color': '#000000'}
        )

        # Sauvegarder le fichier
        net.save_graph(output_file)

        # Ajouter du CSS et JavaScript personnalisé
        self._enhance_html_output(output_file)

        # Calculer les statistiques de regroupement
        grouped_count = sum(1 for _, attr in self.graph.nodes(data=True) if attr.get('is_grouped', False))
        total_chunks = sum(len(group['chunks']) for group in self.entity_groups.values())

        self.logger.info(f"✅ Visualisation sauvegardée dans {output_file}")
        self.logger.info(f"📦 {grouped_count} entités regroupées sur {len(self.graph.nodes)} total")
        self.logger.info(f"🧩 {total_chunks} chunks originaux regroupés")

        # Ouvrir dans le navigateur
        try:
            webbrowser.open('file://' + os.path.abspath(output_file), new=2)
            self.logger.info(f"🌐 Ouverture dans le navigateur...")
        except Exception as e:
            self.logger.warning(f"⚠️ Impossible d'ouvrir le navigateur: {e}")

    def _create_enhanced_legend_html(self) -> str:
        """Crée une légende HTML améliorée qui explique le regroupement."""

        html = "<div style='font-family: Arial, sans-serif; max-width: 350px;'>"
        html += "<h3>Types d'entités</h3>"

        legend_items = [
            ("Module", "module", "box"),
            ("Programme", "program", "diamond"),
            ("Subroutine", "subroutine", "circle"),
            ("Fonction", "function", "ellipse"),
            ("Fonction interne", "internal_function", "dot"),
            ("Type", "type_definition", "triangle"),
            ("Interface", "interface", "star")
        ]

        for label, entity_type, shape in legend_items:
            color = self.entity_colors.get(entity_type, "#666")
            html += f"<p><span style='color: {color}; font-size: 16px;'>●</span> {label} ({shape})</p>"

        html += "<h3>Types de relations</h3>"

        relation_items = [
            ("Contient", "contains", "#1976D2"),
            ("Utilise", "uses", "#388E3C"),
            ("Appelle", "calls", "#F57C00"),
            ("Inter-fichiers", "cross_file", "#455A64")
        ]

        for label, rel_type, color in relation_items:
            html += f"<p><span style='color: {color}; font-size: 16px;'>→</span> {label}</p>"

        html += "<h3>Regroupement</h3>"
        html += "<p>📦 = Entité regroupée (chunks multiples)</p>"
        html += "<p style='border: 2px solid #dc3545; padding: 5px; background: #f8d7da;'>"
        html += "<strong>Bordure rouge:</strong> Entité incomplète"
        html += "</p>"

        html += "</div>"
        return html

    def _enhance_html_output(self, output_file: str):
        """Améliore le fichier HTML avec des informations sur le regroupement."""

        try:
            # Lire le fichier existant
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Calculer les statistiques de regroupement
            grouped_count = sum(1 for _, attr in self.graph.nodes(data=True) if attr.get('is_grouped', False))
            total_chunks = sum(len(group['chunks']) for group in self.entity_groups.values())
            incomplete_count = sum(1 for _, attr in self.graph.nodes(data=True) if not attr.get('is_complete', True))

            # Ajouter du CSS personnalisé dans le head
            custom_css = """
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }

                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }

                .controls {
                    background: white;
                    padding: 15px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    flex-wrap: wrap;
                }

                .control-btn {
                    background: #4285F4;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                    transition: background 0.3s;
                }

                .control-btn:hover {
                    background: #3367D6;
                }

                .stats {
                    background: white;
                    margin: 10px;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    text-align: center;
                }

                .grouping-info {
                    background: #e8f5e8;
                    margin: 10px;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }

                #mynetworkid {
                    border: 1px solid #ddd;
                    margin: 10px;
                    border-radius: 8px;
                }
            </style>
            """

            # Insérer le CSS avant </head>
            content = content.replace('</head>', custom_css + '</head>')

            # Ajouter un header et des contrôles après <body>
            header_html = f"""
            <div class="header">
                <h1>🔍 Graphe de Dépendances Fortran (Regroupé)</h1>
                <p>Visualisation interactive des relations entre entités complètes</p>
            </div>

            <div class="stats">
                <strong>📊 Statistiques:</strong> 
                {len(self.graph.nodes)} entités • 
                {len(self.graph.edges)} relations • 
                {len(set(attr.get('filename', '') for _, attr in self.graph.nodes(data=True)))} fichiers
            </div>

            <div class="grouping-info">
                <strong>📦 Regroupement automatique:</strong> 
                {grouped_count} entités regroupées à partir de {total_chunks} chunks • 
                {len(self.entity_groups)} entités totales indexées
                {f' • ⚠️ {incomplete_count} entités incomplètes' if incomplete_count > 0 else ''}
            </div>

            <div class="controls">
                <button class="control-btn" onclick="network.fit()">🎯 Ajuster la vue</button>
                <button class="control-btn" onclick="togglePhysics()">⚡ Basculer physique</button>
                <button class="control-btn" onclick="showGroupingStats()">📦 Stats regroupement</button>
                <button class="control-btn" onclick="showStats()">📊 Statistiques</button>
                <button class="control-btn" onclick="exportImage()">💾 Exporter image</button>
            </div>
            """

            content = content.replace('<body>', '<body>' + header_html)

            # Ajouter du JavaScript personnalisé avant </body>
            custom_js = f"""
            <script>
                let physicsEnabled = true;

                function togglePhysics() {{
                    physicsEnabled = !physicsEnabled;
                    network.setOptions({{physics: {{enabled: physicsEnabled}}}});
                    console.log('Physics ' + (physicsEnabled ? 'enabled' : 'disabled'));
                }}

                function showGroupingStats() {{
                    let statsText = 'Statistiques de regroupement:\\n\\n';
                    statsText += 'Entités regroupées: {grouped_count}\\n';
                    statsText += 'Chunks originaux: {total_chunks}\\n';
                    statsText += 'Entités totales: {len(self.entity_groups)}\\n';
                    statsText += 'Entités incomplètes: {incomplete_count}\\n\\n';
                    statsText += 'Les entités avec une bordure rouge sont incomplètes\\n';
                    statsText += '(certains chunks manquent).';
                    alert(statsText);
                }}

                function showStats() {{
                    const nodes = network.body.data.nodes.get();
                    const edges = network.body.data.edges.get();

                    // Compter par type
                    const typeCounts = {{}};
                    nodes.forEach(node => {{
                        const type = node.title ? node.title.match(/Type:<\/strong> (\\w+)/) : null;
                        const entityType = type ? type[1] : 'unknown';
                        typeCounts[entityType] = (typeCounts[entityType] || 0) + 1;
                    }});

                    let statsText = 'Types d\\'entités:\\n';
                    Object.entries(typeCounts).forEach(([type, count]) => {{
                        statsText += `- ${{type}}: ${{count}}\\n`;
                    }});

                    alert(statsText);
                }}

                function exportImage() {{
                    alert('Export d\\'image à implémenter');
                }}

                // Ajouter des gestionnaires d'événements
                network.on("selectNode", function (params) {{
                    if (params.nodes.length > 0) {{
                        const nodeId = params.nodes[0];
                        const node = network.body.data.nodes.get(nodeId);
                        console.log('Entité sélectionnée:', node);
                    }}
                }});

                network.on("selectEdge", function (params) {{
                    if (params.edges.length > 0) {{
                        const edgeId = params.edges[0];
                        const edge = network.body.data.edges.get(edgeId);
                        console.log('Relation sélectionnée:', edge);
                    }}
                }});
            </script>
            """

            content = content.replace('</body>', custom_js + '</body>')

            # Réécrire le fichier
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:
            self.logger.warning(f"⚠️ Impossible d'améliorer le HTML: {e}")

    def generate_dependency_report(self, output_file: str = "dependency_report.html"):
        """
        Génère un rapport HTML détaillé incluant les informations de regroupement.
        """

        if not self.graph.nodes:
            self.logger.error("❌ Aucun graphe disponible pour le rapport.")
            return

        # Analyser le graphe avec informations de regroupement
        analysis = self._analyze_dependency_graph_grouped()

        # Générer le HTML
        html = self._generate_enhanced_report_html(analysis)

        # Sauvegarder
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        self.logger.info(f"📋 Rapport de dépendances (avec regroupement) généré: {output_file}")

        # Ouvrir dans le navigateur
        try:
            webbrowser.open('file://' + os.path.abspath(output_file), new=2)
        except:
            pass

    def _analyze_dependency_graph_grouped(self) -> Dict[str, Any]:
        """Analyse le graphe avec informations de regroupement."""

        analysis = {
            'basic_stats': {},
            'grouping_stats': {},
            'centrality': {},
            'cycles': [],
            'components': {},
            'file_analysis': {},
            'entity_types': {},
            'top_entities': {}
        }

        # Statistiques de base
        analysis['basic_stats'] = {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph)
        }

        # Statistiques de regroupement
        grouped_count = sum(1 for _, attr in self.graph.nodes(data=True) if attr.get('is_grouped', False))
        incomplete_count = sum(1 for _, attr in self.graph.nodes(data=True) if not attr.get('is_complete', True))
        total_chunks = sum(len(group['chunks']) for group in self.entity_groups.values())

        analysis['grouping_stats'] = {
            'total_entity_groups': len(self.entity_groups),
            'visualized_entities': len(self.graph.nodes),
            'grouped_entities': grouped_count,
            'incomplete_entities': incomplete_count,
            'total_original_chunks': total_chunks,
            'compression_ratio': total_chunks / len(self.graph.nodes) if len(self.graph.nodes) > 0 else 0
        }

        # Le reste de l'analyse (identique à la version précédente)
        if self.graph.nodes:
            try:
                betweenness = nx.betweenness_centrality(self.graph)
                in_degree = dict(self.graph.in_degree())
                out_degree = dict(self.graph.out_degree())

                analysis['centrality'] = {
                    'most_central': sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10],
                    'most_connected_in': sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10],
                    'most_connected_out': sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:10]
                }
            except:
                pass

        # Cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))
            analysis['cycles'] = cycles[:10]
        except:
            pass

        # Composantes
        try:
            weak_components = list(nx.weakly_connected_components(self.graph))
            analysis['components'] = {
                'count': len(weak_components),
                'largest_size': len(max(weak_components, key=len)) if weak_components else 0
            }
        except:
            pass

        # Analyse par fichier
        files = defaultdict(list)
        entity_types = defaultdict(int)

        for node_id, attr in self.graph.nodes(data=True):
            filename = attr.get('filename', 'unknown')
            entity_type = attr.get('entity_type', 'unknown')

            files[filename].append(node_id)
            entity_types[entity_type] += 1

        analysis['file_analysis'] = {
            'file_count': len(files),
            'entities_per_file': {k: len(v) for k, v in files.items()},
            'largest_file': max(files.items(), key=lambda x: len(x[1])) if files else ('', [])
        }

        analysis['entity_types'] = dict(entity_types)

        return analysis

    def _generate_enhanced_report_html(self, analysis: Dict[str, Any]) -> str:
        """Génère le HTML du rapport avec informations de regroupement."""

        grouping_stats = analysis['grouping_stats']

        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport de Dépendances Fortran (Regroupé)</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f8f9fa;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .stats-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 20px; 
                    margin-bottom: 30px; 
                }}
                .stat-card {{ 
                    background: white; 
                    border-radius: 10px; 
                    padding: 20px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    border-left: 4px solid #4285F4;
                }}
                .stat-card.grouping {{ 
                    border-left: 4px solid #28a745;
                }}
                .stat-value {{ 
                    font-size: 32px; 
                    font-weight: bold; 
                    color: #4285F4; 
                    margin-bottom: 5px;
                }}
                .stat-value.grouping {{ 
                    color: #28a745; 
                }}
                .stat-label {{ 
                    color: #666; 
                    font-size: 14px; 
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .section {{ 
                    background: white; 
                    border-radius: 10px; 
                    padding: 25px; 
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .section h2 {{ 
                    color: #2c3e50; 
                    border-bottom: 2px solid #4285F4;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 15px; 
                }}
                th, td {{ 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd; 
                }}
                th {{ 
                    background-color: #f8f9fa; 
                    font-weight: 600;
                    color: #2c3e50;
                }}
                tr:hover {{ background-color: #f8f9fa; }}
                .badge {{ 
                    display: inline-block; 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 12px; 
                    font-weight: bold; 
                    color: white;
                }}
                .badge-success {{ background-color: #28a745; }}
                .badge-warning {{ background-color: #ffc107; color: #212529; }}
                .badge-danger {{ background-color: #dc3545; }}
                .badge-info {{ background-color: #17a2b8; }}
                .compression-info {{
                    background: #e8f5e8;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #28a745;
                    margin: 15px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Rapport de Dépendances Fortran (Regroupé)</h1>
                    <p>Analyse complète avec regroupement automatique des chunks splittés</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{analysis['basic_stats']['total_nodes']}</div>
                        <div class="stat-label">Entités visualisées</div>
                    </div>
                    <div class="stat-card grouping">
                        <div class="stat-value grouping">{grouping_stats['total_original_chunks']}</div>
                        <div class="stat-label">Chunks originaux</div>
                    </div>
                    <div class="stat-card grouping">
                        <div class="stat-value grouping">{grouping_stats['grouped_entities']}</div>
                        <div class="stat-label">Entités regroupées</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{analysis['basic_stats']['total_edges']}</div>
                        <div class="stat-label">Relations</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{analysis['file_analysis']['file_count']}</div>
                        <div class="stat-label">Fichiers</div>
                    </div>
                    <div class="stat-card grouping">
                        <div class="stat-value grouping">{grouping_stats['compression_ratio']:.1f}x</div>
                        <div class="stat-label">Ratio compression</div>
                    </div>
                </div>

                <div class="section">
                    <h2>📦 Regroupement Automatique</h2>
                    <div class="compression-info">
                        <strong>Regroupement efficace :</strong> 
                        {grouping_stats['total_original_chunks']} chunks originaux regroupés en 
                        {grouping_stats['visualized_entities']} entités visualisées 
                        (ratio de compression: {grouping_stats['compression_ratio']:.1f}x)
                    </div>

                    <table>
                        <tr>
                            <th>Métrique</th>
                            <th>Valeur</th>
                            <th>Description</th>
                        </tr>
                        <tr>
                            <td>Entités totales indexées</td>
                            <td>{grouping_stats['total_entity_groups']}</td>
                            <td>Nombre total d'entités complètes dans le système</td>
                        </tr>
                        <tr>
                            <td>Entités visualisées</td>
                            <td>{grouping_stats['visualized_entities']}</td>
                            <td>Entités affichées dans le graphique (après filtrage)</td>
                        </tr>
                        <tr>
                            <td>Entités regroupées</td>
                            <td>{grouping_stats['grouped_entities']}</td>
                            <td>Entités constituées de plusieurs chunks</td>
                        </tr>
                        <tr>
                            <td>Entités incomplètes</td>
                            <td><span class="badge {'badge-success' if grouping_stats['incomplete_entities'] == 0 else 'badge-warning'}">{grouping_stats['incomplete_entities']}</span></td>
                            <td>Entités avec des chunks manquants</td>
                        </tr>
                        <tr>
                            <td>Chunks originaux</td>
                            <td>{grouping_stats['total_original_chunks']}</td>
                            <td>Nombre total de chunks avant regroupement</td>
                        </tr>
                    </table>
                </div>

                <div class="section">
                    <h2>🏗️ Structure du Graphe</h2>
                    <p><strong>Connexité:</strong> 
                        <span class="badge {'badge-success' if analysis['basic_stats']['is_connected'] else 'badge-warning'}">
                            {'Connecté' if analysis['basic_stats']['is_connected'] else 'Non connecté'}
                        </span>
                    </p>
                    <p><strong>Cycles:</strong> 
                        <span class="badge {'badge-success' if analysis['basic_stats']['is_dag'] else 'badge-danger'}">
                            {'Sans cycle (DAG)' if analysis['basic_stats']['is_dag'] else f"{len(analysis['cycles'])} cycle(s) détecté(s)"}
                        </span>
                    </p>
                    <p><strong>Composantes:</strong> {analysis['components'].get('count', 0)} 
                        (plus grande: {analysis['components'].get('largest_size', 0)} entités)</p>
                    <p><strong>Densité:</strong> {analysis['basic_stats']['density']:.3f}</p>
                </div>

                <div class="section">
                    <h2>📁 Répartition par Fichier</h2>
                    <table>
                        <tr>
                            <th>Fichier</th>
                            <th>Entités</th>
                            <th>Pourcentage</th>
                        </tr>
        """

        # Ajouter les fichiers (reste identique)
        total_entities = analysis['basic_stats']['total_nodes']
        sorted_files = sorted(
            analysis['file_analysis']['entities_per_file'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for filename, count in sorted_files[:15]:
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            html += f"""
                        <tr>
                            <td>{filename}</td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
            """

        html += """
                    </table>
                </div>

                <div class="section">
                    <h2>🏷️ Types d'Entités</h2>
                    <table>
                        <tr>
                            <th>Type</th>
                            <th>Nombre</th>
                            <th>Pourcentage</th>
                        </tr>
        """

        # Ajouter les types d'entités
        for entity_type, count in sorted(analysis['entity_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_entities * 100) if total_entities > 0 else 0
            html += f"""
                        <tr>
                            <td>{entity_type}</td>
                            <td>{count}</td>
                            <td>{percentage:.1f}%</td>
                        </tr>
            """

        html += """
                    </table>
                </div>
        """

        # Le reste du HTML est identique...

        # Ajouter les entités les plus centrales si disponibles
        if 'centrality' in analysis and analysis['centrality']:
            html += """
                <div class="section">
                    <h2>⭐ Entités les Plus Importantes</h2>
                    <table>
                        <tr>
                            <th>Entité</th>
                            <th>Type</th>
                            <th>Centralité</th>
                            <th>Regroupée</th>
                        </tr>
            """

            for node_id, centrality in analysis['centrality'].get('most_central', [])[:10]:
                node_attr = self.graph.nodes.get(node_id, {})
                entity_name = node_attr.get('entity_name', node_id)
                entity_type = node_attr.get('entity_type', 'unknown')
                is_grouped = node_attr.get('is_grouped', False)

                html += f"""
                            <tr>
                                <td>{entity_name}</td>
                                <td>{entity_type}</td>
                                <td>{centrality:.3f}</td>
                                <td>{'✅' if is_grouped else '❌'}</td>
                            </tr>
                """

            html += """
                    </table>
                </div>
            """

        # Ajouter les cycles si détectés
        if analysis['cycles']:
            html += f"""
                <div class="section">
                    <h2>🔄 Cycles Détectés ({len(analysis['cycles'])})</h2>
                    <p class="badge badge-warning">Attention: Les cycles peuvent indiquer des dépendances circulaires</p>
                    <ol>
            """

            for cycle in analysis['cycles'][:5]:
                cycle_names = []
                for node_id in cycle:
                    node_attr = self.graph.nodes.get(node_id, {})
                    cycle_names.append(node_attr.get('entity_name', node_id))

                html += f"<li>{' → '.join(cycle_names)} → {cycle_names[0]}</li>"

            html += """
                    </ol>
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html


# Fonction d'aide mise à jour
async def create_fortran_dependency_visualization(onto_rag_instance,
                                                  output_file: str = "fortran_dependencies.html",
                                                  max_entities: int = None,
                                                  focus_entity: str = None,
                                                  hierarchical: bool = True,
                                                  include_internal_functions: bool = True):
    """
    Fonction helper pour créer facilement une visualisation de dépendances avec regroupement.

    Args:
        onto_rag_instance: Instance de OntoRAG initialisée
        output_file: Nom du fichier HTML de sortie
        max_entities: Nombre maximum d'entités à visualiser
        focus_entity: Entité sur laquelle se concentrer
        hierarchical: Utiliser un layout hiérarchique
        include_internal_functions: Inclure les fonctions internes
    """

    # S'assurer que le système de contexte est initialisé
    if not onto_rag_instance.context_provider:
        await onto_rag_instance.initialize_context_provider()

    # Créer le visualiseur
    visualizer = FortranDependencyVisualizer(onto_rag_instance)

    # Construire le graphe avec regroupement
    await visualizer.build_dependency_graph(
        max_entities=max_entities,
        include_internal_functions=include_internal_functions,
        include_variables=False,
        focus_entity=focus_entity,
        max_depth=3
    )

    # Générer la visualisation
    visualizer.visualize_with_pyvis(
        output_file=output_file,
        hierarchical=hierarchical,
        height="900px",
        width="100%"
    )

    # Générer aussi le rapport
    report_file = output_file.replace('.html', '_report.html')
    visualizer.generate_dependency_report(report_file)

    return visualizer

