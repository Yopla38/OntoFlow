# output/graph_visualizer.py (VERSION SIMPLIFIÉE)
"""
Visualiseur de graphes de dépendances Fortran SIMPLIFIÉ.
Peut fonctionner directement avec des entités sans document_store.
"""
import json
import os
import webbrowser
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import networkx as nx
from pyvis.network import Network

logger = logging.getLogger(__name__)


class SimpleFortranDependencyVisualizer:
    """
    Version simplifiée du visualiseur qui travaille directement avec des entités.
    N'a PAS BESOIN de document_store ou rag_engine.
    """

    def __init__(self):
        # Graphe NetworkX
        self.graph = nx.DiGraph()

        # Entités à visualiser (fournies directement)
        self.entities: List[Any] = []

        # Configuration des couleurs et formes (identique à l'original)
        self.entity_colors = {
            'module': '#4285F4',
            'program': '#EA4335',
            'subroutine': '#34A853',
            'function': '#FBBC05',
            'internal_function': '#FF9800',
            'type_definition': '#9C27B0',
            'interface': '#607D8B',
            'variable_declaration': '#795548',
            'parameter': '#009688',
            'block': '#795548',  # Pour f2py
            'default': '#9AA0A6'
        }

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
            'block': 'square',  # Pour f2py
            'default': 'circle'
        }

        self.edge_colors = {
            'contains': '#1976D2',
            'uses': '#388E3C',
            'calls': '#F57C00',
            'depends_on': '#7B1FA2',
            'internal': '#D32F2F',
            'cross_file': '#455A64'
        }

    def _get_access_level_color(self, base_hex: str, access_level: Optional[str]) -> str:
        """
        Prend une couleur hex de base et la modifie selon le niveau d'accès.
        - 'public' -> plus foncée
        - 'private' -> plus claire
        - None -> inchangée
        """
        if access_level not in ['public', 'private']:
            return base_hex

        # 1. Convertir le hex en RGB
        base_hex = base_hex.lstrip('#')
        try:
            rgb = tuple(int(base_hex[i:i + 2], 16) for i in (0, 2, 4))
            r, g, b = rgb
        except (ValueError, IndexError):
            return base_hex  # Retourner la couleur de base en cas d'erreur

        # 2. Modifier les composantes RGB
        if access_level == 'public':
            # Assombrir en réduisant la valeur (facteur de 0.7)
            r = int(r * 0.7)
            g = int(g * 0.7)
            b = int(b * 0.7)
        elif access_level == 'private':
            # Éclaircir en mélangeant avec du blanc (poids de 60% de blanc)
            r = int(r * 0.4 + 255 * 0.6)
            g = int(g * 0.4 + 255 * 0.6)
            b = int(b * 0.4 + 255 * 0.6)

        # 3. Assurer que les valeurs restent dans l'intervalle [0, 255]
        r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

        # 4. Reconvertir en hexadécimal
        return f'#{r:02x}{g:02x}{b:02x}'

    def build_dependency_graph_from_entities(self,
                                             entities: List[Any],
                                             max_entities: Optional[int] = None,
                                             include_internal_functions: bool = True,
                                             include_variables: bool = False,
                                             focus_entity: Optional[str] = None) -> nx.DiGraph:
        """
        Construit le graphe directement depuis une liste d'entités.
        AUCUN BESOIN de document_store ou EntityManager.
        """
        logger.info(f"🔍 Construction du graphe depuis {len(entities)} entités...")

        # Stocker les entités
        self.entities = entities

        # Réinitialiser le graphe
        self.graph = nx.DiGraph()

        # 1. Filtrer les entités selon les critères
        filtered_entities = self._filter_entities(
            entities, include_internal_functions, include_variables
        )

        # 2. Si focus_entity, filtrer autour de lui
        if focus_entity:
            filtered_entities = self._filter_around_focus_simple(
                focus_entity, filtered_entities
            )

        # 3. Appliquer la limite
        if max_entities:
            filtered_entities = filtered_entities[:max_entities]

        logger.info(f"📋 {len(filtered_entities)} entités sélectionnées pour visualisation")

        # 4. Ajouter les nœuds au graphe
        self._add_entity_nodes_simple(filtered_entities)

        # 5. Ajouter les relations
        self._add_entity_relationships_simple(filtered_entities)

        logger.info(f"✅ Graphe construit: {len(self.graph.nodes)} nœuds, {len(self.graph.edges)} arêtes")

        return self.graph

    def _filter_entities(self,
                         entities: List[Any],
                         include_internal_functions: bool,
                         include_variables: bool) -> List[Any]:
        """Filtre les entités selon les critères - VERSION CORRIGÉE"""

        # Types de base toujours inclus - AJOUT DES PARAMÈTRES
        allowed_types = {
            'module', 'program', 'subroutine', 'function',
            'type_definition', 'interface', 'block',
            'parameter'  # ← PARAMÈTRES TOUJOURS INCLUS
        }

        if include_internal_functions:
            allowed_types.add('internal_function')

        if include_variables:
            allowed_types.update({'variable_declaration', 'variable'})  # Variables normales seulement si demandé

        # Filtrer par type
        filtered = [
            entity for entity in entities
            if entity.entity_type in allowed_types
        ]

        # Trier par importance (mise à jour des priorités)
        type_priorities = {
            'module': 1,
            'program': 1,
            'type_definition': 2,
            'parameter': 3,  # ← PRIORITÉ ÉLEVÉE POUR LES PARAMÈTRES
            'interface': 4,
            'subroutine': 5,
            'function': 5,
            'block': 6,
            'internal_function': 7,
            'variable_declaration': 8,
            'variable': 8
        }

        sorted_entities = sorted(
            filtered,
            key=lambda x: (
                type_priorities.get(x.entity_type, 999),
                -len(x.dependencies) if hasattr(x, 'dependencies') else 0,
                -len(x.called_functions) if hasattr(x, 'called_functions') else 0,
                x.entity_name
            )
        )

        return sorted_entities

    def _filter_around_focus_simple(self,
                                    focus_entity: str,
                                    entities: List[Any]) -> List[Any]:
        """Filtre autour d'une entité focus (version simplifiée)"""

        # Trouver l'entité focus
        focus = None
        for entity in entities:
            if entity.entity_name == focus_entity:
                focus = entity
                break

        if not focus:
            logger.warning(f"⚠️ Entité focus '{focus_entity}' non trouvée")
            return entities

        # Créer un index nom -> entité
        entity_map = {entity.entity_name: entity for entity in entities}

        # Entités à inclure (commence avec l'entité focus)
        included_entities = {focus_entity}
        to_explore = [focus]

        # Exploration en largeur (2 niveaux)
        for level in range(2):
            next_to_explore = []

            for current_entity in to_explore:
                # Ajouter les dépendances
                for dep_info in current_entity.dependencies:
                    # Gérer les deux formats (ancien et nouveau)
                    if isinstance(dep_info, dict):
                        dep_name = dep_info.get('name')
                    else:
                        dep_name = dep_info  # Format ancien (string)

                    if dep_name and dep_name in entity_map and dep_name not in included_entities:
                        included_entities.add(dep_name)
                        next_to_explore.append(entity_map[dep_name])

                # Ajouter les appels
                if hasattr(current_entity, 'called_functions'):
                    for call_info in current_entity.called_functions:
                        # Gérer les deux formats (ancien et nouveau)
                        if isinstance(call_info, dict):
                            call_name = call_info.get('name')
                        else:
                            call_name = call_info  # Format ancien (string)

                        if call_name and call_name in entity_map and call_name not in included_entities:
                            included_entities.add(call_name)
                            next_to_explore.append(entity_map[call_name])

                # Ajouter le parent
                if hasattr(current_entity, 'parent_entity') and current_entity.parent_entity:
                    parent = current_entity.parent_entity
                    if parent in entity_map and parent not in included_entities:
                        included_entities.add(parent)
                        next_to_explore.append(entity_map[parent])

            to_explore = next_to_explore

        # Filtrer les entités pour ne garder que celles incluses
        filtered = [entity for entity in entities if entity.entity_name in included_entities]

        logger.info(f"🎯 Focus sur {focus_entity}: {len(filtered)} entités liées")

        return filtered

    def _add_entity_nodes_simple(self, entities: List[Any]):
        """Ajoute les nœuds d'entités au graphe (version simplifiée)"""

        for entity in entities:
            # Label affiché
            display_label = entity.entity_name

            # Ajouter indicateur si entité groupée
            if hasattr(entity, 'is_grouped') and entity.is_grouped:
                display_label += " [📦]"

            # Description pour tooltip
            description = self._create_entity_description_simple(entity)

            # Taille du nœud
            size = self._calculate_node_size_simple(entity)

            # 1. Obtenir la couleur de base en fonction du type de l'entité.
            base_color = self.entity_colors.get(entity.entity_type, self.entity_colors['default'])

            # 2. Obtenir le niveau d'accès de l'entité (public/private).
            access_level = getattr(entity, 'access_level', None)

            # 3. Calculer la couleur finale en utilisant notre nouvelle fonction helper.
            final_color = self._get_access_level_color(base_color, access_level)

            # Couleur de bordure si incomplète
            border_color = '#dc3545' if not getattr(entity, 'is_complete', True) else None

            # Ajouter le nœud
            self.graph.add_node(
                entity.entity_name,
                label=display_label,
                title=description,
                size=size,
                color=final_color,
                shape=self.entity_shapes.get(entity.entity_type, self.entity_shapes['default']),
                entity_type=entity.entity_type,
                filepath=getattr(entity, 'filepath', ''),
                filename=getattr(entity, 'filename', ''),
                border_color=border_color,
                access_level=access_level
            )

    def _create_entity_description_simple(self, entity) -> str:
        """Crée une description pour le tooltip (version simplifiée)"""

        html = f"<div style='max-width:400px;'>"
        html += f"<h3>{entity.entity_name}</h3>"
        html += f"<p><strong>Type:</strong> {entity.entity_type}</p>"

        access_level = getattr(entity, 'access_level', None)
        if access_level:
            color = "#2C3E50"
            html += f"<p><strong>Visibilité:</strong> <span style='font-weight:bold; color:{color if access_level == 'public' else '#7F8C8D'};'>{access_level.upper()}</span></p>"

        # Fichier
        if hasattr(entity, 'filename') and entity.filename:
            html += f"<p><strong>Fichier:</strong> {entity.filename}</p>"
        elif hasattr(entity, 'filepath') and entity.filepath:
            filename = entity.filepath.split('/')[-1]
            html += f"<p><strong>Fichier:</strong> {filename}</p>"

        # Lignes
        if hasattr(entity, 'start_line') and hasattr(entity, 'end_line'):
            if entity.start_line and entity.end_line:
                html += f"<p><strong>Lignes:</strong> {entity.start_line}-{entity.end_line}</p>"

        # Signature
        if hasattr(entity, 'signature') and entity.signature:
            html += f"<p><strong>Signature:</strong> {entity.signature}</p>"

        # Confiance
        if hasattr(entity, 'confidence'):
            html += f"<p><strong>Confiance:</strong> {entity.confidence:.3f}</p>"

        # Dépendances
        if hasattr(entity, 'dependencies') and entity.dependencies:
            deps_names = []
            for dep_info in entity.dependencies:
                if isinstance(dep_info, dict):
                    dep_name = dep_info.get('name', '')
                    line = dep_info.get('line', 0)
                    if line > 0:
                        deps_names.append(f"{dep_name} (L{line})")
                    else:
                        deps_names.append(dep_name)
                else:
                    deps_names.append(str(dep_info))  # Format ancien

            deps_list = deps_names[:5]
            html += f"<p><strong>Dépendances:</strong> {', '.join(deps_list)}"
            if len(deps_names) > 5:
                html += f" (+{len(deps_names) - 5} autres)"
            html += "</p>"

        # Appels de fonctions
        if hasattr(entity, 'called_functions') and entity.called_functions:
            calls_names = []
            for call_info in entity.called_functions:
                if isinstance(call_info, dict):
                    call_name = call_info.get('name', '')
                    line = call_info.get('line', 0)
                    if line > 0:
                        calls_names.append(f"{call_name} (L{line})")
                    else:
                        calls_names.append(call_name)
                else:
                    calls_names.append(str(call_info))  # Format ancien

            calls_list = calls_names[:3]
            html += f"<p><strong>Appelle:</strong> {', '.join(calls_list)}"
            if len(calls_names) > 3:
                html += f" (+{len(calls_names) - 3} autres)"
            html += "</p>"

        # Parent
        if hasattr(entity, 'parent_entity') and entity.parent_entity:
            html += f"<p><strong>Parent:</strong> {entity.parent_entity}</p>"

        html += "</div>"
        return html

    def _calculate_node_size_simple(self, entity) -> int:
        """Calcule la taille d'un nœud (version simplifiée)"""

        # Taille de base par type
        type_sizes = {
            'module': 40,
            'program': 35,
            'subroutine': 25,
            'function': 25,
            'internal_function': 15,
            'type_definition': 20,
            'interface': 20,
            'block': 25,  # f2py
            'variable_declaration': 10,
            'parameter': 10
        }

        size = type_sizes.get(entity.entity_type, 20)

        # Bonus pour les dépendances
        if hasattr(entity, 'dependencies'):
            size += min(len(entity.dependencies) * 2, 15)

        # Bonus pour les appels
        if hasattr(entity, 'called_functions'):
            size += min(len(entity.called_functions) * 1, 10)

        return min(size, 70)  # Limite max

    def _add_entity_relationships_simple(self, entities: List[Any]):
        """Ajoute les relations entre entités (version simplifiée)"""

        logger.info("🔗 Ajout des relations...")

        # Index nom -> entité
        entity_map = {entity.entity_name: entity for entity in entities}
        edge_count = 0

        for entity in entities:
            source_name = entity.entity_name

            # 1. Relations de dépendance (USE statements)
            if hasattr(entity, 'dependencies'):
                for dep_info in entity.dependencies:
                    # Gérer les deux formats (ancien et nouveau)
                    if isinstance(dep_info, dict):
                        dep_name = dep_info.get('name')
                        dep_line = dep_info.get('line', 0)
                        line_info = f" (ligne {dep_line})" if dep_line > 0 else ""
                    else:
                        dep_name = dep_info  # Format ancien (string)
                        line_info = ""

                    if dep_name and dep_name in entity_map:
                        self.graph.add_edge(
                            source_name, dep_name,
                            label="uses",
                            color=self.edge_colors['uses'],
                            title=f"{source_name} uses {dep_name}{line_info}",
                            width=2.0,
                            edge_type='uses'
                        )
                        edge_count += 1

            # 2. Relations d'appels de fonctions
            if hasattr(entity, 'called_functions'):
                for call_info in entity.called_functions:
                    # Gérer les deux formats (ancien et nouveau)
                    if isinstance(call_info, dict):
                        called_name = call_info.get('name')
                        call_line = call_info.get('line', 0)
                        line_info = f" (ligne {call_line})" if call_line > 0 else ""
                    else:
                        called_name = call_info  # Format ancien (string)
                        line_info = ""

                    if called_name and called_name in entity_map:
                        self.graph.add_edge(
                            source_name, called_name,
                            label="calls",
                            color=self.edge_colors['calls'],
                            title=f"{source_name} calls {called_name}{line_info}",
                            dashes=True,
                            width=1.5,
                            edge_type='calls'
                        )
                        edge_count += 1

            # 3. Relations hiérarchiques (parent-enfant)
            if hasattr(entity, 'parent_entity') and entity.parent_entity:
                if entity.parent_entity in entity_map:
                    self.graph.add_edge(
                        entity.parent_entity, source_name,
                        label="contains",
                        color=self.edge_colors['contains'],
                        title=f"{entity.parent_entity} contains {source_name}",
                        width=3.0,
                        edge_type='contains'
                    )
                    edge_count += 1

        logger.info(f"✅ {edge_count} relations ajoutées")

    def visualize_with_pyvis(self,
                             output_file: str = "fortran_dependencies.html",
                             hierarchical: bool = False,
                             height: str = "800px",
                             width: str = "100%",
                             physics_enabled: bool = True):
        """Génère une visualisation interactive enrichie avec popup et contrôles"""

        if not self.graph.nodes:
            logger.error("❌ Aucun graphe à visualiser")
            return

        logger.info(f"🎨 Génération de la visualisation enrichie...")

        # Créer le réseau PyVis
        net = Network(
            height=height,
            width=width,
            directed=True,
            notebook=False,
            bgcolor="#ffffff",
            font_color="#000000"
        )

        # Configuration des options
        self._configure_pyvis_options_enhanced(net, hierarchical, physics_enabled)

        # Enrichir les données des nœuds avec le code source
        self._enrich_nodes_with_code()

        # Ajouter les nœuds enrichis
        for node_id, attr in self.graph.nodes(data=True):
            color_config = attr.get('color', self.entity_colors['default'])

            if attr.get('border_color'):
                color_config = {
                    'background': color_config,
                    'border': attr['border_color'],
                    'highlight': {'background': color_config, 'border': attr['border_color']}
                }

            # Données enrichies pour le popup
            net.add_node(
                node_id,
                label=attr.get('label', node_id),
                title=attr.get('title', ''),
                color=color_config,
                shape=attr.get('shape', 'circle'),
                size=attr.get('size', 20),
                font={'size': 14},
                borderWidth=3 if attr.get('border_color') else 2,
                # ✅ STOCKER LE CODE SOURCE DANS LE NOEUD
                source_code=attr.get('source_code', ''),
                entity_signature=attr.get('signature', ''),
                entity_filepath=attr.get('filepath', ''),
                entity_filename=attr.get('filename', ''),
                entity_start_line=attr.get('start_line', 0),
                entity_end_line=attr.get('end_line', 0),
                entity_dependencies=attr.get('dependencies', []),
                entity_called_functions=attr.get('called_functions', []),
                entity_parent=attr.get('parent_entity', '')
            )



        # Ajouter les arêtes (identique)
        for source, target, attr in self.graph.edges(data=True):
            edge_type = attr.get('edge_type', 'default')
            width = 1.0
            dashes = False

            if edge_type == 'contains':
                width = 3.0
            elif edge_type == 'uses':
                width = 2.0
            elif edge_type == 'calls':
                width = 1.5
                dashes = True

            net.add_edge(
                source, target,
                label=attr.get('label', ''),
                title=attr.get('title', ''),
                color=attr.get('color', '#666666'),
                width=width,
                dashes=dashes,
                arrows="to"
            )

        # Sauvegarder
        net.save_graph(output_file)

        # Enrichir le HTML avec les nouvelles fonctionnalités
        self._enhance_html_with_developer_features(output_file)

        # Statistiques
        stats = self._calculate_stats()
        logger.info(f"✅ Visualisation enrichie sauvegardée: {output_file}")
        logger.info(f"📊 {stats['nodes']} nœuds, {stats['edges']} arêtes")

        # Ouvrir dans le navigateur
        try:
            webbrowser.open('file://' + os.path.abspath(output_file), new=2)
        except Exception as e:
            logger.warning(f"⚠️ Impossible d'ouvrir le navigateur: {e}")

    def _enrich_nodes_with_code(self):
        """Enrichit les nœuds avec le code source des entités"""

        print("🔍 Enrichissement des nœuds avec le code source...")

        for node_id in self.graph.nodes():
            # Trouver l'entité correspondante
            entity = None
            for e in self.entities:
                if e.entity_name == node_id:
                    entity = e
                    break

            if entity:
                # Extraire le code source réel
                source_code = self._extract_entity_source_code(entity)

                print(f"   📄 Code pour {entity.entity_name}: {len(source_code)} caractères")

                # Enrichir le nœud avec TOUTES les données nécessaires
                self.graph.nodes[node_id].update({
                    'source_code': source_code,  # ✅ VRAI CODE SOURCE
                    'signature': getattr(entity, 'signature', ''),
                    'parent_entity': getattr(entity, 'parent_entity', ''),
                    'dependencies': list(getattr(entity, 'dependencies', [])),
                    'called_functions': list(getattr(entity, 'called_functions', [])),
                    'start_line': getattr(entity, 'start_line', 0),
                    'end_line': getattr(entity, 'end_line', 0),
                    'filepath': getattr(entity, 'filepath', ''),
                    'filename': getattr(entity, 'filename', ''),
                    'entity_type_detailed': entity.entity_type,  # Type détaillé
                    'confidence': getattr(entity, 'confidence', 0.0)
                })
            else:
                print(f"   ⚠️ Entité non trouvée pour le nœud: {node_id}")

    def _extract_entity_source_code(self, entity) -> str:
        """Extrait le VRAI code source d'une entité depuis le fichier"""

        # Méthode 1: Lire directement depuis le fichier avec les numéros de ligne
        if (hasattr(entity, 'filepath') and entity.filepath and
                hasattr(entity, 'start_line') and hasattr(entity, 'end_line') and
                entity.start_line and entity.end_line and
                os.path.exists(entity.filepath)):

            try:
                with open(entity.filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Extraire les lignes correspondantes (conversion 1-indexé vers 0-indexé)
                start_idx = max(0, entity.start_line - 1)
                end_idx = min(len(lines), entity.end_line)

                if start_idx < len(lines):
                    extracted_lines = lines[start_idx:end_idx]
                    code = ''.join(extracted_lines)

                    # Nettoyer le code (supprimer les lignes vides en trop au début/fin)
                    code = code.strip()

                    if code:
                        # Ajouter des informations de contexte
                        header = f"! Fichier: {os.path.basename(entity.filepath)}\n"
                        header += f"! Lignes: {entity.start_line}-{entity.end_line}\n"
                        header += f"! Entité: {entity.entity_name} ({entity.entity_type})\n"
                        header += "! " + "=" * 50 + "\n\n"

                        return header + code

            except Exception as e:
                logger.debug(f"Erreur lecture fichier {entity.filepath}: {e}")

        # Méthode 2: Si l'entité a des chunks avec du texte
        if hasattr(entity, 'chunks') and entity.chunks:
            code_parts = []
            for chunk in entity.chunks:
                if isinstance(chunk, dict) and 'text' in chunk:
                    code_parts.append(chunk['text'])
                elif hasattr(chunk, 'text'):
                    code_parts.append(chunk.text)

            if code_parts:
                full_code = '\n'.join(code_parts)
                header = f"! Code depuis chunks pour {entity.entity_name}\n"
                header += "! " + "=" * 50 + "\n\n"
                return header + full_code

        # Méthode 3: Code par défaut si rien ne fonctionne
        return f"""! Code source non disponible pour {entity.entity_name}
    ! Type: {entity.entity_type}
    ! Fichier: {getattr(entity, 'filepath', 'Non spécifié')}
    ! Lignes: {getattr(entity, 'start_line', '?')}-{getattr(entity, 'end_line', '?')}
    !
    ! Raisons possibles:
    ! - Fichier non trouvé ou non accessible
    ! - Informations de ligne incorrectes
    ! - Entité générée sans code source associé"""

    def _enhance_html_with_developer_features(self, output_file: str):
        """Enrichit le HTML avec CodeMirror pour la coloration syntaxique Fortran"""

        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # CSS enrichi avec support CodeMirror
            enhanced_css = """
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; padding: 0; background-color: #f8f9fa; 
                }

                /* Header avec titre et contrôles */
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 15px; text-align: center;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); position: relative;
                }

                /* Contrôles en haut à droite */
                .controls {
                    position: absolute; top: 10px; right: 10px;
                    display: flex; gap: 8px;
                }
                .control-btn {
                    background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.3);
                    padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;
                    transition: all 0.3s;
                }
                .control-btn:hover { background: rgba(255,255,255,0.3); }
                .control-btn.active { background: rgba(255,255,255,0.4); }

                /* Légende en bas à gauche */
                .legend {
                    position: fixed; bottom: 20px; left: 20px; 
                    background: white; border-radius: 8px; padding: 15px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15); max-width: 300px;
                    font-size: 12px; z-index: 1000; max-height: 400px; overflow-y: auto;
                }
                .legend h4 { margin: 0 0 10px 0; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }
                .legend-item { display: flex; align-items: center; margin: 5px 0; }
                .legend-color { width: 16px; height: 16px; margin-right: 8px; border-radius: 3px; }
                .legend-shape { margin-right: 8px; font-size: 14px; }

                /* Popup pour les détails de l'entité */
                .entity-popup {
                    position: fixed; 
                    top: 50%; left: 50%; 
                    transform: translate(-50%, -50%);
                    background: white; 
                    border-radius: 12px; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    width: 90vw; max-width: 1400px; 
                    height: 85vh; 
                    z-index: 2000; 
                    display: none;        /* ✅ UNE SEULE DÉCLARATION display */
                    overflow: hidden; 
                    flex-direction: column;  /* ✅ Prêt pour flex quand affiché */
                    cursor: default;
                }
                .entity-popup.show {
                    display: flex !important;  /* Devient flex seulement quand .show est ajoutée */
                }
                .popup-overlay {
                    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                    background: rgba(0,0,0,0.5); z-index: 1999; display: none;
                    display: none !important;
                }
                .popup-overlay.show {
                    display: block !important;
                }
                .popup-header {
                    background: #343a40; color: white; padding: 15px; position: relative;
                    border-radius: 12px 12px 0 0; flex-shrink: 0;
                }
                .popup-close {
                    position: absolute; top: 10px; right: 15px; background: none; border: none;
                    color: white; font-size: 24px; cursor: pointer; padding: 0; width: 30px; height: 30px;
                }
                .popup-content {
                    padding: 20px; flex: 1; overflow-y: auto; display: flex; gap: 20px;
                }

                /* Deux colonnes dans le popup */
                .popup-left {
                    flex: 1; min-width: 300px; max-width: 400px;
                }
                .popup-right {
                    flex: 2; min-width: 500px;
                }

                .popup-section { margin-bottom: 20px; }
                .popup-section h4 { 
                    margin: 0 0 8px 0; color: #495057; border-bottom: 1px solid #dee2e6; 
                    padding-bottom: 4px; font-size: 14px;
                }

                /* NOUVEAU : Container CodeMirror */
                .codemirror-container {
                    border: 1px solid #ddd; border-radius: 8px;
                    height: 500px; width: 100%; overflow: hidden;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }

                /* Personnalisation CodeMirror pour Fortran */
                .CodeMirror {
                    height: 100% !important;
                    font-family: 'Fira Code', 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
                    font-size: 13px !important;
                    line-height: 1.4 !important;
                }

                /* Thème sombre pour CodeMirror */
                .cm-s-monokai .CodeMirror {
                    background: #272822 !important;
                    color: #f8f8f2 !important;
                }

                .popup-list {
                    list-style: none; padding: 0; margin: 0;
                }
                .popup-list li {
                    padding: 4px 0; border-bottom: 1px solid #f0f0f0;
                    font-family: monospace; font-size: 12px;
                }

                /* Graphe principal */
                #mynetwork {
                    border: 1px solid #ddd; margin: 0; border-radius: 0;
                    box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
                }
            </style>
            """

            # Ajouter les liens CDN CodeMirror AVANT le CSS personnalisé
            codemirror_links = """
            <!-- CodeMirror CSS -->
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/codemirror.min.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/theme/monokai.min.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/addon/scroll/simplescrollbars.min.css">

            <!-- CodeMirror JavaScript -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/codemirror.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/mode/fortran/fortran.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/addon/scroll/simplescrollbars.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/addon/selection/active-line.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/addon/edit/matchbrackets.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/addon/fold/foldcode.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/addon/fold/foldgutter.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/codemirror/6.65.7/addon/fold/brace-fold.min.js"></script>
            """

            content = content.replace('</head>', codemirror_links + enhanced_css + '</head>')

            # Header enrichi avec contrôles (identique)
            header_html = """
            <div class="header">
                <h1>🔍 Analyseur Fortran Interactif</h1>
                <p>Cliquez sur une entité pour voir son code source avec coloration syntaxique</p>
                <div class="controls">
                    <button class="control-btn" id="togglePhysics" onclick="togglePhysics()">⚡ Physique</button>
                    <button class="control-btn" onclick="network.fit()">🎯 Ajuster</button>
                    <button class="control-btn" onclick="toggleLegend()">📋 Légende</button>
                    <button class="control-btn" onclick="exportView()">💾 Export</button>
                </div>
            </div>
            """

            # Légende interactive (identique)
            legend_html = f"""
                    <div class="legend" id="legend">
                        <h4>🎨 Légende Interactive</h4>

                        <div class="popup-section">
                            <strong>Types d'entités :</strong>
                            <div class="legend-item"><div class="legend-color" style="background:{self.entity_colors['module']}"></div><span>📦 Module</span></div>
                            <div class="legend-item"><div class="legend-color" style="background:{self.entity_colors['program']}"></div><span>🚀 Programme</span></div>
                            <div class="legend-item"><div class="legend-color" style="background:{self.entity_colors['subroutine']}"></div><span>🔧 Subroutine</span></div>
                            <div class="legend-item"><div class="legend-color" style="background:{self.entity_colors['function']}"></div><span>⚙️ Function</span></div>
                            <div class="legend-item"><div class="legend-color" style="background:{self.entity_colors['type_definition']}"></div><span>🏷️ Type défini</span></div>
                        </div>

                        <!-- NOUVELLE SECTION POUR LA VISIBILITÉ -->
                        <div class="popup-section">
                            <strong>Visibilité :</strong>
                            <div class="legend-item"><div class="legend-color" style="background:{self._get_access_level_color('#95A5A6', 'public')}"></div><span>PUBLIC (couleur foncée)</span></div>
                            <div class="legend-item"><div class="legend-color" style="background:{self._get_access_level_color('#95A5A6', 'private')}"></div><span>PRIVATE (couleur claire)</span></div>
                        </div>

                        <div class="popup-section">
                            <strong>Relations :</strong>
                            <div class="legend-item"><div class="legend-color" style="background:{self.edge_colors['uses']}"></div><span>→ Uses (USE)</span></div>
                            <div class="legend-item"><div class="legend-color" style="background:{self.edge_colors['calls']}"></div><span>⤷ Calls (appel)</span></div>
                            <div class="legend-item"><div class="legend-color" style="background:{self.edge_colors['contains']}"></div><span>▣ Contains</span></div>
                        </div>
                    </div>
                    """

            # Popup modifié pour CodeMirror
            popup_html = """
            <div class="popup-overlay" id="popupOverlay" onclick="closeEntityPopup()"></div>
            <div class="entity-popup" id="entityPopup">
                <div class="popup-header">
                    <h3 id="popupTitle">Détails de l'entité</h3>
                    <button class="popup-close" id="popupCloseBtn" title="Fermer (Echap)">&times;</button>                </div>
                <div class="popup-content" id="popupContent">
                    <!-- Contenu dynamique -->
                </div>
            </div>
            """

            content = content.replace('<body>', '<body>' + header_html + legend_html + popup_html)

            # JavaScript enrichi avec CodeMirror
            enhanced_js = """
            <script>
                let physicsEnabled = true;
                let legendVisible = true;
                let nodeData = {}; 
                let isDragging = false;
                let hasDragged = false;
                let dragStartTime = 0;
                let dragStartPos = {x: 0, y: 0};
                let currentCodeMirror = null; // ✅ NOUVEAU : Instance CodeMirror actuelle

                // Fonction pour basculer la physique
                function togglePhysics() {
                    physicsEnabled = !physicsEnabled;
                    network.setOptions({physics: {enabled: physicsEnabled}});

                    const btn = document.getElementById('togglePhysics');
                    btn.classList.toggle('active', physicsEnabled);
                    btn.textContent = physicsEnabled ? '⚡ Physique ON' : '⚡ Physique OFF';

                    console.log('Physics ' + (physicsEnabled ? 'enabled' : 'disabled'));
                }

                // Fonction pour basculer la légende
                function toggleLegend() {
                    legendVisible = !legendVisible;
                    const legend = document.getElementById('legend');
                    legend.style.display = legendVisible ? 'block' : 'none';
                }

                // Fonction pour exporter la vue (placeholder)
                function exportView() {
                    const stats = {
                        nodes: network.body.data.nodes.length,
                        edges: network.body.data.edges.length,
                        physics: physicsEnabled
                    };
                    alert('Export View\\n\\nNodes: ' + stats.nodes + '\\nEdges: ' + stats.edges + '\\nPhysics: ' + stats.physics);
                }

                // ✅ NOUVELLE FONCTION : Créer une instance CodeMirror
                function createCodeMirrorInstance(container, code, language = 'fortran') {
                    // Nettoyer le container
                    container.innerHTML = '';

                    // Créer l'instance CodeMirror
                    const editor = CodeMirror(container, {
                        value: code,
                        mode: language,
                        theme: 'monokai',
                        lineNumbers: true,
                        readOnly: true,
                        lineWrapping: true,
                        styleActiveLine: true,
                        matchBrackets: true,
                        scrollbarStyle: 'simple',
                        viewportMargin: Infinity, // Afficher tout le code
                        foldGutter: true,
                        gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"]
                    });

                    // Forcer le refresh après un délai (nécessaire pour les popups)
                    setTimeout(() => {
                        editor.refresh();
                    }, 100);

                    return editor;
                }

                // Fonction pour ouvrir le popup d'entité
                function openEntityPopup(nodeId) {
                    console.log('🔍 Opening popup for:', nodeId);
                    
                    const node = network.body.data.nodes.get(nodeId);
                    if (!node) {
                        console.error('❌ Node not found:', nodeId);
                        return;
                    }
                    
                    const entityData = extractEntityDataFromNode(node);
                    console.log('📋 Entity data:', entityData); 
                    
                    const popupContent = buildEntityPopupContent(entityData);
                    
                    document.getElementById('popupTitle').textContent = 
                        `📋 ${entityData.entity_name} (${entityData.entity_type})`;
                    document.getElementById('popupContent').innerHTML = popupContent;
                    
                    // ✅ AFFICHER LE POPUP AVEC CLASSES CSS
                    const overlay = document.getElementById('popupOverlay');
                    const popup = document.getElementById('entityPopup');
                    
                    overlay.style.display = 'block';
                    popup.classList.add('show');  // ✅ Ajouter la classe au lieu de style direct
                    
                    setupPopupCloseHandlers();
                    
                    setTimeout(() => {
                        loadCodeWithCodeMirror(entityData);
                    }, 150);
                }
                // ✅ NOUVELLE FONCTION : Configuration des gestionnaires de fermeture
                function setupPopupCloseHandlers() {
                    // Gestionnaire pour l'overlay (clic à côté)
                    const overlay = document.getElementById('popupOverlay');
                    overlay.onclick = function(e) {
                        console.log('🖱️ Clicked on overlay');
                        closeEntityPopup();
                    };
                    
                    // Gestionnaire pour le bouton X
                    const closeBtn = document.getElementById('popupCloseBtn');
                    closeBtn.onclick = function(e) {
                        console.log('🖱️ Clicked close button');
                        e.stopPropagation(); // Empêcher la propagation
                        closeEntityPopup();
                    };
                    
                    // Empêcher la fermeture quand on clique sur le popup lui-même
                    const popup = document.getElementById('entityPopup');
                    popup.onclick = function(e) {
                        e.stopPropagation(); // Empêcher la propagation vers l'overlay
                    };
                }
                
                // ✅ FONCTION POUR FERMER LE POPUP (améliorée)
                function closeEntityPopup() {
    console.log('🔒 Closing popup');
    
    // Nettoyer CodeMirror
    if (currentCodeMirror) {
        try {
            currentCodeMirror.toTextArea();
        } catch (e) {
            console.warn('⚠️ Erreur lors du nettoyage CodeMirror:', e);
        }
        currentCodeMirror = null;
    }
    
    // ✅ CACHER AVEC CLASSES CSS
    const overlay = document.getElementById('popupOverlay');
    const popup = document.getElementById('entityPopup');
    
    if (overlay) overlay.style.display = 'none';
    if (popup) popup.classList.remove('show');  // ✅ Retirer la classe
    
    // Nettoyer les gestionnaires d'événements
    if (overlay) overlay.onclick = null;
    const closeBtn = document.getElementById('popupCloseBtn');
    if (closeBtn) closeBtn.onclick = null;
    
    console.log('✅ Popup closed');
}


                // ✅ NOUVELLE FONCTION : Construire le contenu HTML du popup (modifié pour CodeMirror)
                function buildEntityPopupContent(data) {
                    let html = '<div class="popup-left">';

                    // Section informations générales
                    html += '<div class="popup-section">';
                    html += '<h4>ℹ️ Informations générales</h4>';
                    html += `<p><strong>Type :</strong> ${data.entity_type}</p>`;
                    if (data.filepath) html += `<p><strong>Fichier :</strong> ${data.filepath}</p>`;
                    if (data.start_line) html += `<p><strong>Lignes :</strong> ${data.start_line}-${data.end_line}</p>`;
                    if (data.parent) html += `<p><strong>Parent :</strong> ${data.parent}</p>`;
                    html += '</div>';

                    // Section signature
                    if (data.signature) {
                        html += '<div class="popup-section">';
                        html += '<h4>📝 Signature</h4>';
                        html += `<div style="background: #f8f9fa; padding: 10px; border-radius: 4px; font-family: monospace; font-size: 12px;">${data.signature}</div>`;
                        html += '</div>';
                    }

                    // Section dépendances
                    if (data.dependencies.length > 0) {
                        html += '<div class="popup-section">';
                        html += '<h4>📦 Dépendances (USE)</h4>';
                        html += '<ul class="popup-list">';
                        data.dependencies.forEach(dep => {
                            html += `<li>📦 ${dep}</li>`;
                        });
                        html += '</ul>';
                        html += '</div>';
                    }

                    // Section appels de fonctions
                    if (data.called_functions.length > 0) {
                        html += '<div class="popup-section">';
                        html += '<h4>📞 Appels de fonctions</h4>';
                        html += '<ul class="popup-list">';
                        data.called_functions.forEach(func => {
                            html += `<li>⚙️ ${func}</li>`;
                        });
                        html += '</ul>';
                        html += '</div>';
                    }

                    // Section actions
                    html += '<div class="popup-section">';
                    html += '<h4>🛠️ Actions</h4>';
                    html += `<button class="control-btn" onclick="focusOnEntity('${data.entity_name}')">🎯 Centrer</button><br><br>`;
                    html += `<button class="control-btn" onclick="highlightDependencies('${data.entity_name}')">🔗 Surligner</button><br><br>`;
                    html += `<button class="control-btn" onclick="copyCodeToClipboard()">📋 Copier code</button>`;
                    html += '</div>';

                    html += '</div>'; // Fin popup-left

                    // ✅ NOUVELLE : Colonne droite avec container CodeMirror
                    html += '<div class="popup-right">';
                    html += '<div class="popup-section">';
                    html += '<h4>💻 Code source avec coloration syntaxique</h4>';
                    html += '<div class="codemirror-container" id="codeMirrorContainer">';
                    html += '<div style="padding: 20px; text-align: center; color: #666;">Chargement du code source...</div>';
                    html += '</div>';
                    html += '</div>';
                    html += '</div>'; // Fin popup-right

                    return html;
                }

                // Extraire les données depuis le nœud VIS (identique)
                function extractEntityDataFromNode(node) {
                    return {
                        entity_name: node.id,
                        entity_type: node.entity_type_detailed || extractFromTitle(node.title, /<strong>Type:<\\/strong>\\s*([^<]+)/),
                        signature: node.entity_signature || extractFromTitle(node.title, /<strong>Signature:<\\/strong>\\s*([^<]+)/),
                        filepath: node.entity_filepath || extractFromTitle(node.title, /<strong>Fichier:<\\/strong>\\s*([^<]+)/),
                        dependencies: node.entity_dependencies || [],
                        called_functions: node.entity_called_functions || [],
                        parent: node.entity_parent || extractFromTitle(node.title, /<strong>Parent:<\\/strong>\\s*([^<]+)/),
                        start_line: node.entity_start_line || extractFromTitle(node.title, /<strong>Lignes:<\\/strong>\\s*(\\d+)/),
                        end_line: node.entity_end_line || extractFromTitle(node.title, /<strong>Lignes:<\\/strong>\\s*\\d+-(\\d+)/),
                        source_code: node.source_code || '! Code source non disponible'
                    };
                }

                // Fonction utilitaire pour extraire depuis le HTML (fallback)
                function extractFromTitle(titleHtml, pattern) {
                    if (!titleHtml) return '';
                    const match = titleHtml.match(pattern);
                    return match ? match[1].trim() : '';
                }

                // ✅ NOUVELLE FONCTION : Charger le code avec CodeMirror
                async function loadCodeWithCodeMirror(entityData) {
                    const container = document.getElementById('codeMirrorContainer');
                    if (!container) return;

                    try {
                        let sourceCode = entityData.source_code;

                        if (!sourceCode || sourceCode.trim() === '') {
                            sourceCode = `! Code source non disponible pour ${entityData.entity_name}\\n! Fichier: ${entityData.filepath}\\n! Lignes: ${entityData.start_line}-${entityData.end_line}`;
                        }

                        // ✅ CRÉER L'INSTANCE CODEMIRROR
                        currentCodeMirror = createCodeMirrorInstance(container, sourceCode, 'fortran');

                        // Ajouter des informations dans la ligne de statut (simulée)
                        console.log(`📝 Code chargé pour ${entityData.entity_name}: ${sourceCode.split('\\n').length} lignes`);

                    } catch (error) {
                        container.innerHTML = `<div style="padding: 20px; color: #dc3545;">Erreur lors du chargement du code source:<br>${error.message}</div>`;
                    }
                }

                // ✅ NOUVELLE FONCTION : Copier le code depuis CodeMirror
                function copyCodeToClipboard() {
                    if (currentCodeMirror) {
                        const code = currentCodeMirror.getValue();
                        if (navigator.clipboard) {
                            navigator.clipboard.writeText(code).then(() => {
                                alert('✅ Code source copié dans le presse-papiers !');
                            });
                        } else {
                            // Fallback pour les navigateurs plus anciens
                            const textArea = document.createElement('textarea');
                            textArea.value = code;
                            document.body.appendChild(textArea);
                            textArea.select();
                            document.execCommand('copy');
                            document.body.removeChild(textArea);
                            alert('✅ Code source copié !');
                        }
                    } else {
                        alert('❌ Aucun code source disponible');
                    }
                }

                // Gestionnaires d'événements pour éviter popup sur drag (identiques)
                network.on("dragStart", function (params) {
                    console.log('🔄 Drag start');
                    isDragging = true;
                    hasDragged = false; // Reset au début du drag
                });
                
                network.on("dragging", function (params) {
                    // Ce event se déclenche pendant le mouvement
                    if (isDragging) {
                        hasDragged = true; // On a vraiment bougé
                        console.log('🔄 Actually dragging');
                    }
                });
                
                network.on("dragEnd", function (params) {
                    console.log('🔄 Drag end, hasDragged:', hasDragged);
                    
                    // Petit délai pour éviter le click immédiat après drag
                    setTimeout(() => {
                        isDragging = false;
                        hasDragged = false;
                    }, 50);
                });
                
                // ✅ GESTIONNAIRE CLICK SIMPLIFIÉ
                network.on("click", function (params) {
                    console.log('🖱️ Network click, isDragging:', isDragging, 'hasDragged:', hasDragged);
                    
                    // Si on est en train de dragger ou qu'on vient de dragger, ignorer
                    if (isDragging || hasDragged) {
                        console.log('🚫 Ignoring click (dragging or just dragged)');
                        return;
                    }
                    
                    // C'est un vrai clic !
                    if (params.nodes.length > 0) {
                        const nodeId = params.nodes[0];
                        console.log('✅ Valid click on node:', nodeId);
                        openEntityPopup(nodeId);
                    } else {
                        console.log('🔍 Click on empty space');
                    }
                });
                
                // ✅ DOUBLE-CLICK RESTE IDENTIQUE
                network.on("doubleClick", function (params) {
                    console.log('🖱️ Double click');
                    if (params.nodes.length > 0) {
                        const nodeId = params.nodes[0];
                        focusOnEntity(nodeId);
                    }
                });
                
                // Actions du popup
                function focusOnEntity(entityName) {
                    network.focus(entityName, {
                        scale: 1.5,
                        animation: {duration: 1000, easingFunction: "easeInOutQuad"}
                    });
                    closeEntityPopup();
                }

                function highlightDependencies(entityName) {
                    const connectedNodes = network.getConnectedNodes(entityName);
                    const connectedEdges = network.getConnectedEdges(entityName);
                    console.log(`Highlighting dependencies for ${entityName}:`, connectedNodes);
                    closeEntityPopup();
                }

                // Raccourcis clavier (identiques)
                document.addEventListener('keydown', function(e) {
                    if (e.key === 'Escape') {
                        closeEntityPopup();
                    } else if (e.key === 'p' || e.key === 'P') {
                        togglePhysics();
                    } else if (e.key === 'l' || e.key === 'L') {
                        toggleLegend();
                    }
                });

                // Initialisation
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('🚀 DOM loaded, initializing...');
                    
                    const physicsBtn = document.getElementById('togglePhysics');
                    if (physicsBtn) {
                        physicsBtn.classList.add('active');
                        physicsBtn.textContent = '⚡ Physique ON';
                    }
                
                    // Vérifier que CodeMirror est chargé
                    if (typeof CodeMirror !== 'undefined') {
                        console.log('✅ CodeMirror loaded, version:', CodeMirror.version);
                    } else {
                        console.error('❌ CodeMirror not loaded!');
                    }
                
                    console.log('🎨 Visualiseur Fortran interactif avec CodeMirror initialisé');
                    console.log('💡 Raccourcis: P = Physics, L = Legend, ESC = Fermer popup');
                    
                    // Test de clic simple
                    console.log('🔍 Network object:', network);
                });
            </script>
            """

            content = content.replace('</body>', enhanced_js + '</body>')

            # Réécrire le fichier enrichi
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info("✨ HTML enrichi avec CodeMirror pour coloration syntaxique Fortran")

        except Exception as e:
            logger.warning(f"⚠️ Impossible d'enrichir le HTML: {e}")

    def _configure_pyvis_options_enhanced(self, net: Network, hierarchical: bool, physics_enabled: bool):
        """Configure les options PyVis avec améliorations"""

        if hierarchical:
            net.set_options("""
            var options = {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed",
                        "levelSeparation": 250,
                        "nodeSpacing": 200,
                        "treeSpacing": 300
                    }
                },
                "physics": {
                    "enabled": """ + str(physics_enabled).lower() + """,
                    "hierarchicalRepulsion": {
                        "centralGravity": 0.1,
                        "springLength": 200,
                        "springConstant": 0.01,
                        "nodeDistance": 200,
                        "damping": 0.09
                    },
                    "solver": "hierarchicalRepulsion",
                    "stabilization": {"enabled": true, "iterations": 1000}
                },
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true,
                    "hover": true,
                    "selectConnectedEdges": true,
                    "tooltipDelay": 300
                },
                "edges": {
                    "smooth": {"enabled": true, "type": "dynamic", "roundness": 0.5},
                    "arrows": {"to": {"enabled": true, "scaleFactor": 1.0}},
                    "font": {"size": 12, "strokeWidth": 2, "strokeColor": "#ffffff"}
                },
                "nodes": {
                    "font": {"size": 14, "face": "Arial", "strokeWidth": 3, "strokeColor": "#ffffff"},
                    "borderWidth": 2,
                    "shadow": {"enabled": true, "color": "rgba(0,0,0,0.3)", "size": 10, "x": 2, "y": 2}
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
                        "springLength": 150,
                        "springConstant": 0.08,
                        "damping": 0.4,
                        "avoidOverlap": 0.8
                    },
                    "solver": "forceAtlas2Based",
                    "stabilization": {"enabled": true, "iterations": 1000}
                },
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true,
                    "hover": true,
                    "selectConnectedEdges": true,
                    "tooltipDelay": 300
                }
            }
            """)

    def _configure_pyvis_options(self, net: Network, hierarchical: bool, physics_enabled: bool):
        """Configure les options PyVis"""

        if hierarchical:
            net.set_options("""
            var options = {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed"
                    }
                },
                "physics": {"enabled": """ + str(physics_enabled).lower() + """},
                "interaction": {"navigationButtons": true, "keyboard": true}
            }
            """)
        else:
            net.set_options("""
            var options = {
                "physics": {"enabled": """ + str(physics_enabled).lower() + """},
                "interaction": {"navigationButtons": true, "keyboard": true}
            }
            """)

    def _calculate_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques du graphe"""
        return {
            'nodes': len(self.graph.nodes),
            'edges': len(self.graph.edges),
            'entities_count': len(self.entities)
        }

    # === API SIMPLIFIÉE ===

    def create_visualization_from_entities(self,
                                           entities: List[Any],
                                           output_file: str = "dependencies.html",
                                           max_entities: int = 50,
                                           focus_entity: Optional[str] = None,
                                           include_variables: bool = True,) -> str:
        """API simplifiée - Point d'entrée principal"""


        # Construire le graphe
        self.build_dependency_graph_from_entities(
            entities=entities,
            max_entities=max_entities,
            include_variables=include_variables,
            focus_entity=focus_entity
        )

        # Générer la visualisation
        self.visualize_with_pyvis(output_file=output_file)

        return output_file


# === FONCTION UTILITAIRE POUR LE TEST ===

def create_dependency_visualization_from_parser(entities: List[Any],
                                                output_file: str = "test_dependencies.html",
                                                focus_entity: Optional[str] = None,
                                                include_variables: bool = True,
                                                max_entities: int = 50
                                                ) -> str:
    """
    Fonction utilitaire pour créer une visualisation directement depuis les entités du parser.
    À utiliser dans les tests.
    """

    visualizer = SimpleFortranDependencyVisualizer()

    return visualizer.create_visualization_from_entities(
        entities=entities,
        output_file=output_file,
        include_variables=include_variables,
        focus_entity=focus_entity,
        max_entities=max_entities
    )