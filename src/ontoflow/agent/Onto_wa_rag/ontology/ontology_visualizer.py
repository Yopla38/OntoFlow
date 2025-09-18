"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import os
import random
from collections import defaultdict
import rdflib
from rdflib import RDFS, RDF, OWL


class OntologyVisualizer:
    """
    Classe pour visualiser des ontologies en modélisant les relations comme des arêtes directes.
    Compatible avec différentes ontologies, y compris EMMO.
    """

    # Propriétés standards pour les labels, par ordre de priorité
    LABEL_PROPERTIES = [
        rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),  # skos:prefLabel
        rdflib.RDFS.label,  # rdfs:label
        rdflib.URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),  # skos:altLabel
        rdflib.URIRef("http://purl.org/dc/terms/title"),  # dc:title
        rdflib.URIRef("http://purl.org/dc/elements/1.1/title")  # dc:title (legacy)
    ]

    # Propriétés standards pour les descriptions, par ordre de priorité
    DESCRIPTION_PROPERTIES = [
        rdflib.RDFS.comment,  # rdfs:comment
        rdflib.URIRef("http://purl.org/dc/terms/description"),  # dc:description
        rdflib.URIRef("http://purl.org/dc/elements/1.1/description"),  # dc:description (legacy)
        rdflib.URIRef("http://www.w3.org/2004/02/skos/core#definition")  # skos:definition
    ]

    # Propriétés spécifiques à certaines ontologies (extensible)
    ONTOLOGY_SPECIFIC = {
        "EMMO": {
            "label_props": [
                rdflib.URIRef("http://www.w3.org/2004/02/skos/core#prefLabel")
            ],
            "description_props": [
                rdflib.URIRef("https://w3id.org/emmo#EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9")
            ]
        }
    }

    def __init__(self, ontology_manager, ontology_type=None):
        """
        Initialise le visualiseur avec un gestionnaire d'ontologie.

        Args:
            ontology_manager: Une instance de OntologyManager
            ontology_type: Type d'ontologie ('EMMO', etc.) pour configurations spécifiques
        """
        self.ontology_manager = ontology_manager
        self.graph = nx.DiGraph()
        self.edge_labels = {}
        self.edge_types = {}
        self.ontology_type = ontology_type

        # Propriétés pour les labels et descriptions à utiliser
        self.label_properties = list(self.LABEL_PROPERTIES)
        self.description_properties = list(self.DESCRIPTION_PROPERTIES)

        # Ajouter des propriétés spécifiques à l'ontologie s'il y en a
        if ontology_type and ontology_type in self.ONTOLOGY_SPECIFIC:
            config = self.ONTOLOGY_SPECIFIC[ontology_type]
            # Insérer les propriétés spécifiques en tête de liste pour qu'elles soient prioritaires
            self.label_properties = config.get("label_props", []) + self.label_properties
            self.description_properties = config.get("description_props", []) + self.description_properties

    def build_graph(self, max_concepts=None, focus_uri=None, max_depth=2):
        """Construit un graphe NetworkX à partir de l'ontologie avec les relations comme arêtes."""
        # Réinitialiser le graphe
        self.graph = nx.DiGraph()

        # Sélectionner les concepts à inclure
        concepts_to_include = self._select_concepts(max_concepts, focus_uri, max_depth)

        # Ajouter les concepts sélectionnés comme nœuds
        for uri in concepts_to_include:
            concept = self.ontology_manager.concepts[uri]

            # Extraire le label et la description
            label = self._extract_best_label(uri, concept)
            description = self._extract_best_description(uri, concept)

            self.graph.add_node(
                uri,
                label=label,
                original_uri=uri,
                description=description,
                type="concept"
            )

        # Ajouter les relations hiérarchiques (subClassOf)
        for uri in concepts_to_include:
            concept = self.ontology_manager.concepts[uri]
            for parent in concept.parents:
                if parent.uri in concepts_to_include:
                    self.graph.add_edge(
                        uri, parent.uri,
                        label="subClassOf",
                        type="hierarchy"
                    )
                    # Stocker le type et le label pour la visualisation
                    self.edge_labels[(uri, parent.uri)] = "subClassOf"
                    self.edge_types[(uri, parent.uri)] = "hierarchy"

            # Vérifier si axioms est disponible dans ontology_manager
            semantic_edges = []
            if hasattr(self.ontology_manager, 'axioms'):
                for axiom_type, source, target in self.ontology_manager.axioms:
                    # Vérifier si c'est une relation sémantique
                    if axiom_type.startswith("semantic_"):
                        # Extraire le type spécifique de relation
                        rel_type = axiom_type.replace("semantic_", "")

                        # Vérifier que source et target sont dans les concepts à inclure
                        if source in concepts_to_include and target in concepts_to_include:
                            # Ajouter l'arête
                            self.graph.add_edge(
                                source, target,
                                label=rel_type,
                                type="semantic"
                            )
                            semantic_edges.append((source, target))

                            # Stocker le type et le label pour la visualisation
                            self.edge_labels[(source, target)] = rel_type
                            self.edge_types[(source, target)] = "semantic"
                            print(f"Ajout de relation sémantique: {source} --{rel_type}--> {target}")
            else:
                print("⚠️ Aucun axiome trouvé dans l'ontology_manager")

            print(f"Graphe construit avec {len(self.graph.nodes)} nœuds et {len(self.graph.edges)} arêtes")
            print(f"  - Relations hiérarchiques: {len(self.graph.edges) - len(semantic_edges)}")
            print(f"  - Relations sémantiques: {len(semantic_edges)}")

    def old_build_graph(self, max_concepts=None, focus_uri=None, max_depth=2):
        """
        Construit un graphe NetworkX à partir de l'ontologie avec les relations comme arêtes.

        Args:
            max_concepts: Nombre maximum de concepts à visualiser (None pour tous)
            focus_uri: URI du concept central pour la visualisation (None pour visualiser tout)
            max_depth: Profondeur maximale d'exploration à partir du focus
        """
        # Réinitialiser le graphe
        self.graph = nx.DiGraph()

        # Sélectionner les concepts à inclure
        concepts_to_include = self._select_concepts(max_concepts, focus_uri, max_depth)

        # Ajouter les concepts sélectionnés comme nœuds
        for uri in concepts_to_include:
            concept = self.ontology_manager.concepts[uri]

            # Extraire le label et la description
            label = self._extract_best_label(uri, concept)
            description = self._extract_best_description(uri, concept)

            self.graph.add_node(
                uri,
                label=label,
                original_uri=uri,
                description=description,
                type="concept"
            )

        # Ajouter les relations hiérarchiques (subClassOf)
        for uri in concepts_to_include:
            concept = self.ontology_manager.concepts[uri]
            for parent in concept.parents:
                if parent.uri in concepts_to_include:
                    self.graph.add_edge(
                        uri, parent.uri,
                        label="subClassOf",
                        type="hierarchy"
                    )
                    # Stocker le type et le label pour la visualisation
                    self.edge_labels[(uri, parent.uri)] = "subClassOf"
                    self.edge_types[(uri, parent.uri)] = "hierarchy"

        # Ajouter les relations sémantiques directement entre concepts
        semantic_edges = []  # Pour stocker les relations sémantiques

        for rel_uri, relation in self.ontology_manager.relations.items():
            rel_label = self._extract_best_label(rel_uri, relation)

            # Connecter chaque domaine à chaque portée via cette relation
            for domain_concept in relation.domain:
                if domain_concept.uri not in concepts_to_include:
                    continue

                for range_concept in relation.range:
                    if range_concept.uri not in concepts_to_include:
                        continue

                    # Ajouter l'arête représentant cette relation
                    self.graph.add_edge(
                        domain_concept.uri, range_concept.uri,
                        label=rel_label,
                        relation_uri=rel_uri,
                        type="semantic"
                    )
                    semantic_edges.append((domain_concept.uri, range_concept.uri))

                    # Stocker le type et le label pour la visualisation
                    self.edge_labels[(domain_concept.uri, range_concept.uri)] = rel_label
                    self.edge_types[(domain_concept.uri, range_concept.uri)] = "semantic"

        print(f"Graphe construit avec {len(self.graph.nodes)} nœuds et {len(self.graph.edges)} arêtes")
        print(f"  - Relations hiérarchiques: {len(self.graph.edges) - len(semantic_edges)}")
        print(f"  - Relations sémantiques: {len(semantic_edges)}")

    def _extract_best_label(self, uri, entity):
        """
        Extrait le meilleur label disponible pour une entité.

        Args:
            uri: URI de l'entité
            entity: L'objet Concept ou Relation

        Returns:
            Le label le plus approprié
        """
        # 1. Utiliser le label de l'entité s'il existe et n'est pas simplement l'URI
        if hasattr(entity, 'label') and entity.label and entity.label != self._extract_name_from_uri(uri):
            return entity.label

        # 2. Chercher dans les propriétés de label définies
        uri_ref = rdflib.URIRef(uri)
        for label_prop in self.label_properties:
            for _, _, label_value in self.ontology_manager.graph.triples((uri_ref, label_prop, None)):
                # Préférer en anglais si spécifié
                if hasattr(label_value, 'language') and label_value.language == 'en':
                    return str(label_value)
                # Sinon prendre la première valeur
                return str(label_value)

        # 3. Traitement spécifique pour les URIs spéciales
        if self.ontology_type == "EMMO" and "EMMO_" in uri:
            # Pour EMMO, essayer de construire un label plus lisible
            parts = uri.split("#")
            if len(parts) > 1 and "EMMO_" in parts[1]:
                emmo_id = parts[1]
                # Essayer de créer un nom plus lisible si possible
                if "_" in emmo_id:
                    parts = emmo_id.split("_", 2)
                    if len(parts) > 2:
                        # Si on a un format comme EMMO_class_id, extraire 'class'
                        return parts[1].capitalize()
                # Limiter l'ID pour qu'il soit plus lisible
                if len(emmo_id) > 20:
                    return emmo_id[:17] + "..."
                return emmo_id

        # 4. Fallback: extraire un nom de l'URI
        return self._extract_name_from_uri(uri)

    def _extract_best_description(self, uri, entity):
        """
        Extrait la meilleure description disponible pour une entité.

        Args:
            uri: URI de l'entité
            entity: L'objet Concept ou Relation

        Returns:
            La description la plus appropriée ou chaîne vide
        """
        # 1. Utiliser la description de l'entité si elle existe
        if hasattr(entity, 'description') and entity.description:
            return entity.description

        # 2. Chercher dans les propriétés de description définies
        uri_ref = rdflib.URIRef(uri)
        for desc_prop in self.description_properties:
            for _, _, desc_value in self.ontology_manager.graph.triples((uri_ref, desc_prop, None)):
                # Préférer en anglais si spécifié
                if hasattr(desc_value, 'language') and desc_value.language == 'en':
                    return str(desc_value)
                # Sinon prendre la première valeur
                return str(desc_value)

        # Aucune description trouvée
        return ""

    def _select_concepts(self, max_concepts=None, focus_uri=None, max_depth=2):
        """
        Sélectionne les concepts à visualiser selon les critères spécifiés.

        Args:
            max_concepts: Nombre maximum de concepts
            focus_uri: URI du concept central
            max_depth: Profondeur d'exploration

        Returns:
            Ensemble des URIs des concepts à inclure
        """
        all_concepts = set(self.ontology_manager.concepts.keys())

        # Si aucun focus n'est spécifié ou max_concepts est None, prendre tous les concepts
        if not focus_uri or not max_concepts:
            # Limiter le nombre total si spécifié
            if max_concepts and len(all_concepts) > max_concepts:
                return set(random.sample(list(all_concepts), max_concepts))
            return all_concepts

        # Explorer à partir du focus
        concepts_to_include = {focus_uri}
        current_layer = {focus_uri}

        # Exploration BFS pour trouver les concepts connectés
        for depth in range(max_depth):
            next_layer = set()

            for uri in current_layer:
                if uri in self.ontology_manager.concepts:
                    concept = self.ontology_manager.concepts[uri]

                    # Ajouter les parents
                    for parent in concept.parents:
                        next_layer.add(parent.uri)

                    # Ajouter les enfants
                    for child in concept.children:
                        next_layer.add(child.uri)

                    # Ajouter les concepts liés par relation
                    for rel_uri, relation in self.ontology_manager.relations.items():
                        # Vérifier si ce concept est dans le domaine de la relation
                        for domain in relation.domain:
                            if domain.uri == uri:
                                # Ajouter les concepts de portée
                                for range_concept in relation.range:
                                    next_layer.add(range_concept.uri)

                        # Vérifier si ce concept est dans la portée de la relation
                        for range_concept in relation.range:
                            if range_concept.uri == uri:
                                # Ajouter les concepts de domaine
                                for domain in relation.domain:
                                    next_layer.add(domain.uri)

            # Mettre à jour les concepts à inclure
            concepts_to_include.update(next_layer)
            current_layer = next_layer

            # Arrêter si on dépasse le nombre maximum
            if max_concepts and len(concepts_to_include) >= max_concepts:
                # Prendre un échantillon représentatif
                return set(list(concepts_to_include)[:max_concepts])

        return concepts_to_include

    def _extract_name_from_uri(self, uri):
        """Extrait un nom lisible à partir d'un URI."""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    def visualize_with_matplotlib(self, output_file="ontology_graph.png", figsize=(16, 12)):
        """
        Visualise l'ontologie avec Matplotlib.

        Args:
            output_file: Chemin du fichier de sortie pour l'image
            figsize: Dimensions de la figure (largeur, hauteur) en pouces
        """
        if not self.graph.nodes:
            self.build_graph()

        # Créer un layout pour le graphe (essayer d'abord avec graphviz si disponible)
        try:
            import pygraphviz
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        except ImportError:
            try:
                pos = nx.drawing.nx_pydot.graphviz_layout(self.graph, prog="dot")
            except ImportError:
                # Fallback au spring layout
                pos = nx.spring_layout(self.graph, k=0.3, iterations=50, seed=42)

        plt.figure(figsize=figsize)

        # Dessiner les nœuds (ce sont tous des concepts)
        nx.draw_networkx_nodes(self.graph, pos, node_color='skyblue',
                               node_size=2000, alpha=0.8)

        # Regrouper les arêtes par type
        hierarchy_edges = [(u, v) for u, v, attr in self.graph.edges(data=True)
                           if attr.get('type') == 'hierarchy']
        semantic_edges = [(u, v) for u, v, attr in self.graph.edges(data=True)
                          if attr.get('type') == 'semantic']

        # Dessiner les arêtes de hiérarchie
        nx.draw_networkx_edges(self.graph, pos, edgelist=hierarchy_edges, edge_color='blue',
                               width=1.5, arrows=True)

        # Dessiner les arêtes sémantiques
        nx.draw_networkx_edges(self.graph, pos, edgelist=semantic_edges, edge_color='green',
                               width=1.5, arrows=True, connectionstyle='arc3,rad=0.1', style='dashed')

        # Ajouter les labels des nœuds
        labels = {n: attr['label'] for n, attr in self.graph.nodes(data=True)}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_family='sans-serif')

        # Ajouter les labels des arêtes
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=self.edge_labels,
            font_size=8, font_color='red'
        )

        plt.title("Visualisation de l'Ontologie")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, format="PNG", dpi=300)
        plt.close()

        print(f"Graphe enregistré dans {output_file}")

    def visualize_with_pyvis(self, output_file="ontology_interactive.html", hierarchical=False, height="800px",
                             width="100%"):
        """
        Visualise l'ontologie avec PyVis pour une visualisation interactive améliorée.

        Args:
            output_file: Chemin du fichier HTML de sortie
            hierarchical: Si True, utilise un layout hiérarchique
            height: Hauteur du cadre de visualisation
            width: Largeur du cadre de visualisation
        """
        if not self.graph.nodes:
            self.build_graph()

        # Créer un réseau PyVis
        net = Network(height=height, width=width, directed=True, notebook=False)

        # Configurer les options avancées avec layout hiérarchique
        if hierarchical:
            physics_options = """
            {
                "enabled": true,
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
            }
            """

            net.set_options("""
            var options = {
                "layout": {
                    "hierarchical": {
                        "enabled": true,
                        "direction": "UD",
                        "sortMethod": "directed",
                        "levelSeparation": 150,
                        "nodeSpacing": 200,
                        "treeSpacing": 200
                    }
                },
                "physics": %s,
                "interaction": {
                    "navigationButtons": true,
                    "keyboard": true,
                    "hideEdgesOnDrag": true,
                    "hover": true,
                    "multiselect": true
                },
                "edges": {
                    "smooth": {
                        "enabled": true,
                        "type": "dynamic"
                    },
                    "arrows": {
                        "to": {
                            "enabled": true,
                            "scaleFactor": 0.5
                        }
                    }
                },
                "nodes": {
                    "font": {
                        "size": 14
                    }
                }
            }
            """ % physics_options)
        else:
            # Options non-hiérarchiques originales
            net.set_options("""
            var options = {
                "layout": {
                    "improvedLayout": true
                },
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "maxVelocity": 50,
                    "solver": "forceAtlas2Based",
                    "timestep": 0.35,
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
                    "multiselect": true
                }
            }
            """)

        # Définir les couleurs pour les différents types de nœuds et de domaines
        domain_colors = {
            "sci:Sciences": "#4285F4",  # Bleu Google
            "sci:Technologies": "#EA4335",  # Rouge Google
            "sci:Sante": "#34A853",  # Vert Google
            "sci:Environnement": "#FBBC05"  # Jaune Google
        }

        # Fonction pour déterminer la taille en fonction de l'importance
        def get_node_size(uri, graph):
            # Plus un nœud a de connexions, plus il est important
            importance = len(list(graph.successors(uri))) + len(list(graph.predecessors(uri)))
            # Les nœuds racines sont toujours importants
            if uri in domain_colors:
                return 35  # Taille pour les domaines principaux
            # Taille proportionnelle à l'importance
            return 15 + min(importance * 3, 20)

        # Ajouter les nœuds avec des styles améliorés
        for node_id, attr in self.graph.nodes(data=True):
            label = attr.get('label', node_id.split('/')[-1].split('#')[-1])
            description = attr.get('description', '')

            # Créer un tooltip riche en HTML
            tooltip = f"<div style='max-width:300px;'>"
            tooltip += f"<h3>{label}</h3>"

            if description:
                tooltip += f"<p>{description}</p>"

            tooltip += f"<hr><p><small>URI: {node_id}</small></p></div>"

            # Déterminer la couleur du nœud en fonction du domaine
            color = domain_colors.get(node_id, "#9AA0A6")  # Gris par défaut

            # Les sous-domaines directs héritent d'une version plus claire de la couleur
            for parent_uri in domain_colors:
                parent_concept = self.ontology_manager.concepts.get(parent_uri)
                if parent_concept and parent_concept in self.ontology_manager.concepts.get(node_id,
                                                                                           self.ontology_manager.concepts.get(
                                                                                                   "sci:Sciences")).parents:
                    # Version plus claire de la couleur du parent
                    r = int(domain_colors[parent_uri][1:3], 16)
                    g = int(domain_colors[parent_uri][3:5], 16)
                    b = int(domain_colors[parent_uri][5:7], 16)
                    color = f"#{r:02x}{g:02x}{b:02x}"
                    break

            # Forme selon le type
            shape = "circle"  # Par défaut
            if node_id in domain_colors:
                shape = "diamond"  # Diamant pour les domaines principaux

            # Taille basée sur l'importance
            size = get_node_size(node_id, self.graph)

            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                color=color,
                shape=shape,
                size=size
            )

        # Ajouter les arêtes avec des styles différents selon leur type
        edge_colors = {
            "hierarchy": "#0077be",  # Bleu
            "semantic": "#2ecc71"  # Vert
        }

        edge_styles = {
            "hierarchy": False,  # Non pointillé
            "semantic": True  # Pointillé
        }

        for source, target, attr in self.graph.edges(data=True):
            edge_type = attr.get('type', 'default')
            edge_label = attr.get('label', '')

            # Style différent pour les relations sémantiques
            width = 1.0
            if edge_type == "semantic":
                width = 2.0  # Plus épais pour les relations sémantiques

            net.add_edge(
                source, target,
                title=edge_label,
                label=edge_label,
                color=edge_colors.get(edge_type, "#9AA0A6"),
                dashes=edge_styles.get(edge_type, False),
                width=width,
                arrows="to",
                smooth={"enabled": True, "type": "curvedCW" if edge_type == "semantic" else "dynamic"}
            )

        # Enregistrer et afficher
        net.save_graph(output_file)
        print(f"Graphe interactif amélioré enregistré dans {output_file}")

        # Ouvrir automatiquement dans le navigateur
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_file), new=2)
        except:
            pass

    def visualize_with_graphviz(self, output_file="ontology_graphviz.png"):
        """
        Visualise l'ontologie avec Graphviz.

        Args:
            output_file: Chemin du fichier de sortie
        """
        try:
            import graphviz
            from hashlib import md5

            # Fonction pour créer un ID sûr pour Graphviz
            def safe_id(uri):
                if len(uri) > 64:
                    return "n" + md5(uri.encode()).hexdigest()
                return uri.replace(":", "_").replace("/", "_").replace("#", "_").replace("-", "_")

            # Créer un nouveau graphe
            dot = graphviz.Digraph(comment="Ontology Visualization", format="png")
            dot.attr(rankdir="TB", size="11,8", ratio="fill", fontsize="12")

            # Ajouter les nœuds de concepts
            for uri, attr in self.graph.nodes(data=True):
                node_id = safe_id(uri)
                label = attr.get('label', uri.split('/')[-1].split('#')[-1])
                description = attr.get('description', '')

                tooltip = f"{label}: {description}" if description else label
                dot.node(
                    node_id,
                    label=label,
                    shape="ellipse",
                    style="filled",
                    fillcolor="skyblue",
                    tooltip=tooltip
                )

            # Ajouter les arêtes
            for source, target, attr in self.graph.edges(data=True):
                source_id = safe_id(source)
                target_id = safe_id(target)
                edge_label = attr.get('label', '')
                edge_type = attr.get('type', 'default')

                # Configurer le style selon le type
                if edge_type == "hierarchy":
                    dot.edge(
                        source_id, target_id,
                        label=edge_label,
                        color="blue",
                        style="solid"
                    )
                elif edge_type == "semantic":
                    dot.edge(
                        source_id, target_id,
                        label=edge_label,
                        color="green",
                        style="dashed"
                    )
                else:
                    dot.edge(source_id, target_id, label=edge_label)

            # Rendre et enregistrer
            dot.render(output_file.replace(".png", ""), view=False)
            print(f"Graphe Graphviz enregistré dans {output_file}")

        except ImportError:
            print("La bibliothèque graphviz n'est pas installée. Installez-la avec: pip install graphviz")
            print("Vous devez également installer le logiciel Graphviz: https://graphviz.org/download/")

    def generate_ontology_report(self, output_file="ontology_report.html"):
        """
        Génère un rapport HTML complet sur l'ontologie.

        Args:
            output_file: Chemin du fichier HTML de sortie
        """
        if not self.graph.nodes:
            self.build_graph()

        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport d'Ontologie</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .stats {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
                .stat-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 200px; }}
                .stat-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
                .badge-hierarchy {{ background-color: #d4e6f1; color: #2874a6; }}
                .badge-semantic {{ background-color: #d4efdf; color: #27ae60; }}
                .uri-cell {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
                .uri-cell:hover {{ overflow: visible; white-space: normal; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Rapport d'Analyse d'Ontologie</h1>

                <div class="stats">
                    <div class="stat-card">
                        <h3>Concepts</h3>
                        <div class="stat-value">{len(self.graph.nodes)}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Relations</h3>
                        <div class="stat-value">{len(self.ontology_manager.relations)}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Liens Hiérarchiques</h3>
                        <div class="stat-value">{len([e for e in self.graph.edges(data=True) if e[2].get('type') == 'hierarchy'])}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Liens Sémantiques</h3>
                        <div class="stat-value">{len([e for e in self.graph.edges(data=True) if e[2].get('type') == 'semantic'])}</div>
                    </div>
                </div>

                <h2>Concepts</h2>
                <table>
                    <tr>
                        <th>Label</th>
                        <th>URI</th>
                        <th>Description</th>
                        <th>Parents</th>
                        <th>Enfants</th>
                    </tr>
        """

        # Ajouter les informations sur les concepts
        for uri, concept in self.ontology_manager.concepts.items():
            if uri in [n for n in self.graph.nodes()]:  # Si le concept est dans le graphe
                label = self.graph.nodes[uri].get('label', concept.label or self._extract_name_from_uri(uri))
                parents = ", ".join([p.label for p in concept.parents]) or "-"
                children = ", ".join([c.label for c in concept.children]) or "-"
                description = concept.description or self.graph.nodes[uri].get('description', "-")
                html += f"""
                    <tr>
                        <td>{label}</td>
                        <td class="uri-cell" title="{uri}">{uri}</td>
                        <td>{description}</td>
                        <td>{parents}</td>
                        <td>{children}</td>
                    </tr>
                """

        html += """
                </table>

                <h2>Relations</h2>
                <table>
                    <tr>
                        <th>Label</th>
                        <th>URI</th>
                        <th>Description</th>
                        <th>Domaine</th>
                        <th>Portée</th>
                    </tr>
        """

        # Ajouter les informations sur les relations
        for uri, relation in self.ontology_manager.relations.items():
            label = relation.label or self._extract_best_label(uri, relation)
            domain_labels = ", ".join([self._extract_best_label(d.uri, d) for d in relation.domain]) or "-"
            range_labels = ", ".join([self._extract_best_label(r.uri, r) for r in relation.range]) or "-"
            description = relation.description or "-"
            html += f"""
                    <tr>
                        <td>{label}</td>
                        <td class="uri-cell" title="{uri}">{uri}</td>
                        <td>{description}</td>
                        <td>{domain_labels}</td>
                        <td>{range_labels}</td>
                    </tr>
                """

        html += """
                </table>

                <h2>Liens dans le Graphe</h2>
                <table>
                    <tr>
                        <th>Source</th>
                        <th>Destination</th>
                        <th>Type</th>
                        <th>Label</th>
                    </tr>
        """

        # Ajouter les informations sur les liens
        for source, target, attr in self.graph.edges(data=True):
            source_label = self.graph.nodes[source].get('label', source)
            target_label = self.graph.nodes[target].get('label', target)
            edge_type = attr.get('type', 'default')
            edge_label = attr.get('label', '')

            type_badge = "badge-hierarchy" if edge_type == "hierarchy" else "badge-semantic"
            type_display = "Hiérarchie" if edge_type == "hierarchy" else "Sémantique"

            html += f"""
                    <tr>
                        <td>{source_label}</td>
                        <td>{target_label}</td>
                        <td><span class="badge {type_badge}">{type_display}</span></td>
                        <td>{edge_label}</td>
                    </tr>
                """

        html += """
                </table>
            </div>
        </body>
        </html>
        """

        # Enregistrer le rapport
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"Rapport d'ontologie généré dans {output_file}")

        # Ouvrir automatiquement dans le navigateur
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(output_file), new=2)
        except:
            pass