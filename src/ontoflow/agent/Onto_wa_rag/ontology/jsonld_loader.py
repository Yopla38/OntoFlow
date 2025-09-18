"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/jsonld_loader.py
import json
import os
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from typing import Dict, List, Set, Tuple, Optional


class JsonLDOntologyParser:
    """Parser pour les ontologies au format JSON-LD."""

    def __init__(self, jsonld_file_path: str):
        self.jsonld_file_path = jsonld_file_path
        self.graph = Graph()
        self._load_jsonld()

        # Base URI pour l'ontologie (détecté du fichier)
        self.base_uri = self._extract_base_uri()

        # Extraire les éléments fondamentaux
        self.concepts = self._extract_concepts()
        self.relations = self._extract_relations()
        self.axioms = self._extract_axioms()

        print(
            f"Analyse terminée: {len(self.concepts)} concepts, {len(self.relations)} relations, {len(self.axioms)} axiomes")

    def _extract_base_uri(self) -> str:
        """Extrait l'URI de base de l'ontologie."""
        # Chercher l'URI de base dans les triplets du graphe
        for s, p, o in self.graph:
            if isinstance(s, URIRef):
                uri = str(s)
                if "#" in uri:
                    return uri.split("#")[0] + "#"
                else:
                    parts = uri.split("/")
                    if len(parts) > 3:  # au moins http://example.org/
                        return "/".join(parts[:-1]) + "/"
        return "http://example.org/"

    def _load_jsonld(self):
        """Charge le fichier JSON-LD dans un graphe RDF."""
        try:
            if not os.path.exists(self.jsonld_file_path):
                raise FileNotFoundError(f"Fichier non trouvé: {self.jsonld_file_path}")

            with open(self.jsonld_file_path, 'r', encoding='utf-8') as file:
                jsonld_data = json.load(file)

            # Charger dans le graphe RDF
            self.graph.parse(data=json.dumps(jsonld_data), format='json-ld')
            print(f"Graphe JSON-LD chargé avec {len(self.graph)} triplets")

        except Exception as e:
            print(f"Erreur lors du chargement: {str(e)}")
            raise

    def _extract_concepts(self) -> Set[str]:
        """Extrait tous les concepts (classes) de l'ontologie."""
        concepts = set()

        # Classes RDFS et OWL explicites
        for s, p, o in self.graph.triples((None, rdflib.RDF.type, rdflib.RDFS.Class)):
            concepts.add(str(s))

        for s, p, o in self.graph.triples((None, rdflib.RDF.type, rdflib.OWL.Class)):
            concepts.add(str(s))

        # Classes définies avec @type: Class dans le JSON-LD
        owl_class = URIRef("http://www.w3.org/2002/07/owl#Class")
        for s, p, o in self.graph.triples((None, rdflib.RDF.type, owl_class)):
            concepts.add(str(s))

        # Classes définies simplement avec @type: "Class"
        for s, p, o in self.graph.triples((None, rdflib.RDF.type, None)):
            if "Class" in str(o) and not ("Property" in str(o) or "Statement" in str(o)):
                concepts.add(str(s))

        # Classes participant à des relations subClassOf
        for s, p, o in self.graph.triples((None, rdflib.RDFS.subClassOf, None)):
            concepts.add(str(s))
            if not isinstance(o, BNode):  # Éviter les nœuds vides
                concepts.add(str(o))

        return concepts

    def _extract_relations(self) -> Set[str]:
        """Extrait toutes les relations (propriétés) de l'ontologie."""
        relations = set()

        # Propriétés RDF standard
        for s, p, o in self.graph.triples((None, rdflib.RDF.type, rdflib.RDF.Property)):
            relations.add(str(s))

        # Propriétés OWL (Object, Datatype, Annotation)
        property_types = [
            rdflib.OWL.ObjectProperty,
            rdflib.OWL.DatatypeProperty,
            rdflib.OWL.AnnotationProperty,
            URIRef("http://www.w3.org/2002/07/owl#ObjectProperty"),
            URIRef("http://www.w3.org/2002/07/owl#DatatypeProperty"),
            URIRef("http://www.w3.org/2002/07/owl#AnnotationProperty")
        ]

        for prop_type in property_types:
            for s, p, o in self.graph.triples((None, rdflib.RDF.type, prop_type)):
                relations.add(str(s))

        # Propriétés définies simplement avec @type: "ObjectProperty"
        for s, p, o in self.graph.triples((None, rdflib.RDF.type, None)):
            if "Property" in str(o):
                relations.add(str(s))

        # Relations utilisées dans les statements RDF
        statement_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")
        predicate_uri = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")

        for stmt, _, _ in self.graph.triples((None, rdflib.RDF.type, statement_type)):
            for _, _, pred in self.graph.triples((stmt, predicate_uri, None)):
                relations.add(str(pred))

        return relations

    # Dans jsonld_loader.py - amélioration de l'extraction des relations
    def _extract_axioms(self) -> List[Tuple[str, str, str]]:
        """Extrait tous les types d'axiomes de l'ontologie."""
        axioms = []

        # 1. DEBUG: Afficher tous les prédicats pour analyse
        all_predicates = set()
        for _, p, _ in self.graph:
            all_predicates.add(str(p))
        print(f"Prédicats trouvés dans le graphe: {len(all_predicates)}")
        for pred in sorted(all_predicates):
            if "subClassOf" in pred or "Class" in pred:
                print(f"  - {pred}")

        # 2. Capture explicite de subClassOf dans tous les formats possibles
        subClassOfPreds = [
            rdflib.RDFS.subClassOf,
            URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf"),
            URIRef("subClassOf")
        ]

        for pred in subClassOfPreds:
            for s, _, o in self.graph.triples((None, pred, None)):
                if not isinstance(o, BNode):  # Ignorer les nœuds anonymes
                    axioms.append(("subsumption", str(s), str(o)))
                    print(f"Relation subClassOf trouvée: {s} -> {o}")

        # 3. Capture des relations sémantiques - méthode directe
        semantic_predicates = ["uses", "influencedBy", "appliedTo"]
        for pred_name in semantic_predicates:
            for pred_uri in [
                URIRef(f"http://example.org/scientific-ontology#{pred_name}"),
                URIRef(f"sci:{pred_name}"),
                URIRef(pred_name)
            ]:
                for s, _, o in self.graph.triples((None, pred_uri, None)):
                    rel_type = f"semantic_{pred_name}"
                    axioms.append((rel_type, str(s), str(o)))
                    print(f"Relation sémantique directe: {s} -{pred_name}-> {o}")

        # 4. Recherche des declarations rdf:Statement
        statement_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")

        # Récupérer directement tous les triplets du graphe pour analyse
        statements = {}
        for s, p, o in self.graph:
            if p == rdflib.RDF.type and o == statement_type:
                statements[str(s)] = {"subject": None, "predicate": None, "object": None}

        # Remplir les informations des statements
        for stmt_id in statements:
            stmt = URIRef(stmt_id)

            # Chercher sujet, prédicat, objet
            for s, p, o in self.graph:
                if s != stmt:
                    continue

                pred_str = str(p)
                if "subject" in pred_str.lower():
                    statements[stmt_id]["subject"] = str(o)
                elif "predicate" in pred_str.lower():
                    statements[stmt_id]["predicate"] = str(o)
                elif "object" in pred_str.lower():
                    statements[stmt_id]["object"] = str(o)

        # Ajouter les axiomes basés sur les statements
        for stmt_id, info in statements.items():
            if info["subject"] and info["predicate"] and info["object"]:
                # Déterminer le type de relation
                pred = info["predicate"].split("#")[-1] if "#" in info["predicate"] else info["predicate"].split("/")[
                    -1]

                # Ajouter comme relation sémantique
                axioms.append((f"semantic_{pred}", info["subject"], info["object"]))
                print(f"Relation via Statement: {info['subject']} -{pred}-> {info['object']}")

        return axioms