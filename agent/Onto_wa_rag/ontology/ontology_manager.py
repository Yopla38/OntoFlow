"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/ontology_manager.py
import json
import os
import uuid
from typing import List, Dict, Tuple, Set, Optional, Any
import rdflib
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL


class Concept:
    """Représente un concept dans l'ontologie."""

    def __init__(self, uri: str, label: str = None, description: str = None):
        self.uri = uri
        self.label = label or self._extract_label_from_uri(uri)
        self.description = description
        self.parents = []  # Concepts plus généraux
        self.children = []  # Concepts plus spécifiques
        self.properties = {}  # Propriétés du concept
        self.instances = []  # Instances du concept (documents, etc.)

    def _extract_label_from_uri(self, uri: str) -> str:
        """Extrait un label lisible à partir de l'URI."""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    def add_parent(self, parent_concept):
        """Ajoute un concept parent."""
        if parent_concept not in self.parents:
            self.parents.append(parent_concept)
            if self not in parent_concept.children:
                parent_concept.children.append(self)

    def __str__(self):
        return f"Concept({self.label})"


class Relation:
    """Représente une relation dans l'ontologie."""

    def __init__(self, uri: str, label: str = None, description: str = None):
        self.uri = uri
        self.label = label or self._extract_label_from_uri(uri)
        self.description = description
        self.domain = []  # Concepts source de la relation
        self.range = []  # Concepts cible de la relation

    def _extract_label_from_uri(self, uri: str) -> str:
        """Extrait un label lisible à partir de l'URI."""
        if '#' in uri:
            return uri.split('#')[-1]
        return uri.split('/')[-1]

    def add_domain(self, concept):
        """Définit un concept comme domaine de la relation."""
        if concept not in self.domain:
            self.domain.append(concept)

    def add_range(self, concept):
        """Définit un concept comme portée de la relation."""
        if concept not in self.range:
            self.range.append(concept)

    def __str__(self):
        return f"Relation({self.label})"


class Domain:
    """Représente un domaine de connaissances dans l'ontologie."""

    def __init__(self, name: str, description: str = None):
        self.name = name
        self.description = description
        self.concepts = []  # Concepts principaux du domaine
        self.subdomains = []  # Sous-domaines
        self.parent_domain = None  # Domaine parent
        self.documents = []  # Documents liés à ce domaine

    def add_concept(self, concept):
        """Ajoute un concept au domaine."""
        if concept not in self.concepts:
            self.concepts.append(concept)

    def add_subdomain(self, subdomain):
        """Ajoute un sous-domaine."""
        if subdomain not in self.subdomains:
            self.subdomains.append(subdomain)
            subdomain.parent_domain = self

    def add_document(self, document_id: str, confidence: float = 1.0):
        """Lie un document à ce domaine avec un score de confiance."""
        self.documents.append({"id": document_id, "confidence": confidence})

    def __str__(self):
        return f"Domain({self.name})"


class Text2KGBenchParser:
    """Parser simplifié pour les ontologies au format du benchmark Text2KGBench."""

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.graph = rdflib.Graph()  # Garder un graphe vide mais compatible
        self.data = None
        self.concepts = set()
        self.relations = set()
        self.axioms = []

        # Charger et traiter le fichier JSON
        self._load_json()

    def _load_json(self):
        """Charge et traite le fichier JSON au format Text2KGBench."""
        try:
            if not os.path.exists(self.json_file_path):
                raise FileNotFoundError(f"Fichier non trouvé: {self.json_file_path}")

            with open(self.json_file_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)

            # Récupérer l'ID de l'ontologie ou le nom du fichier
            if 'id' not in self.data:
                base_name = os.path.basename(self.json_file_path)
                self.data['id'] = os.path.splitext(base_name)[0]

            # Générer la base URI pour cette ontologie
            base_uri = f"http://text2kgbench.org/ontology/{self.data.get('id')}#"

            # Traiter les concepts
            for concept in self.data.get('concepts', []):
                if 'qid' in concept:
                    concept_id = concept['qid']
                    concept_uri = f"{base_uri}{concept_id}"
                    self.concepts.add(concept_uri)

                    # Ajouter un triplet RDF minimal pour le label
                    if 'label' in concept:
                        self.graph.add((
                            rdflib.URIRef(concept_uri),
                            rdflib.RDFS.label,
                            rdflib.Literal(concept['label'])
                        ))

            # Traiter les relations
            for relation in self.data.get('relations', []):
                if 'pid' in relation:
                    relation_id = relation['pid']
                    relation_uri = f"{base_uri}{relation_id}"
                    self.relations.add(relation_uri)

                    # Ajouter un triplet RDF minimal pour le label
                    if 'label' in relation:
                        self.graph.add((
                            rdflib.URIRef(relation_uri),
                            rdflib.RDFS.label,
                            rdflib.Literal(relation['label'])
                        ))

                    # Traiter les axiomes domain/range
                    if 'domain' in relation and relation['domain']:
                        domain = relation['domain']
                        domain_uri = f"{base_uri}{domain}"
                        if domain_uri in self.concepts:  # S'assurer que le domaine est un concept connu
                            self.axioms.append(("domain", relation_uri, domain_uri))

                    if 'range' in relation and relation['range']:
                        range_val = relation['range']
                        # Ne traiter que les ranges qui sont des concepts (pas string, number, etc.)
                        if range_val not in ["string", "number", "date", ""]:
                            range_uri = f"{base_uri}{range_val}"
                            if range_uri in self.concepts:  # S'assurer que le range est un concept connu
                                self.axioms.append(("range", relation_uri, range_uri))

            print(
                f"Ontologie {self.data['id']} chargée: {len(self.concepts)} concepts, {len(self.relations)} relations, {len(self.axioms)} axiomes")

        except Exception as e:
            print(f"Erreur lors du chargement de l'ontologie {os.path.basename(self.json_file_path)}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Réinitialiser les structures pour éviter des données partielles
            self.concepts = set()
            self.relations = set()
            self.axioms = []


class OntologyManager:
    """Gère une ontologie et ses domaines de connaissances."""

    def __init__(self, storage_dir: str = "ontology_data"):
        self.storage_dir = storage_dir
        self.concepts = {}  # URI -> Concept
        self.relations = {}  # URI -> Relation
        self.domains = {}  # Nom -> Domaine
        self.axioms = []  # Liste de tuples (type, source, cible)

        # Assurer l'existence du répertoire de stockage
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "export"), exist_ok=True)

        # Graphe RDF pour l'ontologie
        self.graph = rdflib.Graph()

    def load_ontology(self, file_path: str) -> bool:
        """
        Charge une ontologie depuis un fichier (détecte automatiquement le format).
        """
        try:
            # Vérifier si un fichier TTL correspondant existe
            ttl_path = None

            # Si on nous donne un fichier JSON, chercher une version TTL équivalente
            if file_path.lower().endswith('.json'):
                # D'abord, essayer dans le sous-répertoire owl
                base_dir = os.path.dirname(file_path)
                base_name = os.path.basename(file_path).split('.')[0]

                # Construire le chemin vers le fichier TTL potentiel
                owl_dir = os.path.join(os.path.dirname(base_dir), "owl")
                if os.path.exists(owl_dir):
                    ttl_path = os.path.join(owl_dir, f"ont_{base_name}.ttl")
                    if not os.path.exists(ttl_path):
                        # Essayer sans le préfixe "ont_"
                        ttl_path = os.path.join(owl_dir, f"{base_name}.ttl")

                    # Si toujours pas trouvé, essayer d'autres extensions
                    if not os.path.exists(ttl_path):
                        for ext in ['.owl', '.rdf', '.xml']:
                            alt_path = os.path.join(owl_dir, f"{base_name}{ext}")
                            if os.path.exists(alt_path):
                                ttl_path = alt_path
                                break

                # Si un fichier TTL a été trouvé, l'utiliser
                if ttl_path and os.path.exists(ttl_path):
                    print(f"Utilisation du fichier TTL/OWL au lieu du JSON: {ttl_path}")
                    file_path = ttl_path

            # Choisir le parser approprié en fonction de l'extension
            if file_path.lower().endswith(('.ttl', '.owl', '.rdf', '.xml')):
                # Utiliser directement le parser TTL intégré à rdflib
                parser = OntologyParser(file_path)
            elif file_path.lower().endswith('.jsonld'):
                from ontology.jsonld_loader import JsonLDOntologyParser
                parser = JsonLDOntologyParser(file_path)
            elif file_path.lower().endswith('.json'):
                # Si on arrive ici, c'est qu'on n'a pas trouvé de TTL équivalent
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if ('concepts' in data and 'relations' in data and
                                isinstance(data.get('concepts', []), list) and
                                len(data.get('concepts', [])) > 0 and
                                isinstance(data.get('concepts', [])[0], dict) and
                                'qid' in data.get('concepts', [])[0]):
                            # C'est un fichier Text2KGBench, mais nous préférons les TTL
                            print(f"Format Text2KGBench JSON détecté pour {file_path}, mais non préféré")
                            # On peut soit abandonner, soit essayer quand même avec le parser Text2KGBench
                            return False
                except Exception as e:
                    print(f"Erreur lors de l'analyse du JSON: {str(e)}")
                    return False
            else:
                print(f"Format de fichier non supporté: {file_path}")
                return False

            # Stocker les axiomes extraits
            self.axioms.extend(parser.axioms)

            # Créer les concepts
            for concept_uri in parser.concepts:
                uri = str(concept_uri)
                # Récupérer le label s'il existe
                label = None
                for _, _, lbl in parser.graph.triples((rdflib.URIRef(uri), rdflib.RDFS.label, None)):
                    label = str(lbl)
                    break

                self.concepts[uri] = Concept(uri, label)

            # Créer les relations
            for relation_uri in parser.relations:
                uri = str(relation_uri)
                # Récupérer le label s'il existe
                label = None
                for _, _, lbl in parser.graph.triples((rdflib.URIRef(uri), rdflib.RDFS.label, None)):
                    label = str(lbl)
                    break

                self.relations[uri] = Relation(uri, label)

            # Établir les relations hiérarchiques et les domaines/portées
            for axiom_type, source, target in parser.axioms:
                if axiom_type == "subsumption":
                    if source in self.concepts and target in self.concepts:
                        self.concepts[source].add_parent(self.concepts[target])

                elif axiom_type == "domain":
                    if source in self.relations and target in self.concepts:
                        self.relations[source].add_domain(self.concepts[target])

                elif axiom_type == "range":
                    if source in self.relations and target in self.concepts:
                        self.relations[source].add_range(self.concepts[target])

            # Fusionner le graphe RDF
            self.graph += parser.graph

            # Enrichir les concepts avec leurs labels et descriptions
            self._enrich_concepts()

            return True
        except Exception as e:
            print(f"Erreur lors du chargement de l'ontologie: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def old_load_ontology(self, file_path: str) -> bool:
        """
        Charge une ontologie depuis un fichier (détecte automatiquement TTL ou JSON-LD).
        """
        try:

            # Détecter le format du fichier par son extension
            if file_path.lower().endswith(('.jsonld', '.json')):
                from ontology.jsonld_loader import JsonLDOntologyParser
                parser = JsonLDOntologyParser(file_path)
            else:  # Par défaut, utiliser le parser TTL existant
                parser = OntologyParser(file_path)

            # Stocker les axiomes extraits
            self.axioms = parser.axioms

            # Créer les concepts
            for concept_uri in parser.concepts:
                uri = str(concept_uri)
                # Récupérer le label s'il existe
                label = None
                for _, _, lbl in parser.graph.triples((rdflib.URIRef(uri), rdflib.RDFS.label, None)):
                    label = str(lbl)
                    break

                self.concepts[uri] = Concept(uri, label)

            # Créer les relations
            for relation_uri in parser.relations:
                uri = str(relation_uri)
                # Récupérer le label s'il existe
                label = None
                for _, _, lbl in parser.graph.triples((rdflib.URIRef(uri), rdflib.RDFS.label, None)):
                    label = str(lbl)
                    break

                self.relations[uri] = Relation(uri, label)

            # Établir les relations hiérarchiques et les domaines/portées
            for axiom_type, source, target in parser.axioms:
                if axiom_type == "subsumption":
                    if source in self.concepts and target in self.concepts:
                        self.concepts[source].add_parent(self.concepts[target])

                elif axiom_type == "domain":
                    if source in self.relations and target in self.concepts:
                        self.relations[source].add_domain(self.concepts[target])

                elif axiom_type == "range":
                    if source in self.relations and target in self.concepts:
                        self.relations[source].add_range(self.concepts[target])

            # Fusionner le graphe RDF
            self.graph += parser.graph

            # Enrichir les concepts avec leurs labels et descriptions
            self._enrich_concepts()

            return True
        except Exception as e:
            print(f"Erreur lors du chargement de l'ontologie: {str(e)}")
            return False

    def _enrich_concepts(self):
        """Enrichit tous les concepts avec leurs métadonnées sémantiques avec gestion d'erreurs robuste."""
        enriched = 0
        for uri, concept in self.concepts.items():
            try:
                # Extraire les informations sémantiques avec gestion d'erreurs
                semantics = self._extract_concept_semantics_safe(uri)

                # Stocker directement dans l'objet concept avec vérifications
                concept.label = semantics.get("label", "")
                concept.description = semantics.get("description", "")
                concept.alt_labels = semantics.get("alt_labels", [])
                concept.symbols = semantics.get("symbols", [])

                enriched += 1

                # Afficher quelques exemples pour vérification
                #if enriched <= 5 or enriched % 1000 == 0:
                desc = concept.description[:50] + "..." if concept.description and len(
                    concept.description) > 50 else concept.description
                print(f"Concept enrichi: {concept.label} - {desc}")

            except Exception as e:
                # Ne pas laisser une erreur sur un concept arrêter tout le processus
                print(f"⚠️ Erreur lors de l'enrichissement du concept {uri}: {str(e)}")

        print(f"✓ {enriched}/{len(self.concepts)} concepts enrichis avec succès")

    def _extract_concept_semantics_safe(self, concept_uri: str) -> Dict[str, Any]:
        """
        Extrait toutes les informations sémantiques d'un concept avec gestion d'erreurs robuste.

        Args:
            concept_uri: URI du concept

        Returns:
            Dictionnaire d'informations sémantiques
        """
        # Vérifier que l'URI est valide
        if not concept_uri or not isinstance(concept_uri, str):
            return {"label": "Concept inconnu", "description": ""}

        try:
            uri_ref = rdflib.URIRef(concept_uri)

            # Informations à collecter
            info = {
                "label": None,
                "description": None,
                "alt_labels": [],
                "symbols": [],
                "parent_labels": [],
            }

            # 1. Extraire le label principal (avec vérifications)
            try:
                for _, p, o in self.graph.triples((uri_ref, rdflib.SKOS.prefLabel, None)):
                    if o:  # Vérifier que o n'est pas None
                        info["label"] = str(o)
                        break
            except Exception as e:
                print(f"⚠️ Erreur lors de l'extraction du prefLabel pour {concept_uri}: {e}")

            if not info["label"]:
                try:
                    for _, p, o in self.graph.triples((uri_ref, rdflib.RDFS.label, None)):
                        if o:
                            info["label"] = str(o)
                            break
                except Exception:
                    pass

            # 2. Extraire la description (avec vérifications)
            try:
                # EMMO elucidation
                emmo_def = rdflib.URIRef("https://w3id.org/emmo#EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9")
                for _, _, o in self.graph.triples((uri_ref, emmo_def, None)):
                    if o:
                        info["description"] = str(o)
                        break

                # Fallbacks pour la description
                if not info["description"]:
                    for p in [rdflib.RDFS.comment, rdflib.SKOS.definition]:
                        for _, _, o in self.graph.triples((uri_ref, p, None)):
                            if o:
                                info["description"] = str(o)
                                break
                        if info["description"]:
                            break
            except Exception as e:
                print(f"⚠️ Erreur lors de l'extraction de la description pour {concept_uri}: {e}")

            # Fallback pour le label si toujours None
            if not info["label"]:
                # Extraire de l'URI avec gestion d'erreurs
                try:
                    if '#' in concept_uri:
                        info["label"] = concept_uri.split('#')[-1]
                    else:
                        info["label"] = concept_uri.split('/')[-1]
                except Exception:
                    info["label"] = "Inconnu"  # Dernière option

            return info

        except Exception as e:
            # Gestion d'erreur globale - retourner un dict valide avec valeurs par défaut
            print(f"⚠️ Erreur majeure lors de l'extraction des informations pour {concept_uri}: {e}")
            return {
                "label": concept_uri.split('#')[-1] if '#' in concept_uri else concept_uri.split('/')[-1],
                "description": "",
                "alt_labels": [],
                "symbols": [],
                "parent_labels": []
            }

    def resolve_uri(self, uri_or_prefixed: str) -> str:
        """
        Convertit une URI avec préfixe en URI complète si nécessaire.

        Args:
            uri_or_prefixed: URI avec ou sans préfixe

        Returns:
            URI complète
        """
        # Si déjà une URI complète
        if uri_or_prefixed.startswith("http://"):
            return uri_or_prefixed

        # Essayer de résoudre le préfixe
        parts = uri_or_prefixed.split(":", 1)
        if len(parts) == 2:
            prefix, local = parts
            # Mappings de préfixes courants
            prefix_map = {
                "sci": "http://example.org/scientific-ontology#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "xsd": "http://www.w3.org/2001/XMLSchema#"
            }

            if prefix in prefix_map:
                return f"{prefix_map[prefix]}{local}"

        # Si aucun préfixe connu, retourner tel quel
        return uri_or_prefixed

    def build_explicit_hierarchies(self):
        """Construit explicitement toutes les hiérarchies concept-subconcept."""
        # 1. Parcourir tous les concepts pour extraire les relations explicites
        for uri, concept in self.concepts.items():
            # Si le concept a des parents, ajouter des axiomes explicites
            for parent in concept.parents:
                # Vérifier si cette relation est déjà dans les axiomes
                axiom = ("subsumption", uri, parent.uri)
                if axiom not in self.axioms:
                    self.axioms.append(axiom)
                    print(f"Hiérarchie ajoutée: {uri} -> {parent.uri}")

        # 2. Détecter les relations subClassOf dans le graphe
        subClassOf = rdflib.RDFS.subClassOf
        for s, p, o in self.graph.triples((None, subClassOf, None)):
            if not isinstance(o, rdflib.BNode):  # Éviter les nœuds anonymes
                axiom = ("subsumption", str(s), str(o))
                if axiom not in self.axioms:
                    self.axioms.append(axiom)
                    print(f"Hiérarchie depuis RDF: {str(s)} -> {str(o)}")

    def create_domain(self, name: str, description: str = None) -> Domain:
        """
        Crée un nouveau domaine dans l'ontologie.

        Args:
            name: Nom du domaine
            description: Description du domaine

        Returns:
            Le domaine créé
        """
        if name in self.domains:
            return self.domains[name]

        domain = Domain(name, description)
        self.domains[name] = domain
        return domain

    def add_concept(self, uri: str, label: str = None, description: str = None) -> Concept:
        """
        Ajoute un nouveau concept à l'ontologie.

        Args:
            uri: URI du concept
            label: Label humain du concept
            description: Description du concept

        Returns:
            Le concept créé
        """
        if uri in self.concepts:
            return self.concepts[uri]

        # Créer le concept
        concept = Concept(uri, label, description)
        self.concepts[uri] = concept

        # Ajouter au graphe RDF
        self.graph.add((rdflib.URIRef(uri), rdflib.RDF.type, rdflib.OWL.Class))
        if label:
            self.graph.add((rdflib.URIRef(uri), rdflib.RDFS.label, rdflib.Literal(label)))
        if description:
            self.graph.add((rdflib.URIRef(uri), rdflib.RDFS.comment, rdflib.Literal(description)))

        return concept

    def add_relation(self, uri: str, label: str = None, description: str = None) -> Relation:
        """
        Ajoute une nouvelle relation à l'ontologie.

        Args:
            uri: URI de la relation
            label: Label humain de la relation
            description: Description de la relation

        Returns:
            La relation créée
        """
        if uri in self.relations:
            return self.relations[uri]

        # Créer la relation
        relation = Relation(uri, label, description)
        self.relations[uri] = relation

        # Ajouter au graphe RDF
        self.graph.add((rdflib.URIRef(uri), rdflib.RDF.type, rdflib.OWL.ObjectProperty))
        if label:
            self.graph.add((rdflib.URIRef(uri), rdflib.RDFS.label, rdflib.Literal(label)))
        if description:
            self.graph.add((rdflib.URIRef(uri), rdflib.RDFS.comment, rdflib.Literal(description)))

        return relation

    def add_concept_to_domain(self, concept_uri: str, domain_name: str) -> bool:
        """
        Ajoute un concept à un domaine.

        Args:
            concept_uri: URI du concept
            domain_name: Nom du domaine

        Returns:
            True si l'ajout a réussi, False sinon
        """
        if concept_uri not in self.concepts:
            return False

        if domain_name not in self.domains:
            self.create_domain(domain_name)

        self.domains[domain_name].add_concept(self.concepts[concept_uri])
        return True

    def set_concept_hierarchy(self, child_uri: str, parent_uri: str) -> bool:
        """
        Établit une relation de subsomption entre concepts.

        Args:
            child_uri: URI du concept enfant
            parent_uri: URI du concept parent

        Returns:
            True si l'ajout a réussi, False sinon
        """
        if child_uri not in self.concepts or parent_uri not in self.concepts:
            return False

        child = self.concepts[child_uri]
        parent = self.concepts[parent_uri]
        child.add_parent(parent)

        # Ajouter au graphe RDF
        self.graph.add((rdflib.URIRef(child_uri), rdflib.RDFS.subClassOf, rdflib.URIRef(parent_uri)))

        return True

    def set_relation_domain_range(self, relation_uri: str, domain_uri: str, range_uri: str) -> bool:
        """
        Définit le domaine et la portée d'une relation.

        Args:
            relation_uri: URI de la relation
            domain_uri: URI du concept domaine
            range_uri: URI du concept portée

        Returns:
            True si l'opération a réussi, False sinon
        """
        if relation_uri not in self.relations:
            return False

        if domain_uri not in self.concepts or range_uri not in self.concepts:
            return False

        relation = self.relations[relation_uri]
        domain = self.concepts[domain_uri]
        range_concept = self.concepts[range_uri]

        relation.add_domain(domain)
        relation.add_range(range_concept)

        # Ajouter au graphe RDF
        self.graph.add((rdflib.URIRef(relation_uri), rdflib.RDFS.domain, rdflib.URIRef(domain_uri)))
        self.graph.add((rdflib.URIRef(relation_uri), rdflib.RDFS.range, rdflib.URIRef(range_uri)))

        return True

    async def enrich_from_document(self, document_text: str, llm_provider) -> List[Tuple[str, str, str]]:
        """
        Enrichit l'ontologie avec les triplets extraits d'un document.

        Args:
            document_text: Texte du document
            llm_provider: Provider LLM pour l'extraction

        Returns:
            Liste des triplets extraits
        """
        # Créer un extracteur de triplets
        extractor = TripletsExtractor(llm_provider, self)

        # Extraire les triplets
        triplets = await extractor.extract_triplets_from_text(document_text)

        # Pour chaque triplet, mettre à jour l'ontologie
        for subject, relation, object in triplets:
            # S'assurer que le sujet et l'objet sont des concepts
            if subject not in self.concepts:
                self.add_concept(subject)

            if object not in self.concepts:
                self.add_concept(object)

            # S'assurer que la relation existe
            if relation not in self.relations:
                self.add_relation(relation)

            # Mettre à jour les domaines et portées
            self.set_relation_domain_range(relation, subject, object)

        return triplets

    def export_ontology(self, file_path: str, format: str = "ttl") -> bool:
        """
        Exporte l'ontologie dans un fichier.

        Args:
            file_path: Chemin du fichier d'export
            format: Format d'export (ttl, xml, json-ld, n3, etc.)

        Returns:
            True si l'export a réussi, False sinon
        """
        try:
            self.graph.serialize(destination=file_path, format=format)
            return True
        except Exception as e:
            print(f"Erreur lors de l'export de l'ontologie: {str(e)}")
            return False

    def validate_ontology(self) -> Tuple[bool, List[str]]:
        """
        Valide la cohérence de l'ontologie.

        Returns:
            (True, []) si l'ontologie est cohérente,
            (False, [erreurs]) si des incohérences sont détectées
        """
        errors = []

        # Vérifier les cycles dans la hiérarchie des concepts
        for uri, concept in self.concepts.items():
            visited = set()

            def check_cycle(c, path=None):
                if path is None:
                    path = []

                if c.uri in visited:
                    if c.uri in path:
                        # Cycle détecté
                        cycle_path = path[path.index(c.uri):] + [c.uri]
                        error_msg = f"Cycle détecté dans la hiérarchie des concepts: {' -> '.join(cycle_path)}"
                        if error_msg not in errors:
                            errors.append(error_msg)
                    return

                visited.add(c.uri)
                path.append(c.uri)

                for parent in c.parents:
                    check_cycle(parent, path.copy())

            check_cycle(concept)

        # Vérifier les relations sans domaine ou portée
        for uri, relation in self.relations.items():
            if not relation.domain:
                errors.append(f"La relation {uri} n'a pas de domaine défini")

            if not relation.range:
                errors.append(f"La relation {uri} n'a pas de portée définie")

        return len(errors) == 0, errors

    def get_concept_by_label(self, label: str) -> Optional[Concept]:
        """Recherche un concept par son label."""
        for _, concept in self.concepts.items():
            if concept.label.lower() == label.lower():
                return concept
        return None

    def get_domains_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Récupère tous les domaines associés à un document.

        Args:
            document_id: ID du document

        Returns:
            Liste des domaines avec leur score de confiance
        """
        result = []
        for name, domain in self.domains.items():
            for doc in domain.documents:
                if doc["id"] == document_id:
                    result.append({
                        "domain": name,
                        "confidence": doc["confidence"]
                    })
        return result

    def associate_document_with_domain(self, document_id: str, domain_name: str, confidence: float = 1.0) -> bool:
        """
        Associe un document à un domaine avec un score de confiance.

        Args:
            document_id: ID du document
            domain_name: Nom du domaine
            confidence: Score de confiance (0-1)

        Returns:
            True si l'association a réussi, False sinon
        """
        if domain_name not in self.domains:
            return False

        self.domains[domain_name].add_document(document_id, confidence)
        return True

    def extract_concept_semantics(self, concept_uri: str) -> Dict[str, Any]:
        """
        Extrait toutes les informations sémantiques pertinentes d'un concept.

        Args:
            concept_uri: URI du concept

        Returns:
            Dictionnaire d'informations sémantiques
        """
        uri_ref = rdflib.URIRef(concept_uri)

        # Informations à collecter
        info = {
            "label": None,
            "description": None,
            "alt_labels": [],
            "symbols": [],
            "parent_labels": [],
        }

        # 1. Extraire le label principal
        for _, p, o in self.graph.triples((uri_ref, rdflib.SKOS.prefLabel, None)):
            info["label"] = str(o)
            break

        if not info["label"]:
            for _, p, o in self.graph.triples((uri_ref, rdflib.RDFS.label, None)):
                info["label"] = str(o)
                break

        # 2. Extraire les labels alternatifs
        for _, p, o in self.graph.triples((uri_ref, rdflib.SKOS.altLabel, None)):
            info["alt_labels"].append(str(o))

        # 3. Extraire la description - spécifique à EMMO
        emmo_definition = rdflib.URIRef("https://w3id.org/emmo#EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9")
        for _, p, o in self.graph.triples((uri_ref, emmo_definition, None)):
            info["description"] = str(o)
            break

        if not info["description"]:
            for p in [rdflib.RDFS.comment, rdflib.SKOS.definition]:
                for _, _, o in self.graph.triples((uri_ref, p, None)):
                    info["description"] = str(o)
                    break
                if info["description"]:
                    break

        # 4. Extraire les symboles - spécifique à EMMO
        emmo_symbol = rdflib.URIRef("https://w3id.org/emmo#EMMO_7f1dec83_d85e_4e1b_b7bd_c9442d4f5a64")
        for _, _, o in self.graph.triples((uri_ref, emmo_symbol, None)):
            info["symbols"].append(str(o))

        # 5. Extraire les parents et leurs labels
        for _, _, parent in self.graph.triples((uri_ref, rdflib.RDFS.subClassOf, None)):
            if isinstance(parent, rdflib.URIRef):  # Ignorer les restrictions anonymes
                parent_label = self._get_best_label(parent)
                if parent_label:
                    info["parent_labels"].append(parent_label)

        # Si nous n'avons toujours pas de label, utiliser la partie finale de l'URI
        if not info["label"]:
            info["label"] = concept_uri.split('#')[-1].split('/')[-1]

        return info

    def _get_best_label(self, uri_ref):
        """Récupère le meilleur label disponible pour une URI."""
        # Essayer prefLabel
        for _, _, o in self.graph.triples((uri_ref, rdflib.SKOS.prefLabel, None)):
            return str(o)

        # Essayer label standard
        for _, _, o in self.graph.triples((uri_ref, rdflib.RDFS.label, None)):
            return str(o)

        # Extraire de l'URI
        uri = str(uri_ref)
        return uri.split('#')[-1].split('/')[-1]

    def get_concept_hierarchy_chain(self, concept_uri: str) -> List[str]:
        """
        Retourne la chaîne hiérarchique complète d'un concept.

        Args:
            concept_uri: URI du concept

        Returns:
            Liste des noms de concepts dans l'ordre hiérarchique (du plus général au plus spécifique)
        """
        hierarchy = []
        concept = self.concepts.get(concept_uri)

        if not concept:
            return hierarchy

        # Parcourir la hiérarchie vers le haut
        current = concept
        visited = set()  # Pour éviter les cycles

        while current and current.uri not in visited:
            # Ajouter le concept actuel
            label = current.label if current.label else current.uri.split('#')[-1]
            hierarchy.insert(0, label)

            # Marquer comme visité
            visited.add(current.uri)

            # Passer au parent (prendre le premier si plusieurs)
            if hasattr(current, 'parents') and current.parents:
                current = current.parents[0]
            else:
                current = None

        return hierarchy

# Définir la classe OntologyParser ici, en réutilisant votre code
class OntologyParser:
    def __init__(self, ttl_file_path: str):
        """Analyse un fichier ontologie au format TTL."""
        self.graph = rdflib.Graph()
        self.graph.parse(ttl_file_path, format="ttl")

        # Extraire les éléments fondamentaux
        self.concepts = self._extract_concepts()
        self.relations = self._extract_relations()
        self.axioms = self._extract_axioms()

    def _extract_concepts(self) -> Set[str]:
        """Extrait tous les concepts (classes) de l'ontologie."""
        concepts = set()
        for s, p, o in self.graph.triples((None, rdflib.RDF.type, rdflib.OWL.Class)):
            concepts.add(str(s))
        return concepts

    def _extract_relations(self) -> Set[str]:
        """Extrait toutes les relations (propriétés) de l'ontologie."""
        relations = set()
        for s, p, o in self.graph.triples((None, rdflib.RDF.type, rdflib.OWL.ObjectProperty)):
            relations.add(str(s))
        return relations

    def _extract_axioms(self) -> List[Tuple[str, str, str]]:
        """Extrait les axiomes (subsomption, domaine, portée, etc.)."""
        axioms = []

        # Subsomption
        for s, p, o in self.graph.triples((None, rdflib.RDFS.subClassOf, None)):
            if str(s) in self.concepts and str(o) in self.concepts:
                axioms.append(("subsumption", str(s), str(o)))

        # Domaine et portée
        for r in self.relations:
            rel_uri = rdflib.URIRef(r)
            for _, _, domain in self.graph.triples((rel_uri, rdflib.RDFS.domain, None)):
                axioms.append(("domain", r, str(domain)))
            for _, _, range_val in self.graph.triples((rel_uri, rdflib.RDFS.range, None)):
                axioms.append(("range", r, str(range_val)))

        return axioms


class TripletsExtractor:
    def __init__(self, llm_provider, ontology_manager):
        """Extrait des triplets (h,r,t) à partir de texte."""
        self.llm_provider = llm_provider
        self.ontology_manager = ontology_manager  # Remplacer ontology_parser par ontology_manager

    async def extract_triplets_from_text(self, text: str) -> List[Tuple[str, str, str]]:
        """Extrait des triplets d'un texte via LLM."""
        # Récupérer les relations depuis l'ontology_manager
        relations = list(self.ontology_manager.relations.keys())

        # Limiter à 30 relations pour le prompt
        relations_sample = relations[:30] if len(relations) > 30 else relations

        messages = [
            {
                "role": "system",
                "content": "Tu es un expert en extraction d'informations structurées."
            },
            {
                "role": "user",
                "content": f"""
                Extrais TOUS les triplets (sujet, relation, objet) du texte suivant.
                Utilise UNIQUEMENT les relations de cette liste: {', '.join(relations_sample)}

                Texte: {text[:4000]}

                Format de sortie (un triplet par ligne):
                sujet1 | relation1 | objet1
                sujet2 | relation2 | objet2
                """
            }
        ]

        # Appeler le provider LLM
        response = await self.llm_provider.generate_response_for_humain(messages, stream=False)

        # Extraire le contenu de la réponse
        response_text = ""
        if isinstance(response, dict) and "answer" in response:
            response_text = response["answer"]
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = response.get("text",
                                         response.get("content",
                                                      response.get("message", {}).get("content", "")))

        # Analyser la réponse
        triplets = []
        for line in response_text.strip().split('\n'):
            if '|' in line:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) == 3:
                    h, r, t = parts
                    if r in self.ontology_manager.relations:
                        triplets.append((h, r, t))

        return triplets