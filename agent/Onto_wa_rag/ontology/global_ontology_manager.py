"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ontology/global_ontology_manager.py
import os
import rdflib
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from ontology.ontology_manager import OntologyManager, Domain, Concept, Relation, OntologyParser


class GlobalOntologyManager(OntologyManager):
    """
    Gestionnaire d'ontologie global intégrant plusieurs ontologies TTL
    comme des domaines distincts au sein d'une ontologie unifiée.
    """

    def __init__(self, storage_dir: str = "global_ontology"):
        """Initialise le gestionnaire d'ontologie global."""
        super().__init__(storage_dir=storage_dir)

        # Dictionnaire pour les mappings d'URI
        self.uri_mappings = {}  # uri_originale -> uri_prefixée
        self.reverse_mappings = {}  # uri_prefixée -> uri_originale

        # Informations sur les ontologies importées
        self.imported_ontologies = {}  # domaine -> {path, graph, concepts_count, etc.}

    async def integrate_ontology_as_domain(
            self,
            ttl_path: str,
            domain_name: str,
            domain_description: str = None,
            parent_domain: str = None
    ) -> bool:
        """Intègre une ontologie TTL comme un domaine distinct."""
        print(f"Intégration de l'ontologie {ttl_path} comme domaine '{domain_name}'...")

        # Vérifier si le domaine existe déjà
        if domain_name in self.domains:
            print(f"Le domaine '{domain_name}' existe déjà")
            return False

        # Créer un gestionnaire temporaire pour charger l'ontologie séparément
        temp_manager = OntologyManager()
        success = temp_manager.load_ontology(ttl_path)

        if not success:
            print(f"Échec du chargement de l'ontologie {ttl_path}")
            return False

        # Créer le domaine dans l'ontologie globale
        domain = self.create_domain(domain_name, domain_description)

        # Si un parent est spécifié, établir la relation
        if parent_domain and parent_domain in self.domains:
            parent = self.domains[parent_domain]
            parent.add_subdomain(domain)

        # Stocker les informations sur l'ontologie importée
        self.imported_ontologies[domain_name] = {
            "path": ttl_path,
            "graph": temp_manager.graph,
            "concepts_count": len(temp_manager.concepts),
            "relations_count": len(temp_manager.relations)
        }

        # Transférer les concepts avec préfixage
        concepts_map = {}  # uri_originale -> concept_préfixé

        for uri, concept in temp_manager.concepts.items():
            # Créer une URI préfixée
            prefixed_uri = self._create_prefixed_uri(domain_name, uri)

            # Stocker le mapping URI
            self.uri_mappings[uri] = prefixed_uri
            self.reverse_mappings[prefixed_uri] = uri

            # Ajouter le concept à l'ontologie globale
            new_concept = self.add_concept(
                prefixed_uri,
                concept.label,
                concept.description
            )

            # Associer au domaine
            domain.add_concept(new_concept)

            # Mémoriser pour établir les relations plus tard
            concepts_map[uri] = new_concept

        # Transférer les relations avec préfixage
        relations_map = {}  # uri_originale -> relation_préfixée

        for uri, relation in temp_manager.relations.items():
            # Créer une URI préfixée
            prefixed_uri = self._create_prefixed_uri(domain_name, uri)

            # Stocker le mapping URI
            self.uri_mappings[uri] = prefixed_uri
            self.reverse_mappings[prefixed_uri] = uri

            # Ajouter la relation à l'ontologie globale
            new_relation = self.add_relation(
                prefixed_uri,
                relation.label,
                relation.description
            )

            # Mémoriser pour établir les relations plus tard
            relations_map[uri] = new_relation

        # Établir les relations hiérarchiques entre concepts
        for uri, concept in temp_manager.concepts.items():
            if not concept.parents:
                continue

            new_concept = concepts_map[uri]

            for parent in concept.parents:
                if parent.uri in concepts_map:
                    parent_concept = concepts_map[parent.uri]
                    new_concept.add_parent(parent_concept)

        # Établir les relations domaine/portée pour les relations
        for uri, relation in temp_manager.relations.items():
            if uri not in relations_map:
                continue

            new_relation = relations_map[uri]

            # Domaines
            for domain_concept in relation.domain:
                if domain_concept.uri in concepts_map:
                    new_relation.add_domain(concepts_map[domain_concept.uri])

            # Portées
            for range_concept in relation.range:
                if range_concept.uri in concepts_map:
                    new_relation.add_range(concepts_map[range_concept.uri])

        # Transférer les axiomes en préfixant les URIs
        for axiom_type, source, target in temp_manager.axioms:
            if source in self.uri_mappings and target in self.uri_mappings:
                prefixed_source = self.uri_mappings[source]
                prefixed_target = self.uri_mappings[target]
                self.axioms.append((axiom_type, prefixed_source, prefixed_target))

        # Fusionner le graphe RDF
        self.graph += temp_manager.graph

        print(f"✓ Domaine '{domain_name}' créé avec {len(concepts_map)} concepts et {len(relations_map)} relations")
        return True

    def _create_prefixed_uri(self, domain_name: str, uri: str) -> str:
        """Crée une URI préfixée pour éviter les conflits entre domaines."""
        # Si l'URI a un fragment (#)
        if "#" in uri:
            base, fragment = uri.split("#", 1)
            return f"{base}#{domain_name}_{fragment}"

        # Si l'URI se termine par un slash ou a une structure de chemin
        elif uri.endswith("/") or uri.rstrip("/").split("/")[-1]:
            segments = uri.rstrip("/").split("/")
            last_segment = segments[-1]
            base = "/".join(segments[:-1])
            return f"{base}/{domain_name}_{last_segment}"

        # Fallback pour les autres cas
        else:
            return f"{uri}_{domain_name}"

    def get_domain_for_uri(self, prefixed_uri: str) -> Optional[str]:
        """Détermine le domaine d'origine d'une URI préfixée."""
        for domain_name in self.imported_ontologies.keys():
            if f"_{domain_name}" in prefixed_uri or f"#{domain_name}_" in prefixed_uri:
                return domain_name
        return None

    def get_original_uri(self, prefixed_uri: str) -> Optional[str]:
        """Récupère l'URI originale à partir d'une URI préfixée."""
        return self.reverse_mappings.get(prefixed_uri)

    async def integrate_multiple_ontologies(self, ttl_files: List[Tuple[str, str, Optional[str]]]) -> Dict[str, bool]:
        """Intègre plusieurs ontologies TTL comme domaines."""
        results = {}

        for ttl_path, domain_name, description in ttl_files:
            success = await self.integrate_ontology_as_domain(
                ttl_path, domain_name, description
            )
            results[domain_name] = success

        return results

    def get_imported_domains_info(self) -> Dict[str, Dict[str, Any]]:
        """Récupère des informations sur tous les domaines importés."""
        info = {}

        for domain_name, domain_data in self.imported_ontologies.items():
            domain = self.domains.get(domain_name)
            if not domain:
                continue

            info[domain_name] = {
                "path": domain_data["path"],
                "concepts_count": domain_data["concepts_count"],
                "relations_count": domain_data["relations_count"],
                "subconcepts_count": len(domain.subdomains),
                "associated_documents": len(domain.documents)
            }

        return info

    def resolve_cross_domain_uri(self, uri: str, source_domain: str) -> str:
        """Résout une URI potentiellement ambiguë en tenant compte du domaine source."""
        # Si l'URI est déjà une URI complète, la retourner
        if uri.startswith("http://") or uri.startswith("https://"):
            return uri

        # Si c'est une référence simple (ex: "Person"), essayer de la résoudre dans le domaine source
        prefixed_uri = f"{source_domain}_{uri}"

        # Vérifier si cette URI préfixée existe
        for concept_uri in self.concepts:
            if concept_uri.endswith(prefixed_uri):
                return concept_uri

        # Fallback: chercher dans tous les domaines
        for domain_name in self.imported_ontologies:
            test_uri = f"{domain_name}_{uri}"
            for concept_uri in self.concepts:
                if concept_uri.endswith(test_uri):
                    return concept_uri

        # Si rien n'est trouvé, retourner l'URI originale
        return uri