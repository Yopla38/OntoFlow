"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import rdflib
import os

# Supposons que ces constantes de couleur soient définies quelque part
BLUE = '\033[94m'
RED = '\033[91m'
RESET = '\033[0m'


class OntologyMerge:
    def __init__(self, news_ontology_dir, results_dir):
        self.news_ontology_dir = news_ontology_dir
        self.results_dir = results_dir
        # Créer le répertoire de résultats s'il n'existe pas
        os.makedirs(self.results_dir, exist_ok=True)

    def _bind_known_prefixes(self, graph: rdflib.Graph):
        """Lie les préfixes connus au graphe pour une sérialisation propre."""
        print("Liaison des préfixes au graphe unifié...")

        # Définir tous les préfixes utilisés dans vos fichiers
        prefixes = {
            "pna": rdflib.Namespace("http://data.press.net/ontology/asset/"),
            "pnc": rdflib.Namespace("http://data.press.net/ontology/classification/"),
            "pne": rdflib.Namespace("http://data.press.net/ontology/event/"),
            "pni": rdflib.Namespace("http://data.press.net/ontology/identifier/"),
            "pns": rdflib.Namespace("http://data.press.net/ontology/stuff/"),
            "pnt": rdflib.Namespace("http://data.press.net/ontology/tag/"),
            "owl": rdflib.OWL,
            "rdf": rdflib.RDF,
            "rdfs": rdflib.RDFS,
            "xsd": rdflib.XSD,
            "dcterms": rdflib.Namespace("http://purl.org/dc/terms/"),
            "foaf": rdflib.FOAF,
            "dcmitype": rdflib.Namespace("http://purl.org/dc/dcmitype/"),
            "event": rdflib.Namespace("http://purl.org/NET/c4dm/event.owl#"),
            "geo": rdflib.Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#"),
            "time": rdflib.Namespace("http://www.w3.org/2006/time#"),
            "vs": rdflib.Namespace("http://www.w3.org/2003/06/sw-vocab-status/ns#"),
            "psys": rdflib.Namespace("http://proton.semanticweb.org/protonsys#")
        }

        for prefix, namespace in prefixes.items():
            graph.bind(prefix, namespace)

        print("✓ Préfixes liés.")

    def create_unified_ontology(self) -> str:
        """Fusionne tous les modules SNaP en un seul fichier TTL de haute qualité."""
        print(f"{BLUE}Création d'une ontologie unifiée à partir des modules SNaP...{RESET}")

        unified_graph = rdflib.Graph()
        self._bind_known_prefixes(unified_graph)  # Lier les préfixes est toujours crucial

        modules = ["asset.ttl", "classification.ttl", "domain.ttl", "event.ttl", "identifier.ttl", "tag.ttl"]
        loaded_modules = 0

        for module in modules:
            module_path = os.path.join(self.news_ontology_dir, module)
            if os.path.exists(module_path):
                try:
                    unified_graph.parse(module_path, format="turtle")
                    loaded_modules += 1
                    print(f"✓ Module fusionné: {module}")
                except Exception as e:
                    print(f"⚠️ Erreur lors du chargement de {module}: {e}")
            else:
                print(f"⚠️ Module non trouvé: {module}")

        if loaded_modules == 0:
            print(f"{RED}Aucun module n'a pu être chargé!{RESET}")
            return None

        unified_path = os.path.join(self.results_dir, "snap_unified.ttl")

        try:
            # 1. Sérialiser le graphe dans une chaîne de caractères en mémoire
            ttl_data = unified_graph.serialize(format="turtle")

            # 2. --- CORRECTION POST-SÉRIALISATION ---
            # Remplacer les formes incorrectes (ex: "pna: a owl:Ontology") par les formes correctes.
            print("Nettoyage de la sortie pour une qualité maximale...")
            for prefix, namespace in unified_graph.namespaces():
                # Cible la définition de l'ontologie (ex: "pna: a owl:Ontology")
                # Le \n garantit que nous ne remplaçons que les sujets en début de ligne.
                incorrect_onto_def = f"\n{prefix}: a owl:Ontology"
                correct_onto_def = f"\n<{namespace}> a owl:Ontology"
                ttl_data = ttl_data.replace(incorrect_onto_def, correct_onto_def)

                # Cible la définition de la propriété (ex: "isDefinedBy pna:")
                incorrect_prop_def = f"rdfs:isDefinedBy {prefix}:"
                correct_prop_def = f"rdfs:isDefinedBy <{namespace}>"
                ttl_data = ttl_data.replace(incorrect_prop_def, correct_prop_def)

            # 3. Écrire la chaîne de caractères corrigée dans le fichier final
            with open(unified_path, "w", encoding="utf-8") as f:
                f.write(ttl_data)

            print(f"✓ Ontologie unifiée de haute qualité créée: {unified_path}")
            print(f"  - {loaded_modules} modules fusionnés")
            print(f"  - {len(unified_graph)} triplets RDF")
            return unified_path

        except Exception as e:
            print(f"{RED}Erreur lors de la sauvegarde: {e}{RESET}")
            return None

    def old_create_unified_ontology(self) -> str:
        """Fusionne tous les modules SNaP en un seul fichier TTL de haute qualité."""
        print(f"{BLUE}Création d'une ontologie unifiée à partir des modules SNaP...{RESET}")

        # Créer un graphe RDF unifié
        unified_graph = rdflib.Graph()

        # Modules à fusionner
        modules = ["asset.ttl", "classification.ttl", "domain.ttl", "event.ttl", "identifier.ttl", "tag.ttl"]
        loaded_modules = 0

        for module in modules:
            module_path = os.path.join(self.news_ontology_dir, module)

            if os.path.exists(module_path):
                try:
                    # Charger le module dans le graphe unifié
                    unified_graph.parse(module_path, format="turtle")
                    loaded_modules += 1
                    print(f"✓ Module fusionné: {module}")

                except Exception as e:
                    print(f"⚠️ Erreur lors du chargement de {module}: {e}")
            else:
                print(f"⚠️ Module non trouvé: {module}")

        if loaded_modules == 0:
            print(f"{RED}Aucun module chargé!{RESET}")
            return None

        # --- AMÉLIORATION CLÉ ---
        # Lier les préfixes avant de sauvegarder pour garantir un fichier propre.
        self._bind_known_prefixes(unified_graph)

        # Sauvegarder l'ontologie unifiée
        unified_path = os.path.join(self.results_dir, "snap_unified.ttl")

        try:
            # La sérialisation utilisera désormais les préfixes que nous avons liés
            unified_graph.serialize(destination=unified_path, format="turtle")
            print(f"✓ Ontologie unifiée de haute qualité créée: {unified_path}")
            print(f"  - {loaded_modules} modules fusionnés")
            print(f"  - {len(unified_graph)} triplets RDF")

            return unified_path

        except Exception as e:
            print(f"{RED}Erreur lors de la sauvegarde: {e}{RESET}")
            return None

