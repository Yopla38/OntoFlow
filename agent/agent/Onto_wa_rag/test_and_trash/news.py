"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# ragas_benchmark_ontorag_news.py - Version adaptée pour SNaP
import json
import os
import asyncio
from typing import List, Tuple

from CONSTANT import BLUE, RESET, API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, RED, BOLD, GREEN, YELLOW
from RAGAS_benchmark import RAGASOntoRAGBenchmark
from main_app import OntoRAG

# ragas_benchmark_ontorag_news.py - Version corrigée
import os
import asyncio
import rdflib
from typing import List, Tuple

from ontology.merge_ontology import OntologyMerge


class RAGASNewsOntoRAGBenchmark(RAGASOntoRAGBenchmark):
    """Benchmark RAGAS spécialisé pour l'ontologie news SNaP modulaire"""

    def __init__(self, news_ontology_dir: str, max_samples=50):
        super().__init__(ontology_path=None, max_samples=max_samples)
        self.news_ontology_dir = news_ontology_dir
        self.unified_ontology_path = None

    def old_download_multihop_rag_dataset(self):
        """Télécharge le vrai dataset MultiHop-RAG depuis Hugging Face"""
        print(f"{BLUE}Téléchargement du dataset MultiHop-RAG officiel...{RESET}")

        try:
            from datasets import load_dataset

            # Charger le corpus (articles) depuis Hugging Face
            corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")

            # Extraire les articles du knowledge base
            kb_articles = corpus["train"]  # Le corpus est dans "train"

            # Convertir en format attendu par votre code
            processed_articles = []
            for article in kb_articles:
                processed_articles.append({
                    "title": article.get("title", ""),
                    "content": article.get("content", ""),
                    "source": article.get("source", ""),
                    "date": article.get("date", ""),
                    "url": article.get("url", "")
                })

            # Sauvegarder localement
            dataset_path = os.path.join(self.results_dir, "multihop_rag_articles.json")
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(processed_articles, f, indent=2, ensure_ascii=False)

            print(f"{GREEN}✓ {len(processed_articles)} articles téléchargés depuis MultiHop-RAG{RESET}")
            return dataset_path

        except Exception as e:
            print(f"{YELLOW}Erreur téléchargement officiel: {e}{RESET}")
            print(f"{BLUE}Utilisation du dataset manuel...{RESET}")
            return self._create_sample_dataset()

    def _create_real_test_dataset(self):
        """Utilise les vraies questions du dataset MultiHop-RAG"""
        print(f"{BLUE}Chargement des vraies questions MultiHop-RAG...{RESET}")

        try:
            from datasets import load_dataset

            # Charger le dataset principal avec questions
            dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
            data_split = dataset["train"]

            # Extraire les questions avec les bonnes clés
            test_questions = []
            for i in range(min(self.max_samples, len(data_split))):
                item = data_split[i]

                # Utiliser les noms de colonnes corrects
                question = item["query"]
                answer = item["answer"]

                test_questions.append({
                    "question": question,
                    "ground_truth": answer
                })

            # Convertir en format Dataset
            from datasets import Dataset
            self.test_dataset = Dataset.from_dict({
                "question": [q["question"] for q in test_questions],
                "ground_truth": [q["ground_truth"] for q in test_questions]
            })

            print(f"{GREEN}✓ {len(test_questions)} vraies questions MultiHop-RAG chargées{RESET}")
            return True

        except Exception as e:
            print(f"{YELLOW}Erreur chargement questions: {e}{RESET}")
            print(f"{BLUE}Utilisation des questions manuelles...{RESET}")
            return self._create_news_test_dataset()

    def download_multihop_rag_dataset(self):
        """Télécharge et extrait les articles du dataset MultiHop-RAG"""
        print(f"{BLUE}Téléchargement du dataset MultiHop-RAG officiel...{RESET}")

        try:
            from datasets import load_dataset

            # Charger le dataset avec les questions ET les preuves
            questions_dataset = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")

            # Extraire tous les articles depuis les evidence_list
            processed_articles = []
            article_id = 0

            for item in questions_dataset["train"]:
                evidence_list = item.get("evidence_list", [])

                for evidence in evidence_list:
                    # Créer un article à partir de chaque preuve
                    article = {
                        "id": f"article_{article_id}",
                        "title": evidence.get("title", ""),
                        "content": evidence.get("fact", ""),  # Le fait est le contenu
                        "source": evidence.get("source", ""),
                        "author": evidence.get("author", ""),
                        "category": evidence.get("category", ""),
                        "published_at": evidence.get("published_at", ""),
                        "url": evidence.get("url", "")
                    }
                    processed_articles.append(article)
                    article_id += 1

            # Sauvegarder localement
            dataset_path = os.path.join(self.results_dir, "multihop_rag_articles.json")
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(processed_articles, f, indent=2, ensure_ascii=False)

            print(f"{GREEN}✓ {len(processed_articles)} articles extraits des preuves MultiHop-RAG{RESET}")
            return dataset_path

        except Exception as e:
            print(f"{YELLOW}Erreur téléchargement officiel: {e}{RESET}")
            print(f"{BLUE}Utilisation du dataset manuel...{RESET}")
            return self._create_sample_dataset()

    def load_documents(self, dataset_path):
        """Charge les documents depuis le dataset"""
        print(f"{BLUE}Chargement des documents...{RESET}")

        try:
            # Charger directement le corpus MultiHop-RAG
            from datasets import load_dataset

            print(f"{BLUE}Chargement du corpus MultiHop-RAG complet...{RESET}")
            corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")

            # Extraire tous les articles du corpus
            all_articles = corpus["train"]

            print(f"📊 {len(all_articles)} articles disponibles dans le corpus")

            # DEBUG: Examiner la structure du premier article
            if len(all_articles) > 0:
                first_article = all_articles[0]
                print(f"🔍 Type du premier article: {type(first_article)}")
                print(f"🔍 Premier article: {first_article}")
                if hasattr(all_articles, 'column_names'):
                    print(f"🔍 Colonnes disponibles: {all_articles.column_names}")

            # Prendre les articles selon max_samples
            selected_articles = all_articles[:self.max_samples] if self.max_samples > 0 else all_articles

            # Convertir en format attendu
            self.documents = []
            for i, article in enumerate(selected_articles):
                try:
                    # Gérer différents types de structure
                    if isinstance(article, dict):
                        # Si c'est un dict, utiliser get()
                        doc = {
                            "title": article.get("title", f"Article_{i}"),
                            "content": article.get("content", ""),
                            "source": article.get("source", ""),
                            "date": article.get("date", ""),
                            "url": article.get("url", ""),
                            "category": article.get("category", ""),
                            "author": article.get("author", "")
                        }
                    elif isinstance(article, str):
                        # Si c'est une string, l'utiliser comme contenu
                        doc = {
                            "title": f"Article_{i}",
                            "content": article,
                            "source": "",
                            "date": "",
                            "url": "",
                            "category": "",
                            "author": ""
                        }
                    else:
                        # Essayer d'accéder par index si c'est un autre type
                        doc = {
                            "title": article[0] if len(article) > 0 else f"Article_{i}",
                            "content": article[1] if len(article) > 1 else str(article),
                            "source": article[2] if len(article) > 2 else "",
                            "date": article[3] if len(article) > 3 else "",
                            "url": article[4] if len(article) > 4 else "",
                            "category": article[5] if len(article) > 5 else "",
                            "author": article[6] if len(article) > 6 else ""
                        }

                    # Vérifier que le contenu n'est pas vide
                    if doc["content"].strip():
                        self.documents.append(doc)

                except Exception as e:
                    print(f"⚠️ Erreur article {i}: {e}")
                    continue

            print(f"{GREEN}✓ {len(self.documents)} articles chargés avec contenu{RESET}")

            # Afficher quelques exemples
            print(f"\n{YELLOW}📋 Exemples d'articles chargés:{RESET}")
            for i, doc in enumerate(self.documents[:3]):
                print(f"  {i + 1}. {doc['title'][:60]}...")
                print(f"     Source: {doc['source']}, Catégorie: {doc['category']}")
                print(f"     Contenu: {doc['content'][:100]}...")
                print()

            return True

        except Exception as e:
            print(f"{RED}Erreur lors du chargement: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    def old_load_documents(self, dataset_path):
        """Charge les documents avec le contenu des faits"""
        print(f"{BLUE}Chargement des documents...{RESET}")

        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Préparer les documents avec title + content
            processed_documents = []
            for article in data:
                # Combiner titre et contenu pour un document plus riche
                full_content = f"{article['title']}\n\n{article['content']}"
                if article.get('source'):
                    full_content += f"\n\nSource: {article['source']}"

                processed_documents.append({
                    "title": article['title'],
                    "content": full_content,
                    "source": article.get('source', ''),
                    "author": article.get('author', ''),
                    "category": article.get('category', ''),
                    "published_at": article.get('published_at', ''),
                    "url": article.get('url', '')
                })

            self.documents = processed_documents[:self.max_samples]
            print(f"{GREEN}✓ {len(self.documents)} documents chargés avec contenu enrichi{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur lors du chargement: {str(e)}{RESET}")
            return False

    def create_unified_ontology(self) -> str:
        """Fusionne tous les modules SNaP en un seul fichier TTL"""
        print(f"{BLUE}Création d'une ontologie unifiée à partir des modules SNaP...{RESET}")

        unified_path = self.results_dir

        processor = OntologyMerge(
            news_ontology_dir=self.news_ontology_dir,
            results_dir=unified_path
        )

        # 2. Appeler la méthode pour lancer la fusion
        return str(processor.create_unified_ontology())

        """
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

        # Sauvegarder l'ontologie unifiée
        unified_path = os.path.join(self.results_dir, "snap_unified.ttl")
        os.makedirs(self.results_dir, exist_ok=True)

        try:
            unified_graph.serialize(destination=unified_path, format="turtle")
            print(f"✓ Ontologie unifiée créée: {unified_path}")
            print(f"  - {loaded_modules} modules fusionnés")
            print(f"  - {len(unified_graph)} triplets RDF")

            return unified_path

        except Exception as e:
            print(f"{RED}Erreur lors de la sauvegarde: {e}{RESET}")
            return None
        """

    def setup_openai_env(self):
        """Configure la variable d'environnement OpenAI"""
        try:
            # Lire votre clé API
            with open(os.path.join(API_KEY_PATH, 'openAI_key.txt'), 'r') as f:
                api_key = f.read().strip()

            # Définir la variable d'environnement
            os.environ['OPENAI_API_KEY'] = api_key
            print(f"{GREEN}✓ Clé API OpenAI configurée pour RAGAS{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur configuration API: {e}{RESET}")
            return False

    async def initialize_systems(self):
        """Initialise les systèmes avec ontologie news unifiée"""
        print(f"{BLUE}{BOLD}Initialisation avec ontologie SNaP unifiée...{RESET}")

        # 1. Créer l'ontologie unifiée
        self.unified_ontology_path = self.create_unified_ontology()

        if not self.unified_ontology_path:
            print(f"{RED}Impossible de créer l'ontologie unifiée{RESET}")
            return False

        # 2. Initialiser OntoRAG avec l'ontologie unifiée
        storage_dir = "ragas_news_storage"
        if os.path.exists(storage_dir):
            import shutil
            shutil.rmtree(storage_dir)

        self.onto_rag = OntoRAG(
            storage_dir=storage_dir,
            api_key_path=API_KEY_PATH,
            model="gpt-4o",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            ontology_path=self.unified_ontology_path  # Passer le fichier unifié
        )

        await self.onto_rag.initialize()
        print(f"{GREEN}✓ OntoRAG avec ontologie news unifiée initialisé{RESET}")

        # 3. Initialiser baseline sans ontologie
        self.baseline_rag = OntoRAG(
            storage_dir=storage_dir + "_baseline",
            api_key_path=API_KEY_PATH,
            model="gpt-4o",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            ontology_path=None
        )
        await self.baseline_rag.initialize()
        print(f"{GREEN}✓ Baseline RAG initialisé{RESET}")

        return True

    async def generate_test_dataset(self):
        """Génère le dataset de test avec les vraies données"""
        print(f"{BLUE}Génération du dataset de test avec MultiHop-RAG...{RESET}")

        # Essayer d'utiliser les vraies questions d'abord
        if self._create_real_test_dataset():
            return True
        else:
            # Fallback sur les questions manuelles
            print(f"{RED} ERREUR generate dataset")
            return self._create_news_test_dataset()

        """Génère le dataset de test spécifique aux news"""
        #print(f"{BLUE}Génération du dataset de test news...{RESET}")

        # Utiliser les questions manuelles spécifiques aux news
        #return self._create_news_test_dataset()

    def _create_news_test_dataset(self):
        """Dataset de test spécifique aux news"""

        test_questions = [
            {
                "question": "Which artists thrived under Michelle Jubelirer at Capitol Music Group?",
                "ground_truth": "Ice Spice and The Beatles thrived under Michelle Jubelirer at Capitol Music Group."
            },
            {
                "question": "Who hit the Hawks' game-winning buzzer-beater vs. the Taipans?",
                "ground_truth": "Tyler Harvey hit the Hawks' game-winning buzzer-beater vs. the Taipans."
            },
            {
                "question": "What Chrome extension by Steven Tey aids in sharing AI replies?",
                "ground_truth": "ShareGPT is the Chrome extension by Steven Tey that aids in sharing AI replies."
            },
            {
                "question": "What are the characteristics of Nothing Ear Stick earbuds?",
                "ground_truth": "Nothing Ear Stick earbuds are stylish and see-through."
            },
            {
                "question": "How do price-match policies help retailers during big sales?",
                "ground_truth": "Price-match policies help retailers curb customer loss during big sales by ensuring customers get the best price without shopping around."
            },
            {
                "question": "Which NFL team won 26-14 against the Raiders?",
                "ground_truth": "The Detroit Lions won 26-14 against the Raiders in Monday Night Football."
            },
            {
                "question": "Who wrote about AI risks and the creative class?",
                "ground_truth": "Daniel Tencer wrote about AI risks and the creative class."
            },
            {
                "question": "What debuts with Hurricane Season on Netflix Nov. 1?",
                "ground_truth": "Multiple titles debut with Hurricane Season on Netflix Nov. 1, including Locked In, Mysteries of the Faith, and Wingwomen."
            },
            {
                "question": "What organization did Michelle Jubelirer work for?",
                "ground_truth": "Michelle Jubelirer worked for Capitol Music Group."
            },
            {
                "question": "What type of device are the Nothing Ear Stick?",
                "ground_truth": "Nothing Ear Stick are earbuds that are stylish and see-through."
            }
        ]

        # Convertir en format Dataset
        from datasets import Dataset
        self.test_dataset = Dataset.from_dict({
            "question": [q["question"] for q in test_questions],
            "ground_truth": [q["ground_truth"] for q in test_questions]
        })

        print(f"{GREEN}✓ {len(test_questions)} questions spécifiques aux news créées{RESET}")
        return True

    def analyze_ontology_structure(self):
        """Analyse la structure de l'ontologie unifiée créée"""
        if not self.unified_ontology_path:
            return

        print(f"\n{BLUE}Analyse de l'ontologie unifiée:{RESET}")

        try:
            graph = rdflib.Graph()
            graph.parse(self.unified_ontology_path, format="turtle")

            # Compter les classes
            classes = set()
            for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.OWL.Class)):
                classes.add(s)

            # Compter les propriétés
            properties = set()
            for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.OWL.ObjectProperty)):
                properties.add(s)
            for s, p, o in graph.triples((None, rdflib.RDF.type, rdflib.OWL.DatatypeProperty)):
                properties.add(s)

            print(f"  - {len(classes)} classes")
            print(f"  - {len(properties)} propriétés")
            print(f"  - {len(graph)} triplets RDF total")

            # Afficher quelques exemples de classes
            print(f"\n  Exemples de classes:")
            for i, cls in enumerate(list(classes)[:5]):
                label = self._get_label_from_graph(graph, cls)
                print(f"    - {label}")

        except Exception as e:
            print(f"⚠️ Erreur lors de l'analyse: {e}")

    def _get_label_from_graph(self, graph, uri):
        """Extrait le label d'une URI depuis le graphe"""
        for _, _, label in graph.triples((uri, rdflib.RDFS.label, None)):
            return f"{str(label)} ({str(uri).split('#')[-1]})"
        return str(uri).split('#')[-1] if '#' in str(uri) else str(uri).split('/')[-1]


async def main():
    """Fonction principale pour le benchmark news"""
    print(f"{BLUE}{BOLD}Benchmark RAGAS OntoRAG - Ontologies News SNaP{RESET}")

    # Demander le répertoire des ontologies news
    news_dir = input(f"Répertoire contenant les fichiers TTL de news (ex: ./news_ontologies/): ").strip()
    if not news_dir:
        news_dir = "./news_ontologies/"

    if not os.path.exists(news_dir):
        print(f"{RED}Répertoire non trouvé: {news_dir}{RESET}")
        return

    # Lister les fichiers trouvés
    ttl_files = [f for f in os.listdir(news_dir) if f.endswith('.ttl')]
    print(f"Fichiers TTL trouvés: {ttl_files}")

    max_samples = input(f"Nombre maximum de documents (défaut 6): ").strip()
    max_samples = int(max_samples) if max_samples.isdigit() else 6

    # Initialiser le benchmark spécialisé news
    benchmark = RAGASNewsOntoRAGBenchmark(
        news_ontology_dir=news_dir,
        max_samples=max_samples
    )

    # Analyser l'ontologie après création
    if await benchmark.initialize_systems():
        benchmark.analyze_ontology_structure()
    if not benchmark.setup_openai_env():
        print(f"{RED}Impossible de configurer l'API OpenAI{RESET}")
        return False

    results = await benchmark.run_full_benchmark()

    if results:
        print(f"\n{GREEN}{BOLD}✓ Benchmark news terminé avec succès !{RESET}")
        print(f"Consultez les rapports dans : {benchmark.results_dir}")
        print(f"Ontologie unifiée sauvegardée : {benchmark.unified_ontology_path}")
    else:
        print(f"\n{RED}❌ Échec du benchmark{RESET}")


if __name__ == "__main__":
    try:
        # Importer les constantes nécessaires
        from CONSTANT import API_KEY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, BLUE, BOLD, RESET, RED, GREEN, YELLOW
        from main_app import OntoRAG

        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Benchmark interrompu par l'utilisateur{RESET}")
    except Exception as e:
        print(f"\n{RED}Erreur: {str(e)}{RESET}")
        import traceback

        traceback.print_exc()
