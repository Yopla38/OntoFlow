"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# rag_chat_with_history.py
import asyncio
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Dict, Any

from CONSTANT import API_KEY_PATH, LLM_MODEL
from provider.get_key import get_openai_key
from provider.llm_providers import OpenAIProvider

# Importer tous les composants nécessaires
from utils.rag_engine import RAGEngine
from utils.wavelet_rag import WaveletRAG
from ontology.ontology_manager import OntologyManager
from ontology.classifier import OntologyClassifier
from utils.enhanced_document_processor import EnhancedDocumentProcessor
from utils.passage_visualizer import PassageVisualizer
from ontology.ontology_visualizer import OntologyVisualizer



# Configuration des couleurs pour le terminal
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ChatSession:
    """Gère une session de chat avec historique"""

    def __init__(self, max_messages=10):
        """
        Initialise une session de chat

        Args:
            max_messages: Nombre maximum de paires question/réponse à conserver
        """
        self.messages = []  # Format OpenAI: [{"role": "user/assistant/system", "content": "..."}]
        self.max_messages = max_messages

        # Ajouter un message système par défaut
        self.messages.append({
            "role": "system",
            "content": "Tu es un assistant IA qui répond aux questions en utilisant uniquement les informations fournies dans les documents. Garde une cohérence avec les échanges précédents."
        })

    def add_user_message(self, content):
        """Ajoute un message utilisateur à l'historique"""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content):
        """Ajoute un message assistant à l'historique"""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def get_messages(self):
        """Récupère tous les messages de la session"""
        return self.messages.copy()

    def get_context_messages(self):
        """Récupère les messages formatés pour le contexte"""
        # Ne pas inclure le message système dans le contexte
        context_messages = []
        for msg in self.messages[1:]:  # Ignorer le premier message système
            if msg["role"] in ["user", "assistant"]:
                context_messages.append(msg)
        return context_messages

    def get_last_n_exchanges(self, n=3):
        """Récupère les n derniers échanges (paires user/assistant)"""
        # Filtrer les messages système
        conversation = [msg for msg in self.messages if msg["role"] != "system"]
        # Prendre les 2*n derniers messages (n paires user/assistant)
        return conversation[-2 * n:] if len(conversation) >= 2 * n else conversation

    def clear(self):
        """Efface l'historique mais conserve le message système"""
        system_message = next((msg for msg in self.messages if msg["role"] == "system"), None)
        self.messages = [system_message] if system_message else []

    def _trim_history(self):
        """Limite la taille de l'historique au maximum défini"""
        # Compter les messages non-système
        non_system_msgs = [msg for msg in self.messages if msg["role"] != "system"]

        if len(non_system_msgs) > self.max_messages * 2:  # * 2 car on compte les paires
            # Conserver les messages système et les derniers échanges
            system_msgs = [msg for msg in self.messages if msg["role"] == "system"]
            last_messages = non_system_msgs[-self.max_messages * 2:]
            self.messages = system_msgs + last_messages

    def set_system_message(self, content):
        """Modifie ou ajoute le message système"""
        # Chercher un message système existant
        for i, msg in enumerate(self.messages):
            if msg["role"] == "system":
                self.messages[i]["content"] = content
                return

        # Si aucun message système n'existe, en ajouter un au début
        self.messages.insert(0, {"role": "system", "content": content})


class RAGChatSystem:
    """Système de chat RAG avec détection de concepts, sources détaillées, et historique"""

    def __init__(self):
        self.rag_engine = None
        self.wavelet_rag = None
        self.ontology_manager = None
        self.classifier = None
        self.initialized = False
        self.doc_name_map = {}  # Mappage ID du document -> nom lisible

        # Créer une session de chat par défaut
        self.current_session = ChatSession(max_messages=10)

        # Initialiser le visualiseur de passages
        self.passage_visualizer = PassageVisualizer(output_dir="visualizations")

        # Visualiseur d'ontologie (sera initialisé après le chargement de l'ontologie)
        self.ontology_visualizer = None

    async def initialize(self):
        """Initialise tous les composants du système RAG"""
        print(f"{BLUE}{BOLD}Initialisation du système RAG intelligent...{RESET}")

        try:
            # Initialiser les fournisseurs LLM
            OPENAI_KEY = get_openai_key(api_key_path=API_KEY_PATH)

            llm_provider = OpenAIProvider(
                model=LLM_MODEL,
                api_key=OPENAI_KEY
            )

            embedding_provider = llm_provider

            # Initialiser le RAG avec processeur de documents enrichi
            processor = EnhancedDocumentProcessor(chunk_size=1000, chunk_overlap=200)

            self.rag_engine = RAGEngine(
                llm_provider=llm_provider,
                embedding_provider=embedding_provider,
                storage_dir="storage_enhanced"
            )

            # Remplacer le processeur standard par notre processeur enrichi
            self.rag_engine.processor = processor

            await self.rag_engine.initialize()

            # Charger l'ontologie
            self.ontology_manager = OntologyManager(storage_dir="ontology_data")

            ontology_file = "emmo.jsonld"
            if os.path.exists(ontology_file):
                success = self.ontology_manager.load_ontology(ontology_file)
                if not success:
                    print(f"{RED}⚠️ Échec du chargement de l'ontologie, création d'une ontologie basique...{RESET}")
                    self._create_basic_ontology()
            else:
                print(f"{YELLOW}Fichier d'ontologie non trouvé, création d'une ontologie basique...{RESET}")
                self._create_basic_ontology()

            # Initialiser le classifieur ontologique
            self.classifier = OntologyClassifier(
                rag_engine=self.rag_engine,
                ontology_manager=self.ontology_manager,
                storage_dir="classifier_data",
                use_hierarchical=True,
                enable_concept_classification=True
            )

            await self.classifier.initialize()

            # Visualiseur d'ontologie
            try:
                self.ontology_visualizer = OntologyVisualizer(self.ontology_manager,
                                                              ontology_type="EMMO" if "emmo" in ontology_file.lower() else None)
                print(f"{GREEN}✓ Visualiseur d'ontologie initialisé{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠️ Visualiseur d'ontologie non disponible: {str(e)}{RESET}")

            # Initialiser le WaveletRAG
            self.wavelet_rag = WaveletRAG(
                rag_engine=self.rag_engine,
                wavelet="db3",
                levels=3,
                storage_dir="wavelet_storage"
            )

            await self.wavelet_rag.initialize()

            # Charger les documents existants et créer un mappage convivial
            await self._load_document_names()

            self.initialized = True
            print(f"{GREEN}{BOLD}✓ Système initialisé avec succès !{RESET}")
            print(f"Documents chargés: {len(self.doc_name_map)}")

            return True

        except Exception as e:
            print(f"{RED}Erreur lors de l'initialisation : {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return False

    def _create_basic_ontology(self):
        """Crée une ontologie basique si aucune n'est trouvée"""
        domain_ai = self.ontology_manager.create_domain("IntelligenceArtificielle", "Domaine de l'IA")
        domain_physics = self.ontology_manager.create_domain("Physique", "Domaine des sciences physiques")
        domain_math = self.ontology_manager.create_domain("Mathématiques", "Domaine des mathématiques")

        # Ajouter quelques concepts
        concept_ml = self.ontology_manager.add_concept("http://example.org/scientific-ontology#ApprentissageMachine",
                                                       "Apprentissage Machine",
                                                       "Domaine de l'IA qui permet aux machines d'apprendre")
        concept_dl = self.ontology_manager.add_concept("http://example.org/scientific-ontology#ApprentissageProfond",
                                                       "Deep Learning",
                                                       "Sous-domaine basé sur les réseaux neuronaux profonds")
        concept_physics = self.ontology_manager.add_concept("http://example.org/scientific-ontology#PhysiqueQuantique",
                                                            "Physique Quantique",
                                                            "Branche de la physique à l'échelle atomique")

        # Établir des relations hiérarchiques
        self.ontology_manager.set_concept_hierarchy(concept_dl.uri, concept_ml.uri)

        # Ajouter les concepts aux domaines
        self.ontology_manager.add_concept_to_domain(concept_ml.uri, domain_ai.name)
        self.ontology_manager.add_concept_to_domain(concept_dl.uri, domain_ai.name)
        self.ontology_manager.add_concept_to_domain(concept_physics.uri, domain_physics.name)

    async def _load_document_names(self):
        """Charge les noms conviviaux de tous les documents dans le système"""
        all_docs = await self.rag_engine.get_all_documents()
        doc_index = 1

        for doc_id, doc_info in all_docs.items():
            # Extraire un nom lisible pour le document
            if "original_filename" in doc_info:
                name = doc_info["original_filename"]
            elif "filename" in doc_info:
                name = doc_info["filename"]
            elif "path" in doc_info:
                name = os.path.basename(doc_info["path"])
            else:
                name = f"Document {doc_index}"

            # Stocker dans notre mappage
            self.doc_name_map[doc_id] = name
            doc_index += 1

    async def add_document(self, filepath):
        """Ajoute un nouveau document au système"""
        if not self.initialized:
            print(f"{RED}Le système n'est pas initialisé.{RESET}")
            return False

        if not os.path.exists(filepath):
            print(f"{RED}Le fichier {filepath} n'existe pas.{RESET}")
            return False

        try:
            print(f"{BLUE}Ajout du document {filepath}...{RESET}")

            # Ajouter le document
            doc_id = await self.rag_engine.add_document(filepath)

            # Mettre à jour le mappage des noms
            self.doc_name_map[doc_id] = os.path.basename(filepath)

            # Classifier le document
            print(f"{BLUE}Classification du document...{RESET}")
            await self.classifier.classify_document(doc_id, force_refresh=True)
            await self.classifier.classify_document_concepts(doc_id, force_refresh=True)

            print(f"{GREEN}✓ Document ajouté avec succès ! ID: {doc_id}{RESET}")
            return True

        except Exception as e:
            print(f"{RED}Erreur lors de l'ajout du document: {str(e)}{RESET}")
            return False

    async def _calculate_line_numbers(self, doc_id, passage_text):
        """
        Calcule les numéros de ligne pour un passage donné.

        Args:
            doc_id: ID du document
            passage_text: Texte du passage

        Returns:
            Tuple (start_line, end_line)
        """
        # Tenter de récupérer le document complet
        doc_chunks = await self.rag_engine.document_store.get_document_chunks(doc_id)
        if not doc_chunks:
            return ("?", "?")

        # Reconstruire le document complet
        full_text = ""
        for chunk in sorted(doc_chunks, key=lambda x: x["start_pos"]):
            full_text += chunk["text"]

        # Échantillon du début du passage pour la recherche (en évitant les problèmes de correspondance exacte)
        sample_size = min(100, len(passage_text))
        passage_sample = passage_text[:sample_size]

        # Rechercher l'échantillon dans le texte complet
        passage_start = full_text.find(passage_sample)
        if passage_start == -1:
            return ("?", "?")

        # Compter les sauts de ligne jusqu'à cette position
        start_line = full_text[:passage_start].count('\n') + 1

        # Calculer la ligne de fin
        passage_end = passage_start + len(passage_text)
        end_line = full_text[:passage_end].count('\n') + 1

        return (start_line, end_line)

    async def process_query(self, query):
        """Traite une requête avec détection automatique de concepts, sources détaillées et gestion de l'historique"""
        if not self.initialized:
            return {"answer": "Le système n'est pas initialisé.", "error": True}

        try:
            # Ajouter la question à l'historique de conversation
            self.current_session.add_user_message(query)

            print(f"{BLUE}Recherche de concepts pertinents...{RESET}")

            # 1. Récupérer les passages pertinents avec détection de concepts
            result = await self.classifier.auto_concept_search(
                query=query,
                include_semantic_relations=True
            )

            # 2. Si aucun concept n'est détecté, faire un fallback vers WaveletRAG
            if "error" in result:
                print(f"{YELLOW}Aucun concept pertinent trouvé, utilisation du WaveletRAG...{RESET}")
                result = await self.wavelet_rag.chat(query, top_k=5)

            # 3. Récupérer les passages pertinents
            passages = result.get("passages", [])

            # 4. Préparer le prompt enrichi avec le contexte des passages et l'historique
            context = self._build_context_from_passages(passages)

            # 5. Utiliser directement le provider LLM avec l'historique des messages
            # Créer une copie des messages de la session
            messages = self.current_session.get_messages().copy()

            # Ajouter le contexte des documents à la dernière question
            last_user_msg_index = next((i for i in range(len(messages) - 1, -1, -1)
                                        if messages[i]["role"] == "user"), None)

            if last_user_msg_index is not None:
                messages[last_user_msg_index]["content"] = f"{context}\n\nQuestion: {query}"

            # Visualiser les passages pertinents (PDFs uniquement)
            visualizations = {}
            if "passages" in result:
                pdf_passages = [p for p in result["passages"] if
                                p.get("metadata", {}).get("filepath", "").lower().endswith('.pdf')]
                if pdf_passages:
                    visualizations = self.passage_visualizer.visualize_passages(pdf_passages)

            # Ajouter les visualisations au résultat
            result["visualizations"] = visualizations

            # Générer la réponse avec l'historique complet
            response = await self.rag_engine.llm_provider.generate_response(messages)

            # Ajouter la réponse à l'historique
            self.current_session.add_assistant_message(response)

            # Extraire les sources avec les numéros de lignes
            sources = []
            for i, passage in enumerate(passages):
                doc_id = passage.get("document_id", "")
                doc_name = self.doc_name_map.get(doc_id, "Document inconnu")

                # Calculer les numéros de ligne
                start_line, end_line = await self._calculate_line_numbers(doc_id, passage["text"])

                # Extraire les informations de ligne
                metadata = passage.get("metadata", {})
                #start_line = metadata.get("start_line", "?")
                #end_line = metadata.get("end_line", "?")
                section_title = metadata.get("section_title", "")

                # Formater la source
                if section_title:
                    source_info = f"{doc_name} (lignes {start_line}-{end_line}, section: {section_title})"
                else:
                    source_info = f"{doc_name} (lignes {start_line}-{end_line})"

                sources.append({
                    "index": i + 1,
                    "info": source_info,
                    "text": passage["text"][:150] + "..." if len(passage["text"]) > 150 else passage["text"],
                    "similarity": passage.get("similarity", 0)
                })

            # Extraire les concepts détectés
            concepts = []
            if "concepts_detected" in result:
                concepts = result["concepts_detected"]

            self.last_detected_concepts = concepts if concepts else []
            self.last_visualizations = visualizations

            return {
                "answer": response,
                "sources": sources,
                "concepts": concepts,
                "visualizations": visualizations,
                "error": False
            }

        except Exception as e:
            print(f"{RED}Erreur lors du traitement de la requête: {str(e)}{RESET}")
            import traceback
            traceback.print_exc()
            return {"answer": f"Erreur: {str(e)}", "error": True}

    def _build_context_from_passages(self, passages):
        """Construit le contexte à partir des passages pertinents"""
        context = "Contexte fourni :\n\n"

        for i, passage in enumerate(passages, 1):
            doc_id = passage.get("document_id", "")
            doc_name = self.doc_name_map.get(doc_id, "Document inconnu")

            # Extraire les métadonnées utiles
            metadata = passage.get("metadata", {})
            section_title = metadata.get("section_title", "")

            # Ajouter des informations sur la section si disponible
            section_info = f", section: {section_title}" if section_title else ""

            context += f"[Passage {i} de {doc_name}{section_info}]\n{passage['text']}\n\n"

        return context

    def new_session(self):
        """Crée une nouvelle session de chat"""
        self.current_session = ChatSession()
        return True

    def get_document_list(self):
        """Retourne la liste des documents chargés"""
        return [f"{doc_id}: {name}" for doc_id, name in self.doc_name_map.items()]

    def get_conversation_history(self, max_exchanges=3):
        """Récupère l'historique des échanges récents"""
        # Filtrer pour affichage (sans messages systèmes)
        exchanges = []
        conversation = self.current_session.get_messages()

        for msg in conversation:
            if msg["role"] in ["user", "assistant"]:
                exchanges.append(msg)

        # Retourner les derniers échanges
        return exchanges[-max_exchanges * 2:] if len(exchanges) > max_exchanges * 2 else exchanges


async def run_chat():
    """Fonction principale pour exécuter le chat interactif"""
    # Créer et initialiser le système
    chat_system = RAGChatSystem()
    if not await chat_system.initialize():
        print(f"{RED}Initialisation échouée. Arrêt du programme.{RESET}")
        return

    # Vérifier s'il y a des documents
    if not chat_system.doc_name_map:
        print(f"{YELLOW}Aucun document trouvé dans le système.{RESET}")
        add_sample = input("Voulez-vous ajouter des documents d'exemple? (o/n): ").lower()

        if add_sample == 'o':
            # Vérifier les documents PDF dans le dossier documents/
            pdf_files = []
            for file in os.listdir("documents"):
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join("documents", file))

            if pdf_files:
                for pdf in pdf_files:
                    await chat_system.add_document(pdf)
            else:
                print(f"{YELLOW}Aucun document PDF trouvé dans le dossier 'documents/'.{RESET}")
                filepath = input("Entrez le chemin d'un document à ajouter: ")
                if os.path.exists(filepath):
                    await chat_system.add_document(filepath)
                else:
                    print(f"{RED}Chemin invalide.{RESET}")

    # Afficher des informations sur le système
    print(f"\n{GREEN}{BOLD}=== Chat RAG avec conservation de l'historique ==={RESET}")
    print(f"{GREEN}Documents disponibles:{RESET}")
    for i, doc in enumerate(chat_system.doc_name_map.values(), 1):
        print(f"  {i}. {doc}")

    print(f"\n{BOLD}Commandes disponibles:{RESET}")
    print("  !quit, !exit - Quitter le chat")
    print("  !add [chemin] - Ajouter un document")
    print("  !docs - Lister les documents")
    print("  !history - Afficher l'historique de la conversation")
    print("  !clear - Effacer l'historique de la conversation")
    print("  !concepts - Afficher les concepts de l'ontologie")
    print("  !visualize_concepts - Visualiser les concepts détectés dans la dernière recherche")
    print("  !view [numéro] - Ouvrir la visualisation d'une source")
    print("  !ontology - Commandes de visualisation de l'ontologie")
    print("  !help - Afficher l'aide")

    # Boucle principale du chat
    while True:
        print("\n" + "-" * 80)
        query = input(f"{BOLD}Vous:{RESET} ")

        # Traiter les commandes
        if query.lower() in ["!quit", "!exit", "!q"]:
            print(f"{GREEN}Au revoir !{RESET}")
            break

        elif query.lower() == "!help":
            print(f"\n{BOLD}Commandes disponibles:{RESET}")
            print("  !quit, !exit - Quitter le chat")
            print("  !add [chemin] - Ajouter un document")
            print("  !docs - Lister les documents")
            print("  !history - Afficher l'historique de la conversation")
            print("  !clear - Effacer l'historique de la conversation")
            print("  !concepts - Afficher les concepts de l'ontologie")
            print("  !visualize_concepts - Visualiser les concepts détectés dans la dernière recherche")
            print("  !view [numéro] - Ouvrir la visualisation d'une source")
            print("  !ontology - Commandes de visualisation de l'ontologie")
            print("  !help - Afficher l'aide")
            continue

        elif query.lower() == "!docs":
            print(f"\n{GREEN}Documents disponibles:{RESET}")
            for i, doc in enumerate(chat_system.doc_name_map.values(), 1):
                print(f"  {i}. {doc}")
            continue

        elif query.lower().startswith("!add "):
            filepath = query[5:].strip()
            if os.path.exists(filepath):
                await chat_system.add_document(filepath)
            else:
                print(f"{RED}Le fichier {filepath} n'existe pas.{RESET}")
            continue

        elif query.lower() == "!history":
            print(f"\n{YELLOW}Historique de la conversation:{RESET}")
            history = chat_system.get_conversation_history()
            if not history:
                print("  Aucun échange dans l'historique.")
            else:
                for i, msg in enumerate(history):
                    role_color = BOLD if msg["role"] == "user" else GREEN
                    role_display = "Vous" if msg["role"] == "user" else "Assistant"
                    print(f"\n{role_color}{role_display}:{RESET} {msg['content']}")
            continue

        elif query.lower() == "!clear":
            chat_system.current_session.clear()
            print(f"{GREEN}Historique de conversation effacé.{RESET}")
            continue

        elif query.lower() == "!concepts":
            print(f"\n{GREEN}Concepts disponibles dans l'ontologie:{RESET}")
            count = 0
            for uri, concept in chat_system.ontology_manager.concepts.items():
                count += 1
                label = concept.label or uri.split('#')[-1]
                print(f"  - {label}")
                if count >= 20:  # Limiter l'affichage
                    print("  ... et plus")
                    break
            continue

        elif query.lower().startswith("!view "):
            # Format attendu: !view 1 (pour voir l'image de la source 1)
            try:
                source_num = int(query[6:].strip())
                if hasattr(chat_system, 'last_visualizations') and source_num in chat_system.last_visualizations:
                    image_path = chat_system.last_visualizations[source_num]
                    # Ouvrir l'image avec l'application par défaut du système
                    if os.path.exists(image_path):
                        if sys.platform == 'win32':
                            os.startfile(image_path)
                        elif sys.platform == 'darwin':  # macOS
                            subprocess.run(['open', image_path])
                        else:  # Linux
                            subprocess.run(['xdg-open', image_path])
                        print(f"{GREEN}Visualisation ouverte: {image_path}{RESET}")
                    else:
                        print(f"{RED}Fichier non trouvé: {image_path}{RESET}")
                else:
                    print(f"{RED}Visualisation {source_num} non disponible{RESET}")
            except ValueError:
                print(f"{RED}Format incorrect. Utilisez '!view <numéro>'{RESET}")
            continue

        elif query.lower() == "!visualize_concepts":
            # Visualiser les concepts de la dernière recherche
            if not hasattr(chat_system, 'last_detected_concepts') or not chat_system.last_detected_concepts:
                print(f"{YELLOW}Aucun concept détecté dans la dernière recherche.{RESET}")
                continue

            print(f"{BLUE}Génération de la visualisation des concepts détectés...{RESET}")

            # Collecter les URIs des concepts détectés
            concept_uris = []
            for concept in chat_system.last_detected_concepts:
                concept_uri = concept.get('concept_uri') or concept.get('uri')
                if concept_uri:
                    concept_uris.append(concept_uri)

            if not concept_uris:
                print(f"{YELLOW}Aucune URI de concept trouvée.{RESET}")
                continue

            # Construire et visualiser le graphe des concepts détectés
            output_file = "detected_concepts.html"
            chat_system.ontology_visualizer.build_graph(focus_uri=concept_uris[0], max_depth=2) #max_concepts=50
            chat_system.ontology_visualizer.visualize_with_pyvis(output_file=output_file)
            print(f"{GREEN}✓ Visualisation des concepts détectés enregistrée dans: {output_file}{RESET}")
            continue

        elif query.lower() == "!ontology" or query.lower().startswith("!ontology "):
            # Commandes de visualisation d'ontologie
            if chat_system.ontology_visualizer is None:
                print(f"{RED}Visualiseur d'ontologie non disponible{RESET}")
                continue

            # Sous-commandes
            if query.lower() == "!ontology":
                # Afficher l'aide pour les commandes d'ontologie
                print(f"\n{BOLD}Commandes de visualisation d'ontologie:{RESET}")
                print("  !ontology visualize - Visualiser l'ontologie complète")
                print("  !ontology interactive - Créer une visualisation interactive de l'ontologie")
                print("  !ontology concept <uri> - Visualiser un concept spécifique et ses relations")
                print("  !ontology report - Générer un rapport HTML de l'ontologie")

            elif query.lower() == "!ontology visualize":
                # Visualiser l'ontologie avec matplotlib
                print(f"{BLUE}Génération de la visualisation d'ontologie...{RESET}")
                output_file = "ontology_visualization.png"
                chat_system.ontology_visualizer.build_graph() #max_concepts=50
                chat_system.ontology_visualizer.visualize_with_matplotlib(output_file=output_file)
                print(f"{GREEN}✓ Visualisation enregistrée dans: {output_file}{RESET}")

            elif query.lower() == "!ontology interactive":
                # Visualiser l'ontologie de manière interactive
                print(f"{BLUE}Génération de la visualisation interactive d'ontologie...{RESET}")
                output_file = "ontology_interactive.html"
                chat_system.ontology_visualizer.build_graph() #max_concepts=100
                chat_system.ontology_visualizer.visualize_with_pyvis(output_file=output_file)
                print(f"{GREEN}✓ Visualisation interactive enregistrée dans: {output_file}{RESET}")

            elif query.lower().startswith("!ontology concept "):
                # Visualiser un concept spécifique
                concept_uri = query[len("!ontology concept "):].strip()

                # Vérifier si c'est une URI valide
                if concept_uri not in chat_system.ontology_manager.concepts:
                    # Essayer de trouver le concept par son label
                    found = False
                    for uri, concept in chat_system.ontology_manager.concepts.items():
                        if concept.label and concept.label.lower() == concept_uri.lower():
                            concept_uri = uri
                            found = True
                            break

                    if not found:
                        print(f"{RED}Concept non trouvé: {concept_uri}{RESET}")
                        continue

                print(f"{BLUE}Génération de la visualisation du concept {concept_uri}...{RESET}")
                output_file = f"concept_{concept_uri.split('#')[-1].split('/')[-1]}.html"

                # Construire un graphe centré sur ce concept
                chat_system.ontology_visualizer.build_graph(max_concepts=30, focus_uri=concept_uri, max_depth=2)
                chat_system.ontology_visualizer.visualize_with_pyvis(output_file=output_file, hierarchical=True)
                print(f"{GREEN}✓ Visualisation du concept enregistrée dans: {output_file}{RESET}")

            elif query.lower() == "!ontology report":
                # Générer un rapport HTML
                print(f"{BLUE}Génération du rapport d'ontologie...{RESET}")
                output_file = "ontology_report.html"
                chat_system.ontology_visualizer.build_graph()
                chat_system.ontology_visualizer.generate_ontology_report(output_file=output_file)
                print(f"{GREEN}✓ Rapport d'ontologie enregistré dans: {output_file}{RESET}")

            else:
                print(f"{RED}Commande d'ontologie non reconnue. Utilisez !ontology pour voir les options.{RESET}")

            continue

        # Traiter la requête normale
        start_time = datetime.now()
        print(f"{BLUE}Recherche en cours...{RESET}")

        result = await chat_system.process_query(query)

        # Afficher le temps de réponse
        elapsed = (datetime.now() - start_time).total_seconds()

        # Afficher les concepts détectés
        if result.get("concepts") and not result.get("error"):
            print(f"\n{YELLOW}Concepts détectés:{RESET}")
            for concept in result["concepts"][:3]:  # Limiter à 3 concepts
                print(f"  - {concept['label']} ({concept['confidence']:.2f})")

        # Afficher la réponse
        print(f"\n{BOLD}Assistant:{RESET} {result['answer']}")

        # Afficher les sources
        if result.get("sources") and not result.get("error"):
            print(f"\n{YELLOW}Sources:{RESET}")
            for source in result["sources"]:
                print(f"  [{source['index']}] {source['info']}")

        # Afficher les visualisations si disponibles
        if result.get("visualizations") and not result.get("error"):
            print(f"\n{YELLOW}Visualisations:{RESET}")
            for index, image_path in result["visualizations"].items():
                # image_path est le chemin complet, mais nous n'affichons que le nom relatif
                print(f"  [{index}] Capture d'écran enregistrée: {os.path.basename(image_path)}")

        print(f"\n{BLUE}Temps de réponse: {elapsed:.2f} secondes{RESET}")


if __name__ == "__main__":
    try:
        asyncio.run(run_chat())
    except KeyboardInterrupt:
        print(f"\n{GREEN}Chat terminé par l'utilisateur.{RESET}")
    except Exception as e:
        print(f"\n{RED}Erreur: {str(e)}{RESET}")
        import traceback

        traceback.print_exc()
