# rag_routes.py
import os
import json
import time
import threading
import traceback
from flask import request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename

# Import des modules existants pour intégration
from app import app, socketio, user_configs, with_config_lock

# Importez les composants RAG
from utils.enhanced_document_processor import EnhancedDocumentProcessor
from utils.rag_engine import RAGEngine
from utils.wavelet_rag import WaveletRAG
from ontology.ontology_manager import OntologyManager
from ontology.classifier import OntologyClassifier
from provider.get_key import get_openai_key
from provider.llm_providers import OpenAIProvider


# Fonction d'initialisation du système RAG pour un utilisateur
def initialize_rag_system(user_id):
    """Initialise le système RAG pour un utilisateur."""
    if user_id not in user_configs:
        return {"success": False, "message": "Utilisateur non trouvé"}

    try:
        # Créer le répertoire de stockage RAG
        rag_storage_dir = os.path.join(user_configs[user_id]['paths']['user_path'], "rag_storage")
        os.makedirs(rag_storage_dir, exist_ok=True)

        # Initialiser les providers LLM
        api_key = get_openai_key()
        llm_provider = OpenAIProvider(
            model="gpt-4o",
            api_key=api_key
        )
        embedding_provider = llm_provider

        # Initialiser le processeur de documents avec métadonnées enrichies
        processor = EnhancedDocumentProcessor(chunk_size=1000, chunk_overlap=200)

        # Initialiser le moteur RAG
        rag_engine = RAGEngine(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            storage_dir=rag_storage_dir
        )

        # Remplacer le processeur standard par notre processeur enrichi
        rag_engine.processor = processor

        # Initialiser l'OntologyManager
        ontology_dir = os.path.join(rag_storage_dir, "ontology")
        os.makedirs(ontology_dir, exist_ok=True)
        ontology_manager = OntologyManager(storage_dir=ontology_dir)

        # Charger l'ontologie (si disponible, sinon créer une basique)
        ontology_file = os.path.join(ontology_dir, "simple.jsonld")
        if os.path.exists(ontology_file):
            success = ontology_manager.load_ontology(ontology_file)
            if not success:
                create_basic_ontology(ontology_manager)
        else:
            create_basic_ontology(ontology_manager)

        # Initialiser le classifieur ontologique
        classifier_dir = os.path.join(rag_storage_dir, "classifier")
        os.makedirs(classifier_dir, exist_ok=True)

        classifier = OntologyClassifier(
            rag_engine=rag_engine,
            ontology_manager=ontology_manager,
            storage_dir=classifier_dir,
            use_hierarchical=True,
            enable_concept_classification=True
        )

        # Initialiser WaveletRAG
        wavelet_dir = os.path.join(rag_storage_dir, "wavelet")
        os.makedirs(wavelet_dir, exist_ok=True)

        wavelet_rag = WaveletRAG(
            rag_engine=rag_engine,
            wavelet="db3",
            levels=5,
            storage_dir=wavelet_dir
        )

        # Stocker les instances dans la configuration utilisateur
        user_configs[user_id]['rag_engine'] = rag_engine
        user_configs[user_id]['ontology_manager'] = ontology_manager
        user_configs[user_id]['rag_classifier'] = classifier
        user_configs[user_id]['wavelet_rag'] = wavelet_rag

        # Initialisation asynchrone des composants
        threading.Thread(target=async_initialize_rag, args=(user_id,)).start()

        return {"success": True, "message": "Système RAG initialisé"}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "message": f"Erreur d'initialisation: {str(e)}"}


def create_basic_ontology(ontology_manager):
    """Crée une ontologie basique si aucune n'est trouvée."""
    # Créer quelques domaines de base
    ontology_manager.create_domain("IntelligenceArtificielle", "Domaine de l'IA et du ML")
    ontology_manager.create_domain("Mathématiques", "Concepts mathématiques")
    ontology_manager.create_domain("Physique", "Sciences physiques et principes fondamentaux")
    ontology_manager.create_domain("Informatique", "Informatique et sciences de l'information")

    # Ajouter quelques concepts
    ml_concept = ontology_manager.add_concept(
        "http://example.org/ontology#MachineLearning",
        "Apprentissage Machine",
        "Branche de l'IA qui permet aux machines d'apprendre à partir de données"
    )
    dl_concept = ontology_manager.add_concept(
        "http://example.org/ontology#DeepLearning",
        "Apprentissage Profond",
        "Sous-domaine de l'apprentissage machine basé sur des réseaux de neurones profonds"
    )

    # Établir des relations
    ontology_manager.set_concept_hierarchy(dl_concept.uri, ml_concept.uri)

    # Associer aux domaines
    ontology_manager.add_concept_to_domain(ml_concept.uri, "IntelligenceArtificielle")
    ontology_manager.add_concept_to_domain(dl_concept.uri, "IntelligenceArtificielle")


def async_initialize_rag(user_id):
    """Initialise les composants RAG de manière asynchrone."""
    try:
        # Obtenir les instances
        rag_engine = user_configs[user_id]['rag_engine']
        classifier = user_configs[user_id]['rag_classifier']
        wavelet_rag = user_configs[user_id]['wavelet_rag']

        # Initialisation asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Initialiser le RAG
        loop.run_until_complete(rag_engine.initialize())
        socketio.emit('rag_status', {"message": "Moteur RAG initialisé"}, room=user_id)

        # Initialiser le classifieur
        loop.run_until_complete(classifier.initialize())
        socketio.emit('rag_status', {"message": "Classifieur ontologique initialisé"}, room=user_id)

        # Initialiser WaveletRAG
        loop.run_until_complete(wavelet_rag.initialize())
        socketio.emit('rag_status', {"message": "WaveletRAG initialisé"}, room=user_id)

        # Notification de terminaison
        socketio.emit('rag_status', {
            "message": "Initialisation terminée",
            "status": "complete"
        }, room=user_id)

        # Mettre à jour le statut dans la configuration
        user_configs[user_id]['rag_initialized'] = True

        loop.close()
    except Exception as e:
        traceback.print_exc()
        socketio.emit('rag_status', {
            "message": f"Erreur d'initialisation: {str(e)}",
            "status": "error"
        }, room=user_id)


# Routes pour l'API REST

@app.route('/api/rag/initialize', methods=['POST'])
def api_initialize_rag():
    """Route pour initialiser le système RAG."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est déjà initialisé
    if user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": True, "message": "Système RAG déjà initialisé"}), 200

    # Initialiser le système RAG
    result = initialize_rag_system(user_id)
    if result["success"]:
        return jsonify(result), 200
    else:
        return jsonify(result), 500


@app.route('/api/rag/documents', methods=['GET'])
def api_list_rag_documents():
    """Liste tous les documents indexés dans le système RAG."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Récupérer les documents
        rag_engine = user_configs[user_id]['rag_engine']
        documents = loop.run_until_complete(rag_engine.get_all_documents())

        # Formater les documents pour l'affichage
        formatted_docs = []
        for doc_id, doc_info in documents.items():
            formatted_docs.append({
                "id": doc_id,
                "title": doc_info.get("original_filename", "Document sans titre"),
                "path": doc_info.get("path", ""),
                "chunks_count": doc_info.get("chunks_count", 0),
                "date_added": doc_info.get("date_added", "")
            })

        loop.close()
        return jsonify({
            "success": True,
            "documents": formatted_docs
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500


@app.route('/api/rag/documents', methods=['POST'])
def api_add_rag_document():
    """Ajoute un document au système RAG."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    # Vérifier si un fichier est fourni
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Aucun fichier fourni"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Nom de fichier vide"}), 400

    try:
        # Sauvegarder le fichier temporairement
        filename = secure_filename(file.filename)
        temp_path = os.path.join(user_configs[user_id]['paths']['temp_path'], filename)
        file.save(temp_path)

        # Lancer l'ajout de document en arrière-plan
        threading.Thread(target=async_add_document, args=(user_id, temp_path)).start()

        return jsonify({
            "success": True,
            "message": "Document en cours d'ajout. Vous recevrez une notification quand ce sera terminé."
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500


def async_add_document(user_id, file_path):
    """Ajoute un document en arrière-plan et envoie des notifications."""
    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Notifier le début
        socketio.emit('rag_status', {
            "message": f"Ajout du document {os.path.basename(file_path)}...",
            "status": "processing"
        }, room=user_id)

        # Ajouter le document
        rag_engine = user_configs[user_id]['rag_engine']
        doc_id = loop.run_until_complete(rag_engine.add_document(file_path))

        # Classifier le document
        classifier = user_configs[user_id]['rag_classifier']
        domain_result = loop.run_until_complete(classifier.classify_document(doc_id, force_refresh=True))
        concept_result = loop.run_until_complete(classifier.classify_document_concepts(doc_id, force_refresh=True))

        # Notifier la fin
        socketio.emit('rag_status', {
            "message": f"Document ajouté avec succès! ID: {doc_id}",
            "status": "complete",
            "document_id": doc_id,
            "domains": domain_result.get("domains", []),
            "concepts": concept_result.get("concepts", [])
        }, room=user_id)

        # Également émettre un événement pour mettre à jour la liste des documents
        socketio.emit('rag_documents_updated', room=user_id)

        loop.close()

    except Exception as e:
        traceback.print_exc()
        socketio.emit('rag_status', {
            "message": f"Erreur lors de l'ajout du document: {str(e)}",
            "status": "error"
        }, room=user_id)

    finally:
        # Nettoyer le fichier temporaire
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass


@app.route('/api/rag/documents/<document_id>', methods=['DELETE'])
def api_remove_rag_document(document_id):
    """Supprime un document du système RAG."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Supprimer le document
        rag_engine = user_configs[user_id]['rag_engine']
        success = loop.run_until_complete(rag_engine.remove_document(document_id))

        loop.close()

        if success:
            # Émettre un événement pour mettre à jour la liste des documents
            socketio.emit('rag_documents_updated', room=user_id)

            return jsonify({
                "success": True,
                "message": f"Document {document_id} supprimé avec succès"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": f"Document {document_id} non trouvé"
            }), 404

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500


@app.route('/api/rag/documents/<document_id>', methods=['GET'])
def api_get_rag_document(document_id):
    """Récupère les informations détaillées d'un document."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Récupérer le document
        rag_engine = user_configs[user_id]['rag_engine']
        document = loop.run_until_complete(rag_engine.document_store.get_document(document_id))

        if not document:
            loop.close()
            return jsonify({
                "success": False,
                "message": f"Document {document_id} non trouvé"
            }), 404

        # Récupérer les chunks
        chunks = loop.run_until_complete(rag_engine.document_store.get_document_chunks(document_id))

        # Récupérer les classifications
        classifier = user_configs[user_id]['rag_classifier']
        domains = loop.run_until_complete(classifier.classify_document(document_id))
        concepts = loop.run_until_complete(classifier.classify_document_concepts(document_id))

        loop.close()

        return jsonify({
            "success": True,
            "document": {
                "id": document_id,
                "filename": document.get("original_filename", "Document sans titre"),
                "path": document.get("path", ""),
                "chunks_count": len(chunks) if chunks else 0,
                "chunks": [{
                    "id": chunk["id"],
                    "text": chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"],
                    "metadata": chunk.get("metadata", {})
                } for chunk in chunks[:5]],  # Limiter à 5 chunks pour l'affichage
                "domains": domains.get("domains", []),
                "concepts": concepts.get("concepts", [])
            }
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500


@app.route('/api/rag/ask', methods=['POST'])
def api_ask_rag():
    """Pose une question au système RAG avec détection de concepts."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"success": False, "message": "Question requise"}), 400

    question = data['question']
    use_concepts = data.get('use_concepts', True)

    # Lancer la recherche en arrière-plan et utiliser Socket.IO pour le streaming
    threading.Thread(target=async_ask_question, args=(user_id, question, use_concepts)).start()

    return jsonify({
        "success": True,
        "message": "Question en cours de traitement. Les résultats seront envoyés via Socket.IO."
    }), 200


def async_ask_question(user_id, question, use_concepts):
    """Traite une question en arrière-plan et envoie les résultats via Socket.IO."""
    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Obtenir les instances
        classifier = user_configs[user_id]['rag_classifier']
        wavelet_rag = user_configs[user_id]['wavelet_rag']

        # Notifier le début
        socketio.emit('rag_question_processing', {
            "status": "started",
            "question": question
        }, room=user_id)

        # Traiter la question
        if use_concepts:
            # Utiliser la détection automatique de concepts
            socketio.emit('rag_question_processing', {
                "status": "detecting_concepts",
                "message": "Détection des concepts pertinents..."
            }, room=user_id)

            result = loop.run_until_complete(classifier.auto_concept_search(
                query=question,
                include_semantic_relations=True
            ))

            # Vérifier les concepts détectés
            if "concepts_detected" in result and result["concepts_detected"]:
                concepts = result["concepts_detected"]
                socketio.emit('rag_detected_concepts', {
                    "concepts": concepts
                }, room=user_id)
        else:
            # Utiliser WaveletRAG sans détection de concepts
            socketio.emit('rag_question_processing', {
                "status": "searching",
                "message": "Recherche des passages pertinents..."
            }, room=user_id)

            result = loop.run_until_complete(wavelet_rag.chat(question, top_k=5))

        # Si on a une réponse
        if "answer" in result:
            # Émettre les passages récupérés
            if "passages" in result:
                socketio.emit('rag_passages', {
                    "passages": [{
                        "id": p.get("chunk_id", ""),
                        "text": p.get("text", ""),
                        "document_name": p.get("document_name", "Document inconnu"),
                        "similarity": p.get("similarity", 0),
                        "metadata": p.get("metadata", {})
                    } for p in result["passages"]]
                }, room=user_id)

            # Émettre la réponse complète
            socketio.emit('rag_answer', {
                "answer": result["answer"],
                "status": "complete"
            }, room=user_id)
        else:
            socketio.emit('rag_answer', {
                "status": "error",
                "message": result.get("error", "Erreur inconnue lors du traitement de la question")
            }, room=user_id)

        loop.close()

    except Exception as e:
        traceback.print_exc()
        socketio.emit('rag_answer', {
            "status": "error",
            "message": f"Erreur: {str(e)}"
        }, room=user_id)


@app.route('/api/rag/concepts', methods=['GET'])
def api_list_rag_concepts():
    """Liste tous les concepts disponibles dans l'ontologie."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    try:
        # Récupérer l'ontologie
        ontology_manager = user_configs[user_id]['ontology_manager']

        # Convertir les concepts en format JSON
        concepts_list = []
        for uri, concept in ontology_manager.concepts.items():
            concepts_list.append({
                "uri": uri,
                "label": concept.label,
                "description": concept.description if hasattr(concept, 'description') else None,
                "parent_count": len(concept.parents) if hasattr(concept, 'parents') else 0,
                "children_count": len(concept.children) if hasattr(concept, 'children') else 0
            })

        return jsonify({
            "success": True,
            "concepts": concepts_list
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500


@app.route('/api/rag/concepts/<concept_uri>', methods=['GET'])
def api_get_rag_concept(concept_uri):
    """Récupère les informations détaillées d'un concept."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    try:
        # Récupérer l'ontologie
        ontology_manager = user_configs[user_id]['ontology_manager']

        # Résoudre l'URI préfixée si nécessaire
        full_uri = ontology_manager.resolve_uri(concept_uri)

        # Vérifier que le concept existe
        if full_uri not in ontology_manager.concepts:
            return jsonify({
                "success": False,
                "message": f"Concept {concept_uri} non trouvé"
            }), 404

        # Récupérer le concept
        concept = ontology_manager.concepts[full_uri]

        # Formater les données
        concept_data = {
            "uri": full_uri,
            "label": concept.label,
            "description": concept.description if hasattr(concept, 'description') else None,
            "parents": [{
                "uri": parent.uri,
                "label": parent.label
            } for parent in concept.parents] if hasattr(concept, 'parents') else [],
            "children": [{
                "uri": child.uri,
                "label": child.label
            } for child in concept.children] if hasattr(concept, 'children') else []
        }

        # Trouver les documents associés à ce concept
        classifier = user_configs[user_id]['rag_classifier']
        rag_engine = user_configs[user_id]['rag_engine']

        # Exécuter de manière asynchrone pour la recherche des documents
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Récupérer tous les documents
        all_documents = loop.run_until_complete(rag_engine.get_all_documents())

        # Trouver les documents liés à ce concept
        related_documents = []
        for doc_id in all_documents:
            # Classifier le document par concepts
            result = loop.run_until_complete(classifier.classify_document_concepts(doc_id))

            # Vérifier si le concept est présent
            if "concepts" in result:
                for c in result["concepts"]:
                    # Fonction récursive pour chercher le concept dans la hiérarchie
                    def check_concept(concept_obj):
                        if concept_obj.get("concept_uri") == full_uri:
                            return True
                        for sub in concept_obj.get("sub_concepts", []):
                            if check_concept(sub):
                                return True
                        return False

                    # Si le concept est trouvé, ajouter le document
                    if check_concept(c):
                        doc_info = all_documents[doc_id]
                        related_documents.append({
                            "id": doc_id,
                            "title": doc_info.get("original_filename", "Document sans titre"),
                            "confidence": c.get("confidence", 0)
                        })
                        break

        loop.close()

        # Ajouter les documents liés
        concept_data["related_documents"] = related_documents

        return jsonify({
            "success": True,
            "concept": concept_data
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500


@app.route('/api/rag/search', methods=['POST'])
def api_search_rag():
    """Effectue une recherche sémantique dans les documents."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return jsonify({"success": False, "message": "Session invalide"}), 401

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        return jsonify({"success": False, "message": "Système RAG non initialisé"}), 400

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"success": False, "message": "Requête requise"}), 400

    query = data['query']
    document_id = data.get('document_id')  # Optionnel
    top_k = data.get('top_k', 5)

    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Effectuer la recherche avec WaveletRAG
        wavelet_rag = user_configs[user_id]['wavelet_rag']
        passages = loop.run_until_complete(wavelet_rag.search(
            query=query,
            document_id=document_id,
            top_k=top_k
        ))

        loop.close()

        # Formater les résultats
        formatted_passages = []
        for passage in passages:
            formatted_passages.append({
                "id": passage.get("chunk_id", ""),
                "text": passage.get("text", ""),
                "document_id": passage.get("document_id", ""),
                "document_name": passage.get("document_name", "Document inconnu"),
                "similarity": passage.get("similarity", 0),
                "metadata": {
                    "section_title": passage.get("metadata", {}).get("section_title", ""),
                    "section_level": passage.get("metadata", {}).get("section_level", ""),
                    "start_line": passage.get("metadata", {}).get("start_line", ""),
                    "end_line": passage.get("metadata", {}).get("end_line", "")
                }
            })

        return jsonify({
            "success": True,
            "passages": formatted_passages
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        }), 500


# Socket.IO event handlers

@socketio.on('rag_initialize')
def handle_rag_initialize():
    """Initialise le système RAG via Socket.IO."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        socketio.emit('rag_status', {
            "message": "Session invalide",
            "status": "error"
        }, room=user_id)
        return

    # Vérifier si le système est déjà initialisé
    if user_configs[user_id].get('rag_initialized'):
        socketio.emit('rag_status', {
            "message": "Système RAG déjà initialisé",
            "status": "complete"
        }, room=user_id)
        return

    # Initialiser le système RAG
    result = initialize_rag_system(user_id)
    if not result["success"]:
        socketio.emit('rag_status', {
            "message": result["message"],
            "status": "error"
        }, room=user_id)


@socketio.on('rag_ask')
def handle_rag_ask(data):
    """Pose une question au système RAG via Socket.IO."""
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        socketio.emit('rag_answer', {
            "status": "error",
            "message": "Session invalide"
        }, room=user_id)
        return

    # Vérifier si le système est initialisé
    if not user_configs[user_id].get('rag_initialized'):
        socketio.emit('rag_answer', {
            "status": "error",
            "message": "Système RAG non initialisé"
        }, room=user_id)
        return

    if not data or 'question' not in data:
        socketio.emit('rag_answer', {
            "status": "error",
            "message": "Question requise"
        }, room=user_id)
        return

    question = data['question']
    use_concepts = data.get('use_concepts', True)

    # Lancer la recherche en arrière-plan
    threading.Thread(target=async_ask_question, args=(user_id, question, use_concepts)).start()


# Enregistrement des fonctions dans le FunctionHandler

def rag_init():
    """
    Initialise le système RAG pour l'utilisateur courant.

    Returns:
        Informations sur l'initialisation
    """
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return {"status": "error", "message": "Session invalide"}

    if user_configs[user_id].get('rag_initialized'):
        return {"status": "success", "message": "Système RAG déjà initialisé"}

    result = initialize_rag_system(user_id)
    return result


def rag_add_document(file_path):
    """
    Ajoute un document au système RAG.

    Args:
        file_path: Chemin vers le document à ajouter

    Returns:
        Informations sur le document ajouté
    """
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return {"status": "error", "message": "Session invalide"}

    if not user_configs[user_id].get('rag_initialized'):
        return {"status": "error", "message": "Système RAG non initialisé"}

    if not os.path.exists(file_path):
        return {"status": "error", "message": f"Fichier {file_path} introuvable"}

    # Lancer l'ajout en arrière-plan
    threading.Thread(target=async_add_document, args=(user_id, file_path)).start()

    return {
        "status": "success",
        "message": f"Document {os.path.basename(file_path)} en cours d'ajout. Une notification sera envoyée à la fin."
    }


def rag_list_documents():
    """
    Liste tous les documents indexés dans le système RAG.

    Returns:
        Liste des documents avec leurs métadonnées
    """
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return {"status": "error", "message": "Session invalide"}

    if not user_configs[user_id].get('rag_initialized'):
        return {"status": "error", "message": "Système RAG non initialisé"}

    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Récupérer les documents
        rag_engine = user_configs[user_id]['rag_engine']
        documents = loop.run_until_complete(rag_engine.get_all_documents())

        # Formater les documents pour l'affichage
        formatted_docs = []
        for doc_id, doc_info in documents.items():
            formatted_docs.append({
                "id": doc_id,
                "title": doc_info.get("original_filename", "Document sans titre"),
                "path": doc_info.get("path", ""),
                "chunks_count": doc_info.get("chunks_count", 0)
            })

        loop.close()

        # Émettre l'événement pour mettre à jour l'interface
        socketio.emit('rag_documents_list', {"documents": formatted_docs}, room=user_id)

        return {"status": "success", "documents": formatted_docs}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Erreur: {str(e)}"}


def rag_remove_document(document_id):
    """
    Supprime un document du système RAG.

    Args:
        document_id: ID du document à supprimer

    Returns:
        Statut de la suppression
    """
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return {"status": "error", "message": "Session invalide"}

    if not user_configs[user_id].get('rag_initialized'):
        return {"status": "error", "message": "Système RAG non initialisé"}

    try:
        # Exécuter de manière asynchrone
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Supprimer le document
        rag_engine = user_configs[user_id]['rag_engine']
        success = loop.run_until_complete(rag_engine.remove_document(document_id))

        loop.close()

        if success:
            # Émettre un événement pour mettre à jour l'interface
            socketio.emit('rag_documents_updated', room=user_id)

            return {"status": "success", "message": f"Document {document_id} supprimé avec succès"}
        else:
            return {"status": "error", "message": f"Document {document_id} non trouvé"}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Erreur: {str(e)}"}


def rag_ask_question(question, use_concepts=True):
    """
    Pose une question au système RAG.

    Args:
        question: Question à poser
        use_concepts: Si True, utilise la détection automatique de concepts

    Returns:
        Réponse générée
    """
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return {"status": "error", "message": "Session invalide"}

    if not user_configs[user_id].get('rag_initialized'):
        return {"status": "error", "message": "Système RAG non initialisé"}

    # Lancer la recherche en arrière-plan
    threading.Thread(target=async_ask_question, args=(user_id, question, use_concepts)).start()

    return {"status": "processing",
            "message": "Question en cours de traitement. Les résultats seront affichés bientôt."}


def rag_list_concepts():
    """
    Liste les concepts disponibles dans l'ontologie.

    Returns:
        Liste des concepts principaux
    """
    user_id = session.get('user_id')
    if not user_id or user_id not in user_configs:
        return {"status": "error", "message": "Session invalide"}

    if not user_configs[user_id].get('rag_initialized'):
        return {"status": "error", "message": "Système RAG non initialisé"}

    try:
        # Récupérer l'ontologie
        ontology_manager = user_configs[user_id]['ontology_manager']

        # Récupérer les concepts principaux (ceux sans parents)
        top_concepts = []
        for uri, concept in ontology_manager.concepts.items():
            if not concept.parents:
                top_concepts.append({
                    "uri": uri,
                    "label": concept.label,
                    "description": concept.description if hasattr(concept, "description") else None,
                    "children_count": len(concept.children) if hasattr(concept, "children") else 0
                })

        # Émettre l'événement pour mettre à jour l'interface
        socketio.emit('rag_concepts_list', {"concepts": top_concepts}, room=user_id)

        return {"status": "success", "concepts": top_concepts}

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": f"Erreur: {str(e)}"}


# Fonctions d'encodage pour la sérialisation JSON
def register_rag_json_encoders(app):
    """Enregistre des encodeurs JSON personnalisés pour les types du système RAG."""

    class RAGJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            # Gérer les types numpy
            if 'numpy' in str(type(obj)):
                try:
                    return obj.tolist()
                except:
                    return str(obj)

            # Gérer les objets de l'ontologie
            if 'Concept' in str(type(obj)):
                return {
                    "uri": getattr(obj, "uri", None),
                    "label": getattr(obj, "label", None),
                    "description": getattr(obj, "description", None)
                }

            # Gérer d'autres types spéciaux si nécessaire
            return super().default(obj)

    app.json_encoder = RAGJSONEncoder


# Enregistrer les fonctions dans le FunctionHandler au démarrage de l'application
def register_rag_functions_in_handler(function_handler):
    """Enregistre les fonctions RAG dans le FunctionHandler."""

    # Importer le function_handler depuis app.py
    function_definitions = [
        {
            "name": "rag_init",
            "description": "Initialise le système RAG pour l'utilisateur courant.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "rag_add_document",
            "description": "Ajoute un document au système RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Chemin vers le document à ajouter."
                    }
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "rag_list_documents",
            "description": "Liste tous les documents indexés dans le système RAG.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "rag_remove_document",
            "description": "Supprime un document du système RAG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "ID du document à supprimer."
                    }
                },
                "required": ["document_id"]
            }
        },
        {
            "name": "rag_ask_question",
            "description": "Pose une question au système RAG avec détection automatique de concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question à poser au système RAG."
                    },
                    "use_concepts": {
                        "type": "boolean",
                        "description": "Si true, utilise la détection automatique de concepts. Par défaut: true."
                    }
                },
                "required": ["question"]
            }
        },
        {
            "name": "rag_list_concepts",
            "description": "Liste les concepts disponibles dans l'ontologie.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]

    # Ajouter les fonctions au FunctionHandler
    function_handler.add_function("rag_init", rag_init)
    function_handler.add_function("rag_add_document", rag_add_document)
    function_handler.add_function("rag_list_documents", rag_list_documents)
    function_handler.add_function("rag_remove_document", rag_remove_document)
    function_handler.add_function("rag_ask_question", rag_ask_question)
    function_handler.add_function("rag_list_concepts", rag_list_concepts)

    # Retourner les définitions pour l'API function calling
    return function_definitions