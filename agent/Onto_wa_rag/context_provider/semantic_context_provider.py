"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# context_providers/semantic_context_provider.py
import asyncio
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SemanticContextProvider:
    """Fournit le contexte sémantique : entités conceptuellement similaires"""

    def __init__(self, document_store, rag_engine, entity_index):
        self.document_store = document_store
        self.rag_engine = rag_engine
        self.entity_index = entity_index

        # Cache pour les contextes sémantiques
        self._semantic_cache: Dict[str, Dict[str, Any]] = {}

        # Patterns algorithmiques communs en calcul scientifique
        self.algorithmic_patterns = {
            "iterative_solver": ["iteration", "convergence", "tolerance", "residual"],
            "linear_algebra": ["matrix", "vector", "eigenvalue", "decomposition"],
            "numerical_integration": ["quadrature", "integration", "simpson", "gauss"],
            "optimization": ["minimize", "maximize", "gradient", "objective"],
            "differential_equations": ["ode", "pde", "derivative", "boundary"],
            "fft": ["transform", "fourier", "frequency", "spectral"],
            "monte_carlo": ["random", "sampling", "probability", "statistical"],
            "mesh": ["grid", "mesh", "nodes", "elements"],
            "io_operations": ["read", "write", "file", "output"],
            "parallel": ["mpi", "openmp", "parallel", "thread"]
        }

    async def get_semantic_context(self, entity_name: str, max_tokens: int = 2000) -> Dict[str, Any]:
        """Récupère le contexte sémantique d'une entité"""

        # Vérifier le cache
        if entity_name in self._semantic_cache:
            cached = self._semantic_cache[entity_name]
            # Vérifier si le cache n'est pas trop vieux (simple heuristique)
            if len(cached.get("similar_entities", [])) > 0:
                return cached

        # Trouver l'entité principale
        entity_chunks = await self.entity_index.find_entity(entity_name)
        if not entity_chunks:
            return {
                "error": f"Entity '{entity_name}' not found",
                "suggestions": await self._get_similar_entity_names(entity_name)
            }

        main_chunk_id = entity_chunks[0]
        main_chunk = await self._get_chunk_by_id(main_chunk_id)
        entity_info = await self.entity_index.get_entity_info(main_chunk_id)

        if not main_chunk or not entity_info:
            return {"error": f"Could not load entity data for '{entity_name}'"}

        # Extraire les concepts principaux
        concepts = entity_info.get('concepts', [])

        context = {
            "entity": entity_name,
            "type": "semantic",
            "main_concepts": concepts[:3],  # Top 3 concepts
            "similar_entities": [],
            "concept_clusters": {},
            "algorithmic_patterns": [],
            "semantic_neighbors": [],
            "cross_file_relations": [],
            "tokens_used": 0
        }

        tokens_budget = max_tokens

        # 1. Entités similaires via embeddings
        context["similar_entities"] = await self._find_similar_entities_via_embeddings(
            main_chunk, tokens_budget * 0.4
        )
        tokens_budget -= len(str(context["similar_entities"])) // 4

        # 2. Clusters de concepts
        if concepts and tokens_budget > 300:
            context["concept_clusters"] = await self._build_concept_clusters(
                concepts, tokens_budget * 0.25
            )
            tokens_budget -= len(str(context["concept_clusters"])) // 4

        # 3. Patterns algorithmiques
        context["algorithmic_patterns"] = await self._identify_algorithmic_patterns(
            main_chunk['text'], concepts
        )

        # 4. Voisins sémantiques (entités avec concepts similaires)
        if tokens_budget > 200:
            context["semantic_neighbors"] = await self._find_semantic_neighbors(
                concepts, entity_name, tokens_budget * 0.2
            )
            tokens_budget -= len(str(context["semantic_neighbors"])) // 4

        # 5. Relations cross-file
        if tokens_budget > 100:
            context["cross_file_relations"] = await self._find_cross_file_relations(
                entity_info, tokens_budget * 0.15
            )

        context["tokens_used"] = max_tokens - tokens_budget

        # Mettre en cache
        self._semantic_cache[entity_name] = context

        return context

    async def _find_similar_entities_via_embeddings(self, main_chunk: Dict[str, Any], max_tokens: int) -> List[
        Dict[str, Any]]:

        if (hasattr(self.rag_engine, 'classifier') and
                self.rag_engine.classifier and
                hasattr(self.rag_engine.classifier, 'search_with_concepts')):

            try:
                print("🧠 Utilisation du classifier ontologique pour la recherche sémantique")

                # Utiliser la recherche avec concepts
                result = await self.rag_engine.classifier.search_with_concepts(
                    query=main_chunk['text'][:500],  # Utiliser le début du chunk comme requête
                    top_k=8,
                    concept_weight=0.4,
                    min_concept_confidence=0.3
                )

                similar_entities = []
                if 'sources' in result:
                    for source in result['sources']:
                        # Éviter l'entité elle-même
                        if source.get('entity_name') == main_chunk.get('metadata', {}).get('entity_name'):
                            continue

                        entity = {
                            "name": source.get('entity_name', 'Unknown'),
                            "type": source.get('entity_type', 'Unknown'),
                            "similarity": source.get('relevance_score', 0),
                            "file": source.get('filename', ''),
                            "concepts": source.get('detected_concepts', [])[:2],
                            "similarity_reasons": ["Ontological similarity"],
                            "summary": f"Entity from {source.get('filename', 'unknown file')}"
                        }
                        similar_entities.append(entity)

                return similar_entities[:8]

            except Exception as e:
                print(f"⚠️ Erreur avec le classifier ontologique: {e}")
                # Fallback vers la méthode originale

        # FALLBACK : Méthode originale (code existant mais simplifié)
        """Trouve les entités similaires via les embeddings"""
        try:
            # Utiliser le RAG pour trouver des chunks similaires
            similar_chunks = await self.rag_engine.find_similar(
                main_chunk['text'],
                max_results=20,
                min_similarity=0.65  # Seuil plus élevé pour la similarité sémantique
            )

            similar_entities = []
            seen_entities = set()
            tokens_per_entity = max_tokens / 10  # Budget par entité

            for chunk_id, similarity in similar_chunks:
                if chunk_id == main_chunk['id']:  # Skip l'entité elle-même
                    continue

                entity_info = await self.entity_index.get_entity_info(chunk_id)
                if not entity_info:
                    continue

                entity_name = entity_info['name']
                if entity_name in seen_entities:
                    continue

                seen_entities.add(entity_name)

                # Analyser pourquoi c'est similaire
                similarity_reasons = await self._analyze_similarity_reasons(
                    main_chunk, chunk_id, entity_info
                )

                similar_entity = {
                    "name": entity_name,
                    "type": entity_info['type'],
                    "similarity": round(similarity, 3),
                    "file": entity_info.get('filepath', ''),
                    "concepts": [c.get('label', '') for c in entity_info.get('concepts', [])][:2],
                    "similarity_reasons": similarity_reasons,
                    "summary": await self._create_entity_summary(chunk_id, int(tokens_per_entity))
                }

                similar_entities.append(similar_entity)

                if len(similar_entities) >= 8:  # Limiter à 8 entités
                    break

            return similar_entities

        except Exception as e:
            logger.debug(f"Similarity search failed for {main_chunk.get('id', 'unknown')}: {e}")
            return []

    async def _analyze_similarity_reasons(self, main_chunk: Dict[str, Any],
                                          similar_chunk_id: str,
                                          similar_entity_info: Dict[str, Any]) -> List[str]:
        """Analyse pourquoi deux entités sont considérées comme similaires"""
        reasons = []

        # 1. Même type d'entité
        main_metadata = main_chunk.get('metadata', {})
        main_type = main_metadata.get('entity_type', '')
        similar_type = similar_entity_info.get('type', '')

        if main_type == similar_type:
            reasons.append(f"Same entity type ({main_type})")

        # 2. Concepts communs
        main_concepts = set(c.get('label', '') for c in main_metadata.get('detected_concepts', []))
        similar_concepts = set(c.get('label', '') for c in similar_entity_info.get('concepts', []))

        common_concepts = main_concepts.intersection(similar_concepts)
        if common_concepts:
            reasons.append(f"Shared concepts: {', '.join(list(common_concepts)[:3])}")

        # 3. Même fichier ou module
        main_file = main_metadata.get('filepath', '')
        similar_file = similar_entity_info.get('filepath', '')

        if main_file == similar_file:
            reasons.append("Same file")
        elif main_file and similar_file:
            main_module = self._extract_module_name(main_file)
            similar_module = self._extract_module_name(similar_file)
            if main_module == similar_module:
                reasons.append("Same module")

        # 4. Patterns dans le code (analyse textuelle légère)
        similar_chunk = await self._get_chunk_by_id(similar_chunk_id)
        if similar_chunk:
            code_patterns = self._find_common_code_patterns(
                main_chunk['text'], similar_chunk['text']
            )
            if code_patterns:
                reasons.append(f"Common patterns: {', '.join(code_patterns[:2])}")

        return reasons[:4]  # Limiter à 4 raisons

    def _find_common_code_patterns(self, code1: str, code2: str) -> List[str]:
        """Trouve les patterns de code communs entre deux chunks"""
        patterns = []

        # Patterns algorithmiques simples
        pattern_keywords = {
            "loops": ["do ", "while", "for"],
            "conditionals": ["if ", "select case", "where"],
            "math_operations": ["sqrt", "sin", "cos", "exp", "log"],
            "array_operations": ["sum(", "product(", "maxval", "minval"],
            "io_operations": ["read", "write", "print", "open", "close"]
        }

        code1_lower = code1.lower()
        code2_lower = code2.lower()

        for pattern_name, keywords in pattern_keywords.items():
            if (any(kw in code1_lower for kw in keywords) and
                    any(kw in code2_lower for kw in keywords)):
                patterns.append(pattern_name)

        return patterns

    async def _build_concept_clusters(self, concepts: List[Dict[str, Any]], max_tokens: int) -> Dict[str, Any]:
        """Construit des clusters de concepts liés"""
        if not concepts:
            return {}

        clusters = {
            "primary_concepts": [],
            "related_entities": {},
            "concept_network": {}
        }

        # Prendre les concepts principaux
        main_concepts = concepts[:5]  # Top 5 concepts
        clusters["primary_concepts"] = [
            {
                "label": c.get('label', ''),
                "confidence": c.get('confidence', 0),
                "category": c.get('category', '')
            }
            for c in main_concepts
        ]

        # Pour chaque concept, trouver d'autres entités qui partagent ce concept
        for concept in main_concepts:
            concept_label = concept.get('label', '')
            if not concept_label:
                continue

            related_entities = await self._find_entities_with_concept(concept_label)
            clusters["related_entities"][concept_label] = related_entities[:5]  # Top 5 par concept

        return clusters

    async def _find_entities_with_concept(self, concept_label: str) -> List[Dict[str, Any]]:
        """Trouve les entités qui partagent un concept donné"""
        entities_with_concept = []

        # Parcourir tous les chunks pour trouver ceux avec ce concept
        for chunk_id, entity_info in self.entity_index.chunk_to_entity.items():
            entity_concepts = entity_info.get('concepts', [])

            for concept in entity_concepts:
                if concept.get('label', '').lower() == concept_label.lower():
                    entities_with_concept.append({
                        "name": entity_info['name'],
                        "type": entity_info['type'],
                        "confidence": concept.get('confidence', 0),
                        "file": entity_info.get('filepath', '')
                    })
                    break

        # Trier par confiance
        entities_with_concept.sort(key=lambda x: x['confidence'], reverse=True)

        return entities_with_concept

    async def _identify_algorithmic_patterns(self, code: str, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identifie les patterns algorithmiques en utilisant le classifier si disponible"""

        # NOUVEAU : Utiliser le classifier de concepts s'il existe
        if (hasattr(self.rag_engine, 'classifier') and
                self.rag_engine.classifier and
                hasattr(self.rag_engine.classifier, 'smart_concept_detection')):

            try:
                print("🔬 Détection de patterns via le classifier ontologique")

                # Utiliser la détection intelligente de concepts
                detected_concepts = await self.rag_engine.classifier.smart_concept_detection(code[:1000])

                patterns = []
                for concept in detected_concepts[:5]:
                    label = concept.get('label', '').lower()
                    confidence = concept.get('confidence', 0)

                    # Mapper les concepts aux patterns algorithmiques
                    pattern_mapping = {
                        'matrix': 'linear_algebra',
                        'fft': 'fourier_transform',
                        'solver': 'iterative_solver',
                        'integration': 'numerical_integration',
                        'optimization': 'optimization',
                        'parallel': 'parallel_computation',
                        'energy': 'energy_calculation',
                        'density': 'density_functional',
                        'wavelet': 'wavelet_analysis'
                    }

                    for keyword, pattern_name in pattern_mapping.items():
                        if keyword in label:
                            patterns.append({
                                "pattern": pattern_name,
                                "score": int(confidence * 10),
                                "matched_keywords": [label],
                                "description": f"Pattern détecté via concept: {label}",
                                "source": "ontology_classifier"
                            })
                            break

                if patterns:
                    return patterns

            except Exception as e:
                print(f"⚠️ Erreur détection patterns ontologiques: {e}")

        # FALLBACK : Méthode originale (code existant)
        return await self._identify_patterns_fallback(code, concepts)

    async def _identify_patterns_fallback(self, code: str, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Méthode fallback pour identifier les patterns (code original)"""
        identified_patterns = []
        code_lower = code.lower()

        # Analyser les concepts pour identifier les patterns
        concept_labels = [c.get('label', '').lower() for c in concepts]

        for pattern_name, keywords in self.algorithmic_patterns.items():
            score = 0
            matched_keywords = []

            # Chercher les mots-clés dans le code
            for keyword in keywords:
                if keyword in code_lower:
                    score += 2
                    matched_keywords.append(keyword)

            # Chercher les mots-clés dans les concepts
            for keyword in keywords:
                if any(keyword in concept for concept in concept_labels):
                    score += 3
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)

            # Si suffisamment de correspondances
            if score >= 3:
                identified_patterns.append({
                    "pattern": pattern_name,
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "description": self._get_pattern_description(pattern_name),
                    "source": "text_analysis"
                })

        # Trier par score et retourner les meilleurs
        identified_patterns.sort(key=lambda x: x['score'], reverse=True)
        return identified_patterns[:5]

    def _get_pattern_description(self, pattern_name: str) -> str:
        """Retourne une description du pattern algorithmique"""
        descriptions = {
            "iterative_solver": "Algorithme itératif avec convergence",
            "linear_algebra": "Opérations d'algèbre linéaire",
            "numerical_integration": "Intégration numérique",
            "optimization": "Algorithme d'optimisation",
            "differential_equations": "Résolution d'équations différentielles",
            "fft": "Transformée de Fourier rapide",
            "monte_carlo": "Méthode Monte Carlo",
            "mesh": "Opérations sur maillage",
            "io_operations": "Opérations d'entrée/sortie",
            "parallel": "Calcul parallèle"
        }
        return descriptions.get(pattern_name, f"Pattern {pattern_name}")

    async def _find_semantic_neighbors(self, concepts: List[Dict[str, Any]],
                                       current_entity: str, max_tokens: int) -> List[Dict[str, Any]]:

        if (hasattr(self.rag_engine, 'classifier') and
                self.rag_engine.classifier and
                hasattr(self.rag_engine.classifier, 'concept_classifier')):

            try:
                concept_classifier = self.rag_engine.classifier.concept_classifier

                neighbors = []

                # Pour chaque concept principal, trouver les entités liées
                for concept in concepts[:3]:  # Top 3 concepts
                    concept_uri = concept.get('concept_uri')
                    if not concept_uri:
                        continue

                    # Utiliser la recherche par concept
                    search_result = await self.rag_engine.classifier.search_by_concept(
                        query=current_entity,
                        concept_uri=concept_uri,
                        include_subconcepts=True,
                        top_k=5,
                        confidence_threshold=0.3
                    )

                    if 'passages' in search_result:
                        for passage in search_result['passages']:
                            entity_name = passage.get('metadata', {}).get('entity_name', '')

                            if entity_name and entity_name != current_entity:
                                neighbors.append({
                                    "name": entity_name,
                                    "type": passage.get('metadata', {}).get('entity_type', ''),
                                    "semantic_score": passage.get('similarity', 0),
                                    "shared_concepts": [concept.get('label', '')],
                                    "file": passage.get('metadata', {}).get('filename', '')
                                })

                # Dédoublonner par nom d'entité
                seen_names = set()
                unique_neighbors = []
                for neighbor in neighbors:
                    if neighbor['name'] not in seen_names:
                        seen_names.add(neighbor['name'])
                        unique_neighbors.append(neighbor)

                return unique_neighbors[:8]

            except Exception as e:
                print(f"⚠️ Erreur recherche voisins ontologiques: {e}")


        """Trouve les voisins sémantiques basés sur les concepts"""
        if not concepts:
            return []

        # Score des entités basé sur les concepts partagés
        entity_scores = defaultdict(float)
        entity_shared_concepts = defaultdict(list)

        main_concept_labels = [c.get('label', '') for c in concepts]

        # Parcourir tous les chunks
        for chunk_id, entity_info in self.entity_index.chunk_to_entity.items():
            entity_name = entity_info['name']

            if entity_name == current_entity:
                continue

            entity_concepts = entity_info.get('concepts', [])

            # Calculer le score de similarité conceptuelle
            for entity_concept in entity_concepts:
                entity_concept_label = entity_concept.get('label', '')

                if entity_concept_label in main_concept_labels:
                    confidence = entity_concept.get('confidence', 0)
                    entity_scores[entity_name] += confidence
                    entity_shared_concepts[entity_name].append(entity_concept_label)

        # Convertir en liste et trier
        neighbors = []
        for entity_name, score in entity_scores.items():
            if score > 0.1:  # Seuil minimum
                entity_info = await self._get_entity_info_by_name(entity_name)
                if entity_info:
                    neighbors.append({
                        "name": entity_name,
                        "type": entity_info['type'],
                        "semantic_score": round(score, 3),
                        "shared_concepts": entity_shared_concepts[entity_name][:3],
                        "file": entity_info.get('filepath', '')
                    })

        neighbors.sort(key=lambda x: x['semantic_score'], reverse=True)
        return neighbors[:8]

    async def _find_cross_file_relations(self, entity_info: Dict[str, Any], max_tokens: int) -> List[Dict[str, Any]]:
        """Trouve les relations cross-file (entre fichiers différents)"""
        current_file = entity_info.get('filepath', '')
        if not current_file:
            return []

        relations = []

        # Chercher les entités dans d'autres fichiers avec des concepts similaires
        entity_concepts = set(c.get('label', '') for c in entity_info.get('concepts', []))

        for chunk_id, other_entity_info in self.entity_index.chunk_to_entity.items():
            other_file = other_entity_info.get('filepath', '')

            # Skip le même fichier
            if other_file == current_file or not other_file:
                continue

            other_concepts = set(c.get('label', '') for c in other_entity_info.get('concepts', []))

            # Concepts partagés
            shared_concepts = entity_concepts.intersection(other_concepts)

            if shared_concepts:
                relations.append({
                    "entity": other_entity_info['name'],
                    "type": other_entity_info['type'],
                    "file": other_file,
                    "shared_concepts": list(shared_concepts)[:3],
                    "relation_strength": len(shared_concepts) / max(len(entity_concepts), 1)
                })

        # Trier par force de relation
        relations.sort(key=lambda x: x['relation_strength'], reverse=True)
        return relations[:6]

    async def _create_entity_summary(self, chunk_id: str, max_tokens: int) -> str:
        """Crée un résumé d'une entité"""
        chunk = await self._get_chunk_by_id(chunk_id)
        if not chunk:
            return "Summary not available"

        text = chunk['text']

        # Extraire les premières lignes significatives
        lines = text.split('\n')
        summary_lines = []

        for line in lines[:10]:  # Max 10 premières lignes
            line = line.strip()
            if line and not line.startswith('!'):  # Skip commentaires
                summary_lines.append(line)
                if len(summary_lines) >= 3:  # Max 3 lignes significatives
                    break

        summary = '\n'.join(summary_lines)

        # Tronquer si trop long
        max_chars = max_tokens * 4  # Approximation
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."

        return summary

    async def _get_entity_info_by_name(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'une entité par son nom"""
        chunks = await self.entity_index.find_entity(entity_name)
        if chunks:
            return await self.entity_index.get_entity_info(chunks[0])
        return None

    def _extract_module_name(self, filepath: str) -> Optional[str]:
        """Extrait le nom du module depuis le chemin du fichier"""
        if not filepath:
            return None

        filename = filepath.split('/')[-1]

        if '.' in filename:
            module_name = '.'.join(filename.split('.')[:-1])
        else:
            module_name = filename

        return module_name

    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk par son ID depuis le document store"""
        parts = chunk_id.split('-chunk-')
        if len(parts) != 2:
            return None

        document_id = parts[0]

        await self.document_store.load_document_chunks(document_id)
        chunks = await self.document_store.get_document_chunks(document_id)

        if chunks:
            for chunk in chunks:
                if chunk['id'] == chunk_id:
                    return chunk

        return None

    async def _get_similar_entity_names(self, entity_name: str, limit: int = 5) -> List[str]:
        """Trouve des noms d'entités similaires pour les suggestions"""
        all_names = list(self.entity_index.name_to_chunks.keys())

        similar = [name for name in all_names
                   if entity_name.lower() in name.lower() or name.lower() in entity_name.lower()]

        return similar[:limit]