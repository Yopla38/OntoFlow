"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# context_provider/contextual_text_generator.py
import asyncio
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ContextualTextGenerator:
    """
    Générateur de contexte textuel pour LLMs à partir des éléments Fortran.
    Utilise le SmartContextProvider existant et formate le résultat en texte lisible.
    """

    def __init__(self, smart_context_provider):
        """
        Initialise avec un SmartContextProvider existant.

        Args:
            smart_context_provider: Instance de SmartContextProvider
        """
        self.context_provider = smart_context_provider
        self.entity_index = smart_context_provider.entity_index

    async def get_contextual_text(self,
                                  element_name: str,
                                  context_type: str = "complete",
                                  agent_perspective: str = "developer",
                                  task_context: str = "code_understanding",
                                  max_tokens: int = 4000,
                                  format_style: str = "detailed") -> str:
        """
        Génère un contexte textuel complet pour un élément donné.

        Args:
            element_name: Nom de l'élément (fonction, module, variable, fichier...)
            context_type: Type de contexte ("complete", "local", "global", "semantic")
            agent_perspective: Perspective ("developer", "reviewer", "analyzer", etc.)
            task_context: Contexte de la tâche ("code_understanding", "debugging", etc.)
            max_tokens: Nombre maximum de tokens
            format_style: Style de formatage ("detailed", "summary", "bullet_points")

        Returns:
            Texte formaté prêt pour un LLM
        """

        # S'assurer que le context provider est initialisé
        if not self.context_provider._initialized:
            await self.context_provider.initialize()

        # 1. Essayer de résoudre l'élément (pourrait être un nom partiel, nom de fichier, etc.)
        resolved_entity = await self._resolve_element(element_name)

        if not resolved_entity:
            return await self._generate_not_found_text(element_name)

        # 2. Récupérer le contexte selon le type demandé
        if context_type == "complete":
            context_data = await self.context_provider.get_context_for_agent(
                resolved_entity['name'], agent_perspective, task_context, max_tokens
            )
        elif context_type == "local":
            context_data = await self.context_provider.get_local_context(
                resolved_entity['name'], max_tokens
            )
        elif context_type == "global":
            context_data = await self.context_provider.get_global_context(
                resolved_entity['name'], max_tokens
            )
        elif context_type == "semantic":
            context_data = await self.context_provider.get_semantic_context(
                resolved_entity['name'], max_tokens
            )
        else:
            return f"❌ Type de contexte non supporté: {context_type}"

        # 3. Formatter en texte selon le style demandé
        if format_style == "detailed":
            return await self._format_detailed_context(context_data, resolved_entity)
        elif format_style == "summary":
            return await self._format_summary_context(context_data, resolved_entity)
        elif format_style == "bullet_points":
            return await self._format_bullet_context(context_data, resolved_entity)
        else:
            return await self._format_detailed_context(context_data, resolved_entity)

    async def _resolve_element(self, element_name: str) -> Optional[Dict[str, Any]]:
        """
        Résout un nom d'élément qui peut être partiel ou ambigu.
        Utilise l'EntityIndex existant pour la résolution.
        """

        # 1. Recherche directe par nom d'entité
        chunks = await self.entity_index.find_entity(element_name)
        if chunks:
            entity_info = await self.entity_index.get_entity_info(chunks[0])
            if entity_info:
                return {
                    'name': entity_info['name'],
                    'type': entity_info['type'],
                    'resolution_method': 'direct_entity_match',
                    'chunk_id': chunks[0]
                }

        # 2. NOUVEAU : Recherche par concept
        concept_results = await self._search_by_concept(element_name)
        if concept_results:
            return concept_results

        # 2. Recherche par nom de fichier
        if '.' in element_name or '/' in element_name:
            file_entities = await self._find_entities_in_file(element_name)
            if file_entities:
                # Retourner l'entité principale du fichier (module ou programme)
                main_entity = self._find_main_entity_in_file(file_entities)
                if main_entity:
                    return {
                        'name': main_entity['name'],
                        'type': main_entity['type'],
                        'resolution_method': 'file_main_entity',
                        'file_path': element_name,
                        'file_entities_count': len(file_entities)
                    }

        # 3. Recherche fuzzy
        search_results = await self.context_provider.search_entities(element_name)
        if search_results:
            best_match = search_results[0]  # Premier résultat
            return {
                'name': best_match['name'],
                'type': best_match['type'],
                'resolution_method': f"fuzzy_match_{best_match['match_type']}",
                'alternatives': [r['name'] for r in search_results[1:5]]  # 4 alternatives
            }

        return None

    async def _search_by_concept(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """
        NOUVELLE MÉTHODE : Recherche les entités par concept.
        """

        print(f"🔍 Recherche par concept: {concept_name}")

        # 1. Essayer d'abord avec le classifier ontologique si disponible
        if (hasattr(self.context_provider.rag_engine, 'classifier') and
                self.context_provider.rag_engine.classifier):

            try:
                print("🧠 Utilisation du classifier ontologique")

                # Utiliser la recherche par concept du classifier
                result = await self.context_provider.rag_engine.classifier.search_by_concept(
                    query=concept_name,
                    concept_uri=None,  # Laisser le système trouver le concept
                    include_subconcepts=True,
                    top_k=10,
                    confidence_threshold=0.3
                )

                if 'passages' in result and result['passages']:
                    # Prendre le premier résultat et créer un contexte multi-entités
                    return await self._create_concept_context(concept_name, result['passages'])

            except Exception as e:
                print(f"⚠️ Erreur classifier ontologique: {e}")

        # 2. Recherche manuelle dans les métadonnées
        print("🔍 Recherche manuelle dans les concepts")

        matching_entities = []

        for chunk_id, entity_info in self.context_provider.entity_index.chunk_to_entity.items():
            concepts = entity_info.get('concepts', [])

            for concept in concepts:
                concept_label = ''
                if isinstance(concept, dict):
                    concept_label = concept.get('label', '')
                else:
                    concept_label = str(concept)

                # Recherche flexible du concept
                if (concept_label and
                        (concept_name.lower() in concept_label.lower() or
                         concept_label.lower() in concept_name.lower())):
                    confidence = concept.get('confidence', 0) if isinstance(concept, dict) else 0.5

                    matching_entities.append({
                        'entity_name': entity_info['name'],
                        'entity_type': entity_info['type'],
                        'concept_label': concept_label,
                        'confidence': confidence,
                        'filepath': entity_info.get('filepath', ''),
                        'chunk_id': chunk_id
                    })

        if matching_entities:
            # Trier par confiance
            matching_entities.sort(key=lambda x: x['confidence'], reverse=True)

            print(f"✅ Trouvé {len(matching_entities)} entités avec le concept {concept_name}")

            # Prendre la meilleure entité ou créer un contexte multi-entités
            if len(matching_entities) == 1:
                best_match = matching_entities[0]
                return {
                    'name': best_match['entity_name'],
                    'type': best_match['entity_type'],
                    'resolution_method': 'concept_match',
                    'concept_matched': best_match['concept_label'],
                    'concept_confidence': best_match['confidence']
                }
            else:
                # Contexte multi-entités pour un concept
                return await self._create_concept_context_from_entities(concept_name, matching_entities)

        print(f"❌ Aucune entité trouvée pour le concept: {concept_name}")
        return None

    async def _create_concept_context(self, concept_name: str, passages: List[Dict]) -> Dict[str, Any]:
        """
        Crée un contexte spécial pour un concept (résultats du classifier).
        """

        entities = []
        for passage in passages[:5]:  # Top 5
            metadata = passage.get('metadata', {})
            entities.append({
                'name': metadata.get('entity_name', 'Unknown'),
                'type': metadata.get('entity_type', 'Unknown'),
                'confidence': passage.get('similarity', 0),
                'filepath': metadata.get('filepath', '')
            })

        return {
            'name': f"concept_{concept_name}",
            'type': 'concept_group',
            'resolution_method': 'ontology_concept_search',
            'concept_name': concept_name,
            'related_entities': entities,
            'total_entities': len(passages)
        }

    async def _create_concept_context_from_entities(self, concept_name: str, entities: List[Dict]) -> Dict[str, Any]:
        """
        Crée un contexte spécial pour un concept (recherche manuelle).
        """

        return {
            'name': f"concept_{concept_name}",
            'type': 'concept_group',
            'resolution_method': 'manual_concept_search',
            'concept_name': concept_name,
            'related_entities': entities[:20],  # Top 8
            'total_entities': len(entities)
        }

    async def _find_entities_in_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Trouve toutes les entités dans un fichier donné."""
        entities = []

        # Chercher les entités par chemin exact ou nom de fichier
        for chunk_id, entity_info in self.entity_index.chunk_to_entity.items():
            entity_filepath = entity_info.get('filepath', '')

            if (entity_filepath == file_path or
                    entity_filepath.endswith(file_path) or
                    entity_filepath.split('/')[-1] == file_path):
                entities.append(entity_info)

        return entities

    def _find_main_entity_in_file(self, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Trouve l'entité principale d'un fichier (module > programme > première fonction)."""

        # Priorité : module > program > subroutine > function
        type_priorities = {'module': 1, 'program': 2, 'subroutine': 3, 'function': 4}

        entities.sort(key=lambda x: type_priorities.get(x.get('type', ''), 999))

        return entities[0] if entities else None

    async def _format_detailed_context(self, context_data: Dict[str, Any],
                                       resolved_entity: Dict[str, Any]) -> str:
        """Formate un contexte détaillé pour LLM."""

        lines = []
        entity_name = resolved_entity['name']
        entity_type = resolved_entity['type']

        # === EN-TÊTE ===
        lines.append("=" * 80)

        if entity_type == 'concept_group':
            concept_name = resolved_entity.get('concept_name', 'Unknown')
            lines.append(f"📋 CONTEXTE CONCEPT : {concept_name}")
            lines.append("=" * 80)
            lines.append(f"Concept recherché: {concept_name}")
            lines.append(f"Méthode de résolution: {resolved_entity.get('resolution_method', 'unknown')}")

            related_entities = resolved_entity.get('related_entities', [])
            total_entities = resolved_entity.get('total_entities', len(related_entities))

            lines.append(f"Entités liées trouvées: {len(related_entities)} (sur {total_entities} total)")
            lines.append("")

            # === ENTITÉS LIÉES AU CONCEPT AVEC CONTENU ===
            lines.append(f"🔗 ENTITÉS CONTENANT LE CONCEPT '{concept_name}'")
            lines.append("-" * 60)

            for i, entity in enumerate(related_entities, 1):
                name = entity.get('name', entity.get('entity_name', 'Unknown'))
                etype = entity.get('type', entity.get('entity_type', 'Unknown'))
                confidence = entity.get('confidence', 0)
                filepath = entity.get('filepath', '')
                filename = filepath.split('/')[-1] if filepath else 'Unknown'
                chunk_id = entity.get('chunk_id', '')

                lines.append(f"{i}. {name} ({etype})")
                lines.append(f"   └─ Fichier: {filename}")

                if confidence > 0:
                    lines.append(f"   └─ Confiance: {confidence:.3f}")

                if 'concept_label' in entity:
                    lines.append(f"   └─ Concept détecté: {entity['concept_label']}")

                # NOUVEAU : Ajouter le contenu du chunk
                if chunk_id:
                    chunk_content = await self._get_chunk_content_summary(chunk_id, concept_name)
                    if chunk_content:
                        lines.append(f"   └─ Lignes {chunk_content['start_line']}-{chunk_content['end_line']}:")
                        lines.append(f"   └─ {chunk_content['summary']}")

                lines.append("")

            # === ANALYSE DÉTAILLÉE DES TOP 3 ENTITÉS ===
            lines.append(f"🎯 ANALYSE DÉTAILLÉE - TOP 3 ENTITÉS")
            lines.append("-" * 60)

            for i, entity in enumerate(related_entities[:3], 1):
                name = entity.get('name', entity.get('entity_name', 'Unknown'))
                chunk_id = entity.get('chunk_id', '')

                lines.append(f"{i}. ENTITÉ: {name}")
                lines.append("   " + "─" * 50)

                if chunk_id:
                    # Récupérer le contenu complet du chunk
                    chunk_details = await self._get_detailed_chunk_content(chunk_id, concept_name)
                    if chunk_details:
                        lines.append(f"   📍 Position: Lignes {chunk_details['start_line']}-{chunk_details['end_line']}")
                        lines.append(f"   📝 Taille: {chunk_details['size']} caractères")
                        lines.append("")
                        lines.append("   📄 CONTENU DU CHUNK:")
                        lines.append("   " + "─" * 30)

                        # Ajouter le code avec indentation
                        code_lines = chunk_details['content'].split('\n')
                        for line_num, code_line in enumerate(code_lines[:15],
                                                             chunk_details['start_line']):  # Max 15 lignes
                            lines.append(f"   {line_num:3d} | {code_line}")

                        if len(code_lines) > 15:
                            lines.append(f"   ... | ({len(code_lines) - 15} lignes supplémentaires)")

                        lines.append("")

                        # Extraits pertinents pour le concept
                        relevant_extracts = self._find_concept_relevant_lines(chunk_details['content'], concept_name)
                        if relevant_extracts:
                            lines.append("   🔍 EXTRAITS PERTINENTS POUR LE CONCEPT:")
                            lines.append("   " + "─" * 35)
                            for extract in relevant_extracts[:3]:  # Top 3 extraits
                                lines.append(f"   • {extract}")
                            lines.append("")

                # Ajouter les relations (qui appelle/est appelé)
                await self._add_concept_entity_relations(lines, name)
                lines.append("")

            # === SYNTHÈSE DU CONCEPT ===
            lines.append("📊 SYNTHÈSE DU CONCEPT")
            lines.append("-" * 40)

            # Analyser les patterns communs
            common_patterns = await self._analyze_concept_patterns(related_entities, concept_name)
            if common_patterns:
                lines.append("🔄 Patterns détectés:")
                for pattern in common_patterns:
                    lines.append(f"   • {pattern}")
                lines.append("")

            # Répartition par fichier
            file_distribution = self._analyze_concept_file_distribution(related_entities)
            lines.append("📁 Répartition par fichier:")
            for filename, count in file_distribution.items():
                lines.append(f"   • {filename}: {count} entités")
            lines.append("")

            lines.append("")
            lines.append("=" * 80)

            return '\n'.join(lines)

        lines.append(f"📋 CONTEXTE FORTRAN DÉTAILLÉ")
        lines.append("=" * 80)
        lines.append(f"Entité analysée: {entity_name} ({entity_type})")
        lines.append(f"Méthode de résolution: {resolved_entity.get('resolution_method', 'unknown')}")

        if 'alternatives' in resolved_entity:
            lines.append(f"Alternatives trouvées: {', '.join(resolved_entity['alternatives'])}")

        lines.append("")

        # === DÉFINITION PRINCIPALE ===
        if 'contexts' in context_data and 'local' in context_data['contexts']:
            local_ctx = context_data['contexts']['local']
            if 'main_definition' in local_ctx:
                main_def = local_ctx['main_definition']
                lines.append("🔍 DÉFINITION PRINCIPALE")
                lines.append("-" * 40)
                lines.append(f"Nom: {main_def.get('name', 'N/A')}")
                lines.append(f"Type: {main_def.get('type', 'N/A')}")

                location = main_def.get('location', {})
                if location:
                    lines.append(f"Fichier: {location.get('file', 'N/A')}")
                    lines.append(f"Lignes: {location.get('lines', 'N/A')}")

                signature = main_def.get('signature', '')
                if signature and signature != "Signature not found":
                    lines.append(f"Signature: {signature}")

                concepts = main_def.get('concepts', [])
                if concepts:
                    concept_labels = [c.get('label', '') for c in concepts[:3]]
                    lines.append(f"Concepts clés: {', '.join(concept_labels)}")

                lines.append("")

            # === NOUVEAU : QUI APPELLE CETTE FONCTION ===
            await self._add_callers_section(lines, entity_name)

            # === DÉPENDANCES IMMÉDIATES === (identique mais amélioré)
            if 'contexts' in context_data and 'local' in context_data['contexts']:
                local_ctx = context_data['contexts']['local']

                # Dépendances USE
                immediate_deps = local_ctx.get('immediate_dependencies', [])
                if immediate_deps:
                    lines.append("📦 DÉPENDANCES IMMÉDIATES (USE)")
                    lines.append("-" * 40)
                    for dep in immediate_deps:
                        lines.append(f"• {dep.get('name', 'N/A')} ({dep.get('type', 'module')})")
                        summary = dep.get('summary', '')
                        if summary and len(summary) > 0:
                            lines.append(f"  └─ {summary[:100]}...")
                    lines.append("")

                # Fonctions appelées
                called_functions = local_ctx.get('called_functions', [])
                if called_functions:
                    lines.append("🔗 FONCTIONS APPELÉES")
                    lines.append("-" * 40)
                    for func in called_functions:
                        source = func.get('source', 'index')
                        lines.append(f"• {func.get('name', 'N/A')} [{source}]")

                        signature = func.get('signature', '')
                        if signature and signature != "Signature not found":
                            lines.append(f"  └─ {signature}")

                        summary = func.get('summary', '')
                        if summary:
                            lines.append(f"  └─ {summary[:80]}...")
                    lines.append("")

            # === IMPACT ET DÉPENDANTS === (amélioré)
            if 'contexts' in context_data and 'global' in context_data['contexts']:
                global_ctx = context_data['contexts']['global']

                impact_analysis = global_ctx.get('impact_analysis', {})
                if impact_analysis:
                    lines.append("💥 ANALYSE D'IMPACT")
                    lines.append("-" * 40)
                    lines.append(f"Niveau de risque: {impact_analysis.get('risk_level', 'unknown').upper()}")

                    direct_deps = impact_analysis.get('direct_dependents', [])
                    if direct_deps:
                        lines.append(f"Dépendants directs: {', '.join(direct_deps[:5])}")
                        if len(direct_deps) > 5:
                            lines.append(f"  ... et {len(direct_deps) - 5} autres")

                    affected_modules = impact_analysis.get('affected_modules', [])
                    if affected_modules:
                        lines.append(f"Modules affectés: {', '.join(affected_modules)}")

                    recommendations = impact_analysis.get('recommendations', [])
                    if recommendations:
                        lines.append("Recommandations:")
                        for rec in recommendations[:3]:
                            lines.append(f"  • {rec}")

                    lines.append("")

        # === ENTITÉS SIMILAIRES ===
        if 'contexts' in context_data and 'semantic' in context_data['contexts']:
            semantic_ctx = context_data['contexts']['semantic']

            similar_entities = semantic_ctx.get('similar_entities', [])
            if similar_entities:
                lines.append("🔄 ENTITÉS SIMILAIRES")
                lines.append("-" * 40)
                for similar in similar_entities[:4]:
                    similarity = similar.get('similarity', 0)
                    lines.append(f"• {similar.get('name', 'N/A')} (similarité: {similarity:.2f})")

                    reasons = similar.get('similarity_reasons', [])
                    if reasons:
                        lines.append(f"  └─ Raisons: {', '.join(reasons[:2])}")

                    file_name = similar.get('file', '').split('/')[-1]
                    if file_name:
                        lines.append(f"  └─ Fichier: {file_name}")

                lines.append("")

        # === PATTERNS ALGORITHMIQUES ===
        if 'contexts' in context_data and 'semantic' in context_data['contexts']:
            semantic_ctx = context_data['contexts']['semantic']

            patterns = semantic_ctx.get('algorithmic_patterns', [])
            if patterns:
                lines.append("🧠 PATTERNS ALGORITHMIQUES DÉTECTÉS")
                lines.append("-" * 40)
                for pattern in patterns[:3]:
                    lines.append(f"• {pattern.get('pattern', 'N/A')} (score: {pattern.get('score', 0)})")
                    lines.append(f"  └─ {pattern.get('description', 'N/A')}")

                    keywords = pattern.get('matched_keywords', [])
                    if keywords:
                        lines.append(f"  └─ Mots-clés: {', '.join(keywords[:4])}")

                lines.append("")

        # === CONTEXTE FICHIER ===
        if 'contexts' in context_data and 'local' in context_data['contexts']:
            local_ctx = context_data['contexts']['local']

            file_context = local_ctx.get('file_context', {})
            if file_context and file_context.get('other_entities'):
                lines.append("📁 AUTRES ENTITÉS DU MÊME FICHIER")
                lines.append("-" * 40)

                other_entities = file_context['other_entities']
                total = file_context.get('total_entities', len(other_entities))

                for entity in other_entities[:6]:
                    lines.append(f"• {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
                    if entity.get('lines'):
                        lines.append(f"  └─ Lignes: {entity['lines']}")

                if len(other_entities) > 6:
                    lines.append(f"  ... et {total - 6} autres entités")

                lines.append("")

        # === INSIGHTS CLÉS ===
        insights = context_data.get('key_insights', [])
        if insights:
            lines.append("💡 INSIGHTS CLÉS")
            lines.append("-" * 40)
            for insight in insights:
                lines.append(f"• {insight}")
            lines.append("")

        # === RÉSUMÉ FINAL ===
        lines.append("📊 RÉSUMÉ")
        lines.append("-" * 40)

        if 'summary' in context_data:
            summary = context_data['summary']

            entity_overview = summary.get('entity_overview', {})
            if entity_overview:
                lines.append(f"Type d'entité: {entity_overview.get('type', 'N/A')}")

            complexity = summary.get('complexity_indicators', {})
            if complexity:
                level = complexity.get('complexity_level', 'unknown')
                calls = complexity.get('function_calls', 0)
                deps = complexity.get('dependencies', 0)
                lines.append(f"Complexité: {level} ({calls} appels, {deps} dépendances)")

            arch_role = summary.get('architectural_role', '')
            if arch_role:
                lines.append(f"Rôle architectural: {arch_role}")

        # Informations de génération
        generation_info = context_data.get('generation_info', {})
        if generation_info:
            total_tokens = generation_info.get('total_tokens', 0)
            contexts_gen = generation_info.get('contexts_generated', [])
            lines.append(f"Contextes générés: {', '.join(contexts_gen)}")
            lines.append(f"Tokens utilisés: {total_tokens}")

        lines.append("")
        lines.append("=" * 80)

        return '\n'.join(lines)

    async def _get_chunk_content_summary(self, chunk_id: str, concept_name: str) -> Optional[Dict[str, Any]]:
        """Récupère un résumé du contenu d'un chunk."""

        try:
            chunk = await self._get_chunk_by_id(chunk_id)
            if not chunk:
                return None

            content = chunk.get('text', '')
            metadata = chunk.get('metadata', {})

            # Créer un résumé du contenu
            lines = content.split('\n')
            summary_lines = []

            for line in lines[:3]:  # Prendre les 3 premières lignes significatives
                line = line.strip()
                if line and not line.startswith('!'):  # Skip commentaires
                    summary_lines.append(line)

            summary = '; '.join(summary_lines) if summary_lines else 'Contenu non disponible'
            if len(summary) > 100:
                summary = summary[:97] + "..."

            return {
                'start_line': metadata.get('start_pos', 'N/A'),
                'end_line': metadata.get('end_pos', 'N/A'),
                'summary': summary,
                'size': len(content)
            }

        except Exception as e:
            return None

    async def _get_detailed_chunk_content(self, chunk_id: str, concept_name: str) -> Optional[Dict[str, Any]]:
        """Récupère le contenu détaillé d'un chunk."""

        try:
            chunk = await self._get_chunk_by_id(chunk_id)
            if not chunk:
                return None

            content = chunk.get('text', '')
            metadata = chunk.get('metadata', {})

            return {
                'content': content,
                'start_line': metadata.get('start_pos', 1),
                'end_line': metadata.get('end_pos', len(content.split('\n'))),
                'size': len(content),
                'chunk_id': chunk_id
            }

        except Exception as e:
            return None

    def _find_concept_relevant_lines(self, content: str, concept_name: str) -> List[str]:
        """Trouve les lignes les plus pertinentes pour un concept dans le contenu."""

        lines = content.split('\n')
        relevant_lines = []

        # Patterns à chercher liés au concept
        concept_patterns = [
            concept_name.lower(),
            'molecular',
            'dynamics',
            'simulation',
            'particle',
            'force',
            'energy',
            'integration',
            'verlet'
        ]

        for i, line in enumerate(lines):
            line_clean = line.strip().lower()

            # Skip commentaires vides
            if not line_clean or line_clean.startswith('!'):
                continue

            # Chercher les patterns du concept
            for pattern in concept_patterns:
                if pattern in line_clean:
                    # Prendre la ligne originale (avec casse)
                    original_line = line.strip()
                    if len(original_line) > 80:
                        original_line = original_line[:77] + "..."

                    relevant_lines.append(f"L{i + 1}: {original_line}")
                    break

        return relevant_lines

    async def _add_concept_entity_relations(self, lines: List[str], entity_name: str):
        """Ajoute les relations pour une entité dans un contexte de concept."""

        # Simplifier pour ne pas surcharger
        await self.context_provider._ensure_entity_groups()

        # Qui appelle cette entité
        callers = self.context_provider.find_entity_callers(entity_name)
        if callers:
            lines.append("   📞 Appelée par:")
            for caller in callers[:3]:  # Max 3
                lines.append(f"      • {caller['name']} ({caller['type']})")

        # Ce que cette entité appelle
        calls = self.context_provider.find_entity_calls(entity_name)
        if calls:
            lines.append("   🔗 Appelle:")
            for call in calls[:3]:  # Max 3
                lines.append(f"      • {call}")

    async def _analyze_concept_patterns(self, entities: List[Dict], concept_name: str) -> List[str]:
        """Analyse les patterns communs entre les entités d'un concept."""

        patterns = []

        # Types d'entités
        types = [e.get('type', e.get('entity_type', '')) for e in entities]
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1

        most_common_type = max(type_counts.items(), key=lambda x: x[1]) if type_counts else None
        if most_common_type and most_common_type[1] > 1:
            patterns.append(f"Principalement des {most_common_type[0]}s ({most_common_type[1]}/{len(entities)})")

        # Fichiers
        files = [e.get('filepath', '').split('/')[-1] for e in entities if e.get('filepath')]
        unique_files = len(set(files))
        if unique_files == 1:
            patterns.append(f"Concentré dans un seul fichier: {files[0]}")
        elif unique_files > 1:
            patterns.append(f"Réparti sur {unique_files} fichiers différents")

        # Confiance
        confidences = [e.get('confidence', 0) for e in entities if e.get('confidence', 0) > 0]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            patterns.append(f"Confiance moyenne: {avg_conf:.3f}")

        return patterns

    def _analyze_concept_file_distribution(self, entities: List[Dict]) -> Dict[str, int]:
        """Analyse la distribution des entités par fichier."""

        distribution = {}
        for entity in entities:
            filepath = entity.get('filepath', '')
            filename = filepath.split('/')[-1] if filepath else 'Unknown'
            distribution[filename] = distribution.get(filename, 0) + 1

        # Trier par nombre d'entités
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    async def _get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Récupère un chunk par son ID (méthode helper)."""

        try:
            parts = chunk_id.split('-chunk-')
            if len(parts) != 2:
                return None

            document_id = parts[0]

            document_store = self.context_provider.document_store
            await document_store.load_document_chunks(document_id)
            chunks = await document_store.get_document_chunks(document_id)

            if chunks:
                for chunk in chunks:
                    if chunk['id'] == chunk_id:
                        return chunk

            return None

        except Exception as e:
            return None

    async def _add_callers_section_for_concept(self, lines: List[str], entity_name: str):
        """Version simplifiée pour les concepts."""

        await self.context_provider._ensure_entity_groups()
        callers = self.context_provider.find_entity_callers(entity_name)

        if callers:
            lines.append("📞 APPELÉE PAR:")
            for caller in callers[:3]:  # Top 3 seulement
                lines.append(f"   • {caller['name']} ({caller['type']})")
            lines.append("")

    async def _add_callers_section(self, lines: List[str], entity_name: str):
        """Ajoute la section des appelants (logique visualiseur)."""

        # S'assurer que les groupes sont créés
        await self.context_provider._ensure_entity_groups()

        # Utiliser la méthode qui utilise la même logique que le visualiseur
        callers = self.context_provider.find_entity_callers(entity_name)

        if callers:
            lines.append("📞 APPELÉE PAR")
            lines.append("-" * 40)

            for caller in callers[:8]:  # Limiter à 8 appelants
                caller_name = caller['name']
                caller_type = caller['type']
                caller_file = caller['file'].split('/')[-1] if caller['file'] else 'N/A'

                lines.append(f"• {caller_name} ({caller_type})")
                lines.append(f"  └─ Fichier: {caller_file}")

            if len(callers) > 8:
                lines.append(f"  ... et {len(callers) - 8} autres appelants")

            lines.append("")
        else:
            lines.append("📞 APPELÉE PAR")
            lines.append("-" * 40)
            lines.append("• Aucun appelant détecté")
            lines.append("")

    async def _format_summary_context(self, context_data: Dict[str, Any],
                                      resolved_entity: Dict[str, Any]) -> str:
        """Formate un contexte résumé pour LLM."""

        lines = []
        entity_name = resolved_entity['name']
        entity_type = resolved_entity['type']

        lines.append(f"📋 RÉSUMÉ: {entity_name} ({entity_type})")
        lines.append("=" * 60)

        # Définition courte
        if 'contexts' in context_data and 'local' in context_data['contexts']:
            local_ctx = context_data['contexts']['local']
            main_def = local_ctx.get('main_definition', {})

            signature = main_def.get('signature', '')
            if signature and signature != "Signature not found":
                lines.append(f"Signature: {signature}")

            location = main_def.get('location', {})
            if location.get('file'):
                file_name = location['file'].split('/')[-1]
                lines.append(f"Fichier: {file_name} (lignes {location.get('lines', 'N/A')})")

        # Dépendances principales
        deps_count = 0
        calls_count = 0

        if 'contexts' in context_data and 'local' in context_data['contexts']:
            local_ctx = context_data['contexts']['local']
            deps_count = len(local_ctx.get('immediate_dependencies', []))
            calls_count = len(local_ctx.get('called_functions', []))

        if deps_count > 0 or calls_count > 0:
            lines.append(f"Dépendances: {deps_count} modules, {calls_count} fonctions appelées")

        # Impact
        if 'contexts' in context_data and 'global' in context_data['contexts']:
            global_ctx = context_data['contexts']['global']
            impact = global_ctx.get('impact_analysis', {})
            if impact:
                risk = impact.get('risk_level', 'unknown')
                lines.append(f"Impact: niveau {risk}")

        # Patterns
        if 'contexts' in context_data and 'semantic' in context_data['contexts']:
            semantic_ctx = context_data['contexts']['semantic']
            patterns = semantic_ctx.get('algorithmic_patterns', [])
            if patterns:
                pattern_name = patterns[0].get('pattern', '')
                lines.append(f"Pattern principal: {pattern_name}")

        # Insights clés
        insights = context_data.get('key_insights', [])
        if insights:
            lines.append("Insights: " + "; ".join(insights[:3]))

        return '\n'.join(lines)

    async def _format_bullet_context(self, context_data: Dict[str, Any],
                                     resolved_entity: Dict[str, Any]) -> str:
        """Formate un contexte en points pour LLM."""

        lines = []
        entity_name = resolved_entity['name']
        entity_type = resolved_entity['type']

        lines.append(f"• ENTITÉ: {entity_name} ({entity_type})")

        # Points principaux du contexte local
        if 'contexts' in context_data and 'local' in context_data['contexts']:
            local_ctx = context_data['contexts']['local']

            # Dépendances
            deps = local_ctx.get('immediate_dependencies', [])
            if deps:
                dep_names = [d.get('name', 'N/A') for d in deps[:3]]
                lines.append(f"• DÉPEND DE: {', '.join(dep_names)}")

            # Appels
            calls = local_ctx.get('called_functions', [])
            if calls:
                call_names = [c.get('name', 'N/A') for c in calls[:5]]
                lines.append(f"• APPELLE: {', '.join(call_names)}")

            # Parent/Enfants
            parent = local_ctx.get('parent_context')
            if parent:
                lines.append(f"• PARENT: {parent.get('name', 'N/A')}")

            children = local_ctx.get('children_context', [])
            if children:
                child_names = [c.get('name', 'N/A') for c in children[:3]]
                lines.append(f"• CONTIENT: {', '.join(child_names)}")

        # Points du contexte global
        if 'contexts' in context_data and 'global' in context_data['contexts']:
            global_ctx = context_data['contexts']['global']

            impact = global_ctx.get('impact_analysis', {})
            if impact:
                dependents = impact.get('direct_dependents', [])
                if dependents:
                    lines.append(f"• UTILISÉ PAR: {', '.join(dependents[:3])}")

                risk = impact.get('risk_level', '')
                if risk:
                    lines.append(f"• RISQUE MODIFICATION: {risk}")

        # Points sémantiques
        if 'contexts' in context_data and 'semantic' in context_data['contexts']:
            semantic_ctx = context_data['contexts']['semantic']

            similar = semantic_ctx.get('similar_entities', [])
            if similar:
                similar_names = [s.get('name', 'N/A') for s in similar[:3]]
                lines.append(f"• SIMILAIRE À: {', '.join(similar_names)}")

            patterns = semantic_ctx.get('algorithmic_patterns', [])
            if patterns:
                pattern_names = [p.get('pattern', 'N/A') for p in patterns[:2]]
                lines.append(f"• PATTERNS: {', '.join(pattern_names)}")

        # Insights
        insights = context_data.get('key_insights', [])
        for insight in insights[:3]:
            lines.append(f"• INSIGHT: {insight}")

        return '\n'.join(lines)

    async def _generate_not_found_text(self, element_name: str) -> str:
        """Génère un message d'erreur informatif quand l'élément n'est pas trouvé."""

        lines = [
            "❌ ÉLÉMENT NON TROUVÉ",
            "=" * 50,
            f"Élément recherché: '{element_name}'",
            "",
            "💡 Suggestions:"
        ]

        # Essayer de trouver des suggestions
        search_results = await self.context_provider.search_entities(element_name)
        if search_results:
            lines.append("Entités similaires trouvées:")
            for result in search_results[:5]:
                lines.append(f"  • {result['name']} ({result['type']}) - {result['file'].split('/')[-1]}")
        else:
            lines.append("  • Vérifiez l'orthographe")
            lines.append("  • Essayez avec un nom partiel")
            lines.append("  • Utilisez le nom du fichier")

        # Statistiques disponibles
        stats = self.entity_index.get_stats()
        lines.extend([
            "",
            f"📊 Base de données disponible:",
            f"  • {stats['total_entities']} entités totales",
            f"  • {stats['modules']} modules",
            f"  • {stats['functions']} fonctions",
            f"  • {stats['subroutines']} subroutines"
        ])

        return '\n'.join(lines)

    # === MÉTHODES DE CONVENANCE ===

    async def get_quick_context(self, element_name: str) -> str:
        """Contexte rapide en format bullet points."""
        return await self.get_contextual_text(
            element_name,
            context_type="local",
            format_style="bullet_points",
            max_tokens=1000
        )

    async def get_full_context(self, element_name: str) -> str:
        """Contexte complet détaillé."""
        return await self.get_contextual_text(
            element_name,
            context_type="complete",
            format_style="detailed",
            max_tokens=6000
        )

    async def get_dependency_context(self, element_name: str) -> str:
        """Contexte focalisé sur les dépendances."""
        return await self.get_contextual_text(
            element_name,
            context_type="global",
            agent_perspective="analyzer",
            task_context="dependency_analysis",
            format_style="detailed",
            max_tokens=3000
        )

    async def get_semantic_context_text(self, element_name: str) -> str:
        """Contexte sémantique en texte."""
        return await self.get_contextual_text(
            element_name,
            context_type="semantic",
            format_style="detailed",
            max_tokens=2500
        )