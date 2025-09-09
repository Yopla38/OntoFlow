# output/report_generator.py
"""
Générateur de rapports unifiés pour l'analyse Fortran.
Combine tous les types d'analyses en rapports complets et exportables.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from ..providers.smart_orchestrator import SmartContextOrchestrator, get_smart_orchestrator
from ..core.entity_manager import EntityManager, get_entity_manager
from ..core.fortran_analyzer import get_fortran_analyzer
from ..core.concept_detector import get_concept_detector
from .text_generator import ContextualTextGenerator
from .graph_visualizer import FortranDependencyVisualizer

logger = logging.getLogger(__name__)


class FortranAnalysisReport:
    """Classe représentant un rapport d'analyse complet"""

    def __init__(self):
        self.metadata = {}
        self.project_overview = {}
        self.entity_analysis = {}
        self.dependency_analysis = {}
        self.semantic_analysis = {}
        self.quality_metrics = {}
        self.recommendations = []
        self.generation_info = {}


class UnifiedReportGenerator:
    """
    Générateur de rapports unifiés combinant toutes les analyses.
    Utilise tous les composants des phases 1-3 pour des rapports complets.
    """

    def __init__(self, document_store, rag_engine):
        self.document_store = document_store
        self.rag_engine = rag_engine

        # Composants centraux
        self.orchestrator: Optional[SmartContextOrchestrator] = None
        self.entity_manager: Optional[EntityManager] = None
        self.analyzer = None
        self.concept_detector = None
        self.text_generator: Optional[ContextualTextGenerator] = None
        self.graph_visualizer: Optional[FortranDependencyVisualizer] = None

        self._initialized = False

    async def initialize(self):
        """Initialise tous les composants nécessaires"""
        if self._initialized:
            return

        logger.info("🔧 Initialisation du UnifiedReportGenerator...")

        # Composants centraux
        self.orchestrator = await get_smart_orchestrator(self.document_store, self.rag_engine)
        self.entity_manager = await get_entity_manager(self.document_store)
        self.analyzer = await get_fortran_analyzer(self.document_store, self.entity_manager)
        self.concept_detector = get_concept_detector(getattr(self.rag_engine, 'classifier', None))

        # Composants de sortie
        self.text_generator = ContextualTextGenerator(self.document_store, self.rag_engine)
        await self.text_generator.initialize()

        self.graph_visualizer = FortranDependencyVisualizer(self.document_store, self.rag_engine)
        await self.graph_visualizer.initialize()

        self._initialized = True
        logger.info("✅ UnifiedReportGenerator initialisé")

    async def generate_comprehensive_report(self,
                                            output_dir: str = "fortran_analysis_report",
                                            include_visualizations: bool = True,
                                            include_detailed_entities: bool = True,
                                            max_entities_analyzed: int = 50) -> str:
        """
        Génère un rapport complet d'analyse Fortran.

        Returns:
            Chemin vers le répertoire du rapport généré
        """
        await self.initialize()

        logger.info("📊 Génération du rapport complet d'analyse Fortran...")

        # Créer le répertoire de sortie
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Initialiser le rapport
        report = FortranAnalysisReport()

        # 1. Métadonnées du projet
        report.metadata = await self._collect_project_metadata()

        # 2. Vue d'ensemble du projet
        report.project_overview = await self._generate_project_overview()

        # 3. Analyse des entités
        if include_detailed_entities:
            report.entity_analysis = await self._analyze_key_entities(max_entities_analyzed)

        # 4. Analyse des dépendances
        report.dependency_analysis = await self._analyze_project_dependencies()

        # 5. Analyse sémantique
        report.semantic_analysis = await self._analyze_semantic_patterns()

        # 6. Métriques de qualité
        report.quality_metrics = await self._calculate_quality_metrics()

        # 7. Recommandations
        report.recommendations = await self._generate_project_recommendations(report)

        # 8. Informations de génération
        report.generation_info = {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '2.0',
            'components_used': ['EntityManager', 'FortranAnalyzer', 'SmartContextOrchestrator'],
            'total_entities_analyzed': len(self.entity_manager.entities),
            'analysis_duration': 'calculated_later'
        }

        # Générer les fichiers de sortie
        report_files = await self._generate_report_files(report, output_path, include_visualizations)

        logger.info(f"✅ Rapport complet généré dans {output_path}")
        logger.info(f"📁 Fichiers générés: {len(report_files)}")

        return str(output_path)

    async def _collect_project_metadata(self) -> Dict[str, Any]:
        """Collecte les métadonnées du projet"""
        stats = self.entity_manager.get_stats()
        cache_stats = self.orchestrator.get_cache_stats()

        # Analyser la structure des fichiers
        all_files = set()
        for entity in self.entity_manager.entities.values():
            if entity.filepath:
                all_files.add(entity.filepath)

        # Analyser les standards Fortran
        fortran_standards = await self._detect_project_fortran_standards()

        return {
            'project_name': self._extract_project_name(),
            'total_files': len(all_files),
            'total_entities': stats['total_entities'],
            'entity_distribution': stats.get('entity_types', {}),
            'fortran_standards': fortran_standards,
            'indexed_files': list(all_files)[:10],  # Échantillon
            'cache_performance': {
                'entities_cache_hits': cache_stats.get('entities', {}).get('hits', 0),
                'total_cache_entries': sum(
                    cache.get('entries', 0) for cache in cache_stats.values()
                ),
            },
            'quality_indicators': {
                'grouped_entities_ratio': stats['grouped_entities'] / max(1, stats['total_entities']),
                'incomplete_entities_ratio': stats['incomplete_entities'] / max(1, stats['total_entities']),
                'compression_ratio': stats.get('compression_ratio', 1.0)
            }
        }

    def _extract_project_name(self) -> str:
        """Extrait ou génère un nom de projet"""
        # Essayer d'extraire depuis les chemins de fichiers
        all_files = [entity.filepath for entity in self.entity_manager.entities.values() if entity.filepath]

        if all_files:
            # Trouver le répertoire commun
            common_parts = os.path.commonpath(all_files) if len(all_files) > 1 else os.path.dirname(all_files[0])
            project_name = os.path.basename(common_parts) or "fortran_project"
        else:
            project_name = "fortran_analysis"

        return project_name

    async def _detect_project_fortran_standards(self) -> Dict[str, int]:
        """Détecte les standards Fortran utilisés dans le projet"""
        standards_count = {}

        # Échantillonner quelques entités pour détecter les standards
        sample_entities = list(self.entity_manager.entities.values())[:20]

        for entity in sample_entities:
            if entity.chunks:
                # Récupérer le code de l'entité
                chunk_info = entity.chunks[0]
                try:
                    chunk = await self.entity_manager.chunk_access.get_chunk_by_id(chunk_info['chunk_id'])
                    if chunk:
                        # Utiliser le détecteur de standard du text_processor
                        from ..utils.fortran_patterns import FortranTextProcessor
                        processor = FortranTextProcessor()
                        standard = processor.detect_fortran_standard(chunk['text'])
                        standards_count[standard] = standards_count.get(standard, 0) + 1
                except Exception:
                    continue

        return standards_count

    async def _generate_project_overview(self) -> Dict[str, Any]:
        """Génère une vue d'ensemble du projet"""
        # Utiliser GlobalContextProvider pour une vue d'ensemble
        global_context = await self.orchestrator.get_global_context("", max_tokens=2000)

        overview = {
            'project_statistics': global_context.get('project_overview', {}),
            'module_hierarchy': global_context.get('module_hierarchy', {}),
            'architectural_style': '',
            'main_modules': [],
            'circular_dependencies': [],
            'quality_summary': {}
        }

        # Extraire les informations clés
        project_stats = overview['project_statistics']
        if 'statistics' in project_stats:
            overview['project_statistics'] = project_stats['statistics']

        if 'architectural_style' in project_stats:
            overview['architectural_style'] = project_stats['architectural_style']

        if 'main_modules' in project_stats:
            overview['main_modules'] = project_stats['main_modules']

        # Dépendances circulaires
        hierarchy = overview['module_hierarchy']
        if 'circular_dependencies' in hierarchy:
            overview['circular_dependencies'] = hierarchy['circular_dependencies']

        return overview

    async def _analyze_key_entities(self, max_entities: int) -> Dict[str, Any]:
        """Analyse détaillée des entités clés"""
        # Sélectionner les entités les plus importantes
        all_entities = list(self.entity_manager.entities.values())

        # Trier par importance (complexité, dépendances, etc.)
        sorted_entities = sorted(
            all_entities,
            key=lambda x: (
                len(x.dependencies) + len(x.called_functions),  # Complexité
                len(x.concepts),  # Richesse sémantique
                -x.confidence  # Confiance (ordre décroissant)
            ),
            reverse=True
        )

        key_entities = sorted_entities[:max_entities]

        analysis = {
            'total_analyzed': len(key_entities),
            'entities': {}
        }

        # Analyser chaque entité clé
        for entity in key_entities:
            entity_analysis = await self._analyze_single_entity(entity)
            analysis['entities'][entity.entity_name] = entity_analysis

        return analysis

    async def _analyze_single_entity(self, entity) -> Dict[str, Any]:
        """Analyse détaillée d'une entité unique"""
        entity_analysis = {
            'basic_info': {
                'name': entity.entity_name,
                'type': entity.entity_type,
                'file': entity.filename,
                'filepath': entity.filepath,
                'is_grouped': entity.is_grouped,
                'is_complete': entity.is_complete,
                'confidence': entity.confidence
            },
            'complexity_analysis': {},
            'dependencies': {
                'uses': list(entity.dependencies),
                'calls': list(entity.called_functions)
            },
            'semantic_analysis': {
                'concepts': list(entity.concepts),
                'detected_concepts': entity.detected_concepts[:5]  # Top 5
            },
            'impact_analysis': {},
            'quality_indicators': {}
        }

        # Analyse de complexité
        complexity_score = (
                len(entity.dependencies) * 2 +
                len(entity.called_functions) * 1 +
                len(entity.concepts) * 0.5
        )

        entity_analysis['complexity_analysis'] = {
            'score': complexity_score,
            'level': 'high' if complexity_score > 10 else 'medium' if complexity_score > 5 else 'low',
            'factors': {
                'dependencies_count': len(entity.dependencies),
                'function_calls_count': len(entity.called_functions),
                'concepts_count': len(entity.concepts),
                'chunks_count': len(entity.chunks)
            }
        }

        # Analyse d'impact utilisant FortranAnalyzer
        try:
            impact_data = await self.analyzer.analyze_dependencies(entity.entity_name)
            if 'error' not in impact_data:
                entity_analysis['impact_analysis'] = impact_data.get('impact_analysis', {})
        except Exception as e:
            logger.debug(f"Impact analysis failed for {entity.entity_name}: {e}")
            entity_analysis['impact_analysis'] = {'error': str(e)}

        # Indicateurs de qualité
        entity_analysis['quality_indicators'] = {
            'has_documentation': bool(entity.signature and entity.signature != "Signature not found"),
            'has_concepts': len(entity.concepts) > 0,
            'well_structured': entity.is_complete and not entity.is_grouped,
            'high_confidence': entity.confidence > 0.8
        }

        return entity_analysis

    async def _analyze_project_dependencies(self) -> Dict[str, Any]:
        """Analyse globale des dépendances du projet"""
        dependency_analysis = {
            'global_statistics': {},
            'module_dependencies': {},
            'circular_dependencies': [],
            'dependency_graph_metrics': {},
            'critical_entities': []
        }

        # Statistiques globales
        all_entities = list(self.entity_manager.entities.values())
        total_dependencies = sum(len(entity.dependencies) for entity in all_entities)
        total_calls = sum(len(entity.called_functions) for entity in all_entities)

        dependency_analysis['global_statistics'] = {
            'total_use_statements': total_dependencies,
            'total_function_calls': total_calls,
            'avg_dependencies_per_entity': total_dependencies / len(all_entities) if all_entities else 0,
            'avg_calls_per_entity': total_calls / len(all_entities) if all_entities else 0,
            'highly_coupled_entities': len(
                [e for e in all_entities if len(e.dependencies) + len(e.called_functions) > 10])
        }

        # Analyse des modules
        modules = await self.entity_manager.get_entities_by_type('module')
        dependency_analysis['module_dependencies'] = {}

        for module in modules[:10]:  # Top 10 modules
            dependency_analysis['module_dependencies'][module.entity_name] = {
                'dependencies': list(module.dependencies),
                'dependents': await self._find_module_dependents(module.entity_name),
                'internal_entities': len(await self.entity_manager.get_children(module.entity_id)),
                'complexity_score': len(module.dependencies) + len(module.called_functions)
            }

        # Entités critiques (forte connectivité)
        critical_entities = []
        for entity in all_entities:
            # Calculer la centralité (entrants + sortants)
            dependents = await self.entity_manager.find_entity_callers(entity.entity_name)
            centrality = len(entity.dependencies) + len(entity.called_functions) + len(dependents)

            if centrality > 8:  # Seuil de criticité
                critical_entities.append({
                    'name': entity.entity_name,
                    'type': entity.entity_type,
                    'centrality_score': centrality,
                    'dependencies': len(entity.dependencies),
                    'calls': len(entity.called_functions),
                    'dependents': len(dependents)
                })

        # Trier par criticité
        critical_entities.sort(key=lambda x: x['centrality_score'], reverse=True)
        dependency_analysis['critical_entities'] = critical_entities[:15]

        return dependency_analysis

    async def _find_module_dependents(self, module_name: str) -> List[str]:
        """Trouve les entités qui dépendent d'un module"""
        dependents = []
        for entity in self.entity_manager.entities.values():
            if module_name in entity.dependencies:
                dependents.append(entity.entity_name)
        return dependents

    async def _analyze_semantic_patterns(self) -> Dict[str, Any]:
        """Analyse des patterns sémantiques du projet"""
        semantic_analysis = {
            'concept_distribution': {},
            'algorithmic_patterns': {},
            'concept_clusters': {},
            'semantic_quality': {}
        }

        # Distribution des concepts
        all_concepts = []
        concept_confidence = []

        for entity in self.entity_manager.entities.values():
            all_concepts.extend(entity.concepts)
            for concept_data in entity.detected_concepts:
                if isinstance(concept_data, dict):
                    concept_confidence.append(concept_data.get('confidence', 0))

        # Compter les concepts
        from collections import Counter
        concept_counts = Counter(all_concepts)

        semantic_analysis['concept_distribution'] = {
            'total_concepts': len(all_concepts),
            'unique_concepts': len(concept_counts),
            'most_common_concepts': dict(concept_counts.most_common(10)),
            'avg_confidence': sum(concept_confidence) / len(concept_confidence) if concept_confidence else 0
        }

        # Analyser les patterns algorithmiques sur un échantillon
        pattern_analysis = {}
        sample_entities = list(self.entity_manager.entities.values())[:20]

        for entity in sample_entities:
            if entity.entity_type in ['function', 'subroutine']:
                try:
                    patterns = await self.analyzer.detect_algorithmic_patterns(entity.entity_name)
                    if 'error' not in patterns:
                        for pattern in patterns.get('detected_patterns', []):
                            pattern_name = pattern.get('pattern', '')
                            if pattern_name:
                                if pattern_name not in pattern_analysis:
                                    pattern_analysis[pattern_name] = []
                                pattern_analysis[pattern_name].append({
                                    'entity': entity.entity_name,
                                    'confidence': pattern.get('confidence', 0)
                                })
                except Exception:
                    continue

        semantic_analysis['algorithmic_patterns'] = pattern_analysis

        return semantic_analysis

    async def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques de qualité du code"""
        all_entities = list(self.entity_manager.entities.values())

        quality_metrics = {
            'completeness': {},
            'consistency': {},
            'complexity': {},
            'documentation': {},
            'overall_score': 0
        }

        # Métriques de complétude
        complete_entities = sum(1 for e in all_entities if e.is_complete)
        grouped_entities = sum(1 for e in all_entities if e.is_grouped)

        quality_metrics['completeness'] = {
            'complete_entities_ratio': complete_entities / len(all_entities) if all_entities else 0,
            'grouped_entities_ratio': grouped_entities / len(all_entities) if all_entities else 0,
            'average_confidence': sum(e.confidence for e in all_entities) / len(all_entities) if all_entities else 0
        }

        # Métriques de consistance
        entities_with_concepts = sum(1 for e in all_entities if e.concepts)
        entities_with_signature = sum(1 for e in all_entities if e.signature and e.signature != "Signature not found")

        quality_metrics['consistency'] = {
            'entities_with_concepts_ratio': entities_with_concepts / len(all_entities) if all_entities else 0,
            'entities_with_signature_ratio': entities_with_signature / len(all_entities) if all_entities else 0
        }

        # Métriques de complexité
        complexity_scores = []
        for entity in all_entities:
            score = len(entity.dependencies) + len(entity.called_functions) + len(entity.concepts)
            complexity_scores.append(score)

        quality_metrics['complexity'] = {
            'average_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'high_complexity_entities': len([s for s in complexity_scores if s > 10]),
            'complexity_distribution': {
                'low': len([s for s in complexity_scores if s <= 3]),
                'medium': len([s for s in complexity_scores if 3 < s <= 10]),
                'high': len([s for s in complexity_scores if s > 10])
            }
        }

        # Score global (0-100)
        completeness_score = quality_metrics['completeness']['complete_entities_ratio'] * 30
        consistency_score = quality_metrics['consistency']['entities_with_concepts_ratio'] * 25
        confidence_score = quality_metrics['completeness']['average_confidence'] * 25
        complexity_penalty = min(quality_metrics['complexity']['high_complexity_entities'] / len(all_entities),
                                 0.2) * 20

        overall_score = completeness_score + consistency_score + confidence_score - complexity_penalty
        quality_metrics['overall_score'] = max(0, min(100, overall_score))

        return quality_metrics

    async def _generate_project_recommendations(self, report: FortranAnalysisReport) -> List[Dict[str, Any]]:
        """Génère des recommandations basées sur l'analyse complète"""
        recommendations = []

        # Recommandations basées sur la qualité
        quality_score = report.quality_metrics.get('overall_score', 0)

        if quality_score < 70:
            recommendations.append({
                'type': 'quality',
                'priority': 'high',
                'title': 'Améliorer la qualité globale du code',
                'description': f'Score de qualité actuel: {quality_score:.1f}/100. Focus sur la documentation et la complétude.',
                'actions': [
                    'Compléter les signatures manquantes',
                    'Ajouter des commentaires de documentation',
                    'Résoudre les entités incomplètes'
                ]
            })

        # Recommandations basées sur la complexité
        complexity_metrics = report.quality_metrics.get('complexity', {})
        high_complexity_count = complexity_metrics.get('high_complexity_entities', 0)

        if high_complexity_count > len(self.entity_manager.entities) * 0.2:
            recommendations.append({
                'type': 'complexity',
                'priority': 'medium',
                'title': 'Réduire la complexité du code',
                'description': f'{high_complexity_count} entités ont une complexité élevée.',
                'actions': [
                    'Décomposer les fonctions complexes',
                    'Réduire les dépendances croisées',
                    'Introduire des couches d\'abstraction'
                ]
            })

        # Recommandations basées sur les dépendances circulaires
        circular_deps = report.project_overview.get('circular_dependencies', [])
        if circular_deps:
            recommendations.append({
                'type': 'architecture',
                'priority': 'high',
                'title': 'Résoudre les dépendances circulaires',
                'description': f'{len(circular_deps)} cycles de dépendances détectés.',
                'actions': [
                    'Réorganiser la structure des modules',
                    'Introduire des interfaces',
                    'Factoriser le code commun'
                ]
            })

        # Recommandations basées sur les concepts sémantiques
        concept_distribution = report.semantic_analysis.get('concept_distribution', {})
        avg_confidence = concept_distribution.get('avg_confidence', 0)

        if avg_confidence < 0.6:
            recommendations.append({
                'type': 'documentation',
                'priority': 'medium',
                'title': 'Améliorer la clarté conceptuelle',
                'description': f'Confiance moyenne des concepts: {avg_confidence:.3f}. Le code pourrait être plus expressif.',
                'actions': [
                    'Améliorer les noms des fonctions et variables',
                    'Ajouter des commentaires explicatifs',
                    'Structurer le code selon les domaines métier'
                ]
            })

        # Recommandations de performance
        total_entities = report.metadata.get('total_entities', 0)
        if total_entities > 500:
            recommendations.append({
                'type': 'performance',
                'priority': 'low',
                'title': 'Optimiser pour les gros projets',
                'description': f'Projet volumineux ({total_entities} entités). Considérer des optimisations.',
                'actions': [
                    'Implémenter la compilation séparée',
                    'Optimiser les interfaces de modules',
                    'Considérer la parallélisation'
                ]
            })

        return recommendations

    async def _generate_report_files(self,
                                     report: FortranAnalysisReport,
                                     output_path: Path,
                                     include_visualizations: bool) -> List[str]:
        """Génère tous les fichiers du rapport"""
        generated_files = []

        # 1. Rapport HTML principal
        main_report_file = output_path / "index.html"
        await self._generate_html_report(report, main_report_file)
        generated_files.append(str(main_report_file))

        # 2. Rapport JSON (données brutes)
        json_report_file = output_path / "analysis_data.json"
        await self._generate_json_report(report, json_report_file)
        generated_files.append(str(json_report_file))

        # 3. Rapport texte détaillé
        text_report_file = output_path / "detailed_analysis.txt"
        await self._generate_text_report(report, text_report_file)
        generated_files.append(str(text_report_file))

        # 4. Visualisations (si demandées)
        if include_visualizations:
            viz_files = await self._generate_visualizations(output_path)
            generated_files.extend(viz_files)

        # 5. Fichiers d'analyse par entité (échantillon)
        entity_reports_dir = output_path / "entity_reports"
        entity_reports_dir.mkdir(exist_ok=True)

        # Générer des rapports détaillés pour les entités clés
        key_entities = list(report.entity_analysis.get('entities', {}).keys())[:10]
        for entity_name in key_entities:
            entity_file = entity_reports_dir / f"{entity_name}_analysis.txt"
            await self._generate_entity_report(entity_name, entity_file)
            generated_files.append(str(entity_file))

        return generated_files

    async def _generate_html_report(self, report: FortranAnalysisReport, output_file: Path):
        """Génère le rapport HTML principal"""
        html_content = self._create_html_report_template(report)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _create_html_report_template(self, report: FortranAnalysisReport) -> str:
        """Crée le template HTML pour le rapport"""
        metadata = report.metadata
        overview = report.project_overview
        quality = report.quality_metrics
        recommendations = report.recommendations

        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport d'Analyse Fortran - {metadata.get('project_name', 'Projet')}</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f8f9fa;
                    line-height: 1.6;
                }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    margin-bottom: 30px;
                    text-align: center;
                }}
                .section {{ 
                    background: white; 
                    border-radius: 10px; 
                    padding: 25px; 
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .section h2 {{ 
                    color: #2c3e50; 
                    border-bottom: 2px solid #4285F4;
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                .stats-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 20px; 
                    margin-bottom: 30px; 
                }}
                .stat-card {{ 
                    background: white; 
                    border-radius: 10px; 
                    padding: 20px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    border-left: 4px solid #4285F4;
                    text-align: center;
                }}
                .stat-value {{ 
                    font-size: 32px; 
                    font-weight: bold; 
                    color: #4285F4; 
                    margin-bottom: 5px;
                }}
                .stat-label {{ 
                    color: #666; 
                    font-size: 14px; 
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                .quality-score {{
                    font-size: 48px;
                    font-weight: bold;
                    text-align: center;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 20px 0;
                }}
                .score-excellent {{ background: #d4edda; color: #155724; }}
                .score-good {{ background: #d1ecf1; color: #0c5460; }}
                .score-fair {{ background: #fff3cd; color: #856404; }}
                .score-poor {{ background: #f8d7da; color: #721c24; }}
                .recommendation {{ 
                    background: #f8f9fa; 
                    border-left: 4px solid #ffc107; 
                    padding: 15px; 
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .recommendation.high {{ border-left-color: #dc3545; }}
                .recommendation.medium {{ border-left-color: #ffc107; }}
                .recommendation.low {{ border-left-color: #28a745; }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin-top: 15px; 
                }}
                th, td {{ 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd; 
                }}
                th {{ 
                    background-color: #f8f9fa; 
                    font-weight: 600;
                    color: #2c3e50;
                }}
                tr:hover {{ background-color: #f8f9fa; }}
                .badge {{ 
                    display: inline-block; 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 12px; 
                    font-weight: bold; 
                    color: white;
                }}
                .badge-high {{ background-color: #dc3545; }}
                .badge-medium {{ background-color: #ffc107; color: #212529; }}
                .badge-low {{ background-color: #28a745; }}
                .file-list {{ 
                    max-height: 200px; 
                    overflow-y: auto; 
                    background: #f8f9fa; 
                    padding: 10px; 
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Rapport d'Analyse Fortran</h1>
                    <p>Projet: {metadata.get('project_name', 'Fortran Project')}</p>
                    <p>Généré le: {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
                </div>

                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{metadata.get('total_entities', 0)}</div>
                        <div class="stat-label">Entités totales</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{metadata.get('total_files', 0)}</div>
                        <div class="stat-label">Fichiers analysés</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(overview.get('main_modules', []))}</div>
                        <div class="stat-label">Modules principaux</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{len(overview.get('circular_dependencies', []))}</div>
                        <div class="stat-label">Dépendances circulaires</div>
                    </div>
                </div>

                <div class="section">
                    <h2>🎯 Score de Qualité Global</h2>
                    {self._generate_quality_score_html(quality.get('overall_score', 0))}
                </div>

                <div class="section">
                    <h2>🏗️ Vue d'Ensemble du Projet</h2>
                    <h3>Statistiques du Projet</h3>
                    {self._generate_project_stats_html(overview)}

                    <h3>Modules Principaux</h3>
                    {self._generate_main_modules_html(overview.get('main_modules', []))}
                </div>

                <div class="section">
                    <h2>📊 Métriques de Qualité</h2>
                    {self._generate_quality_metrics_html(quality)}
                </div>

                <div class="section">
                    <h2>🔗 Analyse des Dépendances</h2>
                    {self._generate_dependency_analysis_html(report.dependency_analysis)}
                </div>

                <div class="section">
                    <h2>🧠 Analyse Sémantique</h2>
                    {self._generate_semantic_analysis_html(report.semantic_analysis)}
                </div>

                <div class="section">
                    <h2>💡 Recommandations</h2>
                    {self._generate_recommendations_html(recommendations)}
                </div>

                <div class="section">
                    <h2>📁 Fichiers du Projet</h2>
                    <div class="file-list">
                        {self._generate_file_list_html(metadata.get('indexed_files', []))}
                    </div>
                </div>

                <div class="section">
                    <h2>ℹ️ Informations Techniques</h2>
                    <p><strong>Générateur:</strong> UnifiedReportGenerator v2.0</p>
                    <p><strong>Composants utilisés:</strong> EntityManager, FortranAnalyzer, SmartContextOrchestrator</p>
                    <p><strong>Standards Fortran détectés:</strong> {', '.join(metadata.get('fortran_standards', {}).keys())}</p>
                    <p><strong>Cache performance:</strong> {metadata.get('cache_performance', {}).get('total_cache_entries', 0)} entrées en cache</p>
                </div>
            </div>

            <script>
                // Fonctions JavaScript pour l'interactivité
                function showEntityDetails(entityName) {{
                    alert('Détails de ' + entityName + ' - Fonctionnalité à implémenter');
                }}

                function filterRecommendations(priority) {{
                    const recommendations = document.querySelectorAll('.recommendation');
                    recommendations.forEach(rec => {{
                        if (priority === 'all' || rec.classList.contains(priority)) {{
                            rec.style.display = 'block';
                        }} else {{
                            rec.style.display = 'none';
                        }}
                    }});
                }}
            </script>
        </body>
        </html>
        """

        return html

    def _generate_quality_score_html(self, score: float) -> str:
        """Génère le HTML pour le score de qualité"""
        if score >= 80:
            css_class = "score-excellent"
            label = "Excellente"
        elif score >= 65:
            css_class = "score-good"
            label = "Bonne"
        elif score >= 50:
            css_class = "score-fair"
            label = "Correcte"
        else:
            css_class = "score-poor"
            label = "À améliorer"

        return f"""
        <div class="quality-score {css_class}">
            {score:.1f}/100
            <div style="font-size: 16px; margin-top: 10px;">Qualité {label}</div>
        </div>
        """

    def _generate_project_stats_html(self, overview: Dict[str, Any]) -> str:
        """Génère le HTML pour les statistiques du projet"""
        stats = overview.get('project_statistics', {})

        html = "<table>"
        html += "<tr><th>Métrique</th><th>Valeur</th></tr>"

        for key, value in stats.items():
            if isinstance(value, dict):
                continue
            html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"

        html += "</table>"
        return html

    def _generate_main_modules_html(self, main_modules: List[Dict[str, Any]]) -> str:
        """Génère le HTML pour les modules principaux"""
        if not main_modules:
            return "<p>Aucun module principal détecté.</p>"

        html = "<table>"
        html += "<tr><th>Module</th><th>Score</th><th>Enfants</th><th>Dépendances</th><th>Fichier</th></tr>"

        for module in main_modules:
            name = module.get('name', 'Unknown')
            score = module.get('score', 0)
            children = module.get('children_count', 0)
            deps = module.get('dependencies_count', 0)
            filepath = module.get('filepath', '').split('/')[-1]

            html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{score}</td>
                <td>{children}</td>
                <td>{deps}</td>
                <td>{filepath}</td>
            </tr>
            """

        html += "</table>"
        return html

    def _generate_quality_metrics_html(self, quality: Dict[str, Any]) -> str:
        """Génère le HTML pour les métriques de qualité"""
        html = "<h3>Métriques de Complétude</h3>"
        completeness = quality.get('completeness', {})

        html += "<table>"
        html += "<tr><th>Métrique</th><th>Valeur</th><th>Status</th></tr>"

        metrics = [
            ("Entités complètes", completeness.get('complete_entities_ratio', 0), "ratio"),
            ("Entités regroupées", completeness.get('grouped_entities_ratio', 0), "ratio"),
            ("Confiance moyenne", completeness.get('average_confidence', 0), "float")
        ]

        for name, value, value_type in metrics:
            if value_type == "ratio":
                display_value = f"{value:.1%}"
                status = "🟢" if value > 0.8 else "🟡" if value > 0.5 else "🔴"
            else:
                display_value = f"{value:.3f}"
                status = "🟢" if value > 0.8 else "🟡" if value > 0.5 else "🔴"

            html += f"<tr><td>{name}</td><td>{display_value}</td><td>{status}</td></tr>"

        html += "</table>"

        # Métriques de complexité
        complexity = quality.get('complexity', {})
        if complexity:
            html += "<h3>Distribution de Complexité</h3>"
            dist = complexity.get('complexity_distribution', {})

            html += "<table>"
            html += "<tr><th>Niveau</th><th>Nombre d'entités</th></tr>"
            for level, count in dist.items():
                html += f"<tr><td>{level.title()}</td><td>{count}</td></tr>"
            html += "</table>"

        return html

    def _generate_dependency_analysis_html(self, dep_analysis: Dict[str, Any]) -> str:
        """Génère le HTML pour l'analyse des dépendances"""
        html = "<h3>Statistiques Globales</h3>"
        stats = dep_analysis.get('global_statistics', {})

        html += "<table>"
        for key, value in stats.items():
            label = key.replace('_', ' ').title()
            if isinstance(value, float):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            html += f"<tr><td>{label}</td><td>{display_value}</td></tr>"
        html += "</table>"

        # Entités critiques
        critical = dep_analysis.get('critical_entities', [])
        if critical:
            html += "<h3>Entités Critiques (haute connectivité)</h3>"
            html += "<table>"
            html += "<tr><th>Entité</th><th>Type</th><th>Score Centralité</th><th>Dépendances</th><th>Appels</th></tr>"

            for entity in critical[:10]:
                html += f"""
                <tr>
                    <td><strong>{entity['name']}</strong></td>
                    <td>{entity['type']}</td>
                    <td>{entity['centrality_score']}</td>
                    <td>{entity['dependencies']}</td>
                    <td>{entity['calls']}</td>
                </tr>
                """

            html += "</table>"

        return html

    def _generate_semantic_analysis_html(self, semantic_analysis: Dict[str, Any]) -> str:
        """Génère le HTML pour l'analyse sémantique"""
        html = "<h3>Distribution des Concepts</h3>"
        concept_dist = semantic_analysis.get('concept_distribution', {})

        html += "<table>"
        html += "<tr><th>Métrique</th><th>Valeur</th></tr>"

        metrics = [
            ("Concepts totaux", concept_dist.get('total_concepts', 0)),
            ("Concepts uniques", concept_dist.get('unique_concepts', 0)),
            ("Confiance moyenne", f"{concept_dist.get('avg_confidence', 0):.3f}")
        ]

        for name, value in metrics:
            html += f"<tr><td>{name}</td><td>{value}</td></tr>"

        html += "</table>"

        # Concepts les plus fréquents
        most_common = concept_dist.get('most_common_concepts', {})
        if most_common:
            html += "<h3>Concepts les Plus Fréquents</h3>"
            html += "<table>"
            html += "<tr><th>Concept</th><th>Occurrences</th></tr>"

            for concept, count in list(most_common.items())[:10]:
                html += f"<tr><td>{concept}</td><td>{count}</td></tr>"

            html += "</table>"

        # Patterns algorithmiques
        patterns = semantic_analysis.get('algorithmic_patterns', {})
        if patterns:
            html += "<h3>Patterns Algorithmiques Détectés</h3>"
            html += "<table>"
            html += "<tr><th>Pattern</th><th>Entités</th><th>Confiance Moyenne</th></tr>"

            for pattern_name, entities in patterns.items():
                avg_confidence = sum(e['confidence'] for e in entities) / len(entities)
                entity_names = [e['entity'] for e in entities[:3]]
                entity_display = ', '.join(entity_names)
                if len(entities) > 3:
                    entity_display += f" (+{len(entities) - 3} autres)"

                html += f"""
                            <tr>
                                <td><strong>{pattern_name}</strong></td>
                                <td>{entity_display}</td>
                                <td>{avg_confidence:.3f}</td>
                            </tr>
                            """

            html += "</table>"

        return html

    def _generate_recommendations_html(self, recommendations: List[Dict[str, Any]]) -> str:
        """Génère le HTML pour les recommandations"""
        if not recommendations:
            return "<p>🎉 Aucune recommandation spécifique. Le projet semble en bon état !</p>"

        html = f"""
                <div style="margin-bottom: 20px;">
                    <button class="control-btn" onclick="filterRecommendations('all')">Toutes</button>
                    <button class="control-btn" onclick="filterRecommendations('high')">Haute priorité</button>
                    <button class="control-btn" onclick="filterRecommendations('medium')">Moyenne priorité</button>
                    <button class="control-btn" onclick="filterRecommendations('low')">Basse priorité</button>
                </div>
                """

        for rec in recommendations:
            priority = rec.get('priority', 'medium')
            rec_type = rec.get('type', 'general')
            title = rec.get('title', 'Recommandation')
            description = rec.get('description', '')
            actions = rec.get('actions', [])

            priority_emoji = {
                'high': '🔴',
                'medium': '🟡',
                'low': '🟢'
            }.get(priority, '⚪')

            html += f"""
                    <div class="recommendation {priority}">
                        <h4>{priority_emoji} {title}</h4>
                        <p><strong>Type:</strong> {rec_type.title()} | <strong>Priorité:</strong> {priority.title()}</p>
                        <p>{description}</p>
                    """

            if actions:
                html += "<p><strong>Actions suggérées:</strong></p><ul>"
                for action in actions:
                    html += f"<li>{action}</li>"
                html += "</ul>"

            html += "</div>"

        return html

    def _generate_file_list_html(self, files: List[str]) -> str:
        """Génère le HTML pour la liste des fichiers"""
        if not files:
            return "<p>Aucun fichier indexé.</p>"

        html = "<ul>"
        for filepath in files:
            filename = filepath.split('/')[-1]
            html += f"<li><strong>{filename}</strong> <small>({filepath})</small></li>"
        html += "</ul>"

        return html

    async def _generate_json_report(self, report: FortranAnalysisReport, output_file: Path):
        """Génère le rapport JSON avec toutes les données"""
        report_data = {
            'metadata': report.metadata,
            'project_overview': report.project_overview,
            'entity_analysis': report.entity_analysis,
            'dependency_analysis': report.dependency_analysis,
            'semantic_analysis': report.semantic_analysis,
            'quality_metrics': report.quality_metrics,
            'recommendations': report.recommendations,
            'generation_info': report.generation_info
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

    async def _generate_text_report(self, report: FortranAnalysisReport, output_file: Path):
        """Génère un rapport texte détaillé"""
        lines = [
            "=" * 80,
            "RAPPORT D'ANALYSE FORTRAN DÉTAILLÉ",
            "=" * 80,
            f"Projet: {report.metadata.get('project_name', 'Unknown')}",
            f"Généré le: {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
            f"Générateur: UnifiedReportGenerator v2.0",
            "",
            "📊 RÉSUMÉ EXÉCUTIF",
            "-" * 40,
            f"• Entités totales: {report.metadata.get('total_entities', 0)}",
            f"• Fichiers analysés: {report.metadata.get('total_files', 0)}",
            f"• Score de qualité: {report.quality_metrics.get('overall_score', 0):.1f}/100",
            f"• Modules principaux: {len(report.project_overview.get('main_modules', []))}",
            f"• Dépendances circulaires: {len(report.project_overview.get('circular_dependencies', []))}",
            "",
        ]

        # Ajouter l'analyse des entités
        entity_analysis = report.entity_analysis
        if entity_analysis and 'entities' in entity_analysis:
            lines.extend([
                "🔍 ANALYSE DES ENTITÉS CLÉS",
                "-" * 40,
                f"Entités analysées en détail: {entity_analysis['total_analyzed']}",
                ""
            ])

            for entity_name, analysis in list(entity_analysis['entities'].items())[:10]:
                basic_info = analysis['basic_info']
                complexity = analysis['complexity_analysis']

                lines.extend([
                    f"📌 {entity_name}",
                    f"   Type: {basic_info['type']}",
                    f"   Fichier: {basic_info['filename']}",
                    f"   Complexité: {complexity['level']} (score: {complexity['score']:.1f})",
                    f"   Regroupée: {'Oui' if basic_info['is_grouped'] else 'Non'}",
                    f"   Complète: {'Oui' if basic_info['is_complete'] else 'Non'}",
                    ""
                ])

        # Ajouter les recommandations
        if report.recommendations:
            lines.extend([
                "💡 RECOMMANDATIONS",
                "-" * 40
            ])

            for i, rec in enumerate(report.recommendations, 1):
                priority = rec.get('priority', 'medium')
                title = rec.get('title', 'Recommandation')
                description = rec.get('description', '')

                priority_symbol = {
                    'high': '🔴',
                    'medium': '🟡',
                    'low': '🟢'
                }.get(priority, '⚪')

                lines.extend([
                    f"{i}. {priority_symbol} {title} (Priorité: {priority})",
                    f"   {description}",
                    ""
                ])

                actions = rec.get('actions', [])
                if actions:
                    lines.append("   Actions suggérées:")
                    for action in actions:
                        lines.append(f"   • {action}")
                    lines.append("")

        lines.extend([
            "=" * 80,
            "Fin du rapport",
            "=" * 80
        ])

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    async def _generate_visualizations(self, output_path: Path) -> List[str]:
        """Génère les visualisations du projet"""
        viz_files = []

        try:
            # 1. Graphe de dépendances principal
            main_graph_file = output_path / "dependency_graph.html"
            await self.graph_visualizer.build_dependency_graph(
                max_entities=100,
                include_internal_functions=True
            )
            self.graph_visualizer.visualize_with_pyvis(
                output_file=str(main_graph_file),
                hierarchical=True
            )
            viz_files.append(str(main_graph_file))

            # 2. Graphe focalisé sur les modules principaux
            modules = await self.entity_manager.get_entities_by_type('module')
            if modules:
                main_module = modules[0].entity_name
                module_graph_file = output_path / f"module_{main_module}_focus.html"

                await self.graph_visualizer.build_dependency_graph(
                    focus_entity=main_module,
                    max_depth=2
                )
                self.graph_visualizer.visualize_with_pyvis(
                    output_file=str(module_graph_file),
                    hierarchical=False
                )
                viz_files.append(str(module_graph_file))

        except Exception as e:
            logger.warning(f"⚠️ Erreur génération visualisations: {e}")

        return viz_files

    async def _generate_entity_report(self, entity_name: str, output_file: Path):
        """Génère un rapport détaillé pour une entité spécifique"""
        try:
            # Utiliser le générateur de texte pour un contexte complet
            detailed_context = await self.text_generator.get_full_context(entity_name)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(detailed_context)

        except Exception as e:
            logger.warning(f"⚠️ Erreur génération rapport entité {entity_name}: {e}")

            # Rapport minimal en cas d'erreur
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Erreur lors de la génération du rapport pour {entity_name}: {e}")

    # === API simplifiée pour usage externe ===

    async def generate_quick_report(self, output_dir: str = "quick_report") -> str:
        """Génère un rapport rapide avec analyse limitée"""
        return await self.generate_comprehensive_report(
            output_dir=output_dir,
            include_visualizations=False,
            include_detailed_entities=False,
            max_entities_analyzed=20
        )

    async def generate_full_analysis(self, output_dir: str = "full_analysis") -> str:
        """Génère une analyse complète avec toutes les options"""
        return await self.generate_comprehensive_report(
            output_dir=output_dir,
            include_visualizations=True,
            include_detailed_entities=True,
            max_entities_analyzed=100
        )

    async def generate_focused_report(self,
                                      focus_entity: str,
                                      output_dir: str = "focused_report") -> str:
        """Génère un rapport focalisé sur une entité spécifique"""
        await self.initialize()

        # Créer le répertoire
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Rapport texte détaillé pour l'entité
        entity_report_file = output_path / f"{focus_entity}_detailed.txt"
        await self._generate_entity_report(focus_entity, entity_report_file)

        # Visualisation focalisée
        viz_file = output_path / f"{focus_entity}_dependencies.html"
        await self.graph_visualizer.build_dependency_graph(
            focus_entity=focus_entity,
            max_depth=3
        )
        self.graph_visualizer.visualize_with_pyvis(str(viz_file))

        # Rapport JSON avec analyse de dépendances
        analysis_data = await self.analyzer.analyze_dependencies(focus_entity)
        json_file = output_path / f"{focus_entity}_analysis.json"

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"✅ Rapport focalisé généré pour {focus_entity} dans {output_path}")
        return str(output_path)

    def get_generation_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du générateur"""
        return {
            'initialized': self._initialized,
            'components_available': {
                'orchestrator': self.orchestrator is not None,
                'entity_manager': self.entity_manager is not None,
                'analyzer': self.analyzer is not None,
                'text_generator': self.text_generator is not None,
                'graph_visualizer': self.graph_visualizer is not None
            },
            'entity_stats': self.entity_manager.get_stats() if self.entity_manager else {},
            'cache_stats': self.orchestrator.get_cache_stats() if self.orchestrator else {}
        }
