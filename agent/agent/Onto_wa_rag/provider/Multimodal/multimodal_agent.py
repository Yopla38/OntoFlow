"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import hashlib
import json
import base64
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import tempfile
import os

from agent.agent.Onto_wa_rag.CONSTANT import VISION_AGENT_MODEL
from context_provider.contextual_text_generator import logger
from provider.llm_providers import AnthropicProvider
from provider.Multimodal.multimodal_analyse_v3 import AdaptiveMultiAgentAnalyzer


# ==================== SCHEMAS DES TOOLS ====================

TOOLS_SCHEMA = [
    {
        "name": "analyze_image",
        "description": "Analyse complète d'une image (graphique, diagramme, photo) avec extraction de données",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_data": {
                    "type": "string",
                    "description": "Image encodée en base64 ou chemin vers le fichier"
                },
                "force_reanalysis": {
                    "type": "boolean",
                    "description": "Force une nouvelle analyse en ignorant le cache",
                    "default": False
                }
            },
            "required": ["image_data"]
        }
    },
    {
        "name": "query_graph_value",
        "description": "Interroge une valeur Y pour un X donné sur un graphique analysé",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_analysis_id": {
                    "type": "string",
                    "description": "ID de l'analyse d'image précédente"
                },
                "x_value": {
                    "type": "number",
                    "description": "Valeur X pour laquelle chercher Y"
                },
                "curve_index": {
                    "type": "integer",
                    "description": "Index de la courbe à interroger (0 pour la première)",
                    "default": 0
                }
            },
            "required": ["image_analysis_id", "x_value"]
        }
    },
    {
        "name": "recreate_chart",
        "description": "Génère et exécute le code pour recréer fidèlement un graphique analysé",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_analysis_id": {
                    "type": "string",
                    "description": "ID de l'analyse d'image précédente"
                },
                "save_path": {
                    "type": "string",
                    "description": "Chemin pour sauvegarder le graphique recréé",
                    "default": None
                },
                "show_plot": {
                    "type": "boolean",
                    "description": "Afficher le graphique à l'écran",
                    "default": False
                },
                "modify_code": {
                    "type": "object",
                    "description": "Modifications à appliquer au code (couleurs, style, etc.)",
                    "default": {}
                }
            },
            "required": ["image_analysis_id"]
        }
    },
    {
        "name": "get_image_description",
        "description": "Obtient une description détaillée du contenu d'une image analysée",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_analysis_id": {
                    "type": "string",
                    "description": "ID de l'analyse d'image précédente"
                },
                "detail_level": {
                    "type": "string",
                    "enum": ["basic", "detailed", "technical"],
                    "description": "Niveau de détail de la description",
                    "default": "detailed"
                }
            },
            "required": ["image_analysis_id"]
        }
    },
    {
        "name": "requery_image_with_context",
        "description": "Pose une question spécifique sur une image déjà analysée",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_analysis_id": {
                    "type": "string",
                    "description": "ID de l'analyse d'image précédente"
                },
                "question": {
                    "type": "string",
                    "description": "Question spécifique à poser sur l'image"
                }
            },
            "required": ["image_analysis_id", "question"]
        }
    },
    {
        "name": "manage_cache",
        "description": "Gère le cache du système d'analyse (vérifier, nettoyer)",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check", "clear_all", "clear_image"],
                    "description": "Action à effectuer sur le cache"
                },
                "image_analysis_id": {
                    "type": "string",
                    "description": "ID de l'analyse pour actions spécifiques à une image",
                    "default": None
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "get_analysis_history",
        "description": "Récupère l'historique des analyses effectuées",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Nombre maximum d'analyses à retourner",
                    "default": 10
                },
                "content_type_filter": {
                    "type": "string",
                    "description": "Filtrer par type de contenu (bar_chart, line_chart, etc.)",
                    "default": None
                }
            },
            "required": []
        }
    }
]


# ==================== GESTIONNAIRE DE TOOLS ====================

class ImageAnalysisToolsManager:
    """Gestionnaire des tools pour agents IA"""

    def __init__(self, analyzer: AdaptiveMultiAgentAnalyzer, cache_dir: str = "./vision_cache"):
        self.analyzer = analyzer
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "analysis_cache.json"
        self.history_file = self.cache_dir / "analysis_history.json"

        # Créer le répertoire de cache s'il n'existe pas
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Charger le cache existant
        self.analysis_cache = self._load_cache()
        self.analysis_history = self._load_history()

    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Calcule le hash SHA256 du contenu d'une image."""
        return hashlib.sha256(image_bytes).hexdigest()

    def _load_cache(self) -> Dict:
        """Charge le cache depuis le fichier."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

                # Reconvertir les timestamps en objets datetime
                for analysis_id, data in cache_data.items():
                    if 'timestamp' in data:
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])

                logger.info(f"🔄 Cache chargé: {len(cache_data)} analyses")
                return cache_data
            else:
                logger.info("📁 Nouveau cache créé")
                return {}
        except Exception as e:
            logger.error(f"❌ Erreur chargement cache: {e}")
            return {}

    def _load_history(self) -> List:
        """Charge l'historique depuis le fichier."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)

                # Reconvertir les timestamps
                for item in history_data:
                    if 'timestamp' in item:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])

                logger.info(f"📜 Historique chargé: {len(history_data)} entrées")
                return history_data
            else:
                return []
        except Exception as e:
            logger.error(f"❌ Erreur chargement historique: {e}")
            return []

    def _save_cache(self):
        """Sauvegarde le cache dans le fichier."""
        try:
            print(f"🔍 DEBUG: Tentative sauvegarde cache vers {self.cache_file}")
            # Convertir les datetime en string pour JSON
            cache_to_save = {}
            for analysis_id, data in self.analysis_cache.items():
                cache_to_save[analysis_id] = data.copy()
                if 'timestamp' in cache_to_save[analysis_id]:
                    cache_to_save[analysis_id]['timestamp'] = data['timestamp'].isoformat()

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_to_save, f, indent=2, ensure_ascii=False, default=str)

            print(f"💾 Cache sauvegardé: {len(cache_to_save)} analyses")
        except Exception as e:
            print(f"❌ Erreur sauvegarde cache: {e}")

    def _save_history(self):
        """Sauvegarde l'historique dans le fichier."""
        try:
            # Convertir les datetime en string
            history_to_save = []
            for item in self.analysis_history:
                item_copy = item.copy()
                if 'timestamp' in item_copy:
                    item_copy['timestamp'] = item['timestamp'].isoformat()
                history_to_save.append(item_copy)

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_to_save, f, indent=2, ensure_ascii=False, default=str)

            logger.debug(f"📜 Historique sauvegardé: {len(history_to_save)} entrées")
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde historique: {e}")

    def _save_analysis(self, analysis_id: str, results: Dict, image_data: str) -> None:
        """Sauvegarde une analyse dans le cache ET sur disque."""
        print(f"🔍 DEBUG: Sauvegarde analysis_id = {analysis_id}")
        print(f"🔍 DEBUG: Cache dir = {self.cache_dir}")

        self.analysis_cache[analysis_id] = {
            'results': results,
            'image_data': image_data,
            'timestamp': datetime.now()
        }

        self.analysis_history.append({
            'id': analysis_id,
            'content_type': results.get('content_type', 'unknown'),
            'confidence': results.get('confidence', 0.0),
            'timestamp': datetime.now(),
            'success': results.get('success', False)
        })

        print(f"🔍 DEBUG: Cache size = {len(self.analysis_cache)}")
        # Sauvegarder immédiatement sur disque
        self._save_cache()
        self._save_history()
        print(f"🔍 DEBUG: Sauvegarde terminée")

    def old_save_analysis(self, analysis_id: str, results: Dict, image_data: str) -> None:
        """Sauvegarde une analyse dans le cache ET sur disque."""
        self.analysis_cache[analysis_id] = {
            'results': results,
            'image_data': image_data,
            'timestamp': datetime.now()
        }

        # Ajouter à l'historique
        self.analysis_history.append({
            'id': analysis_id,
            'content_type': results.get('content_type', 'unknown'),
            'confidence': results.get('confidence', 0.0),
            'timestamp': datetime.now(),
            'success': results.get('success', False)
        })

        # Sauvegarder immédiatement sur disque
        self._save_cache()
        self._save_history()


    def _generate_analysis_id(self) -> str:
        """Génère un ID unique pour une analyse"""
        return f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(datetime.now()) % 10000}"

    def _get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Récupère une analyse du cache"""
        return self.analysis_cache.get(analysis_id)

    async def analyze_image(self, image_data: str, force_reanalysis: bool = False) -> Dict[str, Any]:
        """Tool: Analyser une image (version corrigée avec cache par hash)."""
        try:
            # --- NOUVELLE LOGIQUE D'OBTENTION DES OCTETS ---
            image_bytes = None
            temp_image_path = None
            if os.path.exists(image_data):
                with open(image_data, 'rb') as f:
                    image_bytes = f.read()
                image_path_for_analysis = image_data
            elif len(image_data) > 200:  # Heuristique simple pour détecter du base64
                try:
                    image_bytes = base64.b64decode(image_data)
                    # Sauver temporairement pour que l'analyseur puisse l'utiliser
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                        temp_file.write(image_bytes)
                        temp_image_path = temp_file.name
                    image_path_for_analysis = temp_image_path
                except Exception:
                    return {'success': False, 'error': 'Invalid base64 data'}
            else:
                return {'success': False, 'error': f'Image path does not exist: {image_data}'}

            if not image_bytes:
                return {'success': False, 'error': 'Could not read image data.'}

            # --- LOGIQUE DE CACHE PAR HASH ---
            image_hash = self._get_image_hash(image_bytes)

            if not force_reanalysis and image_hash in self.analysis_cache:
                logger.info(f"📋 Image trouvée dans le cache via hash: {image_hash}")
                cached_data = self.analysis_cache[image_hash]
                cached_data['from_cache'] = True  # Indiquer que ça vient du cache
                return cached_data

            # --- L'ANALYSE CONTINUE COMME AVANT ---
            logger.info(
                f"🔬 Image non trouvée dans le cache (hash: {image_hash[:10]}...), lancement de l'analyse adaptative...")
            results = await self.analyzer.analyze_image_adaptively(image_path_for_analysis)

            # --- MODIFICATION DE LA SAUVEGARDE ---
            # L'ID de l'analyse EST maintenant le hash. C'est plus simple.
            analysis_id = image_hash

            # Le dictionnaire à sauvegarder est la réponse que l'on veut renvoyer
            response_data = {
                'success': True,
                'analysis_id': analysis_id,
                'content_type': results.get('content_type', 'unknown'),
                'confidence': results.get('confidence', 0.0),
                'quality': results.get('quality', 'unknown'),
                'summary': self._generate_summary(results),
                'queryable': results.get('content_type') in ['bar_chart', 'line_chart', 'scatter_plot', 'table'],
                'recreatable': results.get('content_type') in ['bar_chart', 'line_chart', 'scatter_plot', 'pie_chart',
                                                               'heatmap'],
                'raw_data': results.get('raw_data_points'),
                'results': results  # On sauvegarde tout pour un usage futur
            }

            # Sauvegarder dans le cache en utilisant le hash comme clé
            self.analysis_cache[image_hash] = response_data
            self._save_cache()  # Sauvegarder le fichier cache

            # Ajouter à l'historique (l'historique peut rester une liste simple)
            self.analysis_history.append({
                'id': analysis_id, 'content_type': response_data['content_type'],
                'timestamp': datetime.now()
            })
            self._save_history()

            # Nettoyer le fichier temporaire si créé
            if temp_image_path:
                os.unlink(temp_image_path)

            return response_data

        except Exception as e:
            logger.error(f"❌ Exception dans analyze_image: {e}", exc_info=True)
            return {'success': False, 'error': str(e), 'analysis_id': None}

    async def get_graph_data(self, image_analysis_id: str) -> Dict[str, Any]:
        """Tool: Obtenir les données brutes d'un graphique depuis le cache."""
        try:
            analysis = self._get_analysis(image_analysis_id)
            if not analysis:
                return {'success': False, 'error': 'Analysis ID not found'}

            raw_data = analysis.get('raw_data')
            if not raw_data:
                return {'success': False,
                        'error': 'No raw data found in this analysis. The image might not be a chart.'}

            return {
                'success': True,
                'data': raw_data
            }
        except Exception as e:
            logger.error(f"Erreur dans get_graph_data: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    async def query_graph_value(self, image_analysis_id: str, x_value: float, curve_index: int = 0) -> Dict[str, Any]:
        """Tool: Interroger une valeur sur un graphique"""
        try:
            analysis = self._get_analysis(image_analysis_id)
            if not analysis:
                return {'success': False, 'error': 'Analysis ID not found'}

            results = analysis['results']

            if not results.get('success'):
                return {'success': False, 'error': 'Original analysis failed'}

            content_type = results.get('content_type')
            if content_type not in ['line_chart', 'scatter_plot']:
                return {
                    'success': False,
                    'error': f'Content type {content_type} not queryable for point values'
                }

            # Extraire les données et interpoler
            extracted_data = results.get('extracted_data', {})
            curve_key = f"curve_{curve_index + 1}"

            # Chercher dans les données extraites
            points = None
            for key, value in extracted_data.items():
                if isinstance(value, str) and curve_key in value.lower():
                    # Parser les points depuis le texte
                    points = self._parse_points_from_text(value)
                    break

            if not points or len(points) < 2:
                return {
                    'success': False,
                    'error': f'No sufficient data points found for curve {curve_index}'
                }

            # Interpolation
            from scipy.interpolate import interp1d
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            if x_value < min(x_coords) or x_value > max(x_coords):
                return {
                    'success': False,
                    'error': f'x_value {x_value} outside range [{min(x_coords)}, {max(x_coords)}]'
                }

            f = interp1d(x_coords, y_coords, kind='linear')
            y_value = float(f(x_value))

            return {
                'success': True,
                'x_value': x_value,
                'y_value': y_value,
                'curve_index': curve_index,
                'interpolation_method': 'linear',
                'data_range': {'x': [min(x_coords), max(x_coords)], 'y': [min(y_coords), max(y_coords)]}
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def recreate_chart(self, image_analysis_id: str, save_path: Optional[str] = None,
                             show_plot: bool = False, modify_code: Dict = None) -> Dict[str, Any]:
        """Tool: Recréer un graphique"""
        try:
            analysis = self._get_analysis(image_analysis_id)
            if not analysis:
                return {'success': False, 'error': 'Analysis ID not found'}

            results = analysis['results']

            if not results.get('success'):
                return {'success': False, 'error': 'Original analysis failed'}

            if 'recreation_code' not in results:
                return {'success': False, 'error': 'No recreation code available'}

            # Appliquer les modifications au code si demandées
            code = results['recreation_code']
            if modify_code:
                code = self._apply_code_modifications(code, modify_code)

            # Générer un nom de fichier si non fourni
            if save_path is None:
                save_path = f"recreated_chart_{image_analysis_id}.png"

            # Exécuter le code
            fig = await self.analyzer.execute_recreation_code(
                code,
                save_path=save_path,
                show=show_plot
            )

            return {
                'success': True,
                'saved_to': save_path,
                'code_executed': True,
                'modifications_applied': bool(modify_code),
                'figure_created': True
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def get_image_description(self, image_analysis_id: str, detail_level: str = "detailed") -> Dict[str, Any]:
        """Tool: Obtenir une description de l'image"""
        try:
            analysis = self._get_analysis(image_analysis_id)
            if not analysis:
                return {'success': False, 'error': 'Analysis ID not found'}

            results = analysis['results']

            if not results.get('success'):
                return {'success': False, 'error': 'Original analysis failed'}

            # Générer la description selon le niveau demandé
            description = self._generate_description(results, detail_level)

            return {
                'success': True,
                'description': description,
                'content_type': results.get('content_type'),
                'confidence': results.get('confidence'),
                'detail_level': detail_level
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def requery_image_with_context(self, image_analysis_id: str, question: str) -> Dict[str, Any]:
        """Tool: Poser une question spécifique sur l'image."""
        try:
            analysis = self._get_analysis(image_analysis_id)
            if not analysis:
                return {'success': False, 'error': 'Analysis ID not found'}

            # Obtenir l'image originale
            image_data = analysis['image_data']

            # Préparer l'image avec détection de format
            if os.path.exists(image_data):
                with open(image_data, 'rb') as f:
                    image_bytes = f.read()

                # 🔧 UTILISER LA MÉTHODE DE TRAITEMENT D'IMAGE DU PROVIDER
                image_b64, image_format = await self.analyzer.classifier.llm_provider.process_image_for_vision(
                    image_bytes)
            else:
                image_b64 = image_data
                image_format = "jpeg"  # Par défaut

            # Utiliser generate_vision_response avec le bon format
            answer = await self.analyzer.classifier.llm_provider.generate_vision_response(
                prompt=f"Question de l'utilisateur: {question}\n\nAnalysez cette image et répondez précisément à la question.",
                image_data=image_b64,
                image_format=image_format  # 🔧 AJOUTER LE FORMAT DÉTECTÉ
            )

            return {
                'success': True,
                'question': question,
                'answer': answer,
                'context_used': True
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def manage_cache(self, action: str, image_analysis_id: Optional[str] = None) -> Dict[str, Any]:
        """Tool: Gérer le cache (version persistante)."""
        try:
            if action == "check":
                cache_size_mb = sum(len(str(v)) for v in self.analysis_cache.values()) / (1024 * 1024)
                return {
                    'success': True,
                    'cached_analyses': len(self.analysis_cache),
                    'cache_size_mb': round(cache_size_mb, 2),
                    'analysis_ids': list(self.analysis_cache.keys()),
                    'cache_directory': str(self.cache_dir),
                    'cache_file_exists': self.cache_file.exists(),
                    'history_file_exists': self.history_file.exists()
                }

            elif action == "clear_all":
                cleared_count = len(self.analysis_cache)
                self.analysis_cache.clear()
                self.analysis_history.clear()

                # Supprimer les fichiers
                if self.cache_file.exists():
                    self.cache_file.unlink()
                if self.history_file.exists():
                    self.history_file.unlink()

                return {
                    'success': True,
                    'action': 'clear_all',
                    'cleared_analyses': cleared_count,
                    'files_deleted': True
                }

            elif action == "clear_image":
                if not image_analysis_id:
                    return {'success': False, 'error': 'image_analysis_id required for clear_image'}

                if image_analysis_id in self.analysis_cache:
                    del self.analysis_cache[image_analysis_id]

                    # Supprimer de l'historique aussi
                    self.analysis_history = [
                        h for h in self.analysis_history
                        if h.get('id') != image_analysis_id
                    ]

                    # Sauvegarder les modifications
                    self._save_cache()
                    self._save_history()

                    return {
                        'success': True,
                        'action': 'clear_image',
                        'cleared_analysis_id': image_analysis_id
                    }
                else:
                    return {'success': False, 'error': 'Analysis ID not found in cache'}

            else:
                return {'success': False, 'error': f'Unknown action: {action}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques détaillées du cache."""
        total_size = 0
        oldest_analysis = None
        newest_analysis = None

        for analysis_id, data in self.analysis_cache.items():
            total_size += len(str(data))
            timestamp = data.get('timestamp', datetime.min)

            if oldest_analysis is None or timestamp < oldest_analysis[1]:
                oldest_analysis = (analysis_id, timestamp)
            if newest_analysis is None or timestamp > newest_analysis[1]:
                newest_analysis = (analysis_id, timestamp)

        return {
            'total_analyses': len(self.analysis_cache),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_directory': str(self.cache_dir),
            'oldest_analysis': oldest_analysis[0] if oldest_analysis else None,
            'newest_analysis': newest_analysis[0] if newest_analysis else None,
            'cache_file_size_mb': round(self.cache_file.stat().st_size / (1024 * 1024),
                                        2) if self.cache_file.exists() else 0
        }

    async def get_analysis_history(self, limit: int = 10, content_type_filter: Optional[str] = None) -> Dict[str, Any]:
        """Tool: Récupérer l'historique des analyses"""
        try:
            history = self.analysis_history.copy()

            # Filtrer par type de contenu si demandé
            if content_type_filter:
                history = [h for h in history if h['content_type'] == content_type_filter]

            # Limiter les résultats
            history = history[-limit:] if limit > 0 else history

            return {
                'success': True,
                'total_analyses': len(self.analysis_history),
                'filtered_count': len(history),
                'history': history,
                'content_types': list(set(h['content_type'] for h in self.analysis_history))
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    # Méthodes utilitaires
    def _generate_summary(self, results: Dict) -> str:
        """Génère un résumé de l'analyse"""
        if not results.get('success'):
            return "Analyse échouée"

        content_type = results.get('content_type', 'unknown')
        confidence = results.get('confidence', 0)

        if content_type == 'bar_chart':
            return f"Graphique en barres détecté avec {confidence:.1%} de confiance. Données extractibles."
        elif content_type == 'line_chart':
            return f"Graphique linéaire détecté avec {confidence:.1%} de confiance. Points interrogeables."
        elif content_type == 'table':
            return f"Tableau détecté avec {confidence:.1%} de confiance. Données structurées disponibles."
        else:
            return f"Contenu de type {content_type} détecté avec {confidence:.1%} de confiance."

    def _generate_description(self, results: Dict, detail_level: str) -> str:
        """Génère une description selon le niveau de détail"""
        if detail_level == "basic":
            return self._generate_summary(results)

        elif detail_level == "detailed":
            desc = self._generate_summary(results)
            extracted_data = results.get('extracted_data', {})

            desc += f"\n\nDonnées extraites: {len(extracted_data)} sections analysées."
            if 'metadata_extraction' in extracted_data:
                desc += f"\nÉléments textuels identifiés."

            return desc

        elif detail_level == "technical":
            workflow_steps = results.get('workflow_steps', {})
            desc = f"Analyse technique complète:\n"
            desc += f"- Type: {results.get('content_type')}\n"
            desc += f"- Confiance: {results.get('confidence', 0):.3f}\n"
            desc += f"- Qualité: {results.get('quality')}\n"
            desc += f"- Étapes: {', '.join(workflow_steps.keys())}\n"

            return desc

        return self._generate_summary(results)

    def _parse_points_from_text(self, text: str) -> List[Tuple[float, float]]:
        """Parse les points depuis un texte d'analyse"""
        points = []
        import re

        # Chercher des patterns comme "x: 0.4, y: 0.2" ou "0.4, 0.2"
        patterns = [
            r'x:\s*([0-9.]+).*?y:\s*([0-9.]+)',
            r'([0-9.]+)\s*,\s*([0-9.]+)',
            r'\(([0-9.]+),\s*([0-9.]+)\)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    x, y = float(match[0]), float(match[1])
                    points.append((x, y))
                except:
                    continue

        return points

    def _apply_code_modifications(self, code: str, modifications: Dict) -> str:
        """Applique des modifications au code de recréation"""
        modified_code = code

        # Modification des couleurs
        if 'colors' in modifications:
            new_colors = modifications['colors']
            # Remplacer les couleurs dans le code
            color_pattern = r"colors\s*=\s*\[.*?\]"
            new_color_line = f"colors = {new_colors}"
            modified_code = re.sub(color_pattern, new_color_line, modified_code)

        # Modification du titre
        if 'title' in modifications:
            new_title = modifications['title']
            title_pattern = r"plt\.title\(['\"].*?['\"]\)"
            new_title_line = f'plt.title("{new_title}")'
            modified_code = re.sub(title_pattern, new_title_line, modified_code)

        # Autres modifications...

        return modified_code


# ==================== AGENT DE TEST ====================

class ImageAnalysisAgent:
    """Agent IA qui utilise les tools d'analyse d'images"""

    def __init__(self, tools_manager: ImageAnalysisToolsManager, llm_provider: AnthropicProvider):
        self.tools_manager = tools_manager
        self.llm_provider = llm_provider
        self.available_tools = {
            "analyze_image": self.tools_manager.analyze_image,
            "query_graph_value": self.tools_manager.query_graph_value,
            "recreate_chart": self.tools_manager.recreate_chart,
            "get_image_description": self.tools_manager.get_image_description,
            "requery_image_with_context": self.tools_manager.requery_image_with_context,
            "manage_cache": self.tools_manager.manage_cache,
            "get_analysis_history": self.tools_manager.get_analysis_history
        }

    async def process_user_request(self, user_message: str, image_path: Optional[str] = None) -> str:
        """Traite une demande utilisateur avec les tools disponibles"""

        # Construire le prompt pour l'agent
        system_prompt = f"""Tu es un assistant spécialisé dans l'analyse d'images scientifiques et de graphiques.

Tu as accès aux tools suivants pour analyser et interroger des images :

{json.dumps(TOOLS_SCHEMA, indent=2)}

Quand un utilisateur te donne une image ou pose une question sur une image déjà analysée, 
utilise les tools appropriés pour répondre de manière complète et précise.

Pour analyser une nouvelle image, utilise d'abord analyze_image.
Pour interroger des valeurs spécifiques, utilise query_graph_value.
Pour recréer un graphique, utilise recreate_chart.
Pour des questions contextuelles, utilise requery_image_with_context.

Sois précis et détaillé dans tes réponses."""

        # Construire le message avec l'image si fournie
        messages = []
        if image_path:
            # Encoder l'image
            with open(image_path, 'rb') as f:
                image_b64 = base64.b64encode(f.read()).decode()

            user_message += f"\n\n[Image fournie: {image_path}]"

        messages.append({
            "role": "user",
            "content": user_message
        })

        # Simuler l'appel avec tools (implémentation simplifiée)
        response = await self._simulate_tool_usage(user_message, image_path)

        return response

    async def _simulate_tool_usage(self, user_message: str, image_path: Optional[str] = None) -> str:
        """Simulation de l'usage des tools par l'agent"""

        responses = []

        # Si une image est fournie, l'analyser d'abord
        if image_path:
            responses.append("🔍 Analyse de l'image en cours...")

            analysis_result = await self.tools_manager.analyze_image(image_path)

            if analysis_result['success']:
                analysis_id = analysis_result['analysis_id']
                responses.append(f"✅ Image analysée avec succès!")
                responses.append(f"📊 Type détecté: {analysis_result['content_type']}")
                responses.append(f"🎯 Confiance: {analysis_result['confidence']:.1%}")
                responses.append(f"📝 {analysis_result['summary']}")

                # Obtenir une description détaillée
                desc_result = await self.tools_manager.get_image_description(analysis_id, "detailed")
                if desc_result['success']:
                    responses.append(f"\n📋 Description détaillée:\n{desc_result['description']}")

                # Si c'est un graphique interrogeable
                if analysis_result['queryable']:
                    responses.append(f"\n💡 Ce graphique est interrogeable pour des valeurs spécifiques.")

                # Si c'est recréable
                if analysis_result['recreatable']:
                    responses.append(f"\n🎨 Ce graphique peut être recréé fidèlement.")

                # Stocker l'ID pour usage ultérieur
                self._last_analysis_id = analysis_id

            else:
                responses.append(f"❌ Erreur d'analyse: {analysis_result['error']}")

        # Analyser la demande pour des actions spécifiques
        if any(word in user_message.lower() for word in ['valeur', 'point', 'coordonnée', 'y pour x']):
            if hasattr(self, '_last_analysis_id'):
                # Chercher une valeur x dans le message
                import re
                x_matches = re.findall(r'x\s*=\s*([0-9.]+)', user_message)
                if x_matches:
                    x_value = float(x_matches[0])
                    responses.append(f"\n🔍 Recherche de la valeur Y pour X={x_value}...")

                    query_result = await self.tools_manager.query_graph_value(
                        self._last_analysis_id, x_value
                    )

                    if query_result['success']:
                        responses.append(f"📊 Pour X={x_value}, Y={query_result['y_value']:.6f}")
                    else:
                        responses.append(f"❌ {query_result['error']}")

        if any(word in user_message.lower() for word in ['recréer', 'régénérer', 'reproduire']):
            if hasattr(self, '_last_analysis_id'):
                responses.append(f"\n🎨 Recréation du graphique...")

                recreate_result = await self.tools_manager.recreate_chart(
                    self._last_analysis_id,
                    save_path=f"agent_recreated_{datetime.now().strftime('%H%M%S')}.png"
                )

                if recreate_result['success']:
                    responses.append(f"✅ Graphique recréé et sauvé: {recreate_result['saved_to']}")
                else:
                    responses.append(f"❌ {recreate_result['error']}")

        if any(word in user_message.lower() for word in ['cache', 'historique']):
            responses.append(f"\n📚 Vérification du cache...")

            cache_result = await self.tools_manager.manage_cache("check")
            if cache_result['success']:
                responses.append(f"💾 {cache_result['cached_analyses']} analyses en cache")

            history_result = await self.tools_manager.get_analysis_history(5)
            if history_result['success']:
                responses.append(f"📋 Dernières analyses: {len(history_result['history'])}")

        return "\n".join(responses)


# ==================== FONCTION DE TEST ====================

async def test_image_analysis_agent(api_key):
    """Test complet de l'agent avec tools"""

    print("🚀 INITIALISATION DE L'AGENT D'ANALYSE D'IMAGES")
    print("=" * 60)

    # Initialisation
    llm_provider = AnthropicProvider(
        model=VISION_AGENT_MODEL,
        api_key=api_key,
        system_prompt="Tu es un expert en analyse d'images scientifiques."
    )

    analyzer = AdaptiveMultiAgentAnalyzer(llm_provider)
    tools_manager = ImageAnalysisToolsManager(analyzer)
    agent = ImageAnalysisAgent(tools_manager, llm_provider)

    # Tests avec différents scénarios
    test_scenarios = [
        {
            "name": "Analyse d'image complète",
            "message": "Peux-tu analyser cette image et me dire ce qu'elle contient ?",
            "image": "/home/yopla/Documents/llm_models/python/models/multimodal/test_figure/courbe.jpg"
        },
        {
            "name": "Interrogation de valeur",
            "message": "Quelle est la valeur Y pour X=0.4 sur cette courbe ?",
            "image": None  # Utilise la dernière image analysée
        },
        {
            "name": "Recréation de graphique",
            "message": "Peux-tu recréer ce graphique ?",
            "image": None
        },
        {
            "name": "Question contextuelle",
            "message": "Quelle est la performance de l'ondelette sym3 sur la métrique MAP ?",
            "image": None
        },
        {
            "name": "Gestion du cache",
            "message": "Montre-moi l'état du cache et l'historique des analyses",
            "image": None
        }
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 TEST {i}: {scenario['name']}")
        print("-" * 40)

        try:
            response = await agent.process_user_request(
                scenario['message'],
                scenario['image']
            )

            print(response)

        except Exception as e:
            print(f"❌ Erreur: {e}")

        print("-" * 40)

    print("\n✅ Tests terminés!")


# Pour utiliser les tools dans un vrai agent Anthropic/OpenAI
def get_tools_for_anthropic():
    """Retourne les tools au format Anthropic"""
    return TOOLS_SCHEMA


def get_tools_for_openai():
    """Retourne les tools au format OpenAI"""
    openai_tools = []
    for tool in TOOLS_SCHEMA:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        })
    return openai_tools


# Usage final
if __name__ == "__main__":
    # Test de l'agent
    asyncio.run(test_image_analysis_agent("my_anthropic_key"))

    # Récupérer les tools pour intégration
    anthropic_tools = get_tools_for_anthropic()
    openai_tools = get_tools_for_openai()

    print("🛠️ Tools créés et prêts pour intégration!")
    print(f"📊 {len(anthropic_tools)} tools disponibles")