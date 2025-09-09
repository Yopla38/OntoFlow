# **📊 Adaptive Multi-Agent Image Analyzer**

Système d'analyse d'images adaptatif utilisant des agents IA spécialisés pour extraire des données de graphiques, diagrammes et autres contenus visuels.

## **🚀 Installation**

```bash
pip install anthropic matplotlib numpy pandas scipy pillow instructor
```

## **⚡ Usage Rapide**

```python
from multimodal_analyse_v3 import AdaptiveMultiAgentAnalyzer, AnthropicProvider

# Initialisation
llm_provider = AnthropicProvider(
    model="claude-3-5-sonnet-20241022",
    api_key="your-anthropic-api-key"
)
analyzer = AdaptiveMultiAgentAnalyzer(llm_provider)

# Analyse d'image
results = await analyzer.analyze_image_adaptively("path/to/image.png")
```

## **📤 Envoi d'Images**

### **Méthodes d'envoi**
```python
# 1. Par chemin de fichier
results = await analyzer.analyze_image_adaptively("/path/to/chart.png")

# 2. Par bytes (pour intégration RAG)
with open("image.png", "rb") as f:
    image_bytes = f.read()
# Utiliser image_bytes dans votre système RAG puis sauver temporairement
```

### **Formats supportés**
- **Images** : PNG, JPG, JPEG, WEBP, GIF
- **Taille max** : 20MB (redimensionnement automatique)
- **Résolution** : Optimisée automatiquement pour l'IA

## **📊 Format des Données de Retour**

### **Structure Principal**
```python
{
    'success': bool,                    # Succès de l'analyse
    'content_type': str,               # 'bar_chart', 'line_chart', 'photo', etc.
    'confidence': float,               # Score de confiance (0.0 à 1.0)
    'quality': str,                    # 'excellent', 'good', 'fair', 'poor'
    'extracted_data': dict,            # Données extraites détaillées
    'recreation_code': str,            # Code Python pour recréer le contenu
    'code_explanation': str,           # Explication du code
    'workflow_steps': dict,            # Détails de chaque étape d'analyse
    'timestamp': datetime,             # Horodatage
    'error': str                       # Message d'erreur si échec
}
```

## **📈 Données de Graphiques**

### **Bar Chart (Exemple)**
```python
# Accès aux données extraites
if results['content_type'] == 'bar_chart':
    bar_data = results['extracted_data']['bar_groups_extraction']
    
    # Parsing des valeurs numériques
    import re
    values = {}
    for line in bar_data.split('\n'):
        if ':' in line and '~' in line:
            match = re.search(r'(\w+)\s*:\s*~?(\d+\.?\d*)', line)
            if match:
                metric, value = match.groups()
                values[metric] = float(value)
    
    print(f"Données extraites: {values}")
    # Exemple: {'coif3': 0.83, 'db5': 0.83, 'dmey': 0.74, ...}
```

### **Line Chart (Exemple)**
```python
if results['content_type'] == 'line_chart':
    # Points de données des courbes
    curves_data = results['extracted_data']
    
    # Format attendu pour les courbes
    for curve_id, points in curves_data.items():
        if isinstance(points, list) and len(points) > 0:
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            print(f"Courbe {curve_id}: {len(points)} points")
            print(f"X: {x_values[:5]}...")  # Premiers points
            print(f"Y: {y_values[:5]}...")
```

## **📝 Description d'Images**

### **Pour Photos/Diagrammes**
```python
if results['content_type'] in ['photo', 'diagram', 'text_document']:
    # Description complète
    description = results['extracted_data'].get('general_description', '')
    
    # Éléments détectés
    elements = results['extracted_data'].get('detected_elements', [])
    
    # Texte extrait (si OCR)
    text_content = results['extracted_data'].get('text_content', '')
    
    print(f"Description: {description}")
    print(f"Éléments: {elements}")
    print(f"Texte: {text_content}")
```

## **🔄 Réinterrogation d'Images**

### **Interrogation de Valeurs Spécifiques**
```python
# Pour graphiques : obtenir une valeur Y pour un X donné
if results['content_type'] in ['line_chart', 'scatter_plot']:
    # Méthode directe (nécessite les données de points)
    def query_graph_value(extracted_data, x_target, curve_index=0):
        curve_key = f"curve_{curve_index + 1}"
        if curve_key in extracted_data:
            points = extracted_data[curve_key]
            
            # Interpolation simple
            from scipy.interpolate import interp1d
            if len(points) >= 2:
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                f = interp1d(x_vals, y_vals, kind='linear', bounds_error=False)
                return float(f(x_target))
        return None
    
    y_value = query_graph_value(results['extracted_data'], x_target=0.4)
    print(f"Pour x=0.4, y={y_value}")
```

### **Nouvelle Analyse avec Contexte**
```python
# Réanalyse avec question spécifique
llm_provider.history.add_message("user", "Quelle est la valeur exacte pour x=0.4 sur la courbe rouge?")
specific_answer = await llm_provider.generate_vision_response(
    prompt="Répondez à la question de l'utilisateur en analysant ce graphique.",
    image_data=image_b64  # Image déjà chargée
)
```

## **🎨 Recréation de Graphiques**

### **Exécution du Code Généré**
```python
# Recréer le graphique identique
if 'recreation_code' in results:
    try:
        # Exécution directe
        fig = await analyzer.execute_recreation_code(
            results['recreation_code'],
            save_path="recreated_chart.png",
            show=True
        )
        print("✅ Graphique recréé avec succès")
        
    except Exception as e:
        print(f"❌ Erreur recréation: {e}")
        # Le code est aussi disponible en texte
        print("Code généré:")
        print(results['recreation_code'])
```

### **Modification du Code**
```python
# Récupérer et modifier le code
code = results['recreation_code']

# Exemple : changer les couleurs
modified_code = code.replace(
    "colors = ['#683A95', '#4B75B5', ...]", 
    "colors = ['red', 'blue', 'green', ...]"
)

# Exécuter le code modifié
fig = await analyzer.execute_recreation_code(modified_code)
```

## **💾 Gestion du Cache**

### **Vérification du Cache**
```python
# Vérifier si une image est en cache
with open("image.png", "rb") as f:
    image_bytes = f.read()

cached_analyses = analyzer.classifier.analyzer.get_cached_analyses(image_bytes)
print(f"Analyses en cache: {len(cached_analyses)}")

for analysis in cached_analyses:
    print(f"- {analysis.analysis_type.value}: {analysis.result.confidence}")
```

### **Nettoyage du Cache**
```python
# Vider le cache pour une image spécifique
analyzer.classifier.analyzer.clear_cache(image_bytes)

# Vider tout le cache
analyzer.classifier.analyzer.clear_cache()

# Forcer une nouvelle analyse (ignore le cache)
results = await analyzer.analyze_image_adaptively(
    "image.png", 
    force_reanalysis=True
)
```

## **🔧 Intégration RAG**

### **Exemple d'Intégration**
```python
class RAGImageProcessor:
    def __init__(self, anthropic_api_key):
        self.analyzer = AdaptiveMultiAgentAnalyzer(
            AnthropicProvider(
                model="claude-3-5-sonnet-20241022",
                api_key=anthropic_api_key
            )
        )
    
    async def process_rag_image(self, image_bytes: bytes, user_query: str):
        """Traite une image dans le contexte RAG"""
        
        # 1. Sauvegarder temporairement l'image
        temp_path = f"/tmp/rag_image_{hash(image_bytes)}.png"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        
        # 2. Analyser l'image
        results = await self.analyzer.analyze_image_adaptively(temp_path)
        
        # 3. Préparer les métadonnées pour le RAG
        if results['success']:
            metadata = {
                'content_type': results['content_type'],
                'confidence': results['confidence'],
                'quality': results['quality'],
                'extractable_data': results['content_type'] in [
                    'bar_chart', 'line_chart', 'scatter_plot', 'table'
                ],
                'description': self._generate_description(results),
                'queryable': True
            }
            
            # 4. Stocker les données pour interrogation future
            self._store_extracted_data(results['extracted_data'])
            
            return metadata, results
        
        return None, results
    
    def _generate_description(self, results):
        """Génère une description textuelle pour le RAG"""
        content_type = results['content_type']
        
        if content_type == 'bar_chart':
            return f"Graphique en barres avec {results['confidence']:.1%} de confiance. " \
                   f"Données extraites disponibles pour interrogation."
        
        elif content_type == 'line_chart':
            return f"Graphique linéaire avec {results['confidence']:.1%} de confiance. " \
                   f"Points de données disponibles pour interpolation."
        
        # Autres types...
        return f"Contenu de type {content_type} analysé avec {results['confidence']:.1%} de confiance."

# Usage dans le RAG
rag_processor = RAGImageProcessor("your-api-key")
metadata, analysis = await rag_processor.process_rag_image(image_bytes, user_query)
```

## **📋 Types de Contenu Supportés**

| Type | Description | Données Extraites | Interrogeable |
|------|-------------|-------------------|---------------|
| `bar_chart` | Graphiques en barres | Valeurs par catégorie | ✅ |
| `line_chart` | Graphiques linéaires | Points (x,y) des courbes | ✅ |
| `scatter_plot` | Nuages de points | Coordonnées des points | ✅ |
| `pie_chart` | Graphiques secteurs | Pourcentages/valeurs | ✅ |
| `heatmap` | Cartes de chaleur | Matrice de valeurs | ✅ |
| `table` | Tableaux | Données tabulaires | ✅ |
| `photo` | Photos | Description visuelle | ❌ |
| `diagram` | Diagrammes | Éléments et relations | ❌ |
| `text_document` | Documents texte | Texte extrait (OCR) | ❌ |

## **⚠️ Gestion d'Erreurs**

```python
# Vérification des erreurs
if not results['success']:
    error_msg = results.get('error', 'Erreur inconnue')
    print(f"Erreur d'analyse: {error_msg}")
    
    # Vérifier les étapes qui ont échoué
    for step_name, step_result in results.get('workflow_steps', {}).items():
        if not step_result.success:
            print(f"Échec à l'étape {step_name}: {step_result.error_message}")

# Vérification de la qualité
if results.get('confidence', 0) < 0.7:
    print("⚠️ Confiance faible, résultats à vérifier")

if results.get('quality') in ['fair', 'poor']:
    print("⚠️ Qualité d'extraction faible")
```

## **🔗 Support**

- **Formats d'images** : PNG, JPG, JPEG, WEBP, GIF
- **Types de contenu** : Graphiques, tableaux, diagrammes, photos
- **Taille max** : 20MB par image
- **Concurrent** : Système entièrement asynchrone

Pour des questions ou problèmes, consultez les logs détaillés via `logging.getLogger(__name__)`.