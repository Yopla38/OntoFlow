"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# passage_visualizer.py
import os
import re
import uuid
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
import difflib


class PassageVisualizer:
    """Classe pour créer des captures d'écran visuelles des passages cités dans les documents PDF"""

    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialise le visualiseur de passages

        Args:
            output_dir: Répertoire où les visualisations seront sauvegardées
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Essayer de charger les polices
        try:
            self.title_font = ImageFont.truetype("arial.ttf", 24)
            self.info_font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            # Fallback vers une police par défaut si arial n'est pas disponible
            self.title_font = ImageFont.load_default()
            self.info_font = ImageFont.load_default()

        # Compteur pour rendre les noms de fichiers uniques
        self.counter = 0

    def _extract_keywords(self, text: str, min_length: int = 4, max_keywords: int = 10) -> List[str]:
        """Extrait les mots-clés significatifs d'un texte"""
        stopwords = {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'de', 'du', 'en', 'au', 'aux',
                     'ce', 'cette', 'ces', 'son', 'sa', 'ses', 'leur', 'leurs', 'notre', 'nos', 'votre',
                     'vos', 'qui', 'que', 'quoi', 'dont', 'où', 'quand', 'comment', 'pourquoi', 'car',
                     'si', 'ni', 'ou', 'mais', 'or', 'donc', 'pour', 'par', 'avec', 'sans', 'the',
                     'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like',
                     'through', 'over', 'before', 'between', 'after', 'since', 'without', 'under',
                     'within', 'along', 'following', 'across', 'behind', 'beyond', 'plus', 'except',
                     'but', 'up', 'down', 'off', 'above', 'below'}

        words = re.findall(rf'\b\w{{{min_length},}}\b', text.lower())
        keywords = [word for word in words if word.lower() not in stopwords]

        # Éliminer les doublons et garder l'ordre d'apparition
        unique_keywords = []
        seen = set()
        for word in keywords:
            if word.lower() not in seen:
                unique_keywords.append(word)
                seen.add(word.lower())

        return unique_keywords[:max_keywords]

    def _words_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calcule la similarité entre deux séquences de mots"""
        # Convertir les listes en chaînes pour utiliser difflib
        str1 = ' '.join(seq1).lower()
        str2 = ' '.join(seq2).lower()

        # Calculer la similarité
        similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
        return similarity

    def visualize_passage(self,
                          file_path: str,
                          passage_text: str,
                          doc_title: str = None,
                          context_lines: int = 3) -> Optional[str]:
        """
        Crée une visualisation d'un passage dans un document PDF avec recherche robuste

        Args:
            file_path: Chemin vers le fichier PDF
            passage_text: Texte du passage à visualiser
            doc_title: Titre du document (utilise le nom de fichier si non fourni)
            context_lines: Nombre de lignes de contexte à inclure

        Returns:
            Chemin vers l'image générée ou None en cas d'échec
        """
        if not os.path.exists(file_path) or not file_path.lower().endswith('.pdf'):
            return None

        try:
            # Incrémenter le compteur pour garantir l'unicité
            self.counter += 1

            # 1. Extraire les mots-clés significatifs du passage
            keywords = self._extract_keywords(passage_text)

            # 2. Normaliser le passage pour la comparaison
            normalized_passage = re.sub(r'\s+', ' ', passage_text).strip().lower()
            passage_words = normalized_passage.split()

            # Ouvrir le document PDF
            with fitz.open(file_path) as pdf:
                best_match = None
                best_score = 0
                best_page_num = 0
                best_block_rect = None

                # Parcourir toutes les pages du document
                for page_num, page in enumerate(pdf):
                    # MÉTHODE 1: Utiliser get_text("words") pour localiser les mots-clés
                    words_info = page.get_text("words")

                    # Format de words_info: liste de [x0, y0, x1, y1, "word", block_no, line_no, word_no]
                    word_matches = {}
                    for i, word_info in enumerate(words_info):
                        word = word_info[4].lower()  # Le mot est à l'index 4

                        # Vérifier si ce mot est l'un de nos mots-clés
                        if any(keyword.lower() == word for keyword in keywords):
                            # Stocker sa position et son index
                            block_id = word_info[5]  # ID du bloc
                            if block_id not in word_matches:
                                word_matches[block_id] = []
                            word_matches[block_id].append((i, word_info))

                    # Pour chaque bloc contenant des mots-clés
                    for block_id, matches in word_matches.items():
                        if len(matches) < min(2, len(keywords)):  # Au moins 2 mots-clés ou tous si moins de 2
                            continue

                        # Trier les correspondances par position dans la page (par index de mot)
                        matches.sort(key=lambda x: x[0])

                        # MÉTHODE 2: Utiliser get_text("blocks") pour obtenir le texte complet du bloc
                        blocks = page.get_text("blocks")
                        block_text = None
                        block_rect = None

                        # Trouver le bloc correspondant
                        for block in blocks:
                            if block[5] == block_id:  # L'ID du bloc est à l'index 5
                                block_text = block[4]  # Le texte est à l'index 4
                                block_rect = fitz.Rect(block[0], block[1], block[2], block[3])  # Coordonnées
                                break

                        if block_text:
                            # Normaliser le texte du bloc pour la comparaison
                            normalized_block = re.sub(r'\s+', ' ', block_text).strip().lower()
                            block_words = normalized_block.split()

                            # Calculer la similarité entre le passage et le bloc
                            similarity = self._words_sequence_similarity(passage_words, block_words)

                            # Si ce bloc est le meilleur match jusqu'à présent
                            if similarity > best_score:
                                best_score = similarity
                                best_page_num = page_num
                                best_block_rect = block_rect

                                # Extraire les positions des mots-clés pour le surlignage
                                best_match = [word_info[0:4] for _, word_info in matches]  # Coordonnées (x0,y0,x1,y1)

                # Si un match satisfaisant a été trouvé
                if best_score > 0.4:  # Seuil de confiance
                    # Ouvrir une copie temporaire du PDF pour ajouter les annotations
                    temp_pdf_path = os.path.join(self.output_dir, f"temp_{uuid.uuid4()}.pdf")
                    pdf.save(temp_pdf_path)

                    with fitz.open(temp_pdf_path) as temp_pdf:
                        page = temp_pdf[best_page_num]

                        # 1. Surligner le bloc entier pour une meilleure visibilité
                        if best_block_rect:
                            annot = page.add_highlight_annot(best_block_rect)
                            annot.set_colors(stroke=(1, 0.95, 0.8))  # Jaune très clair
                            annot.update()

                        # 2. Surligner spécifiquement les mots-clés
                        for word_rect in best_match:
                            rect = fitz.Rect(word_rect)  # Convertir en Rect
                            annot = page.add_highlight_annot(rect)
                            annot.set_colors(stroke=(1, 0.8, 0))  # Jaune plus vif pour les mots-clés
                            annot.update()

                        # Calculer la région à capturer
                        if best_block_rect:
                            # Étendre légèrement pour inclure plus de contexte
                            context_height = 75 * context_lines  # Hauteur adaptative selon le contexte
                            view_rect = fitz.Rect(
                                0,  # Prendre toute la largeur de la page
                                max(0, best_block_rect.y0 - context_height / 2),
                                page.rect.width,
                                min(page.rect.height, best_block_rect.y1 + context_height / 2)
                            )
                        else:
                            # Fallback sur les mots-clés s'il n'y a pas de bloc
                            min_y = min(rect[1] for rect in best_match)
                            max_y = max(rect[3] for rect in best_match)
                            context_height = 75 * context_lines
                            view_rect = fitz.Rect(
                                0,
                                max(0, min_y - context_height / 2),
                                page.rect.width,
                                min(page.rect.height, max_y + context_height / 2)
                            )

                        # Créer une capture d'écran avec zoom
                        zoom_factor = 2
                        mat = fitz.Matrix(zoom_factor, zoom_factor)
                        pix = page.get_pixmap(matrix=mat, clip=view_rect)

                        # Créer l'image finale
                        img_data = pix.tobytes("png")
                        img = Image.open(BytesIO(img_data))

                        # Ajouter le bandeau d'information
                        title_height = 50
                        title_img = Image.new('RGB', (img.width, title_height), color='white')
                        combined_img = Image.new('RGB', (img.width, img.height + title_height))
                        combined_img.paste(title_img, (0, 0))
                        combined_img.paste(img, (0, title_height))

                        draw = ImageDraw.Draw(combined_img)
                        title = doc_title or os.path.basename(file_path)

                        # Dessiner le titre et les informations
                        title_bbox = draw.textbbox((0, 0), title, font=self.title_font)
                        title_width = title_bbox[2] - title_bbox[0]

                        draw.text(((combined_img.width - title_width) // 2, 5),
                                  title, font=self.title_font, fill=(0, 0, 0))
                        draw.text((10, 30),
                                  f"Page {best_page_num + 1} (confiance: {best_score:.2f})",
                                  font=self.info_font, fill=(0, 0, 0))

                        # Générer un nom de fichier unique
                        doc_name = os.path.basename(file_path).split('.')[0]
                        filename = f"{doc_name}_p{best_page_num + 1}_{self.counter}_{uuid.uuid4().hex[:6]}.png"
                        output_path = os.path.join(self.output_dir, filename)

                        # Sauvegarder l'image
                        combined_img.save(output_path, format='PNG')

                        # Nettoyer
                        try:
                            os.remove(temp_pdf_path)
                        except:
                            pass

                        return output_path

                # Si aucun match satisfaisant n'a été trouvé
                # Essayons une dernière méthode avec les blocs entiers
                for page_num, page in enumerate(pdf):
                    blocks = page.get_text("blocks")

                    for block in blocks:
                        block_text = block[4]
                        # Normaliser le texte du bloc
                        normalized_block = re.sub(r'\s+', ' ', block_text).strip().lower()

                        # Vérifier combien de mots-clés sont dans ce bloc
                        keyword_count = sum(1 for keyword in keywords if keyword.lower() in normalized_block.lower())

                        if keyword_count >= min(3, len(keywords)):
                            # Ce bloc contient plusieurs mots-clés, c'est probablement notre passage
                            block_rect = fitz.Rect(block[0], block[1], block[2], block[3])

                            # Créer une capture d'écran
                            context_height = 75 * context_lines
                            view_rect = fitz.Rect(
                                0,
                                max(0, block_rect.y0 - context_height / 2),
                                page.rect.width,
                                min(page.rect.height, block_rect.y1 + context_height / 2)
                            )

                            zoom_factor = 2
                            mat = fitz.Matrix(zoom_factor, zoom_factor)
                            pix = page.get_pixmap(matrix=mat, clip=view_rect)

                            # Créer une image avec annotation sur fond jaune très clair
                            img_data = pix.tobytes("png")
                            img = Image.open(BytesIO(img_data))

                            # Ajouter le bandeau d'information
                            title_height = 50
                            title_img = Image.new('RGB', (img.width, title_height), color='white')
                            combined_img = Image.new('RGB', (img.width, img.height + title_height))
                            combined_img.paste(title_img, (0, 0))
                            combined_img.paste(img, (0, title_height))

                            draw = ImageDraw.Draw(combined_img)
                            title = doc_title or os.path.basename(file_path)

                            title_bbox = draw.textbbox((0, 0), title, font=self.title_font)
                            title_width = title_bbox[2] - title_bbox[0]

                            draw.text(((combined_img.width - title_width) // 2, 5),
                                      title, font=self.title_font, fill=(0, 0, 0))
                            draw.text((10, 30),
                                      f"Page {page_num + 1} (correspondance par mots-clés)",
                                      font=self.info_font, fill=(0, 0, 0))

                            # Générer un nom de fichier unique
                            doc_name = os.path.basename(file_path).split('.')[0]
                            filename = f"{doc_name}_p{page_num + 1}_{self.counter}_{uuid.uuid4().hex[:6]}.png"
                            output_path = os.path.join(self.output_dir, filename)

                            # Sauvegarder l'image
                            combined_img.save(output_path, format='PNG')

                            return output_path

                # Dernière solution : prendre la première page avec une capture générique
                page = pdf[0]
                center_y = page.rect.height / 2
                view_rect = fitz.Rect(0, center_y - 200, page.rect.width, center_y + 200)

                zoom_factor = 2
                mat = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=mat, clip=view_rect)

                img_data = pix.tobytes("png")
                img = Image.open(BytesIO(img_data))

                # Ajouter le bandeau d'information
                title_height = 50
                title_img = Image.new('RGB', (img.width, title_height), color='white')
                combined_img = Image.new('RGB', (img.width, img.height + title_height))
                combined_img.paste(title_img, (0, 0))
                combined_img.paste(img, (0, title_height))

                draw = ImageDraw.Draw(combined_img)
                title = doc_title or os.path.basename(file_path)

                title_bbox = draw.textbbox((0, 0), title, font=self.title_font)
                title_width = title_bbox[2] - title_bbox[0]

                draw.text(((combined_img.width - title_width) // 2, 5),
                          title, font=self.title_font, fill=(0, 0, 0))
                draw.text((10, 30),
                          "Page 1 (passage non localisé)", font=self.info_font, fill=(0, 0, 0))

                # Générer un nom de fichier unique
                doc_name = os.path.basename(file_path).split('.')[0]
                filename = f"{doc_name}_p1_{self.counter}_{uuid.uuid4().hex[:6]}.png"
                output_path = os.path.join(self.output_dir, filename)

                combined_img.save(output_path, format='PNG')

                return output_path

        except Exception as e:
            print(f"Erreur lors de la visualisation du passage: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_passages(self, passages: List[Dict[str, Any]]) -> Dict[int, str]:
        """
        Crée des visualisations pour une liste de passages

        Args:
            passages: Liste des passages à visualiser

        Returns:
            Dictionnaire {index_passage: chemin_image}
        """
        visualizations = {}

        for i, passage in enumerate(passages):
            # Extraire les informations nécessaires
            filepath = passage.get("metadata", {}).get("filepath")
            if not filepath or not os.path.exists(filepath) or not filepath.lower().endswith('.pdf'):
                continue

            doc_title = passage.get("document_name", os.path.basename(filepath))
            passage_text = passage.get("text", "")

            # Créer la visualisation
            image_path = self.visualize_passage(
                filepath,
                passage_text,
                doc_title
            )

            # Stocker le résultat si réussi
            if image_path:
                visualizations[i + 1] = image_path

        return visualizations