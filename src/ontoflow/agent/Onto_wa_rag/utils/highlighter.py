"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# highlighter.py
import os
import asyncio
import tempfile
import shutil
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF


class Highlighter:
    """Surlignage des passages pertinents dans les PDFs"""

    def __init__(self, document_store, output_dir: str = "highlighted"):
        """
        Initialise le surligneur

        Args:
            document_store: Stockage de documents
            output_dir: Répertoire pour stocker les PDFs surlignés
        """
        self.document_store = document_store
        self.output_dir = output_dir

        # Créer le répertoire s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)

    async def highlight_passages(
            self,
            document_id: str,
            passages: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Surligne des passages dans un PDF

        Args:
            document_id: ID du document
            passages: Liste des passages à surligner

        Returns:
            Chemin du PDF surligné ou None si échec
        """
        # Vérifier que le document existe
        document = await self.document_store.get_document(document_id)

        if not document:
            return None

        document_path = document["path"]

        # Vérifier que c'est bien un PDF
        if not document_path.lower().endswith('.pdf'):
            return None

        # Créer un fichier de sortie
        filename = os.path.basename(document_path)
        output_path = os.path.join(self.output_dir, f"highlighted_{filename}")

        # Exécuter le surlignage de façon asynchrone
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: self._highlight_pdf_sync(document_path, output_path, passages)
        )

        return output_path if success else None

    def _highlight_pdf_sync(
            self,
            input_path: str,
            output_path: str,
            passages: List[Dict[str, Any]]
    ) -> bool:
        """
        Version synchrone du surlignage de PDF

        Args:
            input_path: Chemin du PDF d'entrée
            output_path: Chemin du PDF de sortie
            passages: Liste des passages à surligner

        Returns:
            True si le surlignage a réussi, False sinon
        """
        try:
            # Ouvrir le document
            doc = fitz.open(input_path)

            for passage in passages:
                # Utiliser les 100 premiers caractères pour trouver le passage dans le PDF
                text_to_find = passage["text"][:100]

                # Chercher le texte dans chaque page
                for page_num, page in enumerate(doc):
                    text = page.get_text()

                    if text_to_find in text:
                        # Rechercher toutes les instances du texte dans la page
                        text_instances = page.search_for(text_to_find)

                        # Surligner chaque instance
                        for inst in text_instances:
                            # Créer un surlignage jaune semi-transparent
                            highlight = page.add_highlight_annot(inst)
                            highlight.set_colors({"stroke": (1, 1, 0)})
                            highlight.update()

            # Sauvegarder le document modifié
            doc.save(output_path)
            doc.close()

            return True

        except Exception as e:
            print(f"Erreur lors du surlignage du PDF: {str(e)}")
            return False