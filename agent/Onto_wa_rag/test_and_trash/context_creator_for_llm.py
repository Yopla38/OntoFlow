"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from typing import List


class ContextBuilderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Générateur de Contexte Code")
        self.root.geometry("800x600")

        # Variables
        self.selected_folder = tk.StringVar()
        self.excluded_folders = [".venv", ".idea", ".git", "__pycache__", ".pytest_cache", "node_modules"]
        self.file_paths = {}  # Mapping item_id -> chemin complet

        self.setup_ui()

    def setup_ui(self):
        """Configure l'interface utilisateur"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configuration de la grille
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Sélection du dossier
        ttk.Label(main_frame, text="Dossier à analyser:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        folder_frame.columnconfigure(0, weight=1)

        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.selected_folder, state="readonly")
        self.folder_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Button(folder_frame, text="Parcourir", command=self.select_folder).grid(row=0, column=1)

        # Dossiers à exclure
        ttk.Label(main_frame, text="Dossiers exclus:").grid(row=1, column=0, sticky=(tk.W, tk.N), pady=(0, 5))

        exclusion_frame = ttk.Frame(main_frame)
        exclusion_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        exclusion_frame.columnconfigure(0, weight=1)

        self.exclusion_text = tk.Text(exclusion_frame, height=3, width=50)
        self.exclusion_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.exclusion_text.insert("1.0", ", ".join(self.excluded_folders))

        # Arborescence des fichiers
        ttk.Label(main_frame, text="Fichiers Python trouvés:").grid(row=2, column=0, sticky=(tk.W, tk.N), pady=(0, 5))

        tree_frame = ttk.Frame(main_frame)
        tree_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        # Treeview avec scrollbars - sélection multiple activée
        self.tree = ttk.Treeview(tree_frame, selectmode="extended")
        self.tree.heading("#0", text="Fichier")

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Boutons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(10, 0))

        ttk.Button(button_frame, text="Actualiser", command=self.refresh_file_list).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Sélectionner tous les fichiers", command=self.select_all_files).pack(
            side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Générer Contexte", command=self.generate_context).pack(side=tk.RIGHT,
                                                                                              padx=(5, 0))

        # Label d'information
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=4, column=0, columnspan=3, pady=(5, 0))
        ttk.Label(info_frame, text="Utilisez Ctrl+Clic ou Shift+Clic pour sélectionner plusieurs fichiers",
                  font=("Arial", 8), foreground="gray").pack()

    def select_folder(self):
        """Ouvre le dialogue de sélection de dossier"""
        folder = filedialog.askdirectory(title="Sélectionner le dossier à analyser")
        if folder:
            self.selected_folder.set(folder)
            self.refresh_file_list()

    def get_excluded_folders(self) -> List[str]:
        """Récupère la liste des dossiers exclus depuis le texte"""
        text = self.exclusion_text.get("1.0", tk.END).strip()
        if not text:
            return []
        return [folder.strip() for folder in text.split(",") if folder.strip()]

    def find_python_files(self, root_path: str) -> List[str]:
        """Trouve tous les fichiers Python dans le dossier, en excluant certains dossiers"""
        python_files = []
        excluded = set(self.get_excluded_folders())
        root_path = Path(root_path)

        try:
            for file_path in root_path.rglob("*.py"):
                # Vérifier si le fichier est dans un dossier exclu
                if any(excluded_folder in file_path.parts for excluded_folder in excluded):
                    continue
                python_files.append(str(file_path))
        except Exception as e:
            print(f"Erreur lors de la recherche de fichiers: {e}")
            return []

        return sorted(python_files)

    def refresh_file_list(self):
        """Met à jour la liste des fichiers dans l'arborescence"""
        if not self.selected_folder.get():
            return

        # Nettoyer l'arborescence
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.file_paths.clear()

        try:
            python_files = self.find_python_files(self.selected_folder.get())
            root_path = Path(self.selected_folder.get())

            if not python_files:
                # Aucun fichier trouvé
                item_id = self.tree.insert("", tk.END, text="Aucun fichier Python trouvé")
                return

            # Construire l'arborescence de manière plus simple
            self._build_simple_tree(python_files, root_path)

            # Sélectionner tous les fichiers par défaut
            self.select_all_files()

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la lecture du dossier: {str(e)}")
            print(f"Erreur détaillée: {e}")

    def _build_simple_tree(self, python_files: List[str], root_path: Path):
        """Construit l'arborescence de manière simple"""
        # Organiser les fichiers par dossier
        folder_structure = {}

        for file_path in python_files:
            file_path_obj = Path(file_path)
            try:
                relative_path = file_path_obj.relative_to(root_path)
                folder_parts = relative_path.parts[:-1]  # Tous sauf le nom du fichier
                file_name = relative_path.parts[-1]

                # Construire la structure de dossiers
                current_level = folder_structure
                for folder in folder_parts:
                    if folder not in current_level:
                        current_level[folder] = {'_files': [], '_subfolders': {}}
                    current_level = current_level[folder]['_subfolders']

                # Ajouter le fichier au bon endroit
                if folder_parts:
                    target_folder = folder_structure
                    for folder in folder_parts[:-1]:
                        target_folder = target_folder[folder]['_subfolders']
                    target_folder[folder_parts[-1]]['_files'].append((file_name, file_path))
                else:
                    # Fichier à la racine
                    if '_root_files' not in folder_structure:
                        folder_structure['_root_files'] = []
                    folder_structure['_root_files'].append((file_name, file_path))

            except Exception as e:
                print(f"Erreur avec le fichier {file_path}: {e}")
                continue

        # Construire l'arborescence dans le treeview
        self._add_to_tree("", folder_structure, root_path)

    def _add_to_tree(self, parent_item, structure, root_path):
        """Ajoute récursivement les éléments à l'arborescence"""
        # Ajouter les fichiers à la racine
        if '_root_files' in structure:
            for file_name, file_path in structure['_root_files']:
                item_id = self.tree.insert(parent_item, tk.END, text=file_name)
                self.file_paths[item_id] = file_path

        # Ajouter les dossiers et leurs contenus
        for folder_name, folder_data in structure.items():
            if folder_name == '_root_files':
                continue

            # Créer le dossier
            folder_item = self.tree.insert(parent_item, tk.END, text=f"📁 {folder_name}")

            # Ajouter les fichiers de ce dossier
            if '_files' in folder_data:
                for file_name, file_path in folder_data['_files']:
                    item_id = self.tree.insert(folder_item, tk.END, text=file_name)
                    self.file_paths[item_id] = file_path

            # Ajouter les sous-dossiers récursivement
            if '_subfolders' in folder_data and folder_data['_subfolders']:
                self._add_to_tree(folder_item, folder_data['_subfolders'], root_path)

    def select_all_files(self):
        """Sélectionne tous les fichiers (pas les dossiers)"""
        file_items = [item_id for item_id in self.file_paths.keys()]
        if file_items:
            self.tree.selection_set(file_items)

    def get_selected_files(self) -> List[str]:
        """Retourne la liste des fichiers sélectionnés"""
        selected_files = []
        selected_items = self.tree.selection()

        for item_id in selected_items:
            if item_id in self.file_paths:  # C'est un fichier, pas un dossier
                selected_files.append(self.file_paths[item_id])

        return selected_files

    def generate_context(self):
        """Génère le fichier de contexte"""
        if not self.selected_folder.get():
            messagebox.showwarning("Attention", "Veuillez sélectionner un dossier.")
            return

        selected_files = self.get_selected_files()
        if not selected_files:
            messagebox.showwarning("Attention", "Aucun fichier sélectionné.")
            return

        try:
            generator = ContextGenerator(self.selected_folder.get())
            output_file = generator.generate_context(selected_files)
            messagebox.showinfo("Succès",
                                f"Contexte généré avec succès!\nFichier: {output_file}\n\nNombre de fichiers traités: {len(selected_files)}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération: {str(e)}")


class ContextGenerator:
    def __init__(self, root_folder: str):
        self.root_folder = Path(root_folder)

    def generate_context(self, file_paths: List[str]) -> str:
        """Génère le fichier de contexte à partir des fichiers sélectionnés"""
        output_file = self.root_folder / "contexte.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            for file_path in file_paths:
                try:
                    file_path_obj = Path(file_path)
                    relative_path = file_path_obj.relative_to(self.root_folder)

                    # Écrire le header du fichier
                    f.write(f"<{relative_path}>\n")

                    # Lire et écrire le contenu du fichier
                    with open(file_path, 'r', encoding='utf-8') as source_file:
                        content = source_file.read()
                        f.write(content)
                        if not content.endswith('\n'):
                            f.write('\n')

                    # Écrire le footer du fichier
                    f.write(f"</{relative_path}>\n\n")

                except Exception as e:
                    # En cas d'erreur de lecture d'un fichier, on l'indique dans le contexte
                    f.write(f"<{relative_path}>\n")
                    f.write(f"ERREUR DE LECTURE: {str(e)}\n")
                    f.write(f"</{relative_path}>\n\n")

        return str(output_file)


def main():
    root = tk.Tk()
    app = ContextBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()