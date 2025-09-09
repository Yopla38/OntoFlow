# pip install Pillow
from PIL import Image

class ImageProcessor:
    """
    Une classe simple pour ouvrir, tourner et sauvegarder des images.
    """

    def __init__(self, image_path=None):
        """
        Initialise le processeur d'image.

        :param image_path: Chemin optionnel vers l'image à ouvrir lors de l'initialisation.
        """
        self.image = None
        self.original_format = None
        if image_path:
            self.open_image(image_path)

    def open_image(self, image_path):
        """
        Ouvre une image à partir du chemin spécifié.

        :param image_path: Le chemin vers le fichier image.
        :raises FileNotFoundError: Si le fichier image n'est pas trouvé.
        :raises Exception: Pour d'autres erreurs liées à l'ouverture d'image.
        """
        try:
            self.image = Image.open(image_path)
            self.original_format = self.image.format
            print(f"Image '{image_path}' ouverte avec succès.")
        except FileNotFoundError:
            print(f"Erreur : Le fichier '{image_path}' n'a pas été trouvé.")
            self.image = None
            self.original_format = None
            raise
        except Exception as e:
            print(f"Erreur lors de l'ouverture de l'image '{image_path}': {e}")
            self.image = None
            self.original_format = None
            raise

    def rotate_image(self, angle):
        """
        Tourne l'image actuellement chargée selon l'angle spécifié.

        :param angle: L'angle de rotation en degrés. Les valeurs positives
                      indiquent une rotation dans le sens antihoraire.
        :raises ValueError: Si aucune image n'est chargée.
        """
        if self.image:
            # La méthode rotate de Pillow fait une rotation antihoraire
            self.image = self.image.rotate(angle, expand=True) # expand=True pour que l'image entière soit visible
            print(f"Image tournée de {angle} degrés.")
        else:
            print("Erreur : Aucune image n'est chargée pour la rotation.")
            raise ValueError("Aucune image chargée.")

    def save_image(self, save_path, file_format=None):
        """
        Sauvegarde l'image actuellement chargée vers le chemin spécifié.

        :param save_path: Le chemin où sauvegarder l'image.
        :param file_format: Format optionnel pour sauvegarder l'image (ex: "PNG", "JPEG").
                            Si None, essaie d'utiliser le format original ou déduit de l'extension.
        :raises ValueError: Si aucune image n'est chargée.
        :raises Exception: Pour les erreurs liées à la sauvegarde d'image.
        """
        if self.image:
            try:
                save_format = file_format if file_format else self.original_format
                if not save_format: # Si toujours None, déduit de l'extension du chemin de sauvegarde
                    save_format = save_path.split('.')[-1].upper()
                    if save_format not in ["JPEG", "PNG", "GIF", "BMP", "TIFF"]: # Formats courants
                        print(f"Attention: Format de sauvegarde '{save_format}' non reconnu, utilisation de PNG par défaut.")
                        save_format = "PNG" # Un format par défaut sûr

                self.image.save(save_path, format=save_format)
                print(f"Image sauvegardée avec succès sous '{save_path}' au format {save_format}.")
            except Exception as e:
                print(f"Erreur lors de la sauvegarde de l'image sous '{save_path}': {e}")
                raise
        else:
            print("Erreur : Aucune image n'est chargée pour la sauvegarde.")
            raise ValueError("Aucune image chargée.")

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Créez un fichier image nommé "test_image.png" (ou utilisez une image existante)
    # pour tester ce script. Par exemple, un simple carré rouge.
    try:
        # Test avec une image exemple (créez-en une si elle n'existe pas)
        try:
            img_test = Image.new('RGB', (200, 100), color = 'red')
            img_test.save("test_image.png")
            print("Image 'test_image.png' créée pour l'exemple.")
        except Exception as e:
            print(f"Impossible de créer l'image de test : {e}")
            # Si la création échoue, l'utilisateur devra fournir sa propre image pour que l'exemple fonctionne.

        # 1. Initialiser la classe et ouvrir une image
        processor = ImageProcessor()
        try:
            processor.open_image("test_image.png") # Remplacez par le chemin de votre image

            # 2. Tourner l'image
            if processor.image:
                processor.rotate_image(90) # Tourne de 90 degrés dans le sens antihoraire

            # 3. Sauvegarder l'image tournée
            if processor.image:
                processor.save_image("test_image_tournee.png")
                # Sauvegarder dans un autre format
                processor.save_image("test_image_tournee.jpg", file_format="JPEG")

            print("\n--- Test avec initialisation directe ---")
            processor2 = ImageProcessor("test_image.png")
            if processor2.image:
                processor2.rotate_image(-45) # Tourne de 45 degrés dans le sens horaire
                processor2.save_image("test_image_tournee_45.png")

        except FileNotFoundError:
            print("Veuillez créer une image 'test_image.png' ou spécifier un chemin d'image valide.")
        except ValueError as ve:
            print(f"Une erreur de valeur s'est produite: {ve}")
        except Exception as e:
            print(f"Une erreur inattendue s'est produite: {e}")


    except ImportError:
        print("La bibliothèque Pillow n'est pas installée. Veuillez l'installer avec 'pip install Pillow'.")
