"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# providers/llm_providers.py
import asyncio
import base64
import gc
import io
import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, AsyncGenerator, Tuple

import instructor
import requests
import torch
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

import openai
from anthropic import Anthropic
from transformers import AutoTokenizer, AutoModel

from ..CONSTANT import API_KEY_PATH, EMBEDDING_MODEL, LOCAL_EMBEDDING_PATH, LANGUAGE, OLLAMA_BASE_URL
from ..provider.get_key import get_openai_key

#  Installation pour le provider local :
# https: // forums.developer.nvidia.com / t / installing - cuda - on - ubuntu - 22 - 04 - rxt4080 - laptop / 292899
# CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=off -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

# attempt to collect openai API key
api_key = get_openai_key(api_key_path=API_KEY_PATH)

try:
    # if no key is found, assume local deployment
    if api_key == "":
        CLIENT_OPENAI = openai.AsyncClient(base_url=OLLAMA_BASE_URL, api_key="ollama")
    else:
        CLIENT_OPENAI = openai.AsyncClient(api_key=api_key)
finally:
    # delete api_key from memory
    del api_key

SPECTER_PATH = LOCAL_EMBEDDING_PATH

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


def _check_llama_cpp_available() -> bool:
    try:
        import llama_cpp
        return True
    except ImportError:
        return False


@dataclass
class Message:
    """Classe représentant un message dans l'historique"""
    role: str  # 'user' ou 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationHistory:
    """Classe gérant l'historique des conversations"""

    def __init__(self, max_messages: int = 10):
        self.messages: List[Message] = []
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """Ajoute un message à l'historique"""
        self.messages.append(Message(role=role, content=content))
        # Garde uniquement les n derniers messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_context_string(self, format_type: str = "anthropic") -> str:
        """Retourne l'historique formaté selon le provider"""
        if format_type == "anthropic":
            return "\n".join([
                f"{msg.role}: {msg.content}"
                for msg in self.messages
            ])
        elif format_type == "deepseek":
            return "\n".join([
                f"<｜{msg.role.capitalize()}｜>{msg.content}"
                for msg in self.messages
            ])

    def clear(self):
        """Efface l'historique"""
        self.messages.clear()


class LLMProvider(ABC):
    def __init__(self):
        self.history = ConversationHistory()
        self.log_file = 'GPT_log.json'

    @abstractmethod
    async def generate_response(self, prompt: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        pass

    def write_log(self, receive_text=None):
        if self.log_file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log = f"Timestamp : {timestamp}\n{receive_text}"
            with open(self.log_file, 'w') as f:
                f.write(log)

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour une liste de textes"""
        pass

    def generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Version synchrone pour générer des embeddings"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_embeddings(texts))
        finally:
            loop.close()

    def generate_response_sync(self, messages, **kwargs):
        """Version synchrone pour générer une réponse"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_response(messages, **kwargs))
        finally:
            loop.close()

    def set_system_prompt(self, system_prompt):
        pass


class OpenAIProvider(LLMProvider):
    def __init__(self,
                 model: str,
                 api_key: str,
                 system_prompt: Optional[str] = None,
                 functions: Optional[List[Dict[str, Any]]] = None
                 ):
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.client = CLIENT_OPENAI  # Utilisation du client asynchrone

    @staticmethod
    def _ensure_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, indent=2)
        elif isinstance(content, (list, tuple)):
            return "\n".join(str(item) for item in content)
        else:
            return str(content)

    @retry(
        wait=wait_random_exponential(min=1, max=40),  # Attente exponentielle entre 1 et 40 secondes
        stop=stop_after_attempt(3),  # Arrêter après 3 tentatives
        retry=retry_if_exception_type((
                openai.APIError,  # Erreurs d'API générales
                openai.APIConnectionError,  # Erreurs de connexion à l'API
                ConnectionError,  # Erreurs de connexion réseau Python
                TimeoutError,  # Dépassements de délai
                requests.exceptions.RequestException  # Erreurs de requête HTTP
        ))  # Type d'erreur à intercepter
    )
    async def generate_response(
            self,
            messages: Union[str, List[Dict[str, str]]],
            stream: bool = False,
            pydantic_model: Optional[BaseModel] = None,
            functions=None,
            **kwargs
    ) -> Union[str, BaseModel, AsyncGenerator]:
        try:
            # Préparer les messages
            if isinstance(messages, str):
                formatted_messages = []
                if self.system_prompt:
                    formatted_messages.append({"role": "system", "content": self.system_prompt})
                formatted_messages.append({"role": "user", "content": self._ensure_text_content(messages)})
            else:
                formatted_messages = messages

            # Préparer les paramètres de base
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": stream,
            }

            # Ajouter les functions si disponibles
            if functions:
                params["functions"] = functions
                params["function_call"] = "auto"

            # Ajouter les kwargs supplémentaires
            params.update(kwargs)

            self.write_log(params)
            # Si streaming est demandé
            if stream:
                return await self.client.chat.completions.create(**params)

            # Si un modèle Pydantic est fourni
            if pydantic_model:
                response = await self.client.beta.chat.completions.parse(
                    messages=formatted_messages,
                    model=self.model,
                    response_format=pydantic_model,
                )

                return response.choices[0].message.parsed.model_dump()

            # Cas standard
            response = await self.client.chat.completions.create(**params)

            return response.choices[0].message.content

        except openai.APIError as e:
            logging.error(f"OpenAI API Error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in generate_response: {str(e)}")
            raise

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=True) -> Any:
        """Méthode spécifique pour la communication humaine avec streaming"""
        try:
            return await self.generate_response(
                messages=messages,
                stream=stream
            )
        except Exception as e:
            logging.error(f"Error in generate_response_for_humain: {str(e)}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=40),  # Attente exponentielle entre 1 et 40 secondes
        stop=stop_after_attempt(3),  # Arrêter après 3 tentatives
        retry=retry_if_exception_type((
                openai.APIError,  # Erreurs d'API générales
                openai.APIConnectionError,  # Erreurs de connexion à l'API
                ConnectionError,  # Erreurs de connexion réseau Python
                TimeoutError,  # Dépassements de délai
                requests.exceptions.RequestException  # Erreurs de requête HTTP
        ))  # Type d'erreur à intercepter
    )
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings avec le modèle OpenAI"""
        try:
            batch_size = 100  # Limite de l'API OpenAI
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except Exception as e:
            logging.error(f"Erreur d'embeddings OpenAI: {str(e)}")
            # Retourner une erreur explicite plutôt que des embeddings aléatoires
            raise


class AnthropicProvider(LLMProvider):
    """
    Provider pour l'API Claude d'Anthropic avec support des modèles Pydantic.
    """

    def __init__(self, model: str, api_key: str, system_prompt: Optional[str] = None):
        """
        Initialise le provider Anthropic.

        Args:
            model: Nom du modèle à utiliser (ex: "claude-3-opus-20240229")
            api_key: Clé API Anthropic
            system_prompt: Prompt système optionnel
        """
        # Configuration de base
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt

        # Création du client instructor pour les réponses structurées
        self.structured_client = instructor.from_anthropic(
            Anthropic(api_key=api_key)
        )

        # Client standard pour les réponses textuelles
        self.text_client = Anthropic(api_key=api_key)

        # ThreadPoolExecutor pour les appels synchrones
        self.executor = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def _ensure_text_content(content: Any) -> str:
        """Convertit le contenu en texte si nécessaire"""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, indent=2)
        elif isinstance(content, (list, tuple)):
            return "\n".join(str(item) for item in content)
        else:
            return str(content)

    async def generate_response(
            self,
            prompt: str,
            pydantic_model: Optional[BaseModel] = None,
            **kwargs
    ) -> Union[str, BaseModel]:
        """
        Génère une réponse à partir du prompt.

        Args:
            prompt: Le prompt à envoyer au modèle
            pydantic_model: Modèle Pydantic optionnel pour structurer la réponse
            **kwargs: Arguments additionnels pour l'API

        Returns:
            Soit une chaîne de caractères, soit une instance du modèle Pydantic
        """
        # Ajoute le message utilisateur à l'historique
        self.history.add_message("user", prompt)

        # Construit le contexte avec l'historique
        context = self.history.get_context_string("anthropic")

        try:
            # Conversion du prompt en texte si nécessaire
            # text_prompt = self._ensure_text_content(context)
            if pydantic_model:
                response = await self._generate_structured_response(context, pydantic_model, **kwargs)
            else:
                response = await self._generate_text_response(context, **kwargs)

            # Ajoute la réponse à l'historique
            if isinstance(response, str):
                self.history.add_message("assistant", response)
            else:
                self.history.add_message("assistant", str(response))

            return response
        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse: {str(e)}")

    async def _generate_structured_response(
            self,
            prompt: str,
            pydantic_model: BaseModel,
            **kwargs
    ) -> BaseModel:
        """
        Génère une réponse structurée selon un modèle Pydantic.
        """
        print("Entering _generate_structured_response")

        try:
            # Préparation des paramètres de base
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 8192),
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }],
                "response_model": pydantic_model
            }

            # Ajout du prompt système si présent
            if self.system_prompt:
                request_params["system"] = self.system_prompt

            # Mise à jour avec les kwargs additionnels
            request_params.update(kwargs)

            print(f"Request params: {request_params}")
            # Exécution de la requête dans le thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.structured_client.messages.create(**request_params)
            )
            print(f"LA REPONSE: ")
            print(response)
            return response

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse structurée: {str(e)}")

    async def _generate_text_response(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse textuelle simple.
        """
        try:
            # Préparation des paramètres de base
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "messages": [{"role": "user", "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]}]
            }

            # Ajout du prompt système si présent
            if self.system_prompt:
                request_params["system"] = self.system_prompt

            # Mise à jour avec les kwargs additionnels
            request_params.update(kwargs)

            # Exécution de la requête dans le thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.text_client.messages.create(**request_params)
            )

            # Extraction du contenu de la réponse
            content = response.content[0].text if hasattr(response, 'content') else response.content
            return content

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse textuelle: {str(e)}")

    async def generate_response_from_messages(
            self,
            messages: List[Dict[str, str]],
            pydantic_model: Optional[BaseModel] = None,
            **kwargs
    ) -> Union[str, BaseModel]:
        """Génère une réponse à partir d'une liste de messages (compatibilité OpenAI)"""

        # Extraire le system prompt s'il existe
        system_prompt = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(f"{msg['role'].upper()}: {msg['content']}")

        # Construire le prompt final
        prompt = "\n\n".join(user_messages)

        # Sauvegarder l'ancien system prompt et utiliser le nouveau temporairement
        old_system_prompt = self.system_prompt
        if system_prompt:
            self.system_prompt = system_prompt

        try:
            result = await self.generate_response(prompt, pydantic_model, **kwargs)
            return result
        finally:
            # Restaurer l'ancien system prompt
            self.system_prompt = old_system_prompt

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        pass
        return {}

    def set_system_prompt(self, system_prompt: str):
        """
        Définit ou met à jour le prompt système.
        """
        self.system_prompt = system_prompt

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    async def generate_vision_response(
            self,
            prompt: str,
            image_data: Union[str, bytes],
            pydantic_model: Optional[BaseModel] = None,
            image_format: str = "jpeg",
            **kwargs
    ) -> Union[str, BaseModel]:
        """
        Génère une réponse à partir d'un prompt et d'une image.

        Args:
            prompt: Le prompt à envoyer au modèle
            image_data: Image en base64 (str) ou bytes
            pydantic_model: Modèle Pydantic optionnel pour structurer la réponse
            image_format: Format de l'image (jpeg, png, webp, gif)
            **kwargs: Arguments additionnels pour l'API

        Returns:
            Soit une chaîne de caractères, soit une instance du modèle Pydantic
        """
        # Préparation de l'image
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        else:
            image_b64 = image_data

        # Ajoute le message utilisateur à l'historique (sans l'image pour économiser l'espace)
        self.history.add_message("user", f"[IMAGE] {prompt}")

        try:
            if pydantic_model:
                response = await self._generate_vision_structured_response(
                    prompt, image_b64, pydantic_model, image_format, **kwargs
                )
            else:
                response = await self._generate_vision_text_response(
                    prompt, image_b64, image_format, **kwargs
                )

            # Ajoute la réponse à l'historique
            if isinstance(response, str):
                self.history.add_message("assistant", response)
            else:
                self.history.add_message("assistant", str(response))

            return response

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse vision: {str(e)}")

    async def _generate_vision_structured_response(
            self,
            prompt: str,
            image_b64: str,
            pydantic_model: BaseModel,
            image_format: str = "jpeg",
            **kwargs
    ) -> BaseModel:
        """Génère une réponse vision structurée selon un modèle Pydantic."""

        try:
            # Préparation des paramètres
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 8192),
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }],
                "response_model": pydantic_model
            }

            # Ajout du prompt système si présent
            if self.system_prompt:
                request_params["system"] = self.system_prompt

            # Mise à jour avec les kwargs additionnels
            request_params.update(kwargs)

            # Exécution de la requête dans le thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.structured_client.messages.create(**request_params)
            )

            return response

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse vision structurée: {str(e)}")

    async def _generate_vision_text_response(
            self,
            prompt: str,
            image_b64: str,
            image_format: str = "jpeg",
            **kwargs
    ) -> str:
        """Génère une réponse vision textuelle simple."""

        try:
            # Préparation des paramètres
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 4096),
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image_format}",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            }

            # Ajout du prompt système si présent
            if self.system_prompt:
                request_params["system"] = self.system_prompt

            # Mise à jour avec les kwargs additionnels
            request_params.update(kwargs)

            # Exécution de la requête dans le thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.text_client.messages.create(**request_params)
            )

            # Extraction du contenu de la réponse
            content = response.content[0].text if hasattr(response, 'content') else response.content
            return content

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse vision textuelle: {str(e)}")

    async def process_image_for_vision(self, image_data: bytes) -> Tuple[str, str]:
        """
        Traite une image pour l'optimiser pour l'API vision.

        Args:
            image_data: Données binaires de l'image

        Returns:
            Tuple (image_base64, format_detected)
        """
        try:
            # Ouvrir l'image
            image = Image.open(io.BytesIO(image_data))

            # Convertir en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Redimensionner si nécessaire (limite Claude)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Sauvegarder en JPEG optimisé dans un buffer
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)

            # Encoder en base64
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            return image_b64, "jpeg"

        except Exception as e:
            raise ValueError(f"Erreur traitement image: {e}")

    def __del__(self):
        """
        Nettoyage des ressources à la destruction de l'instance.
        """
        self.executor.shutdown(wait=False)


class LocalDeepSeek_R1_Provider(LLMProvider):
    def __init__(self, model: str, api_key: str, cached_model=None, system_prompt: Optional[str] = None,
                 structured_response: Optional[str] = None):

        super().__init__()
        if not _check_llama_cpp_available():
            raise ImportError(
                """Ce provider nécessite llama_cpp. Installez-le avec: CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
"""
            )

        from llama_cpp import Llama

        self.system_prompt = system_prompt
        self.structured_response = structured_response
        self.executor = ThreadPoolExecutor(max_workers=1)  # Créer l'executor avant le modèle

        # Utiliser le modèle en cache ou en charger un nouveau
        if cached_model:
            self.llm = cached_model
        else:
            #  Vérification du gpu et optimisation du nb de couche sur gpu
            gpu_layers = get_optimal_n_gpu_layers(default=0)
            print(f"Nombre de couches gpu utilisées : {str(gpu_layers)}")
            self.llm = Llama(
                model_path=model,
                n_gpu_layers=gpu_layers,
                n_ctx=22000,
                verbose=False
            )

    def get_model(self):
        """Retourne l'instance du modèle pour le cache."""
        return self.llm

    def _format_prompt(self, prompt: str, structured: bool = False, pydantic_model: BaseModel = None) -> str:
        """Formate le prompt selon le template DeepSeek."""
        formatted = ""

        json_instructions = ""
        if " json " in prompt.lower():
            # Instructions spécifiques pour le format JSON
            json_instructions = """
                IMPORTANT - Format de sortie JSON :
                1. Utilisez uniquement des guillemets doubles (") pour les clés et les valeurs JSON
                2. Dans les chaînes de caractères Python (content), utilisez des guillemets simples (')
                3. Échappez les guillemets doubles dans le code Python avec un backslash (\")
                4. Pour les caractères spéciaux, utilisez leur forme Unicode directe (é, è, à) plutôt que leur forme encodée (\x27, \xc3\xa9)
                5. Les retours à la ligne dans le code doivent être représentés par \n

                Exemple de format attendu :
                {
                    "files": [{
                        "file_path": "example.py",
                        "code_field": {
                            "language": "python",
                            "content": "def example():\n    print('Hello world!')\n    value = \"test\""
                        },
                        "dependencies": ["package1"]
                    }]
                }\n\n
                """

        if self.system_prompt:
            formatted += f"{self.system_prompt}"

        # Ajout de l'historique
        history_context = self.history.get_context_string("deepseek")
        if history_context:
            formatted += f"{history_context}\n"

        formatted += f"<｜User｜>{prompt}\n\n"

        formatted += json_instructions if json_instructions else ""
        formatted += self.structured_response if self.structured_response else ""

        formatted += "<｜Assistant｜>"
        return formatted

    @staticmethod
    def _ensure_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, indent=2)
        elif isinstance(content, (list, tuple)):
            return "\n".join(str(item) for item in content)
        else:
            return str(content)

    async def generate_response(
            self,
            prompt: str,
            pydantic_model: Optional[BaseModel] = None,
            stream: bool = False,
            callback=None,
            **kwargs
    ) -> Union[str, BaseModel]:
        """
        Génère une réponse, avec support optionnel du streaming.

        Args:
            prompt: Le prompt utilisateur
            pydantic_model: Modèle Pydantic optionnel pour structurer la réponse
            stream: Si True, génère la réponse en streaming
            callback: Fonction à appeler pour chaque chunk (utilisée en mode streaming)
            **kwargs: Arguments supplémentaires

        Returns:
            La réponse générée, sous forme de texte ou d'objet Pydantic
        """
        # Ajoute le message utilisateur à l'historique
        self.history.add_message("user", prompt)

        structured_prompt = self._format_prompt(prompt)
        kwargs["callback"] = callback  # Passer le callback aux méthodes internes

        try:
            if stream and not self.structured_response:
                # Mode streaming pour les réponses non structurées
                async for chunk in self._stream_response(structured_prompt, **kwargs):
                    # Les chunks sont déjà traités par la méthode _stream_response
                    # qui met à jour l'historique à la fin du streaming
                    pass

                # Récupérer la réponse complète depuis l'historique
                if self.history.messages and self.history.messages[-1].role == "assistant":
                    return self.history.messages[-1].content
                return ""
            else:
                # Version non-streaming originale pour les réponses structurées
                # ou quand le streaming n'est pas demandé
                response = await self._generate_text_response(structured_prompt, **kwargs)

                # Ajoute la réponse à l'historique
                if isinstance(response, str):
                    self.history.add_message("assistant", response)
                else:
                    self.history.add_message("assistant", str(response))

                structured_message = None
                if self.structured_response:
                    if " json " in self.structured_response.lower():
                        structured_message = extract_json_only(response)
                        structured_message = self.extract_json(structured_message)
                    elif "```language" in self.structured_response.lower():
                        structured_message = extract(response,
                                                     ["python", "css", "html", "js", "jsx", "javascript", "markdown",
                                                      "",
                                                      "plaintext", "xml", "json", "yaml"])
                else:
                    structured_message = response

                try:
                    if structured_message:
                        structured_message = self.encode_json(structured_message)
                except Exception as e:
                    return structured_message if structured_message else response

                if pydantic_model and structured_message:
                    # Extraire les données selon le schéma
                    return pydantic_model.parse_obj(structured_message)
                else:
                    return structured_message if structured_message else response

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse: {str(e)}")

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=True) -> Any:
        """Méthode spécifique pour le streaming compatible avec l'application cliente"""
        try:
            # Extraire le dernier message utilisateur
            last_user_message = next((msg["content"] for msg in reversed(messages)
                                      if msg["role"] == "user"), None)

            if last_user_message:
                if stream:
                    # Version streaming
                    async def stream_generator():
                        async for chunk in self._stream_response(
                                self._format_prompt(last_user_message),
                                temperature=0.7
                        ):
                            yield {"chunk": chunk}

                    return stream_generator()
                else:
                    # Version non-streaming
                    return await self.generate_response(last_user_message, stream=False)
            return "Aucun message utilisateur trouvé"
        except Exception as e:
            logging.error(f"Error in generate_response_for_humain: {str(e)}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour une liste de textes"""
        pass

    def extract_json(self, texte):
        code_match = re.search(r"```json\n(.*?)```", texte, re.DOTALL | re.IGNORECASE)
        if code_match:
            texte = re.sub(r'("bloc de code": ")[^"]*\n[^"]*(")',
                           lambda m: m.group(1) + m.group(0).replace('\n', '\\n')[
                                                  len(m.group(1)):-len(m.group(2))] + m.group(2), code_match.group(1))

            return texte
        return texte

    def encode_json(self, response: str) -> dict:
        """Extrait et valide le JSON de la réponse."""
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        try:
            # Si la réponse est déjà un dictionnaire
            if isinstance(response, dict):
                return response

            # Si c'est une chaîne, on essaie de la parser
            if isinstance(response, str):
                # Première désérialisation pour gérer les \n et \"
                json_str = json.loads(response)

                # Si json_str est encore une chaîne (double encodage), on parse à nouveau
                if isinstance(json_str, str):
                    json_str = json.loads(json_str)

                return json_str

        except Exception as e:
            raise ValueError(f"Erreur lors du parsing JSON: {str(e)}\nRéponse reçue: {response}")

    async def _generate_text_response(self, prompt: str, stream: bool = False, **kwargs) -> str:
        try:
            if stream:
                # Si le streaming est demandé, on utilise une autre méthode
                async for chunk in self._stream_response(prompt, **kwargs):
                    pass  # On consomme tous les chunks
                # Retourner la réponse complète depuis l'historique
                if self.history.messages and self.history.messages[-1].role == "assistant":
                    return self.history.messages[-1].content
                return ""  # Fallback si pas d'historique
            else:
                # Version non-streaming existante
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.executor,
                    lambda: self.llm(
                        prompt,
                        max_tokens=kwargs.get("max_tokens", 4000),
                        temperature=kwargs.get("temperature", 0.7),
                        stop=["<｜end▁of▁sentence｜>", "<｜User｜>"]
                    )
                )
                return response["choices"][0]["text"].strip()
        except Exception as e:
            raise Exception(f"Erreur lors de la génération textuelle: {str(e)}")

    async def _stream_response(self, prompt: str, **kwargs):
        """Génère une réponse en streaming, chunk par chunk"""
        try:
            loop = asyncio.get_event_loop()

            # Créer un générateur qui appelle llama_cpp avec stream=True
            stream_gen = await loop.run_in_executor(
                self.executor,
                lambda: self.llm(
                    prompt,
                    max_tokens=kwargs.get("max_tokens", 4000),
                    temperature=kwargs.get("temperature", 0.7),
                    stop=["<｜end▁of▁sentence｜>", "<｜User｜>"],
                    stream=True  # Activer le streaming
                )
            )

            full_response = ""

            # Parcourir les chunks générés
            for chunk in stream_gen:
                if "choices" in chunk and chunk["choices"]:
                    chunk_text = chunk["choices"][0]["text"]
                    if chunk_text:
                        full_response += chunk_text
                        # Si le callback est fourni, l'appeler avec le chunk
                        if kwargs.get("callback"):
                            kwargs["callback"](chunk_text)
                        yield chunk_text

            # Stocker la réponse complète dans l'historique
            self.history.add_message("assistant", full_response.strip())

        except Exception as e:
            raise Exception(f"Erreur lors du streaming de la réponse: {str(e)}")

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def set_structured_local_model(self, structured_response_for_local_model):
        self.structured_response = structured_response_for_local_model + "\nAssurez-vous d'échapper correctement les caractères spéciaux dans le code, notamment les backslashes doivent être doublés (\\\\)."

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class LocalEmbeddingProvider(LLMProvider):
    """
    Provider pour générer des embeddings localement avec llama-cpp.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048):
        """
        Initialise le provider d'embedding local.

        Args:
            model_path: Chemin vers le modèle d'embedding
            n_ctx: Taille du contexte pour le modèle
        """
        super().__init__()
        if not _check_llama_cpp_available():
            raise ImportError(
                """Ce provider nécessite llama_cpp. Installez-le avec: CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
"""
            )
        from llama_cpp import Llama

        # Déterminer le nombre optimal de couches GPU
        gpu_layers = get_optimal_n_gpu_layers(default=0)
        print(f"Nombre de couches GPU utilisées pour l'embedding : {str(gpu_layers)}")

        # Initialiser le modèle
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=gpu_layers,
            n_ctx=n_ctx,
            embedding=True,  # Activer le mode embedding
            verbose=False
        )

        self.executor = ThreadPoolExecutor(max_workers=2)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Génère des embeddings pour une liste de textes en utilisant le modèle local.

        Args:
            texts: Liste de textes à encoder

        Returns:
            Liste d'embeddings (vecteurs de nombres flottants)
        """
        try:
            # Traiter les textes par lots si nécessaire
            batch_size = 10  # Ajuster selon les capacités de votre GPU
            embeddings = []

            # Traiter par lots
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._process_batch(batch)
                embeddings.extend(batch_embeddings)

            return embeddings
        except Exception as e:
            logging.error(f"Erreur d'embeddings locale: {str(e)}")
            raise

    async def _process_batch(self, batch: List[str]) -> List[List[float]]:
        """
        Traite un lot de textes pour générer des embeddings.

        Args:
            batch: Lot de textes à traiter

        Returns:
            Liste d'embeddings pour le lot
        """
        loop = asyncio.get_event_loop()
        batch_results = []

        for text in batch:
            # Exécuter dans le thread pool pour ne pas bloquer la boucle asyncio
            embedding = await loop.run_in_executor(
                self.executor,
                lambda t=text: self.llm.embed(t)
            )
            batch_results.append(embedding)

        return batch_results

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Cette classe est spécialisée pour l'embedding, pas pour la génération de texte.
        Cette méthode n'est implémentée que pour respecter l'interface LLMProvider.
        """
        raise NotImplementedError("Ce provider est uniquement pour les embeddings, pas pour la génération de texte.")

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        """
        Cette classe est spécialisée pour l'embedding, pas pour la génération de texte.
        Cette méthode n'est implémentée que pour respecter l'interface LLMProvider.
        """
        raise NotImplementedError("Ce provider est uniquement pour les embeddings, pas pour la génération de texte.")

    def __del__(self):
        """
        Nettoyage des ressources à la destruction de l'instance.
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class SpecterProvider(LLMProvider):
    def __init__(self, device: str = "cuda"):
        """Initialise le modèle SPECTER en local."""
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        model_path = SPECTER_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings en mode asynchrone."""
        return await asyncio.to_thread(self._generate_embeddings_sync, texts)

    def generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings en mode synchrone."""
        return self._generate_embeddings_sync(texts)

    def _generate_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """Méthode privée pour générer les embeddings."""
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token

            return embeddings.tolist()
        except Exception as e:
            logging.error(f"Erreur dans generate_embeddings: {e}")
            raise

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """SPECTER ne génère pas de texte, donc cette méthode ne fait rien."""
        logging.warning("SPECTER est un modèle d'embedding, pas un modèle conversationnel.")
        return "SPECTER ne génère pas de réponse textuelle."

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        """SPECTER ne génère pas de texte."""
        logging.warning("SPECTER est un modèle d'embedding, pas un modèle conversationnel.")
        return {"error": "SPECTER ne supporte pas la génération de texte."}


class LocalMultimodalProvider(LLMProvider):
    def __init__(self, model: str, clip_model_path: str, api_key: Optional[str] = None, cached_model=None, system_prompt: Optional[str] = None,
                 structured_response: Optional[str] = None):

        super().__init__()

        if not _check_llama_cpp_available():
            raise ImportError(
                """Ce provider nécessite llama_cpp. Installez-le avec: CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade
"""
            )

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler

        self.system_prompt = system_prompt
        self.structured_response = structured_response
        self.executor = ThreadPoolExecutor(max_workers=1)

        if cached_model:
            self.llm = cached_model
        else:
            gpu_layers = get_optimal_n_gpu_layers(default=-1) # -1 pour tout mettre sur le GPU
            print(f"Nombre de couches gpu utilisées : {str(gpu_layers)}")

            chat_handler = Llava15ChatHandler(clip_model_path=clip_model_path, verbose=False)

            # 2. Initialiser Llama en lui passant le chat_handler
            self.llm = Llama(
                model_path=model,
                chat_handler=chat_handler,  # Le point crucial
                n_gpu_layers=gpu_layers,
                n_ctx=4096,
                verbose=False
            )

    def get_model(self):
        return self.llm

    def _prepare_image(self, image_path: str) -> str:
        """Charge une image et l'encode en base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise IOError(f"Erreur lors du chargement de l'image {image_path}: {e}")

    async def generate_response(
            self,
            prompt: str,
            image_path: Optional[str] = None, # Nouveau paramètre pour l'image
            pydantic_model: Optional[BaseModel] = None,
            stream: bool = False,
            callback=None,
            **kwargs
    ) -> Union[str, BaseModel]:
        """
        Génère une réponse, avec support optionnel de l'image et du streaming.
        """
        self.history.add_message("user", prompt)

        # Construction des messages pour le modèle multimodal
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Construction du contenu utilisateur (texte + image)
        user_content = [{"type": "text", "text": prompt}]
        if image_path:
            try:
                base64_image = self._prepare_image(image_path)
                image_url = f"data:image/jpeg;base64,{base64_image}"
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
            except IOError as e:
                # Gérer l'erreur si l'image ne peut pas être chargée
                # On peut choisir de continuer sans l'image ou de lever une exception
                logging.error(str(e))
                # Pour cet exemple, on continue sans l'image
                pass

        messages.append({"role": "user", "content": user_content})

        try:
            loop = asyncio.get_event_loop()
            response_generator = await loop.run_in_executor(
                self.executor,
                lambda: self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", 2048),
                    temperature=kwargs.get("temperature", 0.7),
                    stream=stream
                )
            )

            if stream:
                full_response = ""
                async def stream_handler():
                    nonlocal full_response
                    for chunk in response_generator:
                        content = chunk["choices"][0].get("delta", {}).get("content")
                        if content:
                            full_response += content
                            if callback:
                                callback(content)
                            yield content
                    self.history.add_message("assistant", full_response.strip())

                # Retourne un générateur asynchrone pour le streaming
                return "".join([chunk async for chunk in stream_handler()])

            else:
                response_text = response_generator["choices"][0]["message"]["content"]
                self.history.add_message("assistant", response_text)
                return response_text

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse multimodale: {str(e)}")

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        """SPECTER ne génère pas de texte."""
        logging.warning("Not implemented")
        return {"error": "Not implemented"}

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class Qwen3EmbeddingProvider(LLMProvider):
    def __init__(self, device: str = "cuda", model_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = "Qwen/Qwen3-Embedding-8B"

        if self.device == "cuda" and model_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            logging.warning("Le device CUDA actuel ne supporte pas bfloat16. Passage en float16.")
            self.dtype = torch.float16
        else:
            self.dtype = model_dtype if self.device == "cuda" else torch.float32

        logging.info(f"Chargement du modèle {self.model_name} sur le device: {self.device} avec dtype: {self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=True
        )

        self.model.eval()
        logging.info("Modèle chargé et en mode évaluation.")

    async def generate_embeddings(
            self,
            texts: List[str],
            batch_size: int = 128,
            prefix_type: str = ""
    ) -> List[List[float]]:
        """
        Génère les embeddings pour une liste de textes.

        Utilise automatiquement un traitement par lots optimisé pour la performance et la stabilité
        sur de grands volumes de données. Pour un petit nombre de textes (inférieur à batch_size),
        le traitement se fait en un seul lot.

        Args:
            texts (List[str]): La liste des chaînes de caractères à encoder.
            batch_size (int): La taille des lots à traiter. À ajuster en fonction de la VRAM disponible.

        Returns:
            List[List[float]]: Une liste d'embeddings, où chaque embedding est une liste de floats.
        """
        if not texts:
            return []

        return await asyncio.to_thread(
            self._process_in_batches,
            texts,
            batch_size
        )

    def _process_in_batches(
            self,
            texts: List[str],
            batch_size: int
    ) -> List[List[float]]:
        """
        Méthode de travail interne, synchrone, qui gère la boucle de traitement par lots et le nettoyage.
        """
        all_embeddings = []

        # La barre de progression s'affichera pour tout traitement, même un seul lot.
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenisation et placement sur le GPU
            inputs = self.tokenizer(
                batch_texts, padding=True, truncation=True, return_tensors="pt",
                max_length=self.model.config.max_position_embeddings
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Inférence
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                embeddings = self._mean_pooling(last_hidden_state, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Ajout au résultat et nettoyage
                all_embeddings.extend(embeddings.cpu().numpy().tolist())

            # Nettoyage de la mémoire après chaque lot
            del inputs, outputs, last_hidden_state, embeddings
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return all_embeddings

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    async def generate_response(self, prompt: str, **kwargs) -> str:
        """ModernBERT ne génère pas de texte, donc cette méthode ne fait rien."""
        logging.warning("ModernBERT est un modèle d'embedding, pas un modèle conversationnel.")
        return "ModernBERT ne génère pas de réponse textuelle."

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        """ModernBERT ne génère pas de texte."""
        logging.warning("ModernBERT est un modèle d'embedding, pas un modèle conversationnel.")
        return {"error": "ModernBERT ne supporte pas la génération de texte."}

    def get_model_info(self) -> Dict[str, Any]:
        """Retourne des informations sur le modèle."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_length": 8192,
            "embedding_dim": self.model.config.hidden_size
        }


def extract(texte, balises):
    code_blocks = {}
    current_file = None
    # Expression régulière pour trouver les blocs de code
    texte = texte.lstrip('*').rstrip('*')
    pattern = r"Fichier\s*:\s*(.*?)\n```(.*?)\n(.*?)```"
    matches = re.findall(pattern, texte, re.DOTALL)
    for match in matches:
        fichier = match[0].strip()
        balise = match[1].strip()
        code = match[2].strip()
        fichier = fichier.replace("`", "")
        if balise in balises:
            # Recherche de la fin du bloc de code en remontant jusqu'à ```
            end_index = code.rfind("```")
            while end_index != -1:
                if code[end_index - 1] != '`':
                    code = code[:end_index]
                    break
                end_index = code.rfind("```", 0, end_index)
            code_blocks[fichier] = code
    return convert_to_json_format(list(code_blocks.items()))


def convert_to_json_format(file_tuples):
    """
    Convertit une liste de tuples (nom_fichier, contenu) en format JSON spécifique.

    Args:
        file_tuples (list): Liste de tuples contenant (nom_fichier, contenu)

    Returns:
        dict: Dictionnaire au format JSON spécifié
    """
    result = {"files": []}

    for file_name, content in file_tuples:
        # Déterminer le langage en fonction de l'extension du fichier
        extension = file_name.split('.')[-1].lower()
        language_mapping = {
            'py': 'python',
            'js': 'javascript',
            'java': 'java',
            'cpp': 'cpp',
            'txt': 'plaintext',
            "html": 'html',
            "xml": 'xml'
            # Ajoutez d'autres mappings selon vos besoins
        }

        language = language_mapping.get(extension, 'plaintext')

        # Créer l'objet fichier
        file_obj = {
            "file_path": file_name,
            "code_field": {
                "language": language,
                "content": content
            },
            "dependencies": []  # Liste vide par défaut
        }

        # Si c'est un requirements.txt, extraire les dépendances
        if file_name == 'requirements.txt':
            dependencies = [dep.strip() for dep in content.split('\n') if dep.strip()]
            # Mettre à jour les dépendances pour tous les fichiers précédents
            for file in result["files"]:
                file["dependencies"] = dependencies
            # Continuer sans ajouter le requirements.txt comme fichier
            continue

        result["files"].append(file_obj)

    return result


def extract_json_only(text: str) -> str:
    """Extrait uniquement la partie JSON d'une réponse"""
    # Trouve la première accolade ouvrante
    start = text.find('{')
    if start == -1:
        return text

    # Compte les accolades pour trouver la fin du JSON
    count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            count += 1
        elif text[i] == '}':
            count -= 1
            if count == 0:
                return text[start:i + 1]

    return text


def get_optimal_n_gpu_layers(default: int = 0) -> int:
    """
    Détermine dynamiquement le nombre optimal de couches à exécuter sur le GPU
    en fonction de la mémoire libre sur le GPU.

    Args:
        default (int): Valeur par défaut si aucun GPU n'est détecté.

    Returns:
        int: Nombre de couches à utiliser sur le GPU.
    """
    try:
        # On interroge nvidia-smi pour obtenir la mémoire libre (en MB) du premier GPU
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader']
        )
        # Extraire la mémoire libre du premier GPU
        free_memory_str = output.decode('utf-8').split('\n')[0].strip()
        free_memory = int(free_memory_str)

        # Définir des seuils (en MB) pour adapter le nombre de couches.
        # Ces valeurs sont à ajuster en fonction de ton modèle et de ton hardware.
        if free_memory >= 16000:
            return 35  # Par exemple, si 16GB ou plus sont libres, on peut allouer 30 couches au GPU
        elif free_memory >= 8000:
            return 20  # Entre 8 et 16GB disponibles
        elif free_memory >= 4000:
            return 12  # Entre 4 et 8GB disponibles
        else:
            return 6  # Moins de 4GB disponibles
    except Exception as e:
        # Si nvidia-smi n'est pas disponible ou une erreur survient, on utilise la valeur par défaut (par ex. 0)
        print("Aucun GPU détecté ou erreur lors de la détection de la mémoire GPU :", e)
        return default


def normalize_encoded_chars(text: str) -> str:
    """
    Remplace les caractères encodés (comme \xc3\xa9) par leurs équivalents Unicode.

    Args:
        text: Le texte à normaliser

    Returns:
        Le texte avec les caractères normalisés
    """
    # Dictionnaire des remplacements courants en français
    replacements = {
        "\\xc3\\xa9": "é",  # é
        "\\xc3\\xa8": "è",  # è
        "\\xc3\\xaa": "ê",  # ê
        "\\xc3\\xa0": "à",  # à
        "\\xc3\\xa7": "ç",  # ç
        "\\xc3\\xb4": "ô",  # ô
        "\\xc3\\xae": "î",  # î
        "\\xc3\\xbb": "û",  # û
        "\\xc3\\xa2": "â",  # â
        "\\xc3\\xab": "ë",  # ë
        "\\xc3\\xaf": "ï",  # ï
        "\\xc3\\xbc": "ü",  # ü
        "\\xc3\\xb9": "ù",  # ù
        "\\xc3\\xa4": "ä",  # ä
        "\\xc3\\xb6": "ö",  # ö
        "\\xc3\\x89": "É",  # É
        "\\xc3\\x88": "È",  # È
        "\\xc3\\x8a": "Ê",  # Ê
        "\\xc3\\x80": "À",  # À
        "\\xc3\\x87": "Ç",  # Ç
        "\\xc3\\x94": "Ô",  # Ô
        "\\xc3\\x8e": "Î",  # Î
        "\\xc3\\x9b": "Û",  # Û
        "\\xc3\\x82": "Â",  # Â
        "\\xc3\\x8b": "Ë",  # Ë
        "\\xc3\\x8f": "Ï",  # Ï
        "\\xc3\\x9c": "Ü",  # Ü
        "\\xc3\\x99": "Ù",  # Ù
        "\\xc3\\x84": "Ä",  # Ä
        "\\xc3\\x96": "Ö",  # Ö
    }

    normalized_text = text
    for encoded, decoded in replacements.items():
        normalized_text = normalized_text.replace(encoded, decoded)

    return normalized_text


def clean_json_string(json_str: str) -> str:
    """Nettoie les chaînes de caractères dans un JSON pour les rendre valides."""

    def clean_string(match):
        """Nettoie une chaîne de caractères individuelle."""
        string_content = match.group(1)
        # Échappe les guillemets doubles qui ne sont pas déjà échappés
        string_content = string_content.replace('"', '\\"')
        # Échappe les backslashes
        string_content = string_content.replace('\\', '\\\\')
        return f'"{string_content}"'

    # Trouve toutes les chaînes de caractères dans le JSON (entre guillemets doubles)
    # et les nettoie une par une
    cleaned = re.sub(r'"([^"]*)"', clean_string, json_str)
    return cleaned


# -------------------------------------------------------------------------------------------------
# EXEMPLE DEEPSEEK R1 en local
# -------------------------------------------------------------------------------------------------

class CodeResponse(BaseModel):
    language: str
    content: str
    explanation: str


# Exemple d'utilisation du provider dans une application console
async def main_deepseek():
    # Chemin vers votre modèle DeepSeek
    model_path = "/home/yopla/Documents/llm_models/python/models/txt2txt/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S.gguf"

    # Création du provider
    provider = LocalDeepSeek_R1_Provider(
        model=model_path,
        api_key="",  # Non utilisé mais requis par l'interface
        system_prompt="Vous êtes un assistant d'IA expert en programmation Python. Répondez de manière concise et claire."
    )

    print("Assistant DeepSeek initialisé. Vous pouvez commencer à discuter.")
    print("Tapez 'exit' pour quitter ou 'clear' pour effacer l'historique.")

    while True:
        user_input = input("\nVous: ")

        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            provider.history.clear()
            print("Historique effacé.")
            continue

        print("\nAttendez pendant que l'IA génère une réponse...\n")

        try:
            # Génération d'une réponse standard
            response = await provider.generate_response(user_input, temperature=0.7, stream=True)
            print(f"Assistant: {response}")

        except Exception as e:
            print(f"Erreur: {str(e)}")


async def main_deepseek_streaming():
    # Chemin vers votre modèle DeepSeek
    model_path = "/home/yopla/Documents/llm_models/python/models/txt2txt/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S.gguf"

    # Création du provider
    provider = LocalDeepSeek_R1_Provider(
        model=model_path,
        api_key="",  # Non utilisé mais requis par l'interface
        system_prompt="Vous êtes un assistant d'IA expert en programmation Python. Répondez de manière concise et claire."
    )

    print("Assistant DeepSeek initialisé avec support du streaming. Vous pouvez commencer à discuter.")
    print("Tapez 'exit' pour quitter ou 'clear' pour effacer l'historique.")

    while True:
        user_input = input("\nVous: ")

        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'clear':
            provider.history.clear()
            print("Historique effacé.")
            continue

        print("\nAssistant: ", end="", flush=True)  # Pas de saut de ligne pour le streaming

        try:
            # Fonction de callback pour afficher les chunks au fur et à mesure
            def print_chunk(chunk):
                print(chunk, end="", flush=True)

            # Utilisation du streaming
            async for chunk in provider._stream_response(
                    provider._format_prompt(user_input),
                    callback=print_chunk,
                    temperature=0.7
            ):
                pass  # Le callback s'occupe de l'affichage

            print()  # Saut de ligne final

        except Exception as e:
            print(f"\nErreur: {str(e)}")


# Exemple de génération de code structuré avec Pydantic
async def generate_structured_code_deepseek():
    # Chemin vers votre modèle DeepSeek
    model_path = "/home/yopla/Documents/llm_models/python/models/txt2txt/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S/DeepSeek-R1-Distill-Qwen-14B-Q5_K_S.gguf"

    # Configuration pour la génération de code
    provider = LocalDeepSeek_R1_Provider(
        model=model_path,
        api_key="",
        system_prompt="Vous êtes un expert en programmation Python.",
        structured_response="""
        Veuillez fournir votre réponse au format JSON suivant:
        ```json
        {
            "language": "python",
            "content": "# votre code ici",
            "explanation": "Explication détaillée du code"
        }
        ```
        """
    )

    # Exemple de requête pour générer du code
    prompt = "Écrivez une fonction Python qui filtre une liste pour ne garder que les nombres premiers."

    try:
        # Obtenir une réponse structurée
        response = await provider.generate_response(prompt, pydantic_model=CodeResponse)

        print("=== Réponse structurée ===")
        print(f"Langage: {response.language}")
        print(f"Code:\n{response.content}")
        print(f"Explication: {response.explanation}")

    except Exception as e:
        print(f"Erreur lors de la génération de code: {str(e)}")


def demo_deepseek():
    print("=== Démonstration du LocalDeepSeek_R1_Provider ===")
    print("1. Démarrer une conversation interactive (sans streaming)")
    print("2. Générer du code structuré avec Pydantic")
    print("3. Démarrer une conversation interactive avec streaming")
    choice = input("Choisissez une option (1/2/3): ")

    if choice == "1":
        asyncio.run(main_deepseek())
    elif choice == "2":
        asyncio.run(generate_structured_code_deepseek())
    elif choice == "3":
        asyncio.run(main_deepseek_streaming())
    else:
        print("Option non valide")


async def extract_value_from_data(image_path, legende):
    # Langue pour les prompts
    LANGUAGE = "français"

    print("Initialisation du fournisseur multimodal...")
    provider = LocalMultimodalProvider(
        model="/home/yopla/Documents/llm_models/python/models/multimodal/llava-v1.5-13b-Q6_K.gguf",
        clip_model_path="/home/yopla/Documents/llm_models/python/models/multimodal/mmproj-model-f16.gguf",
        system_prompt=f"Tu es un expert en analyse de données visuelles."
    )

    # La légende est une information cruciale qu'on fournit dans les deux étapes

    # --- ÉTAPE 1 : FORCER L'ANALYSE ET LA DESCRIPTION ---
    print("\n--- ÉTAPE 1 : Demande de description du graphique ---")
    prompt_description = (
        f"Voici la légende du graphique : {legende}\n\n"
        "Ta première tâche est de décrire ce graphique en détail, sans encore extraire de valeur numérique précise. "
        "Ta description doit inclure :\n"
        "1. Les axes X et Y, y compris leurs noms et l'intervalle approximatif de leurs valeurs.\n"
        "2. Les différentes courbes présentes (couleur et style, ex: 'ligne continue bleue').\n"
        "3. À quoi chaque courbe correspond d'après la légende fournie.\n"
        "4. La tendance générale de chaque courbe (ex: 'commence à X, augmente jusqu'à un maximum, puis diminue').\n\n"
        f"Fournis cette description entièrement et uniquement en {LANGUAGE}."
    )

    description_reponse = await provider.generate_response(
        prompt=prompt_description,
        image_path=image_path,
        stream=False  # Pas besoin de stream pour la description
    )

    print("\n[RÉPONSE DU MODÈLE - DESCRIPTION] :")
    print(description_reponse)
    # À ce stade, le prompt et la description sont dans l'historique du 'provider'.

    # --- ÉTAPE 2 : POSER LA QUESTION CIBLÉE ---
    print("\n--- ÉTAPE 2 : Demande d'extraction de la valeur spécifique ---")
    prompt_extraction = (
        "Parfait. Maintenant, en te basant sur ta description précédente et en analysant à nouveau le graphique très attentivement :\n"
        "Quelle est la valeur numérique approximative de εzz lorsque Rs est égal à 0.4 ? Pour trouver la bonne valeur, vous devez tracer une ligne verticale en partant de l'abscisse jusqu'à la courbe puis un droite horizontale de la courbe à l'ordonnée. la valeur sur l'axe correspond à la valeur à retourner\n"
        "**Réponds UNIQUEMENT avec la valeur numérique en pourcent.** Ne formule aucune phrase."
    )

    # Note : Pas besoin de renvoyer l'image_path !
    # Le chat handler de LLaVA a déjà traité l'image lors du premier appel.
    valeur_extraite = await provider.generate_response(
        prompt=prompt_extraction,
        stream=False
    )

    print("\n[RÉPONSE DU MODÈLE - VALEUR EXTRAITE] :")
    print(valeur_extraite)

    # Vous pouvez maintenant essayer de convertir la réponse en nombre
    try:
        valeur_numerique = float(valeur_extraite.strip().replace(',', '.'))
        print(f"\nValeur numérique parsée avec succès : {valeur_numerique}")
    except ValueError:
        print(f"\nImpossible de convertir la réponse '{valeur_extraite}' en nombre.")

    del provider


async def demo_multimodal(model, clip):
    legende = "La légende de l'image est : εzz et εxx en fonction de la surface relative Rs. La courbe continue bleue représente εzz et la courbe en tirets rouges représente εxx."

    # Prompt ultra-spécifique
    prompt_cible = (

        "Tâche : Analyse le graphique attentivement.\n"
        "Question : Sur la courbe εzz, quelle est la valeur approximative de la déformation Strain en % lorsque Rs est égal à 0.4 ? "
        "Réponds par une seule valeur numérique ou une phrase très courte."
    )
    provider = LocalMultimodalProvider(
        model="/home/yopla/Documents/llm_models/python/models/multimodal/llava-v1.5-13b-Q6_K.gguf",
        clip_model_path="/home/yopla/Documents/llm_models/python/models/multimodal/mmproj-model-f16.gguf",
        system_prompt=f"Tu es un assistant multimodal d'extraction de données très performant. L'utilisateur veut une réponse en {LANGUAGE}."
    )

    # Génération avec une image
    response = await provider.generate_response(
        prompt=prompt_cible,
        image_path="/home/yopla/Documents/llm_models/python/models/multimodal/test_figure/Strain_de_Rs.png",
        stream=True
    )
    print(response)
    del provider

# Test en cours ...
if __name__ == "__main__":
    legende = """ εzz (blue line) and εxx (red line). Strain(Rs)"""
    asyncio.run(extract_value_from_data(
        "/home/yopla/Documents/llm_models/python/models/multimodal/test_figure/courbe.jpg",
        legende=legende))
    #asyncio.run(demo_multimodal(None, None))






    """
    # Exemple d'utilisation du LocalEmbeddingProvider
    model_path = "/home/yopla/Documents/llm_models/python/embedding/nomic-embed-text-v1.5.Q8_0.gguf"
    embedding_provider = LocalEmbeddingProvider(model_path=model_path)

    # Test d'utilisation
    async def test_embeddings():
        texts = ["Ceci est un exemple de texte en français"]
        embeddings = await embedding_provider.generate_embeddings(texts)
        print(f"Dimension du vecteur: {len(embeddings[0])}")
        # Devrait afficher "Dimension du vecteur: 3072"

    # Ou de manière synchrone
    texts = ["Exemple de texte"]
    embeddings = embedding_provider.generate_embeddings_sync(texts)
    print(f"Dimension du vecteur: {len(embeddings[0])}")
"""

"""
# exemple d'utilisation de SPECTER pour embedding de texte scientifique
    async def main():
        provider = SpecterProvider()
        texts = ["Un article sur l'IA", "Un autre sur la physique quantique"]
        embeddings = await provider.generate_embeddings(texts)
        print(len(embeddings[0]))


    asyncio.run(main())

# ou de manière synchrone
    provider = SpecterProvider()
    texts = ["Un article sur l'IA", "Un autre sur la physique quantique"]
    embeddings = provider.generate_embeddings_sync(texts)
    print(len(embeddings[0]))

"""


"""
modele a tester sur cluster pour embedding : SGPT-5.8B
"""