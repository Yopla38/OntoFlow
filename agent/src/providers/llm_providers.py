"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# src/providers/llm_providers.py
import asyncio
import json
import logging
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field, dataclass
from datetime import datetime
from typing import Optional, Union, Any, List, Dict, Type, AsyncGenerator


import instructor
from pydantic import BaseModel
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type

from agent.KEY import LOCAL_MODEL_PATH
#from agent.src.providers.local_CEA_API import LLM_CEA
from agent.src.types.interfaces import LLMProvider
import openai
from anthropic import Anthropic

from ...CONSTANT import CLAUDE_MAX_TOKEN


#  Installation pour le provider local :
# https: // forums.developer.nvidia.com / t / installing - cuda - on - ubuntu - 22 - 04 - rxt4080 - laptop / 292899
# CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_CUBLAS=on -DLLAVA_BUILD=off -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: str, system_prompt: Optional[str] = None):
        super().__init__()
        self.model = model
        self.system_prompt = system_prompt
        self.client = openai.AsyncClient(api_key=api_key)  # Utilisation du client asynchrone

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
        wait=wait_random_exponential(min=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(openai.APIError)
    )
    async def generate_response(
            self,
            messages: Union[str, List[Dict[str, str]]],
            stream: bool = False,
            pydantic_model: Optional[BaseModel] = None,
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

            # Si streaming est demandé
            if stream:
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    stream=True,
                    **kwargs
                )

            # Si un modèle Pydantic est fourni
            if pydantic_model:
                response = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=formatted_messages,
                    response_format=pydantic_model,
                    **kwargs
                )
                return response.choices[0].message.parsed.model_dump()

            # Cas standard
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                stream=False,
                **kwargs
            )
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

class VLLMProvider(OpenAIProvider):
    """
    OpenAI-compatible provider for a vLLM server.
=    """
    def __init__(
        self,
        model: str,
        system_prompt: Optional[str] = None,
        served_url: str = "http://127.0.0.1:8000",
        api_key: str = "EMPTY",
    ):
        super().__init__(model=model, api_key=api_key, system_prompt=system_prompt)
        served_url = served_url.rstrip("/")
        self.base_url = f"{served_url}/v1"
        try:
            self.client = openai.AsyncClient(base_url=self.base_url, api_key=api_key)
        except TypeError:
            self.client = openai.AsyncOpenAI(base_url=self.base_url, api_key=api_key)

    @retry(
        wait=wait_random_exponential(min=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(openai.APIError)
    )
    async def generate_response(
        self,
        messages: Union[str, List[Dict[str, str]]],
        stream: bool = False,
        pydantic_model: Optional[BaseModel] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], AsyncGenerator]:
        """
        Matches OpenAIProvider’s contract, but replaces the `beta.chat.completions.parse`
        path with a local JSON→Pydantic validation (vLLM doesn’t implement that endpoint).
        """
        # Prepare messages (copying your OpenAIProvider behavior)
        if isinstance(messages, str):
            formatted_messages: List[Dict[str, str]] = []
            if self.system_prompt:
                formatted_messages.append({"role": "system", "content": self.system_prompt})
            formatted_messages.append({"role": "user", "content": self._ensure_text_content(messages)})
        else:
            formatted_messages = messages

        try:
            if stream:
                # Return the async streaming iterator (your code expects the raw stream)
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    stream=True,
                    **kwargs
                )

            # Regular non-streaming completion
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                stream=False,
                **kwargs
            )
            content = (resp.choices[0].message.content if resp.choices else "") or ""

            if pydantic_model is not None:
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as je:
                    raise ValueError(
                        "vLLM cannot use `beta.chat.completions.parse`. "
                        "Ask the model to respond with strict JSON; received non-JSON content."
                    ) from je
                model_cls = pydantic_model if isinstance(pydantic_model, type) else type(pydantic_model)
                parsed = model_cls.model_validate(data)
                return parsed.model_dump()

            return content

        except openai.APIError as e:
            logging.error(f"vLLM (OpenAI-compatible) API Error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in VLLMProvider.generate_response: {str(e)}")
            raise


    @staticmethod
    def check_server(url: str = "http://127.0.0.1:8000") -> bool:
        """
        True if vLLM server responds 200 to GET /health.
        Uses curl to avoid extra dependencies.
        """
        try:
            subprocess.run(
                ["curl", "-fsS", f"{url.rstrip('/')}/health"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    @classmethod
    def serve(
        cls,
        model: str,
        port: int = 8000,
        host: str = "0.0.0.0",
        python_executable: str = sys.executable,
        extra_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        detach: bool = True,
    ) -> subprocess.Popen:
        """
        Launch a vLLM OpenAI-compatible server:
        python -m vllm.entrypoints.openai.api_server --model <model> --port <port> --host <host> [extra_args...]
        Example extra_args:
            ["--tensor-parallel-size", "2", "--max-model-len", "32768", "--trust-remote-code"]
        """
        cmd = [
            python_executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--port", str(port),
            "--host", host,
        ]
        if extra_args:
            cmd += list(extra_args)

        popen_kwargs = {"env": (env or os.environ.copy())}
        if detach:
            popen_kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE})

        return subprocess.Popen(cmd, **popen_kwargs)

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
            Anthropic(api_key=api_key, timeout=900),
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

    @staticmethod
    def _convert_openai_messages_to_anthropic(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convertit le format de messages OpenAI au format Anthropic.

        OpenAI: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
        Anthropic: [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}, ...]
        """
        anthropic_messages = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            # Ignorer les messages système dans la conversion directe
            # (ils seront gérés séparément par le paramètre "system")
            if role == "system":
                continue

            # Convertir en format Anthropic
            anthropic_message = {
                "role": role,
                "content": [{"type": "text", "text": content}] if isinstance(content, str) else content
            }

            anthropic_messages.append(anthropic_message)

        return anthropic_messages

    async def generate_response(
            self,
            prompt: Union[str, List[Dict[str, str]]],
            pydantic_model: Optional[BaseModel] = None,
            **kwargs
    ) -> Union[str, BaseModel]:
        """
        Génère une réponse à partir du prompt.

        Args:
            prompt: Le prompt à envoyer au modèle (str ou liste de messages OpenAI)
            pydantic_model: Modèle Pydantic optionnel pour structurer la réponse
            **kwargs: Arguments additionnels pour l'API

        Returns:
            Soit une chaîne de caractères, soit une instance du modèle Pydantic
        """

        # Traitement selon le type de prompt
        if isinstance(prompt, str):
            # Ajoute le message utilisateur à l'historique
            self.history.add_message("user", prompt)
            # Construit le contexte avec l'historique
            context = self.history.get_context_string("anthropic")

            # Messages pour l'API
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": context}]
            }]

        elif isinstance(prompt, list):
            # Cas où prompt est une liste de messages au format OpenAI
            # Extraire un éventuel message système
            system_message = None
            for message in prompt:
                if message["role"] == "system":
                    system_message = message["content"]
                    break

            # Convertir la liste de messages au format Anthropic
            messages = self._convert_openai_messages_to_anthropic(prompt)

            # Ajouter tous les messages à l'historique
            for message in prompt:
                if message["role"] != "system":  # Ignorer les messages système dans l'historique
                    self.history.add_message(message["role"], message["content"])

            # Si un message système est présent dans les messages, il a priorité
            if system_message:
                kwargs["system"] = system_message
        else:
            raise ValueError("Le prompt doit être une chaîne de caractères ou une liste de messages")

        try:
            if pydantic_model:
                response = await self._generate_structured_response(messages, pydantic_model, **kwargs)
            else:
                response = await self._generate_text_response(messages, **kwargs)

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
            messages: Union[str, List[Dict[str, Any]]],
            pydantic_model: BaseModel,
            **kwargs
    ) -> BaseModel:
        """
        Génère une réponse structurée selon un modèle Pydantic.
        """

        try:
            # Préparation des paramètres de base
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", CLAUDE_MAX_TOKEN),
                "response_model": pydantic_model
            }

            # Gestion des messages selon le type d'entrée
            if isinstance(messages, str):
                request_params["messages"] = [{
                    "role": "user",
                    "content": [{"type": "text", "text": messages}]
                }]
            else:
                request_params["messages"] = messages

            # Ajout du prompt système si présent et non déjà spécifié dans kwargs
            if self.system_prompt and "system" not in kwargs:
                request_params["system"] = self.system_prompt

            # Mise à jour avec les kwargs additionnels
            for key, value in kwargs.items():
                if key != "messages":  # Éviter d'écraser les messages déjà configurés
                    request_params[key] = value

            # Exécution de la requête dans le thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.structured_client.messages.create(**request_params)
            )

            return response

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse structurée: {str(e)}")

    async def _generate_text_response(self, messages: Union[str, List[Dict[str, Any]]], **kwargs) -> str:
        """
        Génère une réponse textuelle simple.
        """
        try:
            # Préparation des paramètres de base
            request_params = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            # Gestion des messages selon le type d'entrée
            if isinstance(messages, str):
                request_params["messages"] = [{
                    "role": "user",
                    "content": [{"type": "text", "text": messages}]
                }]
            else:
                request_params["messages"] = messages

            streaming = kwargs.get("streaming", False)
            streaming_callback = kwargs.get("streaming_callback", None)

            # Ajout du prompt système si présent et non déjà spécifié dans kwargs
            if self.system_prompt and "system" not in kwargs:
                request_params["system"] = self.system_prompt

            # Mise à jour avec les kwargs additionnels
            for key, value in kwargs.items():
                if key != "messages" and key != "streaming" and key != "streaming_callback":
                    request_params[key] = value

            # Gestion du streaming
            if streaming and streaming_callback:
                request_params["stream"] = True

                # Conteneur pour la réponse complète
                full_response = ""

                # Exécution de la requête en streaming
                async with self.text_client.messages.stream(**request_params) as stream:
                    async for chunk in stream:
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                            text_chunk = chunk.delta.text
                            full_response += text_chunk
                            # Appel du callback avec le fragment
                            streaming_callback(text_chunk)

                return full_response
            else:
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

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        # TODO: Implémentation manquante
        pass
        return {}

    def set_system_prompt(self, system_prompt: str):
        """
        Définit ou met à jour le prompt système.
        """
        self.system_prompt = system_prompt

    def __del__(self):
        """
        Nettoyage des ressources à la destruction de l'instance.
        """
        self.executor.shutdown(wait=False)


class LocalDeepSeek_R1_Provider(LLMProvider):
    def __init__(self, model: str, api_key: str, cached_model=None, system_prompt: Optional[str] = None,
                 structured_response: Optional[str] = None):

        super().__init__()
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
            **kwargs
    ) -> Union[str, BaseModel]:

        # Ajoute le message utilisateur à l'historique
        self.history.add_message("user", prompt)

        structured_prompt = self._format_prompt(prompt)
        try:

            response = await self._generate_text_response(structured_prompt, **kwargs)

            # Ajoute la réponse à l'historique
            if isinstance(response, str):
                self.history.add_message("assistant", response)
            else:
                self.history.add_message("assistant", str(response))
            if self.structured_response:
                if " json " in self.structured_response.lower():
                    structured_message = extract_json_only(response)
                    structured_message = self.extract_json(structured_message)
                elif "```language" in self.structured_response.lower():
                    structured_message = extract(response, ["python", "css", "html", "js", "jsx", "javascript", "markdown", "", "plaintext", "xml", "json", "yaml"])

            else:
                structured_message = response
            #  TODO bof pas beau
            try:
                structured_message = self.encode_json(structured_message)
            except Exception as e:
                return structured_message

            if pydantic_model:
                # Extraire les données selon le schéma
                return pydantic_model.parse_obj(structured_message)
            else:
                return pydantic_model.parse_obj(structured_message)

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse: {str(e)}")

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        pass
        return {}

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

    async def _generate_text_response(self, prompt: str, **kwargs) -> str:
        try:
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

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def set_structured_local_model(self, structured_response_for_local_model):
        self.structured_response = structured_response_for_local_model + "\nAssurez-vous d'échapper correctement les caractères spéciaux dans le code, notamment les backslashes doivent être doublés (\\\\)."

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class MistralProvider(LLMProvider):
    """
    Provider pour l'API Mistral avec support des modèles Pydantic.
    """

    def __init__(self, model: str, api_key: str, mixtral_api_url: str, system_prompt: Optional[str] = None):
        """
        Initialise le provider Mistral.

        Args:
            model: Nom du modèle à utiliser (ex: "mistralsmall-22b")
            api_key: Clé API Mistral
            mixtral_api_url: URL de l'API Mistral
            system_prompt: Prompt système optionnel
        """
        super().__init__()  # Initialise ConversationHistory
        self.model = model
        self.system_prompt = system_prompt

        # Client Mistral standard
        self.client = LLM_CEA(
            api_key=api_key,
            mixtral_api_url=mixtral_api_url
        )


        # ThreadPoolExecutor pour les appels synchrones
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def generate_response(self, prompt: str, pydantic_model: dict = None, **kwargs) -> Union[str, BaseModel]:
        """
        Génère une réponse à partir du prompt.
        """
        # Ajoute le message utilisateur à l'historique
        self.history.add_message("user", prompt)

        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            if pydantic_model:
                # Utilisation d'instructor pour les réponses structurées
                response = await self._generate_structured_response(messages, pydantic_model, **kwargs)
            else:
                # Réponse textuelle standard
                response = await self._generate_text_response(messages, **kwargs)

            # Ajoute la réponse à l'historique
            self.history.add_message("assistant", str(response))

            return response
        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse: {str(e)}")

    async def generate_response_for_humain(self, messages: List[Dict[str, str]], stream=None) -> Dict[str, Any]:
        """
        Génère une réponse formatée pour l'interface utilisateur.
        """
        try:
            # Formatage des messages
            formatted_messages = []
            if self.system_prompt:
                formatted_messages.append({
                    "role": "system",
                    "content": self.system_prompt
                })
            formatted_messages.extend(messages)

            # Génération de la réponse
            response = await self._generate_text_response(formatted_messages, stream=stream)

            return {
                "role": "assistant",
                "content": response if not stream else "",
                "stream": stream,
                "response_stream": response if stream else None
            }

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse pour l'humain: {str(e)}")

    async def _generate_text_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Génère une réponse textuelle simple.
        """
        try:
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_token": kwargs.get("max_tokens", 4096),
                "stream": kwargs.get("stream", False)
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.client.chat.create(**request_params)
            )

            return response.messages[0] if not kwargs.get("stream") else response

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse textuelle: {str(e)}")

    async def _generate_structured_response(self, messages: List[Dict[str, str]], pydantic_model: dict, **kwargs) -> str:
        """
        Génère une réponse textuelle simple.
        """
        try:
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_token": kwargs.get("max_tokens", 4096),
                "stream": kwargs.get("stream", False),
                "response_format": pydantic_model
            }

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self.executor,
                lambda: self.client.chat.create(**request_params)
            )

            return response.messages[0] if not kwargs.get("stream") else response

        except Exception as e:
            raise Exception(f"Erreur lors de la génération de la réponse textuelle: {str(e)}")


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
                if code[end_index-1] != '`':
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

