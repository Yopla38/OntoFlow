"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# coding: utf-8
# -----------------------------------------------------------------------------------
# Author: Yoann CURE
# Date: 2024
# Description: Interface pour appeler des modèles de langage sur différents serveurs
#              avec prise en charge du streaming des API Ollama et Mixtral.
# -----------------------------------------------------------------------------------

import json
import requests

# Serveurs CEA
OLLAMA_API_URL = "https://holiagen.ixia.intra.cea.fr/"
MIXTRAL_API_URL = "https://litellm-dev.ixia.intra.cea.fr"
USER = 'nobody'

# Mapping pour modèles Mixtral
MIXTRAL_MODELS_MAPPING = {
    "mistralsmall-22b": "mistralsmall-22b",
    "LITELLM2OLLAMA-llama3.2:3b": "LITELLM2OLLAMA-llama3.2:3b",
    "LITELLM2OLLAMA-deepseek-coder-v2-16b": "LITELLM2OLLAMA-deepseek-coder-v2-16b",
    "LITELLM2OLLAMA-bge-m3": "LITELLM2OLLAMA-bge-m3"
}

VERIFY_CERTIFICAT = False


class ChatCompletion:
    def __init__(self, api_key, llama_api_url: str = OLLAMA_API_URL, mixtral_api_url: str = MIXTRAL_API_URL, user: str = 'yc170903'):
        self.llama_api_url = llama_api_url
        self.mixtral_api_url = mixtral_api_url
        self.api_key = api_key
        self.user = user

    @staticmethod
    def handle_ollama_stream(response):
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if 'message' in data:
                    yield Completion(data['message']['content'])
        # Yield a final completion object to indicate the end
        yield Completion('')

    @staticmethod
    def handle_mixtral_stream(response):
        for line in response.iter_lines():
            if line.startswith(b"data: "):  # Mixtral's stream format
                data_json = line[6:]  # Strip the "data: " prefix
                data = json.loads(data_json.decode('utf-8'))
                if 'choices' in data and data['choices']:
                    yield Completion(data['choices'][0]['delta']['content'])

    def create(self, model, messages, temperature=1.0, top_p=1.0, max_token=1, stream=False, response_format="", stop=None,
               presence_penalty=0.0, frequency_penalty=0.0, options=None, keep_alive=None):
        # Determine the endpoint and URL based on the model

        if model.lower() in MIXTRAL_MODELS_MAPPING:
            url = f"{self.mixtral_api_url}/chat/completions"
            headers = {
                'Authorization': self.api_key,
                'Content-Type': 'application/json'
            }
            data = {
                "model": MIXTRAL_MODELS_MAPPING[model.lower()],
                "messages": [{"role": msg["role"], "content": msg["content"]} for msg in messages if
                             msg["role"] == "user"],
                "user": self.user,
                "temperature": temperature,
                "stream": stream,

            }
            if response_format is not "":
                data["response_format"] = {
                    "type": "json_schema",
                    "json_schema": response_format
                }

            response = requests.post(url, json=data, headers=headers, verify=VERIFY_CERTIFICAT, stream=stream)

        else:
            url = f"{self.llama_api_url}/api/chat"
            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "n": max_token,
                "stream": stream,
                "stop": stop,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "options": options,
                "keep_alive": keep_alive
            }
            response = requests.post(url, json=data, verify=VERIFY_CERTIFICAT, stream=stream)

        if response.status_code == 200:
            if stream:
                if model.lower() in MIXTRAL_MODELS_MAPPING:
                    return self.handle_mixtral_stream(response)
                else:
                    return self.handle_ollama_stream(response)
            else:
                response_json = response.json()
                return StandardResponse(response_json)
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")


class Embedding:
    def __init__(self, api_url):
        self.api_url = api_url

    def create(self, model, input, options=None, keep_alive=None):
        url = f"{self.api_url}/api/embeddings"
        data = {
            "model": model,
            "input": input,
            "options": options,
            "keep_alive": keep_alive
        }
        response = requests.post(url, json=data, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return EmbeddingResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")


class Model:
    def __init__(self, api_url):
        self.api_url = api_url

    def list(self):
        url = f"{self.api_url}/models"
        response = requests.get(url, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return ModelListResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")

    def show(self, name):
        url = f"{self.api_url}/api/show"
        data = {"name": name}
        response = requests.post(url, json=data, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return ModelShowResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")

    def create(self, name, modelfile=None, stream=False, path=None):
        url = f"{self.api_url}/api/create"
        data = {
            "name": name,
            "modelfile": modelfile,
            "stream": stream,
            "path": path
        }
        response = requests.post(url, json=data, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return ModelCreateResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")

    def delete(self, name):
        url = f"{self.api_url}/api/delete"
        data = {"name": name}
        response = requests.delete(url, json=data, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return ModelDeleteResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")

    def copy(self, source, destination):
        url = f"{self.api_url}/api/copy"
        data = {
            "source": source,
            "destination": destination
        }
        response = requests.post(url, json=data, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return ModelCopyResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")

    def pull(self, name, insecure=False, stream=False):
        url = f"{self.api_url}/api/pull"
        data = {
            "name": name,
            "insecure": insecure,
            "stream": stream
        }
        response = requests.post(url, json=data, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return ModelPullResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")

    def push(self, name, insecure=False, stream=False):
        url = f"{self.api_url}/api/push"
        data = {
            "name": name,
            "insecure": insecure,
            "stream": stream
        }
        response = requests.post(url, json=data, verify=VERIFY_CERTIFICAT)
        if response.status_code == 200:
            return ModelPushResponse(response.json())
        else:
            raise Exception(f"Erreur {response.status_code}: {response.text}")


class BaseResponse:
    def __init__(self, data):
        self.data = data


class ChatCompletionChoice:
    def __init__(self, data):
        self.data = data
        self.index = self.data.get("index")
        self.text = self.data.get("text")
        self.logprobs = self.data.get("logprobs")
        self.finish_reason = self.data.get("finish_reason")
        self.stop_reason = self.data.get("stop_reason")


class ChatCompletionMessage:
    def __init__(self, data):
        self.data = data
        self.role = self.data.get("role")
        self.content = self.data.get("content")


class ChatCompletionResponse(BaseResponse):
    def __init__(self, data):
        super().__init__(data)
        if "choices" in self.data:
            # Mixtral response format
            self.model = self.data.get("model")
            self.created_at = self.data.get("created")
            self.choices = [ChatCompletionChoice(c) for c in self.data.get("choices", [])]
        else:
            # Ollama response format
            self.model = self.data.get("model")
            self.created_at = self.data.get("created_at")
            self.message = ChatCompletionMessage(self.data.get("message", {}))
            self.done = self.data.get("done")
            self.total_duration = self.data.get("total_duration")
            self.load_duration = self.data.get("load_duration")
            self.prompt_eval_duration = self.data.get("prompt_eval_duration")
            self.eval_count = self.data.get("eval_count")
            self.eval_duration = self.data.get("eval_duration")


class StandardResponse:
    def __init__(self, data):
        self.data = data
        self.model = data.get("model")
        self.created_at = data.get("created") if "choices" in data else data.get("created_at")

        if "choices" in data and data["choices"]:
            # Mixtral response format
            if "message" in data["choices"][0]:
                self.messages = [data["choices"][0]["message"]["content"]]
            else:
                self.messages = [choice["text"] for choice in data["choices"]]
        else:
            # Ollama response format
            self.messages = [data["message"]["content"]] if "message" in data else []

    @property
    def message(self):
        return self.messages[0].strip() if self.messages else ""

    @property
    def all_messages(self):
        return self.messages


class EmbeddingResponse(BaseResponse):
    @property
    def embedding(self):
        return self.data["embedding"]


class ModelListResponse:
    def __init__(self, data):
        self.data = data

    @property
    def models(self):
        return self.data["models"]


class ModelShowResponse:
    def __init__(self, data):
        self.data = data

    @property
    def details(self):
        return self.data["details"]


class ModelCreateResponse:
    def __init__(self, data):
        self.data = data

    @property
    def status(self):
        return self.data["status"]


class ModelDeleteResponse:
    def __init__(self, data):
        self.data = data

    @property
    def status(self):
        return self.data["status"]


class ModelCopyResponse:
    def __init__(self, data):
        self.data = data

    @property
    def status(self):
        return self.data["status"]


class ModelPullResponse:
    def __init__(self, data):
        self.data = data

    @property
    def status(self):
        return self.data["status"]


class ModelPushResponse:
    def __init__(self, data):
        self.data = data

    @property
    def status(self):
        return self.data["status"]


class LLM_CEA:
    def __init__(self, api_key: str = None, llama_api_url: str = OLLAMA_API_URL, mixtral_api_url: str = MIXTRAL_API_URL, user: str = 'me'):
        self.api_key = api_key
        self.chat = ChatCompletion(api_key=api_key, llama_api_url=llama_api_url, mixtral_api_url=mixtral_api_url, user=user)
        self.embedding = Embedding(llama_api_url)
        self.model = Model(llama_api_url)


class Delta:
    def __init__(self, content):
        self.content = content


class Choice:
    def __init__(self, delta):
        self.delta = delta


class Completion:
    def __init__(self, content):
        self.choices = [Choice(Delta(content))]


