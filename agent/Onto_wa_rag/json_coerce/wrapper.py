"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Louis BEAL
Entité: IRIG
Année: 2025
Description: Agent IA d'Intégration Continue
------------------------------------------
"""

import json
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from json_coerce.json_parser import clean_output
from json_coerce.model_convert import convert_model_to_struct


JSON_RETRY_PROMPT = """You provided the following JSON:
{input}

This cannot be parsed as valid JSON for the reason:
{error}

Please correct the JSON so that it can be parsed correctly.

Return only the output of this operation, do not add your own commentry, explanation or formatting.
"""

VALIDATION_RETRY_PROMPT = """You provided the following structured output:
{output}

However it fails to validate for the following reason:
{error}

Please ensure it fits the structure exactly, and return a corrected JSON object that validates correctly against this structure:
{structure}

Return only the output of this operation, do not add your own commentry, explanation or formatting."""


class StructuredWrapper:
    def __init__(self, client: OpenAI, structure: BaseModel.__class__) -> None:
        self.client = client
        self.structure_model = structure
        self.structure = convert_model_to_struct(structure)

        self.history = []

    @classmethod
    def json_coerced_chat_oneshot(
        cls,
        client: OpenAI,
        structure: BaseModel.__class__,
        model: str,
        prompt: str | list[dict[str, str]],
        max_retries: int = 3,
    ) -> dict[str, str]:
        wrapper = cls(client=client, structure=structure)
        return wrapper.chat(prompt=prompt, model=model, max_retries=max_retries)

    def _chat(self, model: str, prompt: str | None = None) -> str:
        """
        Act on the current message history, appending the response to the history.
        """
        if prompt is not None and (
            len(self.history) == 0 or self.history[-1]["role"] != "user"
        ):
            self.history.append({"role": "user", "content": prompt})

        if len(self.history) > 0 and self.history[-1]["role"] != "user":
            raise ValueError("Last message in history must be from user")

        completion = self.client.chat.completions.create(
            model=model,
            messages=self.history,
        )

        if len(completion.choices) == 0:
            return ""
        if completion.choices[0].message is None:
            return ""
        if completion.choices[0].message.content is None:
            return ""

        self.history.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )

        return completion.choices[0].message.content

    def chat(
        self, prompt: str | list[dict[str, str]], model: str, max_retries: int = 3
    ) -> dict[str, str]:
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        current_retries = 0

        # get the content of the last user prompt
        user_prompt_content = prompt.pop(-1)["content"]
        # inject our structure into the prompt
        modified_prompt = f"""Extract details or perform tasks according to the following text:

"{user_prompt_content}"

Your response should contain ONLY a valid JSON object with the following fields.
Do not respond with any other content, only the JSON object with the following fields:
{self.structure}"""

        # get a response
        result = self._chat(model=model, prompt=modified_prompt)
        if result == "":
            return {}

        parsed = self._validate_output(result, current_retries, max_retries, model)

        return json.loads(parsed)

    def _validate_output(
        self, output: str, current_retries: int, max_retries: int, retry_model: str
    ) -> str:
        output = self._get_json(output, current_retries, max_retries, retry_model)
        output = self._validate_structure(
            output, current_retries, max_retries, retry_model
        )

        return output

    def _get_json(
        self, text: str, current_retries: int, max_retries: int, retry_model: str
    ) -> str:
        """
        Attempt to extract JSON from the text, retrying if necessary.
        """

        try:
            json.loads(clean_output(text))
            return clean_output(text)
        except json.JSONDecodeError as e:
            if current_retries >= max_retries:
                raise ValueError(
                    f"Max retries reached, unable to extract valid JSON. Last attempt:\n{text}"
                )
            print(
                f"Failed to parse JSON, asking {retry_model} to retry... (attempt {current_retries + 1}/{max_retries})"
            )
            retry = self._chat(
                model=retry_model,
                prompt=JSON_RETRY_PROMPT.format(input=text, error=str(e)),
            )

            return self._validate_output(
                retry, current_retries + 1, max_retries, retry_model
            )

    def _validate_structure(
        self,
        data: str,
        current_retries: int,
        max_retries: int,
        retry_model: str,
    ) -> str:
        try:
            self.structure_model.model_validate(json.loads(data))
            return data

        except ValidationError as e:
            if current_retries >= max_retries:
                raise ValueError(
                    f"Max retries reached, unable to produce valid structure. Last attempt:\n{json.dumps(data, indent=2)}"
                )
            print(
                f"Validation failed, asking {retry_model} to fix the issue... (attempt {current_retries + 1}/{max_retries})"
            )
            retry = self._chat(
                model=retry_model,
                prompt=VALIDATION_RETRY_PROMPT.format(
                    output=data,
                    error=str(e),
                    structure=self.structure,
                ),
            )

            return self._validate_output(
                retry, current_retries + 1, max_retries, retry_model
            )

    @property
    def last_response(self) -> str:
        """
        Get the content of the last assistant response
        """
        for message in self.history[::-1]:
            if message["role"] == "assistant":
                return message["content"]
        return ""
