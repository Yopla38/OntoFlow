"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

from typing import Dict, Any

from agent.src.agent import Agent


class HPCAgent(Agent):
    async def analyze_documentation(self, doc_path: str):
        """Analyse la documentation pour générer des function_calls"""
        analysis_prompt = f"""Analysez la documentation du HPC dans {doc_path}.
        Identifiez toutes les opérations possibles (connexion, soumission de jobs, etc.)
        et créez des function_calls correspondants.

        Pour chaque opération, identifiez :
        1. Le nom de la fonction
        2. Les paramètres requis et leur type
        3. Les paramètres optionnels
        4. La description de l'opération
        5. Les contraintes spécifiques
        """
        analysis_prompt += """
        Format attendu:
        {
        "functions": [
                {
        "name": "submit_job",
                    "description": "Soumet un job sur le cluster",
                    "parameters": {
        "type": "object",
                        "properties": {
        "job_script": {"type": "string", "description": "Chemin du script"},
                            "queue": {"type": "string", "description": "Queue de calcul"},
                            "cores": {"type": "integer", "description": "Nombre de coeurs"}
                        },
                        "required": ["job_script"]
                    }
                }
            ]
        }"""

        functions = await self.llm_provider.generate_response(analysis_prompt)
        self.register_functions(functions)

    def register_functions(self, functions: Dict[str, Any]):
        """Enregistre les function_calls générés dynamiquement"""
        self.available_functions = functions["functions"]

        # Créer dynamiquement les méthodes de l'agent
        for func in self.available_functions:
            async def function_wrapper(**kwargs):
                # Vérifier les arguments requis
                missing_args = self.check_missing_arguments(func["name"], kwargs)
                if missing_args:
                    return {
                        "status": "need_info",
                        "missing_arguments": missing_args,
                        "description": func["description"]
                    }
                # Exécuter la fonction réelle sur le HPC
                return await self.execute_hpc_command(func["name"], kwargs)

            # Attacher la fonction à l'instance
            setattr(self, func["name"], function_wrapper)
