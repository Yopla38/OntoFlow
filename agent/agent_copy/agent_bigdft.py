"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# agent/agent_bigdft.py
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import shutil
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from agent.CONSTANT_BIGDFT import INPUT_YAML_FORMAT, Stage, REQUIRED_FIELDS
from agent.src.agent import setup_workspace
from agent.src.components.file_manager import FileManager
from agent.src.components.task_manager import TaskManager
from agent.src.factory.simple_agent_factory import SimpleAgentFactory
from agent.src.factory.development_agent_factory import DevelopmentAgentFactory
from agent.src.interface.html_manager import HTMLManager
from agent.src.types.roles_definition import select_role

LOGGER = logging.getLogger("agent_bigdft")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------------------------ #
#                     PROMPTS  DÉVELOPPPEUR  DYNAMIQUES              #
# ------------------------------------------------------------------ #
#  Bloc obligatoire à concaténer en fin de prompt
_FORMAT_BLOCK = """
Votre format de réponse doit suivre cette structure:
{
  "files": [
    {
      "file_path": chemin_du_fichier,
      "file_exists": boolean qui indique si le fichier existe déjà,
      "modification_type": "create" ou "update" ou "delete",
      "code_field": {
        "language": "str",
        "content": contenu complet pour les nouveaux fichiers, sinon vide
      },
      "code_changes": {
        # pour les fichiers existants
        "operation": "diff" ou "jsonpatch",
        "location": {
          "context_before": "",
          "match_code": "",
          "context_after": ""
        },
        "new_code": "contenu du patch (diff ou jsonpatch)"
      },
      "dependencies": ["liste des dépendances"]
    }
  ]
}
""".strip()


def prompt_xyz() -> str:
    return f"""
Tu es un générateur expert de fichiers de structure atomique au format XYZ
(pour BigDFT). Règles :

1. Le fichier doit se nommer strictement "posinp.xyz".
2. Première ligne: nombre total d’atomes.
3. Deuxième ligne: commentaire libre (ex. paramètre de maille).
4. À partir de la 3ᵉ ligne: «Element␣␣x␣␣y␣␣z» (deux espaces, coordonnées
   en Å).
5. Aucune colonne superflue, pas de tabulation.

Ne fournis AUCUN script Python/bash: écris directement le contenu complet
du fichier dans le champcode_field.content (language="xyz").

{_FORMAT_BLOCK}
""".strip()


def prompt_input() -> str:
    return f"""
Tu dois produire un fichier YAML d’entrée BigDFT «input.yaml» en suivant ce format:
{INPUT_YAML_FORMAT}.


Écris le contenu complet dans code_field.content (language="yaml").
{_FORMAT_BLOCK}
""".strip()


def prompt_slurm() -> str:
    return f"""
Génère un script SLURM «submit_job.sh» compatible avec la simulation BigDFT.

Exigences:
1. Shebang#!/bin/bash
2. Directives #SBATCH (partition, nœuds, temps, etc.) issues des réponses
   de l’utilisateur.
3. Chargement des modules nécessaires puis appel de bigdft.
4. Fichier exécutable (+x).

Le script complet est à placer dans code_field.content (language="bash").
{_FORMAT_BLOCK}
""".strip()


STAGE_PROMPTS: Dict[Stage, Callable[[], str]] = {
    Stage.GENERATE_XYZ: prompt_xyz,
    Stage.GENERATE_INPUT: prompt_input,
    Stage.GENERATE_SLURM: prompt_slurm,
    # Stage.GENERATE_ANALYSIS garde le prompt Python d’origine
}


class BigDFTAssistant:
    SUPPORTED_ROLES = ["SuperAgent_BIGDFT", "Convertisseur_dial2tec", "Developpeur"]

    # ---------------------------- INITIALISATION ------------------------- #
    def __init__(self, workspace: Path):
        self.workspace_dir = Path(workspace)
        self.stage: Stage = Stage.ASK_SYSTEM
        self.history: List[Dict[str, str]] = []
        self.exclude_dir = {".agent_workspace", ".git", "__pycache__", "venv"}
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        for sub in ("structures", "input_files", "scripts", "analysis"):
            (self.workspace_dir / sub).mkdir(exist_ok=True)

        # Initialisation du logger
        self._setup_logger()

    @classmethod
    async def create(cls, workspace_dir: str | Path | None = None):
        self = cls(workspace_dir or Path.cwd() / "bigdft_workspace")
        await self._init_agents()
        return self

    async def _init_agents(self):
        ws_root, proj_dir, _, venvs = await setup_workspace(self.workspace_dir)
        venv_dir = venvs / "bigdft_assistant"
        venv_dir.mkdir(exist_ok=True)

        self.file_manager = FileManager(
            project_path=str(self.workspace_dir), venv_path=str(venv_dir)
        )
        await self.file_manager.initialize_venv()

        self.task_manager = TaskManager()
        self.html_manager = HTMLManager(str(ws_root))

        DevelopmentAgentFactory.SUPPORTED_ROLES = self.SUPPORTED_ROLES

        self.super_agent = SimpleAgentFactory.create_agent_from_role(
            select_role(
                "SuperAgent_BIGDFT",
                base_dir=self.workspace_dir,
                excluded_dir=self.exclude_dir,
            ),
            project_path=str(proj_dir),
            project_name="bigdft_assistant",
            workspace_root=str(ws_root),
            file_manager=self.file_manager,
            task_manager=self.task_manager,
            streaming=True,
        )

        self.idea_agent = SimpleAgentFactory.create_agent_from_role(
            select_role(
                "Convertisseur_dial2tec",
                base_dir=self.workspace_dir,
                excluded_dir=self.exclude_dir,
            ),
            project_path=str(proj_dir),
            project_name="bigdft_assistant",
            workspace_root=str(ws_root),
            file_manager=self.file_manager,
            task_manager=self.task_manager,
        )

        # ----- développeur ------------------------------------------------
        self.developer = DevelopmentAgentFactory.create_agent_from_role(
            select_role(
                "Developpeur",
                base_dir=self.workspace_dir,
                excluded_dir=self.exclude_dir,
            ),
            project_path=str(proj_dir),
            project_name="bigdft_assistant",
            workspace_root=str(ws_root),
            file_manager=self.file_manager,
            task_manager=self.task_manager,
        )
        # Sauvegarde du prompt d’origine
        self._dev_prompt_default: str = self.developer.system_prompt

    # ------------------------------------------------------------------ #
    #                         UTILITAIRES PRIVÉS                        #
    # ------------------------------------------------------------------ #
    def _push(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    @staticmethod
    def _is_clean_question(txt: str) -> bool:
        banned = ("CONVERSATION", "{", "}", "champs obligatoires", "Il manque")
        if any(b.lower() in txt.lower() for b in banned):
            return False
        return txt.strip().endswith("?") and len(txt.strip().split()) <= 30

    # ------------------------------------------------------------------ #
    #                 CHANGEMENT DYNAMIQUE DU PROMPT DEV                #
    # ------------------------------------------------------------------ #
    def _update_developer_prompt(self):
        """Ajuste le system_prompt du développeur pour le stage courant."""
        if self.stage in STAGE_PROMPTS:
            new_prompt = STAGE_PROMPTS[self.stage]()  # prompt adapté
            self.developer.system_prompt = new_prompt
        else:
            # On revient au prompt d’origine
            self.developer.system_prompt = self._dev_prompt_default

    # --------------------- COLLECTE ÉTAT (known / missing) -------------- #
    async def _collect_state(self) -> Tuple[Dict[str, str], List[str]]:
        required = REQUIRED_FIELDS.get(self.stage, [])
        if not required:
            return {}, []

        system_message = (
            f"Vous êtes un expert en BigDFT chargé d'extraire des informations techniques précises. "
            f"Nous sommes à l'étape {self.stage.name}. "
            f"Analysez UNIQUEMENT les réponses explicites de l'utilisateur concernant: {', '.join(required)}. "
            f"Ne devinez PAS les informations non fournies clairement sauf lorsque le dialogue fait référence au dialogue (comme 'oui', 'd'accord', 'la valeur standard'). "
            f"Répondez strictement au format JSON: {{ \"known\": {{champ: valeur}}, \"missing\": [liste] }}"
        )

        # Convertir l'historique en format structuré
        messages = [{"role": "system", "content": system_message}]
        messages.extend(self.history)

        self._log("PROMPT", f"Prompt d'extraction: {messages}", is_internal=True)

        for attempt in range(3):
            try:
                answer = await self.super_agent.process_message(messages)
                self._log("RESPONSE", f"Réponse brute {attempt + 1}: {answer}", is_internal=True)

                # Normaliser la réponse selon son type
                data = None
                if isinstance(answer, dict):
                    data = answer
                elif isinstance(answer, str):
                    try:
                        data = json.loads(answer)
                    except json.JSONDecodeError:
                        # Si ce n'est pas du JSON valide, demander une correction
                        messages.append({"role": "assistant", "content": answer})
                        messages.append({"role": "user", "content":
                            "La réponse précédente n'était pas un JSON valide. "
                            "Répondez strictement au format JSON: { \"known\": {champ: valeur}, \"missing\": [liste] }"
                                         })
                        continue
                else:
                    # Type non supporté
                    messages.append({"role": "assistant", "content": str(answer)})
                    messages.append({"role": "user", "content":
                        "Format de réponse non supporté. "
                        "Répondez strictement au format JSON: { \"known\": {champ: valeur}, \"missing\": [liste] }"
                                     })
                    continue

                if data:
                    self._log("PARSED", f"Données extraites: {data}", is_internal=True)
                    return data.get("known", {}), data.get("missing", [])
            except Exception as e:
                self._log("ERROR", f"Erreur extraction d'état: {str(e)}", is_internal=True)
                # Continuer la boucle

        # En cas d'échec total
        return {}, required

    async def _ask_question(self, missing_field: str) -> str:
        """
        Génère une question concise pour obtenir l'information manquante.
        Utilise un format de messages structuré pour réduire les hallucinations.
        """
        system_message = (
            f"Vous êtes un expert en simulations BigDFT qui pose des questions techniques précises. "
            f"Votre rôle est d'obtenir l'information manquante: {missing_field}. "
            f"Formulez UNE SEULE question concise en français pour obtenir précisément cette information. "
            f"Votre réponse doit contenir UNIQUEMENT cette question, sans explication ni commentaire. "
            f"La question doit être courte, claire et se terminer par un point d'interrogation."
        )

        # Sélectionner les 3-5 derniers messages pertinents pour le contexte
        recent_context = [m for m in self.history[-5:] if m["role"] in ["user", "assistant"]]

        messages = [
            {"role": "system", "content": system_message},
            *recent_context
        ]

        for attempt in range(3):
            try:
                # Obtenir la réponse du modèle
                response = await self.super_agent.process_message(messages)

                # Extraire la question selon le type de réponse
                q = ""
                if isinstance(response, dict):
                    # Si c'est un dictionnaire, essayer d'extraire le texte
                    if 'content' in response:
                        q = response['content']
                    elif 'message' in response and 'content' in response['message']:
                        q = response['message']['content']
                    else:
                        # Log pour debugging
                        self._log("DEBUG", f"Format de dict inconnu: {response}", is_internal=True)
                        continue
                else:
                    # Si c'est une chaîne
                    q = str(response)

                # Nettoyer la question
                q = q.strip() if q else ""

                # Vérifier si c'est une question propre
                if q and self._is_clean_question(q):
                    return q

                # Construire un prompt plus simple pour la prochaine tentative
                messages = [
                    {"role": "system", "content": "Posez une seule question courte et directe."},
                    {"role": "user", "content": f"Demandez-moi simplement: {missing_field}"}
                ]
            except Exception as e:
                self._log("ERROR", f"Erreur dans _ask_question: {str(e)}", is_internal=True)
                # Continuer avec la tentative suivante

        # Solution de repli si toutes les tentatives échouent
        return f"Pouvez‑vous préciser {missing_field.replace('_', ' ')} ?"

    # ---------------------- GEN PROMPT & TASKS ------------------------- #
    async def _build_gen_prompt(self) -> Tuple[List[str], List]:
        files_map = {
            Stage.GENERATE_XYZ: ["structures/posinp.xyz"],
            Stage.GENERATE_INPUT: ["input_files/input.yaml"],
            Stage.GENERATE_SLURM: ["scripts/submit_job.sh"],
            Stage.GENERATE_ANALYSIS: ["analysis/analyze_results.py"],
        }
        files = files_map[self.stage]

        extra = ""
        if self.stage == Stage.GENERATE_XYZ:
            extra = (
                "\nIMPORTANT : Générer directement le contenu du fichier posinp.xyz "
                "sans script de génération."
            )

        messages = [
            {"role": "system", "content": f"Vous êtes un expert en génération de fichiers pour BigDFT. "
                                          f"OBJECTIF: {self.stage.name}\n"
                                          f"Fichier(s) à produire: {', '.join(files)}{extra}\n"
                                          f"Produisez une réponse conforme au format pydantic attendu."},
            # Inclure l'historique pertinent
            *self.history
        ]
        return files, messages

    @staticmethod
    def _task_to_dict(task: Any) -> Dict:
        if isinstance(task, dict):
            return task
        if hasattr(task, "model_dump"):
            return task.model_dump()
        if hasattr(task, "dict"):
            return task.dict()
        raise TypeError("Unsupported task format")

    async def _apply_tasks(
        self, tasks: List[Any], emit: Callable[[str], None]
    ) -> Tuple[bool, List[str]]:
        ok = True
        files: List[str] = []
        for t in tasks:
            td = self._task_to_dict(t)
            emit(f"\n• {td.get('title','tâche')}\n")
            try:
                await self.developer.process_task(td)
                files.extend(td.get("files", []))
                emit(" ✅\n")
            except Exception as e:
                ok = False
                emit(f" ❌ {e}\n")
        # Restaurer le prompt développeur
        self._update_developer_prompt()
        return ok, files

    def _copy_xyz(self, files: List[str]):
        for f in files:
            if f.endswith(".xyz"):
                src = self.workspace_dir / f
                uploads = self.workspace_dir.parent / "uploads"
                uploads.mkdir(exist_ok=True)
                dest = uploads / "posinp.xyz"
                shutil.copy2(src, dest)
                return str(dest)
        return None

    async def _analyze_user_intent(self, user_msg: str, context_field: str) -> Dict[str, Any]:
        """Analyse l'intention de l'utilisateur de façon générique en utilisant le LLM"""
        system_message = (
            f"Vous êtes un expert en BigDFT spécialisé dans l'analyse d'intention. "
            f"L'utilisateur configure un calcul et doit spécifier '{context_field}'. "
            f"Analysez UNIQUEMENT son intention, sans élaborer ni poursuivre la conversation. "
            f"Répondez strictement au format JSON: {{\"intent\": \"answer\"|\"clarification\"|\"other\", \"explanation\": \"brève explication\"}}"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_msg}
        ]

        for _ in range(3):  # Quelques tentatives pour obtenir un JSON valide
            try:
                response = await self.super_agent.process_message(messages)

                # Gestion des différents types de retour possibles
                if isinstance(response, dict):
                    # Si c'est déjà un dictionnaire, utiliser directement
                    return response
                elif isinstance(response, str):
                    # Si c'est une chaîne, essayer de la parser comme JSON
                    try:
                        return json.loads(response)
                    except json.JSONDecodeError:
                        # Si ce n'est pas du JSON valide, demander une correction
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content":
                            "La réponse précédente n'était pas un JSON valide. "
                            "Répondez strictement au format JSON: {\"intent\": \"answer\"|\"clarification\"|\"other\", \"explanation\": \"brève explication\"}"
                                         })
                        continue
                else:
                    # Type non pris en charge, convertir en chaîne
                    messages.append({"role": "assistant", "content": str(response)})
                    messages.append({"role": "user", "content":
                        "Format de réponse non supporté. "
                        "Répondez strictement au format JSON: {\"intent\": \"answer\"|\"clarification\"|\"other\", \"explanation\": \"brève explication\"}"
                                     })
                    continue
            except Exception as e:
                self._log("ERROR", f"Erreur pendant l'analyse d'intention: {str(e)}", is_internal=True)
                # Continuer la boucle pour nouvelle tentative

        # Fallback par défaut - en cas d'échec total
        return {"intent": "other", "explanation": "Impossible de déterminer l'intention"}

    def _setup_logger(self):
        """Configure un logger pour enregistrer toute la conversation et les étapes internes"""
        log_dir = self.workspace_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Nom de fichier avec timestamp pour ne pas écraser les anciennes sessions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"conversation_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger(f"bigdft_conversation_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.info("=== DÉBUT DE CONVERSATION BigDFT ===")

    def _log(self, category, message, is_internal=False):
        """Journalise un message avec son contexte"""
        level = logging.DEBUG if is_internal else logging.INFO
        prefix = "[INTERNE] " if is_internal else ""
        self.logger.log(level, f"{prefix}{category}: {message}")

    # ------------------------------------------------------------------ #
    #                             MAIN LOOP                              #
    # ------------------------------------------------------------------ #

    async def process_web_message(
        self, user_msg: str, emit: Callable[[str], None]
    ) -> Dict[str, Any]:
        self._push("user", user_msg)
        self._log("USER", user_msg)
        self._log("STAGE", f"Étape actuelle: {self.stage}", is_internal=True)

        # -------- COLLECTE D'INFOS OBLIGATOIRES ------------------------ #
        if self.stage in REQUIRED_FIELDS:
            self._log("PROCESS", "Début de collecte d'informations", is_internal=True)
            known, missing = await self._collect_state()
            self._log("STATE", f"Connu: {known}, Manquant: {missing}", is_internal=True)
            if known:
                recap = "\n".join(f" • {k} : {v}" for k, v in known.items())
                emit(f"\nRécapitulatif actuel :\n{recap}\n")

            if missing:
                # Analyser l'intention de l'utilisateur de façon générique
                intent_analysis = await self._analyze_user_intent(user_msg, missing[0])

                if intent_analysis.get("intent") == "clarification":
                    # L'utilisateur demande des explications
                    messages = [
                        {"role": "system",
                         "content": f"Vous êtes un expert en BigDFT spécialisé dans les explications pédagogiques. "
                                    f"L'utilisateur a besoin d'explications sur '{missing[0]}'. "
                                    f"Fournissez une explication claire, pédagogique et précise, suivie d'une question "
                                    f"concise pour obtenir cette information."}, *self.history]
                    explanation = await self.super_agent.process_message(messages)
                    emit(f"\nAssistant: {explanation}\n")
                    return {"status": "need_info"}
                else:
                    # Poser la question pour le champ manquant
                    question = await self._ask_question(missing[0])
                    emit(f"\nAssistant: {question}\n")
                    return {"status": "need_info"}

            # On passe à la génération
            self.stage = {
                Stage.ASK_SYSTEM: Stage.GENERATE_XYZ,
                Stage.ASK_ANALYSIS: Stage.ASK_CALC,  # Nouveau: Analyse → Calcul
                Stage.ASK_CALC: Stage.GENERATE_INPUT,
                Stage.ASK_HPC: Stage.GENERATE_SLURM,
                Stage.GENERATE_SLURM: Stage.GENERATE_ANALYSIS,  # Ajout: SLURM → Analyse
            }[self.stage]

        # ------------------- GÉNÉRATION DE FICHIERS ------------------- #
        if self.stage in {
            Stage.GENERATE_XYZ,
            Stage.GENERATE_INPUT,
            Stage.GENERATE_SLURM,
            Stage.GENERATE_ANALYSIS,
        }:
            # Adapter le prompt du développeur AVANT création des tâches
            self._update_developer_prompt()

            _, g_prompt = await self._build_gen_prompt()
            emit("\nAssistant: génération des fichiers…\n")
            raw = await self.idea_agent.process_message(g_prompt)

            tasks = (
                list(raw.improvement_proposal.tasks)
                if hasattr(raw, "improvement_proposal")
                else json.loads(raw)["improvement_proposal"]["tasks"]
            )
            ok, files = await self._apply_tasks(tasks, emit)
            if not ok:
                return {"status": "error"}

            if self.stage == Stage.GENERATE_XYZ:
                self._copy_xyz(files)
                self.stage = Stage.CONFIRM_XYZ
                xyz_abs_path = self._copy_xyz(files)
                self.stage = Stage.CONFIRM_XYZ
                emit("\nLe fichier posinp.xyz est prêt. Tapez 'ok' s'il vous convient.\n")
                return {
                    "status": "need_info",  # toujours en attente de 'ok'
                    "xyz_path": xyz_abs_path,  # <– chemin absolu
                    "files": files,
                    "workspace": str(self.workspace_dir),
                    }
            elif self.stage == Stage.GENERATE_INPUT:
                self.stage = Stage.CONFIRM_INPUT
                emit("\nLe fichier input.yaml est prêt. Tapez 'ok' pour valider.\n")
            elif self.stage == Stage.GENERATE_SLURM:
                self.stage = Stage.GENERATE_ANALYSIS
            else:  # ANALYSIS
                self.stage = Stage.FINAL

            return {
                "status": "success",
                "mode": "files_generated",
                "files": files,
                "workspace": str(self.workspace_dir),
            }

        # ------------------- CONFIRMATIONS UTILISATEUR ---------------- #
        if self.stage == Stage.CONFIRM_XYZ:
            if user_msg.lower().strip() in {"ok", "oui", "yes"}:
                self.stage = Stage.ASK_ANALYSIS
                emit("\nTraitement de la structure confirmée. Préparation des paramètres d'analyse'...\n")
                return await self.process_web_message("", emit)  # Appel récursif avec message vide
            else:
                emit("\nAssistant: quelles corrections sur la structure?\n")
                self.stage = Stage.ASK_SYSTEM
            return {"status": "need_info"}

        if self.stage == Stage.CONFIRM_INPUT:
            if user_msg.lower().strip() in {"ok", "oui", "yes"}:
                self.stage = Stage.ASK_HPC
                emit("\nTraitement du type de calcul confirmé. Préparation du HPC...\n")
                return await self.process_web_message("", emit)  # Appel récursif avec message vide
            else:
                emit("\nAssistant: précisez les modifications de paramètres DFT.\n")
                self.stage = Stage.ASK_CALC
            return {"status": "need_info"}

        # ------------------------------ FIN --------------------------- #
        if self.stage == Stage.FINAL:
            emit(
                "\n🎉 Tous les fichiers sont prêts! "
                "Soumettez le job puis exécutez analyze_results.py avec v_sim "
                "pour visualiser les résultats.\n"
            )
            return {"status": "completed"}

        emit("\nAssistant: Je reste à votre écoute.\n")
        return {"status": "asking"}

    # ------------------------------------------------------------------ #
    #                               CLEANUP                              #
    # ------------------------------------------------------------------ #
    def cleanup(self):
        for ag in (self.super_agent, self.idea_agent, self.developer):
            try:
                ag.cleanup()
            except Exception:
                pass


async def deploy_bigdft_assistant(workspace_dir: str | Path | None = None):
    try:
        LOGGER.info("Démarrage de l’assistant BigDFT…")
        return await BigDFTAssistant.create(workspace_dir)
    except Exception as exc:
        LOGGER.error("Erreur lors du déploiement: %s", exc)
        traceback.print_exc()
        raise