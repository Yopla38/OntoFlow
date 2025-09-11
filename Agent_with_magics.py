import asyncio
import tempfile
import shutil
from pathlib import Path

# --- Imports de votre projet ---
from agent.Agent_fortran import deploy_improvment_agent_fortran, SimpleAgentFactory, DevelopmentAgentFactory, \
    AiderCodeAggregator
from agent.src.components.file_manager import FileManager # <-- On importe le VRAI FileManager
from agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG
from agent.src.components.task_manager import TaskManager
from agent.src.types.roles_definition import select_role
from agent.src.utils.utilitaires import generate_tree_llm
from agent.CONSTANT import MODEL_AGREGATEUR

# --- Imports IPython Magic ---
from IPython.core.magic import Magics, magics_class, line_cell_magic
from IPython.display import display, Markdown
import nest_asyncio

nest_asyncio.apply()


class NotebookFileManager:
    """
    Un FileManager qui intercepte les écritures de fichiers pour un usage en notebook.
    Il se fait passer pour le vrai FileManager de l'agent.
    """

    def __init__(self, project_path):
        self._project_path = Path(project_path)
        self.generated_code = ""
        self.file_written = None

    async def write_file(self, file_path: str, content: str):
        """Au lieu d'écrire sur le disque, on stocke le contenu."""
        print(f"Intercepted write to '{file_path}'")
        self.generated_code = content
        self.file_written = file_path
        return True

    def get_generated_code(self) -> str:
        """Récupère le dernier code qui a été "écrit"."""
        return self.generated_code

    # Implémenter d'autres méthodes que l'agent pourrait appeler, même si elles sont vides.
    async def read_file(self, file_path: str) -> str:
        return ""  # Le contexte vient du RAG, pas besoin de lire de vrais fichiers.

    async def initialize_venv(self):
        pass  # Pas de venv nécessaire pour la génération de code simple.


# ==============================================================================
# 2. CLASSE D'ORCHESTRATION DE L'AGENT (UTILISANT VOS AGENTS)
# ==============================================================================

class NotebookCodeAgent:
    def __init__(self, rag_instance: OntoRAG):
        self.rag_agent = rag_instance
        self.conversation_state = {}
        self.idea_generator = None
        self.developpeur = None

        # Créer un espace de travail temporaire pour les agents
        self.workspace_dir = Path(tempfile.mkdtemp(prefix="agent_notebook_"))

        # Initialiser les vrais agents
        self._initialize_agents()

        print("🤖✨ Agents réels (Planner/Developer) initialisés et prêts.")

    def _initialize_agents(self):
        """Crée les instances réelles des agents en utilisant les factories."""
        file_manager = NotebookFileManager(self.workspace_dir)
        task_manager = TaskManager()
        file_manager = FileManager(project_path=str(self.workspace_dir))
        # Initialisation du venv n'est pas nécessaire si on ne compile/exécute pas
        # await file_manager.initialize_venv()

        self.idea_generator = SimpleAgentFactory.create_agent_from_role(
            role_config=select_role("GenerateurIdees", base_dir=self.workspace_dir),
            project_path=str(self.workspace_dir), project_name="notebook_project",
            workspace_root=str(self.workspace_dir), file_manager=file_manager,
            task_manager=task_manager
        )

        self.developpeur = DevelopmentAgentFactory.create_agent_from_role(
            role_config=select_role("Developpeur", self.workspace_dir, excluded_dir=set()),  # Correction ici
            project_path=str(self.workspace_dir),
            project_name="notebook_project",
            workspace_root=str(self.workspace_dir),
            file_manager=file_manager,
            task_manager=task_manager,
            agregateur=AiderCodeAggregator(MODEL_AGREGATEUR)
        )

    async def generate_plan(self, user_idea: str):
        """Étape 1 : Le VRAI Planner génère un plan d'action."""
        self.clear_state()
        self.conversation_state['initial_idea'] = user_idea

        print("🧠 Agent Planificateur : Sollicitation...")
        rag_context = await self.rag_agent.query(user_idea, use_ontology=True)

        # Le prompt est simplifié car le contexte projet est vide. Le RAG est la source de vérité.
        improvement_prompt = f"DEMANDE UTILISATEUR: {user_idea}\nCONTEXTE RAG: {rag_context.get('answer')}\n\nTA MISSION : Décompose cette demande en tâches pour un agent développeur qui doit générer UN SEUL fichier de code Python."

        # Appel réel à votre agent
        plan_response = await self.idea_generator.process_message(improvement_prompt)

        if not plan_response or not hasattr(plan_response, 'improvement_proposal'):
            return "❌ Le planificateur n'a pas pu générer de plan."

        tasks = plan_response.improvement_proposal.tasks
        self.conversation_state['plan_tasks'] = [t.model_dump() for t in tasks]  # Stocker les tâches

        # Formatter le plan pour l'affichage
        md_plan = "### Plan d'action de l'Agent\n"
        for i, task in enumerate(tasks):
            md_plan += f"{i + 1}. **{task.title}**\n   - {task.description}\n"

        self.conversation_state['status'] = 'AWAITING_EXECUTION'
        return md_plan

    async def execute_plan_and_generate_code(self, shell):
        """Le vrai Développeur exécute le plan dans le sandbox, puis nous lisons le résultat."""
        if self.conversation_state.get('status') != 'AWAITING_EXECUTION':
            return

        print("✍️ Agent Développeur : Exécution du plan dans un environnement temporaire...")
        plan_tasks = self.conversation_state.get('plan_tasks', [])

        # L'agent va maintenant appeler file_exists, create_or_update_file, etc.
        # sur le vrai FileManager, qui écrira dans self.workspace_dir.
        for task_dict in plan_tasks:
            await self.developpeur.process_task(task_dict)

        # Maintenant, on cherche le fichier .py que l'agent a dû créer
        generated_files = list(Path(self.workspace_dir).glob('*.py'))

        if generated_files:
            # On prend le premier fichier python trouvé
            output_file_path = generated_files[0]
            print(f"Code généré par l'agent dans le fichier : {output_file_path.name}")

            # On lit son contenu
            final_code = output_file_path.read_text(encoding='utf-8')

            # On l'injecte dans la cellule suivante
            shell.set_next_input(final_code, replace=False)
            display(Markdown("### ✅ Code Généré\nLe plan a été exécuté. Une nouvelle cellule a été créée."))
        else:
            display(Markdown(
                "### ❌ Erreur\nLe développeur a terminé son travail mais aucun fichier de code Python n'a été trouvé dans le workspace."))

        self.clear_state()  # Nettoie le workspace et réinitialise pour la prochaine fois

    def clear_state(self):
        """Nettoie le workspace temporaire et réinitialise les agents."""
        if self.workspace_dir and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
        self.workspace_dir = Path(tempfile.mkdtemp(prefix="agent_notebook_"))
        self.conversation_state = {}
        self._initialize_agents()
        print(f"🗑️ État de l'agent et workspace ({self.workspace_dir.name}) réinitialisés.")


# ==============================================================================
# 3. CLASSE DE LA MAGIC IPYTHON (L'INTERFACE UTILISATEUR)
# ==============================================================================

@magics_class
class CIAgentMagic(Magics):
    def __init__(self, shell):
        super(CIAgentMagic, self).__init__(shell)
        self.rag_instance = None
        self.code_agent = None
        if 'ontorag_magic' or 'Onto_RAG_with_magics' in shell.extension_manager.loaded:
            magic_instance = shell.magics_manager.magics['line']['rag'].__self__
            # Maintenant, récupérer l'instance du moteur RAG stockée dans cette classe
            self.rag_instance = magic_instance.rag
            if self.rag_instance:
                self.code_agent = NotebookCodeAgent(self.rag_instance)
                print("✅ Connexion à l'instance RAG existante réussie.")

        print("🪄 Magic Agent CI (Intégration Réelle) prête.")

    @line_cell_magic
    def agent(self, line, cell=None):
        async def main():
            if not self.code_agent:
                display(Markdown("### ❌ Erreur : Moteur RAG non trouvé."))
                return

            query = cell.strip() if cell else line.strip()

            if not query.startswith('/'):
                command, args = '/plan', query
            else:
                command, *args_list = query.split(' ', 1)
                args = args_list[0] if args_list else ""

            if command == '/plan':
                plan_md = await self.code_agent.generate_plan(args)
                display(Markdown(plan_md))
                display(Markdown("\n**Si ce plan vous convient, exécutez :** `%agent /execute_plan`"))

            elif command == '/execute_plan':
                await self.code_agent.execute_plan_and_generate_code(self.shell)

            elif command == '/clear':
                self.code_agent.clear_state()

            else:
                display(Markdown(
                    "### Aide Agent CI\n- `%%agent <idée>` pour générer un plan.\n- `%agent /execute_plan` pour générer le code."))

        asyncio.run(main())


def load_ipython_extension(ipython):
    ipython.register_magics(CIAgentMagic)

