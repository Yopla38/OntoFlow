# OntoRAG: The Agent-First RAG for Code Analysis

Welcome, hackers! OntoRAG is a high-powered Retrieval-Augmented Generation (RAG) system designed to build AI agents that can deeply understand and reason about complex codebases, especially **Fortran**.

Forget simple keyword search. This is your toolkit to create agents that can analyze dependencies, explain scientific concepts hidden in the code, and even suggest refactors.

## 🚀 Core Concept: The Two-Tiered Agent System

OntoRAG provides two ways to query the system, designed for different agent behaviors. Understanding this is key to building awesome projects.

| Feature | **Conversational Agent (`run`)** | **Quick Query Router (`ask`)** |
| :--- | :--- | :--- |
| **Use Case** | **Complex tasks, dialogue, planning** | **Simple, factual, one-shot questions** |
| **How it Works** | A stateful agent that uses memory, plans multi-step actions, and can ask for clarification. | A stateless router that finds the best single tool for a direct question. |
| **Best For** | Building code assistants, documentation writers, architectural analyzers. | Getting quick facts to feed into a larger agent logic. |
| **Memory** | **Yes.** Remembers the conversation history. | **No.** Each query is independent. |
| **Example** | `"Summarize the entire architecture of the Poisson solver."` | `"Who calls the 'cleanup' function?"` |
| **Code Call** | `rag.agent_fortran.run(query)` | `rag.ask(query)` |


## 💻 Quickstart: Get an Agent Running in 5 Minutes

### 1. Installation

```bash
# Clone and install dependencies
git clone [YOUR_REPO_URL]
cd OntoRAG
pip install -r requirements.txt
```

### 2. Configuration

Place your OpenAI API key in a file. The default path is `keys/openAI_key.txt`.
```bash
mkdir keys
echo "sk-YourSuperSecretApiKey" > keys/openAI_key.txt
```

### 3. Your First Agent Interaction (`main.py`)

This complete example initializes the system, loads a directory of Fortran code, and uses the powerful conversational agent to answer a complex question.

```python
import asyncio
from onto_rag import OntoRAG

# --- Configuration ---
# Directory where OntoRAG will store its indexes and data
STORAGE_DIR = "ontorag_storage_hackathon" 
# Path to your project's code
CODE_DIRECTORY = "/path/to/your/fortran/project/src"
# (Optional) Path to a domain ontology file (.ttl, .jsonld)
ONTOLOGY_PATH = "path/to/your/ontology.ttl"

async def main():
    """Initializes OntoRAG, ingests code, and runs the agent."""

    # 1. Initialize the system
    print("🚀 Initializing OntoRAG...")
    rag = OntoRAG(
        storage_dir=STORAGE_DIR,
        ontology_path=ONTOLOGY_PATH,
        model="gpt-4o" # Powerful model recommended for agent reasoning
    )
    await rag.initialize()
    print("✅ System ready!")

    # 2. Scan and add documents to the knowledge base
    print(f"📁 Scanning code from {CODE_DIRECTORY}...")
    documents = rag.scan_directory(directory_path=CODE_DIRECTORY)
    
    if not documents:
        print(f"❌ No documents found in {CODE_DIRECTORY}. Please check the path.")
        return
        
    print(f"🧠 Indexing {len(documents)} files...")
    await rag.add_documents_batch(documents, max_concurrent=5)
    print("✅ Indexing complete!")

    # 3. Unleash the Conversational Agent!
    print("\n🤖 --- Starting Agent Session ---")
    
    # This is the agent you'll build upon. It can plan, reason, and remember.
    question = "Give me a high-level summary of the main purpose of the 'wave_functions_mod' module and list its key public subroutines."
    
    print(f"\nUser Query: {question}")
    print("Agent is thinking...")
    
    agent_response = await rag.agent_fortran.run(question, use_memory=True)
    
    print("\n--- Agent's Final Answer ---")
    print(agent_response)
    print("--- End of Answer ---")

if __name__ == "__main__":
    asyncio.run(main())```
```
## 🤖 The Conversational Agent in Depth (`/agent`)

This is your primary tool for building sophisticated AI applications.

### Core Features

-   **Multi-Step Reasoning**: The agent can break down a complex query like "Analyze the impact of changing X" into smaller steps: find X, find its callers, find its dependencies, and then synthesize a summary.
-   **Stateful Memory**: The agent remembers the context of your conversation. You can ask follow-up questions.
-   **Interactive Clarification**: If a query is ambiguous, the agent can ask you for more details instead of failing.

### Managing the Agent's Memory

Memory is powerful but can sometimes lead to confusion if you switch topics. You have full control.

**To clear the agent's memory:**
This is essential when you want to start a fresh conversation or switch tasks.

*From the CLI:*
```
💫 Commande : /agent_clear
```

*From your Python code:*
```python
# Clears the entire conversation history and internal thought process.
rag.agent_fortran.clear_memory()
print("🧠 Agent memory has been cleared.")
```

**To inspect the agent's memory:**

*From the CLI:*
```
💫 Commande : /agent_memory
```

*From your Python code:*
```python
summary = rag.agent_fortran.get_memory_summary()
print(summary)
```

## 🛠️ The Quick Query Router (`ask`)

Use `rag.ask()` when your agent needs a quick, factual piece of information without conversational context. It's faster and more direct.

```python
# Example: Your agent needs to know the exact file path of a function
# before performing a more complex analysis.

result_dict = await rag.ask("Where is the 'compute_energy' subroutine located?")

# The result is structured data, not just text.
# Your agent can then parse this to get the information it needs.
```

## 📜 CLI Cheatsheet

The provided `example_usage()` function gives you a powerful command-line interface for rapid testing.

| Command | Description |
| :--- | :--- |
| `/agent <your complex task>` | **(Primary Tool)** Deploys the conversational agent. |
| `<your simple question>` | Uses the quick `ask` router for direct queries. |
| `/find <entity_name>` | Searches for Fortran entities (functions, modules, etc.). |
| `/consult_entity <name>` | Gets a raw, detailed data report on a code entity. |
| `/visualization` | Generates an interactive dependency graph HTML file. |
| `/stats` or `/stats entity`| Shows statistics about the indexed content. |
| `/agent_clear` | **(Important!)** Wipes the agent's memory clean. |
| `/help` | Shows all available commands. |
| `/quit` | Exits the CLI. |

