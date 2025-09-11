## 1. Installation
Create a new environnement (tested with python 3.10)
```bash
cd 2-aiengine/OntoFlow/agent
pip install -r requirements.txt
python -m pip install aider-install
aider-install
```

## 2. Configuration

Place your OpenAI API key and Anthropic API key in two files.
```bash
mkdir keys
echo "sk-YourSuperSecretApiKey" > keys/openAI_key.txt
echo "sk-YourSuperSecretApiKey" > keys/anthropicAI_key.txt
```

Modify the constant API_KEY_PATH in agent/agent/Onto_wa_rag/CONSTANT.py

## 3. Run

```bash
python agent/test_agent_fortran.py --program-path test_folder/ --idea test/idea.txt
```

# Importants files for agents:
    - /agent/CONSTANT.py to define somes constants
    - /agent/ROLES.py to define agents roles
    - /agent/test_agent_fortran.py an entry point for an agent working on bigdft code
    - /agent/Agent_fortran.py The core of an agent

# Importants files for Onto RAG:
    - /agent/agent/Onto_wa_rag/CONSTANT.py to define somes constans
    - /agent/agent/Onto_wa_rag/Integration_fortran_RAG.py
    - /agent/agent/Onto_wa_rag/jupyter_analysis/jupyter_notebook_parser.py This parser is actually not include in the rag workflow. 
    - /agent/agent/Onto_wa_rag/Bibliotheque_d_ontologie somes files with ttl ontology (if you pass this file in a LLM, he can construct another)
    - /agent/agent/Onto_wa_rag/main_app.py is a CLI for ontorag but not used by agent

# Jupyter env
    
```bash
    
pip install ipykernel
python -m ipykernel install --user --name=hackathon-venv --display-name="Python (Hackathon Venv)"
```

Changez le noyau de votre notebook.

    Ouvrez votre fichier .ipynb.

    En haut à droite, vous verrez probablement "Python 3 (ipykernel)" ou le nom de votre ancien noyau. Cliquez dessus.

    Dans la liste qui apparaît, sélectionnez le NOUVEAU noyau : "Python (Hackathon Venv)".

    Le notebook va redémarrer et se connecter au bon interpréteur.

  
