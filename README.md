## 1. Installation (only Onto_RAG)
Create a new environnement (tested with python 3.10)
```bash
cd 2-aiengine/OntoFlow/agent
pip install -r requirements.txt
```

## 2. Configuration

Place your OpenAI API key and Anthropic API key in two files.
```bash
mkdir keys
echo "sk-YourSuperSecretApiKey" > keys/openAI_key.txt
echo "sk-YourSuperSecretApiKey" > keys/anthropicAI_key.txt
```

Modify the constant API_KEY_PATH in agent/Onto_wa_rag/CONSTANT.py

## 3. Run
Go to 2-aiengine/OntoFlow

CLI version
```bash
python Only_RAG.py 
```

# Jupyter env
    
```bash
    
pip install ipykernel
python -m ipykernel install --user --name=hackathon-venv --display-name="Python (Hackathon Venv)"
```

Go to 2-aiengine/OntoFlow
open the demo_.ipynb  (use the notebook kernel : "Python (Hackathon Venv)")