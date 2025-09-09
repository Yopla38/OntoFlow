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

