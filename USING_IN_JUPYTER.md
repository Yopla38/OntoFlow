# Quick Guide: Interacting with the OntoRAG Agent in your Notebook

NOTE : Actually, the jupyter parser is not functional, so only f90 files works

This guide summarizes the essential commands for using the RAG agent via the `%rag` magic command.
### HOW TO
After README.md installation instructions
```bash
source [your_path]/llm-hackathon-2025/venv/bin/activate
cd [your_path]/llm-hackathon-2025/
pip install ipykernel
python -m ipykernel install --user --name=hackathon-venv --display-name="Python (Hackathon Venv)"
# start jupyter lab
jupyter lab
```

#### Change the notebook kernel:
At the top right, you’ll probably see “Python 3 (ipykernel)” or the name of your old kernel. Click on it.

From the list that appears, select the NEW kernel: “Python (Hackathon Venv)”.

The notebook will restart and connect to the correct interpreter.

### **Step 1: Initialization (Once per session)**

1.  **Load the Magic Command**
    Run this command to enable the `%rag` tool.
    ```ipython
    %load_ext Onto_RAG_with_magics
    ```

2.  **Index the Documents**
    Tell the agent which documents to analyze. Only run this command if the index doesn't exist or if your documents have changed.
    ```ipython
    # Define `documents_info` variable
    documents_info = [
    {"filepath": "[your_path]/llm-hackathon-2025/2-aiengine/OntoFlow/test_folder/PSbox.f90", "project_name": "BigDFT", "version": "1.9"},
    # Ajoutez ici d'autres documents si vous le souhaitez]
    
    # Rag indexation
    %rag /add_docs documents_info
    ```

---

#### **Step 2: The Conversational Agent (Main Usage)**

Use this mode for complex questions that require a dialogue.

*   **Start a Conversation**: `%rag /agent <your question>`
    > Clears the previous memory and asks your first question.
    ```ipython
    %rag /agent A quoi sert la subroutine reduce_energies.
    ```

*   **Reply to the Agent**: `%rag /agent_reply <your answer>`
    > Use this only when the agent asks for clarification.
    ```ipython
    %rag /agent_reply Give me a practical code example.
    ```

*   **Clear the Memory**: `%rag /agent_clear`
    > Resets the current conversation.

---

#### **Step 3: Direct Search (For simple questions)**

*   **Natural Language Query**: `%rag <your question>`
    > For a simple, single-line question. The answer is often more direct.
    ```ipython
    %rag What is the purpose of the System module in PyBigDFT?
    ```
    > For a long or multi-line question, use `%%rag` at the beginning of the cell.
    ```ipython
    %%rag
    I want to visualize the electron density from a calculation.
    What are the steps to follow after the calculation is finished?
    ```

*   **Semantic Search**: `%rag /search <your question>`
    > Finds the most relevant passages in the documents and presents them as sources.

---

#### **Step 4: Utility Commands**

*   **Display Help**: `%rag /help`
    > Displays the list of all available commands.

*   **List Documents**: `%rag /list`
    > Shows all documents that have been loaded into the index.

*   **View Statistics**: `%rag /stats`
    > Displays information about the index (number of documents, chunks, etc.).