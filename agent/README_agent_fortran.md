# Documentation for the `agent_fortran.py` script

This document explains the principle and functionality of the `agent_fortran.py` script. This script implements a system of autonomous agents to iteratively improve a Fortran codebase based on an idea or request formulated in natural language.

## 1. General Principle

The objective of `agent_fortran.py` is to automate the process of developing and refactoring Fortran code. To achieve this, it relies on a multi-agent architecture where each agent has a specific role and collaborates with others to achieve a common goal. [5, 7, 9]

The key concepts are as follows:
*   **Autonomous Agents**: Programs capable of acting independently to accomplish tasks. [7, 12, 13]
*   **Multi-Agent System**: A collection of agents that interact to solve complex problems. [7]
*   **Retrieval-Augmented Generation (RAG)**: A technique that allows language models to draw upon external knowledge to generate more accurate and contextual responses. [3, 11, 14, 15, 16]
*   **Iterative Improvement**: The script operates in cycles, where in each iteration, it attempts to improve the existing code based on the initial request.

## 2. Architecture and Components

The script is built around the `deploy_improvment_agent_fortran` class and several components that interact within the `auto_improve_cycle` method.

### 2.1. The Agents

*   **`idea_generator` ("Idea Generator" Agent)**:
    *   **Role**: Technical Project Manager.
    *   **Function**: Analyzes the user's request and the current state of the code to generate a detailed action plan. This plan is a list of precise technical tasks for the developer agent. It is also responsible for formulating queries for the RAG agent.

*   **`developpeur` ("Developer" Agent)**:
    *   **Role**: Software Engineer.
    *   **Function**: Executes the tasks defined by the "Idea Generator". Each task results in the modification or creation of a code file.

*   **`rag_agent` ("RAG" Agent)**:
    *   **Role**: Source Code Expert.
    *   **Function**: Provides precise information about the existing code. It can either **analyze** the code to extract general concepts or **extract** specific portions of the source code (like subroutines or functions). It relies on a vector knowledge base of the project's code.

### 2.2. The Managers

*   **`FileManager`**: Manages all file-related operations (reading, writing, directory tree management).
*   **`TaskManager`**: Manages the queue and status of tasks to be performed.

## 3. How the Improvement Cycle (`auto_improve_cycle`) Works

The core of the script is the `auto_improve_cycle` method. It orchestrates the collaboration between the agents by following a multi-step process:

**Step 1: Preliminary Plan**
1.  The `idea_generator` analyzes the user's request (`user_idea`) and the current project's directory structure.
2.  It breaks down the objective into a series of actionable tasks for the `developpeur`.
3.  For each task, it determines if contextual information about the code is needed and formulates one or more "RAG consultations" (queries for the RAG agent).

**Step 2: Knowledge Gathering (via the RAG agent)**
1.  If RAG consultations have been defined, the `rag_agent` is queried.
2.  For each query, the `rag_agent` interrogates its knowledge base (the previously indexed project code) and returns either an analysis or a source code snippet.

**Step 3: Plan Correction**
1.  The results obtained from the RAG agent are provided to the `idea_generator`.
2.  The latter uses this new context to correct the initial plan. The main goal of this phase is to ensure that the file names associated with each task are correct.

**Step 4: Final Plan Enrichment**
1.  The descriptions of the tasks in the corrected plan are "enriched" with the contextual information obtained from the RAG agent. This gives the `developpeur` all the necessary information to accomplish its task effectively.

**Step 5: Final Plan Execution**
1.  The `developpeur` sequentially executes each task from the final, enriched plan.
2.  Each task translates into a concrete operation on the code (file modification or creation).

**Step 6 & 7: Verification and Post-Verification Corrections**
1.  Once all tasks are executed, a verification phase is launched to ensure that the initial request has been successfully fulfilled.
2.  If corrections are necessary, a new, shorter cycle of planning and execution is initiated to apply them.

## 4. How It Works: A Concrete Example

Let's take the example provided in the script:
```python
test_idea = ("Please write the complete docstring for the function inspect_rototranslation. "
             "Also, write the README.md file for the reformatting module")
```

1.  **Preliminary Plan**:
    *   The `idea_generator` creates two main tasks.
    *   For the first task, it does not know the exact file containing `inspect_rototranslation`. It therefore generates a RAG consultation: `"Please extract the complete code for the function inspect_rototranslation."`.
    *   For the second task, it generates another RAG consultation to get an overview: `"Provide a summary of the project's overall architecture for the purpose of writing a README."`.

2.  **Knowledge Gathering**:
    *   The `rag_agent` searches for `inspect_rototranslation` and returns its source code along with the name of the file where it is located.
    *   It analyzes the entire project to provide a summary.

3.  **Plan Correction**:
    *   The `idea_generator` receives the RAG's response and updates the first task by adding the correct filename.

4.  **Enrichment**:
    *   The description of the first task is enriched with the function's source code.
    *   The description of the second task is enriched with the architecture summary.

5.  **Execution**:
    *   The `developpeur` receives the first task, and thanks to the provided context, it has both the code to be documented and the description of what needs to be done. It modifies the file to add the docstring.
    *   It then executes the second task and creates a `README.md` file with content based on the provided summary.

This process repeats for the number of defined iterations, allowing for the gradual refinement of the result.

## 5. Running the Script

To use this script, you must:
1.  Ensure that all necessary dependencies are installed.
2.  Modify the `test_library_path` and `test_idea` variables in the `example_run` function to point to the Fortran project to be improved and to describe the desired objective.
3.  Execute the script from a terminal: `python agent_fortran.py`.

The script will then initialize the environment, index the project files for the RAG, and start the continuous improvement loop.