# Agent Task Usage Guide

## Task Structure

Each task must follow a precise structure with specific keywords. These keywords are essential for the system to function properly.

### Required Keywords

```python
{
    "id": int,          # Unique task identifier
    "title": str,       # Short and descriptive title
    "description": str, # Detailed description
    "files": List[str], # List of file paths
}
```
Full Task Format
```python
{
    # Identification fields
    "id": int,                    # Example: 1
    "title": str,                 # Example: "Calculator model creation"
    "description": str,           # Example: "Implement the base classes..."
    "assigned_role": str,         # Example: "Coder", "UI_Designer", "Tester"

    # Files and Structure
    "files": List[str],           # Example: ["app/models/calculator.py", "app/models/operations.py"]
                                  # IMPORTANT: Always use relative paths with '/'

    # Dependencies and Packages
    "dependencies": List[int],    # Example: [1, 2] (IDs of tasks this one depends on)
    "additional_packages": List[str],  # Example: ["numpy", "pandas"]

    # Validation
    "acceptance_criteria": List[str]   # Example: ["Complete unit tests", "Full documentation"]
}
```
Task Examples
Example 1: Development Task created by the main agent
```python
{
    "id": 1,
    "title": "Calculator model implementation",
    "description": "Create the base classes for the scientific calculator with fundamental operations",
    "assigned_role": "Coder",
    "files": [
        "app/models/calculator.py",
        "app/models/operations.py"
    ],
    "additional_packages": ["math", "decimal"],
    "acceptance_criteria": [
        "All basic operations implemented",
        "Unit tests present",
        "Complete documentation"
    ]
}
```
Example 2: Interface Task created by the main agent
```python
{
    "id": 2,
    "title": "User interface creation",
    "description": "Develop the web interface with Flask for the calculator",
    "assigned_role": "UI_Designer",
    "files": [
        "templates/calculator.html",
        "static/css/style.css",
        "static/js/calculator.js"
    ],
    "dependencies": [1],
    "additional_packages": ["flask", "bootstrap-flask"],
    "acceptance_criteria": [
        "Responsive interface",
        "Dark theme implemented",
        "All buttons functional"
    ]
}
```
Important Rules

    File Paths

        Always use relative paths

        Use / as the separator (even on Windows)

        Respect the project structure

    Available Roles

        "Architect"

        "Coder"

        "UI_Designer"

        "Tester"

    Dependencies

        Use the IDs of the tasks your task depends on

        Ensure the dependent tasks exist

    Additional Packages

        Specify only the necessary Python packages

        Use the exact names of the PyPI packages

Best Practices

    Description

        Be precise and detailed

        Include the necessary context

        Explain interactions with other components

    Files

        List all required files

        Respect the project structure

        Include test files if needed

    Acceptance Criteria

        Be specific and measurable

        Include quality criteria

        Specify required tests

Task Validation

The system will automatically check:

    The presence of all required fields

    The validity of file paths

    The existence of dependencies

    The consistency of assigned roles

See improvment_agent_for_MIA.py for an example of simple agents
See Agent_fortran.py for an example of more complex agents

