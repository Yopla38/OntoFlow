"""
------------------------------------------
Copyright: CEA Grenoble
Auteur: Louis BEAL
Entité: IRIG
Année: 2025
Description: Agent IA d'Intégration Continue
------------------------------------------
"""


def clean_output(content: str) -> str:
    """
    Parse the JSON content from the LLM response.

    Args:
        content (str): The content string from the LLM response.

    Returns:
        str: cleaned JSON string.
    """
    cache = []
    indent = 0
    for line in content.splitlines():
        append = False
        if "{" in line:
            append = True
            indent += line.count("{")
        if "}" in line:
            if indent >= 1:
                append = True
            indent -= line.count("}")
        if indent > 0:
            append = True

        if append:
            cache.append(line)

        if indent == 0 and append:
            break

    return "\n".join(cache)
