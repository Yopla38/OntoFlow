"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

import pathlib
import xml.etree.ElementTree as ET
import open_fortran_parser
interesting_tags = {
    'module', 'subroutine', 'function', 'program', 'interface',
    'module_procedure', 'block_data', 'procedure_stmt'
}

def extract_defs(f90_path):
    root = open_fortran_parser.parse(pathlib.Path(f90_path))
    file_node = root.find('file')

    results = []

    def recursive_scan(node):
        # Cas spécial : vrai type utilisateur

        if node.tag == 'variable':
            name = node.get('name')
            lb = node.get('line_begin')
            le = node.get('line_end')
            if name and lb and le:
                results.append(('derived_type', name, int(lb), int(le)))

        elif node.tag in interesting_tags:
            name = node.get('name')
            lb = node.get('line_begin')
            le = node.get('line_end')
            if name and lb and le:
                results.append((node.tag, name, int(lb), int(le)))

        # descente récursive
        for child in node:
            recursive_scan(child)

    recursive_scan(file_node)
    return results


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    for kind, name, lb, le in extract_defs(path):
        print(f"{kind.capitalize():15s} {name:20s} lines {lb}-{le}")

