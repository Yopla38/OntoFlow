"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

# utils/f2py_fortran_semantic_chunker.py
import os
import re
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

# Import f2py components
try:
    import numpy.f2py.crackfortran as crackfortran

    F2PY_AVAILABLE = True
except ImportError:
    F2PY_AVAILABLE = False
    print("⚠️ F2PY not available, falling back to regex parser")

from .semantic_chunker import SemanticChunker, DocumentSection




