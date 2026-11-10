"""Gemeinsame Fixtures: das Handbeispiel, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def six():
    """Sechs Container: Zeiten 2, 3+, 4, 5, 5+, 8 (+ zensiert). Handrechnung: vier Ereignisse, Summe der Zeiten 27, also ρ̂ = 4/27."""
    return [2, 3, 4, 5, 5, 8], [1, 0, 1, 1, 0, 1]
