from pathlib import Path

import pytest


@pytest.fixture()
def simple_edges() -> list[tuple[str, str, float]]:
    """Simple edges list of tuples"""
    return [
        ("A", "B", 10),
        ("B", "A", 8.0),
        ("A", "C", 10),
        ("B", "C", 2),
        ("D", "E", 5),
        ("F", "G", 2),
        ("H", "I", 0),
    ]


@pytest.fixture()
def simple_edges_file() -> Path:
    """Simple edges file"""
    return Path(__file__).parent / "resources" / "simple_edges.tsv"
