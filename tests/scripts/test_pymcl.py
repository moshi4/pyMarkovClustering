import shlex
import subprocess as sp


def test_pymcl_cli(simple_edges_file, tmp_path):
    """Test pymcl cli"""
    clusters_file = tmp_path / "clusters.tsv"

    cmd = f"pymcl {simple_edges_file} -o {clusters_file} -I 1.5"
    result = sp.run(shlex.split(cmd))

    assert result.returncode == 0
    assert clusters_file.exists()
    assert clusters_file.read_text(encoding="utf-8") == "A\tB\tC\nD\tE\nF\tG\nH\nI"


def test_pymcl__main__(simple_edges_file, tmp_path):
    """Test pymcl __main__ call"""
    cmd = f"python -m pymarkovclustering {simple_edges_file} -I 1.5"
    result = sp.run(shlex.split(cmd), capture_output=True, text=True, encoding="utf-8")

    assert result.returncode == 0
    assert result.stdout.strip() == "A\tB\tC\nD\tE\nF\tG\nH\nI"  # Check stdout
