from pymarkovclustering.utils import random_edges, write_clusters, write_edges


def test_write_edges(tmp_path):
    """Test write edges"""
    edges = [("A", "B", 0.5), ("B", "C", 1.0), ("C", "A", 0.8)]
    outfile = tmp_path / "edges.tsv"
    write_edges(outfile, edges)

    expected_content = "A\tB\t0.5\nB\tC\t1.0\nC\tA\t0.8"
    assert outfile.read_text(encoding="utf-8") == expected_content


def test_write_clusters(tmp_path):
    """Test write clusters"""
    clusters = [["A", "B"], ["C"], ["D", "E", "F"]]
    outfile = tmp_path / "clusters.tsv"
    write_clusters(outfile, clusters)

    expected_content = "A\tB\nC\nD\tE\tF"
    assert outfile.read_text(encoding="utf-8") == expected_content


def test_random_edges():
    """Test random edges generation"""
    node_size = 1000
    edges = random_edges(node_size, max_cluster_size=20)

    nodes = set()
    for source, target, weight in edges:
        assert 0.0 <= weight <= 1.0
        nodes.add(source)
        nodes.add(target)

    assert len(nodes) == node_size
