import pymarkovclustering as pymcl


def test_mclviz(tmp_path):
    """Test pymclviz"""
    edges = pymcl.random_edges(100, random_add_rate=0.1, random_remove_rate=0.1)
    matrix, labels = pymcl.edges_to_sparse_matrix(edges)
    mcl_matrix = pymcl.mcl(matrix)
    clusters = pymcl.extract_clusters(mcl_matrix, labels)
    fig = pymcl.mclviz(matrix, labels, clusters)

    clusters_viz_file = tmp_path / "clusters.png"
    fig.savefig(clusters_viz_file, dpi=100)
    assert clusters_viz_file.exists()


def test_easymclviz(tmp_path):
    """Test pymclviz"""
    edges = pymcl.random_edges(100, random_add_rate=0.1, random_remove_rate=0.1)
    fig = pymcl.easymclviz(edges, show_label=True)

    clusters_viz_file = tmp_path / "clusters.png"
    fig.savefig(clusters_viz_file, dpi=100)  # type: ignore
    assert clusters_viz_file.exists()
