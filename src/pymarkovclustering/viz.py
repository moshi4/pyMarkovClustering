from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

import pymarkovclustering as pymcl

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def mclviz(
    matrix: np.ndarray | csr_matrix,
    labels: list[str],
    clusters: list[list[str]],
    /,
    *,
    ax: Axes | None = None,
    node_size: int = 20,
    node_cmap: str = "gist_rainbow",
    node_alpha: float = 1.0,
    edge_width: float = 1.0,
    edge_color: str = "lightgray",
    show_label: bool = False,
    label_va: str = "bottom",
    font_size: int = 8,
) -> Figure:
    """Visualize Markov Clustering clusters using networkx

    Parameters
    ----------
    matrix : np.ndarray | csr_matrix
        Adjacency matrix used as MCL input
    labels : list[str]
        Matrix labels
    clusters : list[list[str]]
        MCL clusters
    ax : Axes | None, optional
        Matplotlib axes. If None, auto created.
    node_size : int, optional
        Node plot size
    node_cmap : str, optional
        Node colormap (e.g. `gist_rainbow`, `jet`, `viridis`)
    node_alpha : float, optional
        Node color alpha parameter
    edge_width : float, optional
        Edge line width
    edge_color : str, optional
        Edge color
    show_label : bool, optional
        If True, show node label
    label_va : str, optional
        Node label vertical alignment (`top`|`center`|`bottom`|`baseline`|`center_baseline`)
    font_size : int, optional
        Node label size

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Notes
    -----
    Additional installation of `networkx` and `matplotlib` are required for MCL clusters visualization.
    For better position layout, extra packages `pygraphviz`, `pydot`, `lxml` installation is preferred.
    See <https://networkx.org/documentation/stable/install.html> in details.
    """  # noqa: E501
    try:
        # Check extra module import
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise

    # Create matplotlib axes if ax is None
    if ax is None:
        fig = plt.figure(figsize=(8, 6), tight_layout=True)
        ax = fig.add_subplot()
        ax.axis(False)

    # Plot network graph
    G = nx.Graph(matrix)
    G.remove_edges_from(nx.selfloop_edges(G))
    try:
        # Better position layout using pygraphviz
        pos = nx.drawing.nx_agraph.pygraphviz_layout(G)
    except ImportError:
        pos = nx.drawing.spring_layout(G)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_size,
        node_color=_get_node_colors(clusters, labels, node_cmap),  # type: ignore
        alpha=node_alpha,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(s, t) for s, t, d in G.edges.data() if d["weight"] > 0],
        ax=ax,
        width=edge_width,
        edge_color=edge_color,
    )
    if show_label:
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            labels={i: label for i, label in enumerate(labels)},
            font_size=font_size,
            verticalalignment=label_va,
        )
    return ax.figure  # type: ignore


def easymclviz(
    edges: str | Path | list[tuple[str, str, float]],
    /,
    *,
    inflation: float = 2.0,
    max_iter: int = 100,
    quiet: bool = True,
    ax: Axes | None = None,
    node_size: int = 20,
    node_cmap: str = "gist_rainbow",
    node_alpha: float = 1.0,
    edge_width: float = 1.0,
    edge_color: str = "lightgray",
    show_label: bool = False,
    label_va: str = "bottom",
    font_size: int = 8,
) -> Figure:
    """Run Markov Clustering and visualize clusters using networkx

    easymclviz automates load edges as matrix, MCL, extract clusters, visualization

    Parameters
    ----------
    edges : str | Path | list[tuple[str, str, float]]
        Edges(source, target, weight) file or list of tuples
    inflation : float, optional
        Inflation factor
    max_iter : int, optional
        Max number of iterations
    quiet : bool, optional
        If True, pring log on screen
    ax : Axes | None, optional
        Matplotlib axes. If None, auto created.
    node_size : int, optional
        Node plot size
    node_cmap : str, optional
        Node colormap (e.g. `gist_rainbow`, `jet`, `viridis`, `tab20`)
    node_alpha : float, optional
        Node color alpha parameter
    edge_width : float, optional
        Edge line width
    edge_color : str, optional
        Edge color
    show_label : bool, optional
        If True, show node label
    label_va : str, optional
        Node label vertical alignment (`top`|`center`|`bottom`|`baseline`|`center_baseline`)
    font_size : int, optional
        Node label size

    Returns
    -------
    fig : Figure
        Matplotlib figure

    Notes
    -----
    Additional installation of `networkx` and `matplotlib` are required for MCL clusters visualization.
    For better position layout, extra packages `pygraphviz`, `pydot`, `lxml` installation is preferred.
    See <https://networkx.org/documentation/stable/install.html> in details.

    Examples
    --------
    >>> import pymarkovclustering as pymcl
    >>> # easymclviz automates load edges as matrix, MCL, extract clusters, visualization
    >>> fig = pymcl.easymclviz("edges.tsv")
    >>> # easymclviz is same as code below
    >>> matrix, labels = pymcl.edges_to_sparse_matrix("edges.tsv")
    >>> mcl_matrix = pymcl.mcl(matrix)
    >>> clusters = pymcl.extract_clusters(mcl_matrix, labels)
    >>> fig = pymcl.mclviz(matrix, labels, clusters)
    """  # noqa: E501
    matrix, labels = pymcl.edges_to_sparse_matrix(edges)
    mcl_matrix = pymcl.mcl(matrix, inflation=inflation, max_iter=max_iter, quiet=quiet)
    clusters = pymcl.extract_clusters(mcl_matrix, labels)
    return mclviz(
        matrix,
        labels,
        clusters,
        ax=ax,
        node_size=node_size,
        node_cmap=node_cmap,
        node_alpha=node_alpha,
        edge_width=edge_width,
        edge_color=edge_color,
        show_label=show_label,
        label_va=label_va,
        font_size=font_size,
    )


def _get_node_colors(
    clusters: list[list[str]],
    labels: list[str],
    cmap: str = "gist_rainbow",
) -> list[str]:
    """Get appropriate node colors from clusters & labels & cmap"""
    try:
        import matplotlib as mpl
        from matplotlib.colors import to_hex
    except:
        raise
    label2node_idx = {label: i for i, label in enumerate(labels)}
    node_idx2color = {}
    for cluster_idx, cluster in enumerate(clusters):
        for node_idx in map(label2node_idx.get, cluster):
            color = to_hex(mpl.colormaps[cmap](cluster_idx / (len(clusters) - 1)))
            node_idx2color[node_idx] = color
    return [v for _, v in sorted(node_idx2color.items())]
