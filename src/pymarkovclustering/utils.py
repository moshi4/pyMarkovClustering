from __future__ import annotations

import csv
import itertools
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy.sparse import csr_matrix


def edges_to_sparse_matrix(
    edges: str | Path | list[tuple[str, str, float]],
    /,
) -> tuple[csr_matrix, list[str]]:
    """Convert edges file or list of tuples to sparse matrix

    Parameters
    ----------
    edges : str | Path | list[tuple[str, str, float]]
        Edges(source, target, weight) file or list of tuples

    Returns
    -------
    matrix : csr_matrix
        Sparse matrix representation of edges
    nodes : list[str]
        List of node labels
    """
    # Load edges data
    edges = _load_edges(edges)

    # Prepare data for sparse matrix
    node2index = {}
    weights = []
    source_indices, target_indices = [], []
    for source, target, weight in edges:
        if source not in node2index:
            node2index[source] = len(node2index)
        if target not in node2index:
            node2index[target] = len(node2index)
        source_indices.append(node2index[source])
        target_indices.append(node2index[target])
        weights.append(weight)

    # Create sparse matrix
    nodes = list(node2index.keys())
    matrix = csr_matrix(
        (weights, (source_indices, target_indices)),
        shape=(len(nodes), len(nodes)),
    )
    return matrix, nodes


def _load_edges(
    edges: str | Path | list[tuple[str, str, float]],
) -> list[tuple[str, str, float]]:
    """Load edges file and return a list of tuples"""
    if isinstance(edges, (list, tuple)):
        return edges
    data: list[tuple[str, str, float]] = []
    with open(edges, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3:
                source, target, weight = row[0], row[1], row[2]
            else:
                source, target, weight = row[0], row[1], 1.0
            data.append((source, target, float(weight)))
    return data


def extract_clusters(
    matrix: csr_matrix,
    labels: list[str] | None = None,
    /,
) -> list[list[str]]:
    """
    Extract clusters from MCL result matrix and map them to labels

    Parameters
    ----------
    matrix : csr_matrix
        MCL result matrix
    labels : list[str] | None, optional
        List of labels corresponding to matrix indices.
        If None, `'0','1','2'...'X'` index label is used.

    Returns
    -------
    clusters : list[list[str]]
        List of clusters, where each cluster is a list of labels
    """
    if labels is None:
        labels = list(map(str, range(matrix.shape[0])))
    if len(labels) != matrix.shape[0]:
        raise ValueError("Length of labels must match matrix size")

    # Cluster labels based on row indices of max value in each column of matrix
    row_idx2cluster = defaultdict(list)
    for col_idx, row_idx in enumerate(matrix.argmax(axis=0).A1):  # type:ignore
        row_idx2cluster[row_idx].append(labels[col_idx])
    clusters: list[list[str]] = list(row_idx2cluster.values())

    # Check cluster labels match input labels
    cluster_labels = itertools.chain.from_iterable(clusters)
    if sorted(cluster_labels) != sorted(labels):
        raise ValueError("Input labels missing in result clusters. MCL bug? Failed to extract clusters.")  # fmt: skip  # noqa: E501

    return sorted(clusters, key=lambda c: len(c), reverse=True)


def random_edges(
    node_size: int = 100,
    /,
    *,
    min_cluster_size: int = 1,
    max_cluster_size: int = 10,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    random_add_rate: float = 0,
    random_remove_rate: float = 0,
    seed: int | None = 0,
) -> list[tuple[str, str, float]]:
    """
    Simple random edges generator for clustering test

    Node name is set as `{ClusterID}_{NodeID in Cluster}` e.g. `18_2`

    Parameters
    ----------
    node_size : int, optional
        Total number of nodes to generate
    min_cluster_size, max_cluster_size : int, optional
        Min-Max size of each cluster
    min_weight, max_weight : float, optional
        Min-Max weight for edges
    random_add_rate, random_remove_rate : float, optional
        Random add, remove edges rate for noisy dataset generation
    seed : int | None, optional
        Random seed for reproducibility

    Returns
    -------
    edges : list[tuple[str, str, float]]
        Edges list of tuples
    """  # noqa: E501
    random.seed(seed)
    if not 1 <= min_cluster_size <= max_cluster_size:
        raise ValueError(f"Invalid min-max cluster size (1 <= {min_cluster_size=} <= {max_cluster_size})")  # fmt: skip  # noqa: E501
    if not 0.0 <= min_weight <= max_weight:
        raise ValueError(f"Invalid min-max weight (0.0 <= {min_weight=} <= {max_weight=})")  # fmt: skip  # noqa: E501

    # Generate clusters until the total node size is reached
    data_count = 0
    cluster_count = 0
    clusters: list[list[str]] = []
    while True:
        cluster_count += 1
        cluster_size = random.randint(min_cluster_size, max_cluster_size)
        if data_count + cluster_size > node_size:
            cluster_size = node_size - data_count
        # Create a cluster with unique node labels
        cluster = []
        for i in range(1, cluster_size + 1):
            cluster.append(f"{cluster_count}_{i}")
        clusters.append(cluster)
        data_count += cluster_size
        # Stop if all nodes have been assigned to clusters
        if data_count == node_size:
            break

    # Generate edges list of tuples from clusters
    edges: list[tuple[str, str, float]] = []
    for cluster in clusters:
        if len(cluster) == 1:
            edges.append((cluster[0], cluster[0], 1.0))
        else:
            for s, t in combinations(cluster, 2):
                w = round(random.uniform(min_weight, max_weight), 3)
                edges.append((s, t, w))

    # Random remove edges
    edges = random.sample(edges, int(len(edges) * (1 - random_remove_rate)))
    # Random add edges
    labels = list(itertools.chain.from_iterable(clusters))
    for _ in range(int(len(edges) * random_add_rate)):
        s, t = random.sample(labels, 2)
        w = round(random.uniform(min_weight, max_weight) / 2, 3)
        edges.append((s, t, w))

    return edges


def write_edges(
    outfile: str | Path,
    edges: list[tuple[str, str, float]],
    /,
) -> None:
    """Write edges to file

    Parameters
    ----------
    outfile : str | Path
        Output tab-delimited edges file
    edges : list[tuple[str, str, float]]
        Edges list of tuples
    """
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(["\t".join((s, t, str(w))) for s, t, w in edges]))


def write_clusters(
    outfile: str | Path,
    clusters: list[list[str]],
    /,
) -> None:
    """Write clusters to file

    Parameters
    ----------
    outfile : str | Path
        Output tab-delimited clusters file
    clusters : list[list[str]]
        List of clusters, each cluster is a list of node labels
    """
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("\n".join(["\t".join(c) for c in clusters]))
