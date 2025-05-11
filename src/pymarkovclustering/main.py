from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import sklearn.preprocessing
from scipy.sparse import csr_matrix, sparray

from pymarkovclustering.logger import init_logger, init_null_logger, logging_timeit
from pymarkovclustering.utils import edges_to_sparse_matrix, extract_clusters

logger = logging.getLogger(__name__)


@logging_timeit(msg="Finished MCL")
def mcl(
    matrix: np.ndarray | csr_matrix,
    /,
    *,
    inflation: float = 2.0,
    max_iter: int = 100,
    quiet: bool = True,
) -> csr_matrix:
    """
    Run Markov Clustering

    Parameters
    ----------
    matrix : np.ndarray | csr_matrix
        Adjacency matrix
    inflation : float, optional
        Inflation factor
    max_iter : int, optional
        Max number of iterations
    quiet : bool, optional
        If True, pring log on screen

    Returns
    -------
    mcl_matrix : csr_matrix
        MCL result matrix

    Notes
    -----
    This function is designed to produce MCL results similar to `$ mcl abc.tsv --abc -I 2.0` command.
    See <https://micans.org/mcl/man/mcl.html> in detail.

    References
    ----------
    MCL - a cluster algorithm for graphs (<https://micans.org/mcl/>)
    """  # noqa: E501
    init_null_logger() if quiet else init_logger()

    row_num, col_num = matrix.shape
    if row_num != col_num:
        raise ValueError(f"Input matrix must be square ({row_num=}, {col_num=}).")

    matrix = to_undirected_sparse_matrix(matrix)

    logger.info(f"Starting MCL ({inflation=}, {max_iter=})")
    logger.info(f"Load {row_num}x{col_num} matrix with {len(matrix.data)} entries")
    set_self_loops(matrix)
    matrix = normalize(matrix)
    matrix = prune(matrix)
    # logger.info(f"Initial Normalization\n{matrix.toarray()}")
    for i in range(1, max_iter + 1):
        start_time = time.time()
        prev_matrix = matrix.copy()
        matrix = expand(matrix)
        matrix = inflate(matrix, inflation)
        matrix = prune(matrix)
        elapsed_time = time.time() - start_time
        logger.info(f"Iteration {i:03d} done --- {elapsed_time:.2f}[s]")
        # logger.info(f"After Expansion & Inflation\n{matrix.toarray()}")
        if allclose(matrix, prev_matrix):
            break
        if i == max_iter:
            warn = warnings.warn if quiet else logger.warning
            warn(f"MCL reached {max_iter=} and not converged yet. Consider increasing max_iter.")  # fmt: skip  # noqa: E501
    logger.info(f"{len(extract_clusters(matrix))} clusters found")
    return matrix


def easymcl(
    edges: str | Path | list[tuple[str, str, float]],
    /,
    *,
    inflation: float = 2.0,
    max_iter: int = 100,
    quiet: bool = True,
) -> list[list[str]]:
    """
    Run Markov Clustering from edges file or list of tuples

    easymcl automates load edges as matrix, MCL, extract clusters

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

    Returns
    -------
    clusters : list[list[str]]
        List of clusters, each cluster is a list of node labels

    Notes
    -----
    This function is designed to produce MCL results similar to `$ mcl abc.tsv --abc -I 2.0` command.
    See <https://micans.org/mcl/man/mcl.html> in detail.

    References
    ----------
    MCL - a cluster algorithm for graphs (<https://micans.org/mcl/>)

    Examples
    --------
    >>> import pymarkovclustering as pymcl
    >>> # easymcl automates load edges as matrix, MCL, extract clusters
    >>> clusters = pymcl.easymcl("edges.tsv")
    >>> # easymcl is same as code below
    >>> matrix, labels = pymcl.edges_to_sparse_matrix("edges.tsv")
    >>> mcl_matrix = pymcl.mcl(matrix)
    >>> clusters = pymcl.extract_clusters(mcl_matrix, labels)
    """  # noqa: E501
    matrix, labels = edges_to_sparse_matrix(edges)
    mcl_matrix = mcl(matrix, inflation=inflation, max_iter=max_iter, quiet=quiet)
    return extract_clusters(mcl_matrix, labels)


def to_undirected_sparse_matrix(matrix: np.ndarray | csr_matrix) -> csr_matrix:
    """
    Convert to undirected sparse matrix

    `A'ij = A'ji = Max(Aij, Aji)` [A = input matrix, A' = result matrix]

    Parameters
    ----------
    matrix : np.ndarray | csr_matrix
        Target matrix

    Returns
    -------
    matrix : csr_matrix
        Undirected sparse matrix
    """
    if isinstance(matrix, (np.ndarray, sparray)):
        matrix = csr_matrix(matrix)
    return matrix.maximum(matrix.T)


def set_self_loops(matrix: csr_matrix) -> None:
    """
    Set self-loops in matrix to max value of each column.
    If max column value is 0, set it to 1.

    Parameters
    ----------
    matrix : csr_matrix
        Target matrix
    """
    diag = matrix.max(axis=0).toarray().flatten()
    diag[diag == 0] = 1
    matrix.setdiag(diag)


def normalize(matrix: csr_matrix) -> csr_matrix:
    """
    Normalize matrix columns (each column sums to 1)

    Parameters
    ----------
    matrix : csr_matrix
        Target matrix

    Returns
    -------
    norm_matrix : csr_matrix
        Normalized matrix
    """
    return sklearn.preprocessing.normalize(matrix, norm="l1", axis=0)


def expand(matrix: csr_matrix) -> csr_matrix:
    """
    Expand matrix

    Parameters
    ----------
    matrix : csr_matrix
        Target matrix

    Returns
    -------
    expand_matrix : csr_matrix
        Expanded matrix
    """
    return matrix @ matrix


def inflate(matrix: csr_matrix, inflation_factor: float) -> csr_matrix:
    """
    Inflate matrix

    Parameters
    ----------
    matrix : csr_matrix
        Target matrix
    inflation_factor : float
        Inflation factor

    Returns
    -------
    inflate_matrix : csr_matrix
        Inflated matrix
    """
    return normalize(matrix.power(inflation_factor))


def prune(matrix: csr_matrix, threshold: float = 1e-4) -> csr_matrix:
    """
    Prune matrix value below threshold to 0

    Parameters
    ----------
    matrix : csr_matrix
        Target matrix
    threshold : float, optional
        Prune threshold

    Returns
    -------
    prune_matrix : csr_matrix
        Pruned matrix
    """
    matrix.data[matrix.data < threshold] = 0
    matrix.eliminate_zeros()
    return matrix


def allclose(
    m1: csr_matrix,
    m2: csr_matrix,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Returns True if two arrays are element-wise equal within a tolerance

    Parameters
    ----------
    m1, m2 : csr_matrix
        Matrix to compare
    rtol : float, optional
        Relative tolerance
    atol : float, optional
        Absolute tolerance

    Returns
    -------
    result : bool
        Check result
    """
    if m1.data.shape == m2.data.shape:
        return np.allclose(m1.data, m2.data, rtol=rtol, atol=atol)
    else:
        return False
