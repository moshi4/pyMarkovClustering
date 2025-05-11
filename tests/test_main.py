import numpy as np
from scipy.sparse import csr_matrix

import pymarkovclustering as pymcl
from pymarkovclustering.main import (
    allclose,
    expand,
    inflate,
    normalize,
    prune,
    set_self_loops,
    to_undirected_sparse_matrix,
)


def test_mcl_simple_edges(simple_edges):
    """Test MCL with simple input"""
    matrix, _ = pymcl.edges_to_sparse_matrix(simple_edges)
    mcl_matrix = pymcl.mcl(matrix, inflation=1.5, max_iter=100)
    assert np.array_equal(
        mcl_matrix.toarray(),
        np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ),
    )


def test_easymcl_simple_edges(simple_edges):
    """Test easymcl with simple edges input"""
    clusters = pymcl.easymcl(simple_edges, inflation=1.5, max_iter=100)
    assert clusters == [["A", "B", "C"], ["D", "E"], ["F", "G"], ["H"], ["I"]]


def test_easymcl_simple_edges_file(simple_edges_file):
    """Test easymcl with simple edges file input"""
    clusters = pymcl.easymcl(simple_edges_file, inflation=2.0, max_iter=50, quiet=False)
    assert clusters == [["A", "B", "C"], ["D", "E"], ["F", "G"], ["H"], ["I"]]


def test_set_self_loops():
    """Test setting self-loops in matrix"""
    matrix = csr_matrix(
        np.array(
            [
                [1, 2, 0, 3],
                [4, 5, 0, 6],
                [7, 8, 0, 9],
                [0, 0, 0, 0],
            ]
        )
    )
    set_self_loops(matrix)
    assert np.array_equal(
        matrix.toarray(),
        np.array(
            [
                [7, 2, 0, 3],
                [4, 8, 0, 6],
                [7, 8, 1, 9],
                [0, 0, 0, 9],
            ]
        ),
    )


def test_normalize():
    """Test matrix normalization"""
    matrix = csr_matrix(
        np.array(
            [
                [5, 1, 1],
                [0, 1, 1],
                [0, 0, 3],
            ],
        ),
    )
    norm_matrix = normalize(matrix)
    assert np.array_equal(
        norm_matrix.toarray(),
        np.array(
            [
                [1.0, 0.5, 0.2],
                [0.0, 0.5, 0.2],
                [0.0, 0.0, 0.6],
            ],
        ),
    )


def test_expand():
    """Test matrix expansion"""
    matrix = csr_matrix(
        np.array(
            [
                [1.0, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5],
            ],
        ),
    )
    expand_matrix = expand(matrix)
    assert np.array_equal(
        expand_matrix.toarray(),
        np.array(
            [
                [1.0, 0.75, 0.25],
                [0.0, 0.25, 0.50],
                [0.0, 0.00, 0.25],
            ],
        ),
    )


def test_inflate():
    """Test matrix inflation"""
    matrix = csr_matrix(
        np.array(
            [
                [0.5, 0.5],
                [1.0, 1.0],
            ],
        ),
    )
    inflate_matrix = inflate(matrix, inflation_factor=2)
    assert np.array_equal(
        inflate_matrix.toarray(),
        np.array(
            [
                [0.2, 0.2],
                [0.8, 0.8],
            ],
        ),
    )


def test_to_undirected_sparse_matrix():
    """Test directed to undirected sparse matrix"""
    matrix = np.array(
        [
            [1, 0, 0, 2],
            [0, 0, 0, 5],
            [3, 0, 1, 0],
            [1, 2, 1, 0],
        ],
    )
    undirected_matrix = to_undirected_sparse_matrix(matrix)
    assert np.array_equal(
        undirected_matrix.toarray(),
        np.array(
            [
                # M'ij=M'ji=Max(Mij, Mji)
                [1, 0, 3, 2],
                [0, 0, 0, 5],
                [3, 0, 1, 1],
                [2, 5, 1, 0],
            ],
        ),
    )


def test_prune():
    """Test sparse matrix pruning"""
    matrix = csr_matrix(
        np.array(
            [
                [1, 0, 0.1],
                [0, 1, 0.001],
                [1, 0.00001, 1],
            ]
        )
    )
    prune_matrix = prune(matrix, threshold=0.001)
    assert np.array_equal(
        prune_matrix.toarray(),
        np.array(
            [
                [1, 0, 0.1],
                [0, 1, 0.001],  # not pruned: 0.001 threshold boundary
                [1, 0, 1],  # pruned: 0.00001 < 0.001 threshold
            ]
        ),
    )


def test_allclose():
    """Test allclose function"""
    assert allclose(
        csr_matrix(np.array([[1, 2], [3, 4]])),
        csr_matrix(np.array([[1, 2], [3, 4]])),
    )
    assert not allclose(
        csr_matrix(np.array([[1, 2], [3, 4]])),
        csr_matrix(np.array([[1, 2], [3, 5]])),
    )
