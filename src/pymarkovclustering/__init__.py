import numpy as _np

from pymarkovclustering import logger as _logger
from pymarkovclustering.main import easymcl, mcl
from pymarkovclustering.utils import (
    edges_to_sparse_matrix,
    extract_clusters,
    random_edges,
    write_clusters,
    write_edges,
)
from pymarkovclustering.viz import easymclviz, mclviz

__version__ = "0.1.0"
__all__ = [
    "mcl",
    "easymcl",
    "edges_to_sparse_matrix",
    "extract_clusters",
    "random_edges",
    "write_edges",
    "write_clusters",
    "mclviz",
    "easymclviz",
]

_logger.init_null_logger()

# Better matrix display in jupyter notebook for debugging
_np.set_printoptions(
    edgeitems=10,
    linewidth=500,
    formatter={"float": lambda v: "   --" if v == 0 else f"{v:.3f}"},
)
