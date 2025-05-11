from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pymarkovclustering as pymcl
from pymarkovclustering.scripts import exit_handler

logger = logging.getLogger(__name__)


def main():
    """Main function called from CLI"""
    args = get_args()
    run(**args.__dict__)


@exit_handler
def run(
    edges: str | Path,
    *,
    outfile: str | Path | None,
    inflation: float = 2.0,
    max_iter: int = 100,
    quiet: bool = False,
) -> None:
    """
    Parameters
    ----------
    edges : str | Path
        Edges(source, target, weight) file
    outfile : str | Path | None
        Output cluster file
    inflation : float, optional
        Inflation factor
    max_iter : int, optional
        Max number of iteration
    quiet : bool, optional
        If True, print log on screen
    """
    clusters = pymcl.easymcl(
        edges,
        inflation=inflation,
        max_iter=max_iter,
        quiet=quiet,
    )
    if outfile is None:
        for cluster in clusters:
            print("\t".join(cluster))
    else:
        pymcl.write_clusters(outfile, clusters)
        logger.info(f"Write MCL clusters result to '{outfile}'")


def get_args() -> argparse.Namespace:
    """Get arguments

    Returns
    -------
    args : argparse.Namespace
        Argument parameters
    """
    description = "Markov Clustering in Python"
    parser = argparse.ArgumentParser(
        description=description,
        usage="pymcl [options] edges.tsv -o clusters.tsv",
        add_help=False,
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "edges",
        type=Path,
        help="Input edges(source, target, weight) tab-delimited file",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="Output tab-delimited clusters file (default: stdout)",
        default=None,
        metavar="",
    )
    default_inflation = 2.0
    parser.add_argument(
        "-I",
        "--inflation",
        type=float,
        help=f"Inflation factor (default: {default_inflation})",
        default=default_inflation,
        metavar="",
    )
    default_max_iter = 100
    parser.add_argument(
        "--max_iter",
        type=int,
        help=f"Max number of iteration (default: {default_max_iter})",
        default=default_max_iter,
        metavar="",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="No print log on screen (default: OFF)",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--version",
        version=f"v{pymcl.__version__}",
        help="Print version information",
        action="version",
    )
    parser.add_argument(
        "-h",
        "--help",
        help="Show this help message and exit",
        action="help",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
