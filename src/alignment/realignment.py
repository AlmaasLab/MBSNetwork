from __future__ import annotations

from itertools import combinations
from pathlib import Path

import networkx as nx
from config import config
from database import Session
from network.network import read_network_json
from preprocessing.dataset import Dataset

from alignment.alignment import PairwiseAligner


def to_realignment(
    network: nx.Graph,
    to_threshold: float,
    coarse_alignments: int,
    save_path: Path,
    n_jobs: int,
) -> None:
    """Realign unconnected nodes with unweighted topological overlap >= threshold."""
    to_pairs = compute_high_to_pairs(network, to_threshold)

    with Session() as session:
        dataset = Dataset(config)
        dataset.get_representatives(session)

    mbs_dict = {site.id: i for i, site in enumerate(dataset.mbss)}
    pair_indices = [(mbs_dict[u], mbs_dict[v]) for u, v in to_pairs]
    alignment_config = config.alignment
    alignment_config.coarse_alignments = coarse_alignments
    alignment_config.path = save_path

    aligner = PairwiseAligner(dataset.mbss, alignment_config, pair_indices=pair_indices)
    aligner.align(n_jobs=n_jobs, save_flag=True)


def compute_high_to_pairs(G: nx.Graph, threshold: float) -> list[tuple[int, int]]:
    high_to_pairs: list[tuple[int, int]] = []

    for component in nx.connected_components(G):
        if len(component) < 3:
            continue

        nodes = list(component)
        for u, v in combinations(nodes, 2):
            if G.has_edge(u, v):
                continue

            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))

            intersection_size = len(neighbors_u & neighbors_v)
            to_score = (
                intersection_size
                / (  # ignore delta_uv (as u and v are not connected)
                    min(len(neighbors_u), len(neighbors_v)) + 1
                )
            )

            if to_score >= threshold:
                high_to_pairs.append((int(u), int(v)))

    return high_to_pairs


if __name__ == "__main__":
    network = read_network_json("MBSNetwork")
    to_threshold = 0.5
    to_realignment(
        network,
        to_threshold=to_threshold,
        coarse_alignments=300,
        save_path=Path("/MBSNetwork/data/alignments/to_realignment"),
        n_jobs=-3,
    )
