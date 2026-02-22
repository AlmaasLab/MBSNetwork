from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from alignment.alignment import PairwiseAligner
from config import config
from database import Session
from database.datamodel.models import MetalBindingSite
from sqlalchemy import select
from tqdm import tqdm

if TYPE_CHECKING:
    from pathlib import Path


def compute_rmsd_distribution(
    mbs_ids: list[int],
    pairs: list[tuple[int, int]],
    samples: int = 10000,
    save_path: Path | None = None,
    coarse_alignments: int = 1,
    top_ranking_fraction: float = 1.0,
) -> np.ndarray:
    """Using pairs of MBSs, calculate the distribution of RMSD values for `samples`
    repeated point cloud alignments."""
    results = np.zeros((samples, len(pairs)))

    with Session() as session:
        mbss = (
            session.execute(
                select(MetalBindingSite).where(MetalBindingSite.id.in_(mbs_ids))
            )
            .scalars()
            .all()
        )
        mbs_dict = {site.id: site for site in mbss}
        sorted_mbss = [mbs_dict[site_id] for site_id in mbs_ids]

        config.alignment.coarse_alignments = coarse_alignments
        config.alignment.top_ranking_fraction = top_ranking_fraction
        config.alignment.n_jobs = 10

        if samples == 1:
            aligner = PairwiseAligner(sorted_mbss, config.alignment, pair_indices=pairs)
            alignments = aligner.align()
            results[0, :] = alignments[:, 2]
        else:
            for i, pair in tqdm(enumerate(pairs)):
                pair_indices = [pair for _ in range(samples)]
                aligner = PairwiseAligner(
                    sorted_mbss, config.alignment, pair_indices=pair_indices
                )
                alignments = aligner.align()
                results[:, i] = alignments[:, 2]

    if save_path:
        with save_path.open("wb") as f:
            np.save(f, results)

    return results


def evaluate_pairwise_alignment(
    mbs_ids: list[int], pair_indices: list[tuple[int, int]]
):
    """Evaluate performance and time complexity of the pairwise alignment on selected
    point cloud pairs. Measure the fraction of successful alignments and runtime of the
    approach as a function of the number of coarse alignments N."""

    def sample_alignment():
        aligner = PairwiseAligner(
            sorted_mbss, config.alignment, pair_indices=pair_indices
        )
        start = time.perf_counter()
        all_alignments = aligner.align()
        runtime = time.perf_counter() - start

        # Estimate average runtime per alignment
        alignment_dict["Runtime"].append(
            runtime * config.alignment.n_jobs / len(pair_indices)
        )

        # Calculate fraction of successful alignments
        key = "Successful"
        successful_counts = {key: 0}
        total_counts = {key: 0}
        for (_, _, rmsd), (_, _, best_rmsd) in zip(all_alignments, best_alignments):
            total_counts[key] += 1
            if rmsd < max(1.2 * best_rmsd, 0.3):
                successful_counts[key] += 1

        alignment_dict[key].append(successful_counts[key] / total_counts[key])

        alignment_dict["N"].append(config.alignment.coarse_alignments)

    path = config.directory.analysis / "alignment_performance_test.csv"

    with Session() as session:
        mbss = (
            session.execute(
                select(MetalBindingSite).where(MetalBindingSite.id.in_(mbs_ids))
            )
            .scalars()
            .all()
        )
        mbs_dict = {site.id: site for site in mbss}
        sorted_mbss = [mbs_dict[site_id] for site_id in mbs_ids]

        # Find best alignment with large N.
        print("Computing best alignments for reference...")
        config.alignment.coarse_alignments = 500
        aligner = PairwiseAligner(
            sorted_mbss, config.alignment, pair_indices=pair_indices
        )
        best_alignments = aligner.align()

        # Evaluate performance of centered and non-centered pairwise alignment
        alignment_dict = {"Successful": [], "N": [], "Runtime": []}
        n_range = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
        for i, n in enumerate(n_range):
            print(f"Sample {i + 1}/{len(n_range)}")
            config.alignment.coarse_alignments = n
            sample_alignment()

    df = pd.DataFrame.from_dict(alignment_dict)
    df.to_csv(path)


def prepare_pair_indices(
    alignment_path: Path,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Prepare MBS IDs and pair indices from alignment data."""
    data = np.load(alignment_path)

    # Extract unique MBS IDs and pair indices.
    mbs_ids: list[int] = []
    pair_indices: list[tuple[int, int]] = []
    for src_id, tgt_id, _ in data:
        src_id = int(src_id)
        tgt_id = int(tgt_id)
        if src_id not in mbs_ids:
            mbs_ids.append(int(src_id))
        src_idx = mbs_ids.index(int(src_id))
        if tgt_id not in mbs_ids:
            mbs_ids.append(int(tgt_id))
        tgt_idx = mbs_ids.index(int(tgt_id))
        pair_indices.append((src_idx, tgt_idx))

    return mbs_ids, pair_indices


def compute_standard_and_heuristic_rmsd_distributions(
    mbs_ids: list[int], pair_indices: list[tuple[int, int]]
):
    """Compute RMSD distributions for standard and heuristic alignments."""
    print("Computing standard alignment RMSD distribution...")
    save_path = config.directory.analysis / "standard_alignment_0_4.npy"
    compute_rmsd_distribution(
        mbs_ids,
        pair_indices,
        samples=1,
        save_path=save_path,
        coarse_alignments=1,
        top_ranking_fraction=1.0,
    )

    print("Computing heuristic alignment RMSD distribution...")
    save_path = config.directory.analysis / "heuristic_alignment_0_4.npy"
    compute_rmsd_distribution(
        mbs_ids,
        pair_indices,
        samples=1,
        save_path=save_path,
        coarse_alignments=200,
        top_ranking_fraction=0.05,
    )


if __name__ == "__main__":
    # Subfigures 1. A and B
    alignment_path = config.directory.alignments / "preliminary/alignments_0_4.npy"
    mbs_ids, pair_indices = prepare_pair_indices(alignment_path)
    compute_standard_and_heuristic_rmsd_distributions(mbs_ids, pair_indices)

    # Subfigures 1. C and D
    evaluate_pairwise_alignment(mbs_ids, pair_indices)
