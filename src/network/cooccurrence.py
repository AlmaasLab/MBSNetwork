from __future__ import annotations

import random
import re
from typing import Literal

import networkx as nx
import numpy as np
from config import config
from database import Session
from database.datamodel.models import Ligand
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from scipy.stats import false_discovery_control  # type: ignore[import-untyped]
from tqdm import tqdm

from network.network import read_network_json
from network.utils import tqdm_joblib


def compute_ligand_to_ligand_distribution(
    network: nx.Graph,
    ligands: list[str],
    rotate: bool = True,
    save: bool = True,
) -> np.ndarray:
    """Compute the distribution of metal ligands to metal ligands along edges in the MBSNetwork.

    Returns an N x N matrix of metal ligand distributions (N unique ligand types)."""
    n_ligands = len(ligands)
    array = np.zeros((n_ligands, n_ligands))
    ligand_to_idx = dict(zip(ligands, range(n_ligands)))
    for node_u, node_v in network.edges:
        # Skip if same protein.
        uniprot_u = network.nodes[node_u]["uniprot"]
        uniprot_v = network.nodes[node_v]["uniprot"]
        if uniprot_u is not None and uniprot_u == uniprot_v:
            continue

        ligand_i = network.nodes[node_u]["ligand"]
        ligand_j = network.nodes[node_v]["ligand"]
        i = ligand_to_idx[ligand_i]
        j = ligand_to_idx[ligand_j]
        array[i, j] += 1
        if i != j:
            array[j, i] += 1

    if rotate:
        array = np.rot90(array, k=1)

    if save:
        path = config.directory.networks / f"{name}/ligand_to_ligand_distribution.npz"
        np.savez(path, ligands=ligands, distribution=array)

    return array


def shuffle_and_compute_ligand_distribution(
    args: tuple[nx.Graph, list[str], list[str]],
):
    network, ligand_array, ligand_pdb_ids = args
    random.shuffle(ligand_array)
    node_attributes = {
        node: {"ligand": ligand} for node, ligand in zip(network.nodes, ligand_array)
    }
    nx.set_node_attributes(network, node_attributes)
    return compute_ligand_to_ligand_distribution(network, ligand_pdb_ids, save=False)


def compute_ligand_zscores(
    network: nx.Graph, ligand_pdb_ids: list[str], shuffles: int
) -> None:
    """Compute Z-scores of each metal ligand based on the distribution of the neighboring
    ligands in the MBSNetwork and that of the MBSNetwork with randomized metal ligand tags."""
    # Set of all node attributes
    ligand_array = [attr["ligand"] for _, attr in network.nodes(data=True)]
    ligand_pdb_ids.sort()

    # Run shuffles in parallel
    with tqdm_joblib(tqdm(desc="", total=shuffles)) as _:  # type: ignore
        arrays = Parallel(n_jobs=-2)(
            delayed(shuffle_and_compute_ligand_distribution)(
                (network, ligand_array, ligand_pdb_ids)
            )
            for _ in range(shuffles)
        )

    samples = np.dstack(arrays)  # type: ignore

    # Calculate mean, std, and Z-scores
    means = np.mean(samples, axis=2)
    stds = np.std(samples, axis=2)
    x = compute_ligand_to_ligand_distribution(network, ligand_pdb_ids, save=False)
    z_scores = (x - means) / stds

    p_adjusted = bh_correct_symmetric_permutation_pvalues(x, samples)

    # Save Z-scores and adjusted p-values.
    path = config.directory.networks / f"{name}/ligand_zscores.npz"
    np.savez(path, ligands=ligand_pdb_ids, z_scores=z_scores, p_adjusted=p_adjusted)


def build_ec_number_array(
    network: nx.Graph, digit: Literal[2, 3]
) -> tuple[list[str], list[str]]:
    # Group EC numbers using classification digit
    ec_array: list[str] = []  # list of generalized EC numbers
    if digit == 2:
        pattern = r"^[0-7]\.[0-9]*"
    elif digit == 3:
        pattern = r"^[0-7]\.[0-9]*\.[0-9]*"
    for node, attr in network.nodes(data=True):
        ec = attr["ec"]
        try:
            ec_group = re.match(pattern, ec).group(0)  # type: ignore[union-attr]
            ec_array.append(ec_group)
            network.nodes[node]["ec"] = ec_group
        except (AttributeError, TypeError) as _:
            ec_array.append("")
            network.nodes[node]["ec"] = ""

    # Set of all EC number groups (defined by digit)
    ec_numbers = list(set(ec_array))
    ec_numbers.sort()

    return ec_numbers, ec_array


def compute_ec_zscore(network: nx.Graph, digit: Literal[2, 3] = 2, shuffles: int = 100):
    """
    Compute Z-scores of each EC number based on the distribution of the neighboring EC numbers
    in the MBSNetwork and that of the MBSNetwork with randomized EC number tags.

    EC numbers are grouped based on the second (digit = 2, default) or
    third classification digit (digit = 3).
    """
    ec_numbers, ec_array = build_ec_number_array(network, digit)

    with tqdm_joblib(tqdm(desc="", total=shuffles)) as _:
        arrays = Parallel(n_jobs=-2)(
            delayed(compute_ec_to_ec_distribution)(
                network, ec_numbers, ec_array[:], digit=digit, save=False
            )
            for _ in range(shuffles)
        )

    samples = np.dstack(arrays)  # type: ignore

    # Compute mean, std, and Z-scores.
    means = np.mean(samples, axis=2)
    stds = np.std(samples, axis=2)
    x = compute_ec_to_ec_distribution(network, ec_numbers, digit=digit, save=False)
    z_scores = (x - means) / stds

    # Replace NaN with most positive or negative Z-score.
    most_positive = np.nanmax(z_scores)
    most_negative = np.nanmin(z_scores)
    nan_indices = np.isnan(z_scores)
    nan_indices = np.isnan(z_scores)
    z_scores[nan_indices & (x > 0)] = most_positive
    z_scores[nan_indices & (x == 0)] = most_negative

    # Adjusted p-values.
    p_adjusted = bh_correct_symmetric_permutation_pvalues(x, samples)

    path = config.directory.networks / f"{name}/ec_zscores_{digit}.npz"
    np.savez(path, ec_numbers=ec_numbers, z_scores=z_scores, p_adjusted=p_adjusted)


def compute_ec_to_ec_distribution(
    network: nx.Graph,
    ec_numbers: list[str],
    ec_array: list[str] | None = None,
    digit: Literal[2, 3] = 2,
    rotate: bool = True,
    save: bool = True,
):
    """Compute the distribution of grouped EC numbers along edges in the MBSNetwork."""
    # Shuffle EC numbers
    if ec_array:
        random.shuffle(ec_array)
        ec_map = dict(zip(network.nodes, ec_array))
    else:
        ec_map = {n: network.nodes[n]["ec"] for n in network.nodes}

    # Compute EC-to-EC distribution
    n_ec = len(ec_numbers)
    array = np.zeros((n_ec, n_ec))
    ec_to_idx = dict(zip(ec_numbers, range(n_ec)))
    for node_u, node_v in network.edges:
        # Skip if same protein.
        uniprot_u = network.nodes[node_u]["uniprot"]
        uniprot_v = network.nodes[node_v]["uniprot"]
        if uniprot_u is not None and uniprot_u == uniprot_v:
            continue

        i = ec_to_idx[ec_map[node_u]]
        j = ec_to_idx[ec_map[node_v]]
        array[i, j] += 1
        if i != j:
            array[j, i] += 1

    if rotate:
        array = np.rot90(array, k=1)

    if save:
        path = config.directory.networks / f"{name}/ec_to_ec_distribution_{digit}.npz"
        np.savez(path, ec_numbers=ec_numbers, distribution=array)

    return array


def bh_correct_symmetric_permutation_pvalues(
    observed: np.ndarray,
    null_samples: np.ndarray,
) -> np.ndarray:
    n = observed.shape[0]

    # Upper-left triangle including anti-diagonal: i + j <= n - 1.
    mask = np.add.outer(np.arange(n), np.arange(n)) <= (n - 1)

    # Compute empirical p-values on that triangle.
    p_values = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(n):
            if not mask[i, j]:
                continue
            obs = observed[i, j]
            null = null_samples[i, j, :]

            p_values[i, j] = (
                np.sum(np.abs(null - null.mean()) >= np.abs(obs - null.mean())) + 1
            ) / (len(null) + 1)

    # BH only on unique tests (the masked triangle).
    flat_pvals = p_values[mask]
    p_adj_flat = false_discovery_control(flat_pvals, method="bh")

    # Reconstruct full matrix by anti-diagonal mirroring.
    p_adjusted = np.full((n, n), np.nan, dtype=float)
    p_adjusted[mask] = p_adj_flat

    # Fill the complement via reflection across the anti-diagonal:
    # (i, j) -> (n-1-j, n-1-i).
    ii, jj = np.where(~mask)
    p_adjusted[ii, jj] = p_adjusted[n - 1 - jj, n - 1 - ii]

    return p_adjusted


def pretty_print_matrix(
    M: np.ndarray,
    labels: list[str],
    title: str,
    mask: np.ndarray | None = None,
    fmt: str = "{:7.3g}",
) -> None:
    print("\n" + title)
    header = " " * 8 + "".join(f"{lbl:>8}" for lbl in labels)
    print(header)
    for i, row in enumerate(M):
        line = f"{labels[-i-1]:>6}  "
        for j, val in enumerate(row):
            if mask is not None and not mask[i, j]:
                line += "   ·    "
            else:
                if np.isnan(val):
                    line += "   NaN  "
                else:
                    line += fmt.format(val).rjust(8)
        print(line)


if __name__ == "__main__":
    with Session() as session:
        ligands = session.query(Ligand).all()
        ligand_pdb_ids = [ligand.pdb_id for ligand in ligands]
        ligand_pdb_ids.sort()

    name = "MBSNetwork"
    network = read_network_json(name)

    # Ligand-to-ligand distribution and Z-scores.
    compute_ligand_to_ligand_distribution(
        network, ligand_pdb_ids, rotate=False, save=True
    )
    compute_ligand_zscores(network, ligand_pdb_ids, shuffles=10000)

    # EC-to-EC Z-scores (digit = 3).
    digit = 3
    ec_numbers, ec_array = build_ec_number_array(network, digit)
    compute_ec_to_ec_distribution(
        network, ec_numbers, ec_array, digit=digit, rotate=False, save=True
    )
    compute_ec_zscore(network, digit=digit, shuffles=10000)
