from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np
from config import config
from database import Session
from database.datamodel.models import MetalBindingSite
from preprocessing.dataset import Dataset
from sqlalchemy import select
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dataset_summary():
    """Calculate dataset properties.

    Produces data for Table {dataset_summary} in the manuscript.
    """
    with Session() as session:
        # Get original and filtered MBS datasets
        original_mbss = session.execute(select(MetalBindingSite)).scalars().all()
        dataset = Dataset(config)
        dataset.get_representatives(session)
        final_mbss = dataset.mbss

        # Helper function to extract unique entries
        def extract_unique_entries(sites: Sequence[MetalBindingSite], attribute: str):
            return set(getattr(site.assembly, attribute) for site in sites)

        # Number of protein structures (i.e. PDB entries)
        original_structures = extract_unique_entries(original_mbss, "entry")
        final_structures = extract_unique_entries(final_mbss, "entry")

        # Number of peptides
        original_peptides = set(site.peptide.uniprot for site in original_mbss)
        final_peptides = set(site.peptide.uniprot for site in final_mbss)

        # Number of MBSs with/without EC numbers
        original_ec = sum(site.peptide.ec is not None for site in original_mbss)
        final_ec = sum(site.peptide.ec is not None for site in final_mbss)

        # Print as Latex table
        print(f"MBSs & {len(original_mbss)} & {len(final_mbss)} \\\\")
        print(
            f"PDB entries & {len(original_structures)} & {len(final_structures)} \\\\"
        )
        print(f"Proteins & {len(original_peptides)} & {len(final_peptides)} \\\\")
        print(f"MBSs with EC & {original_ec} & {final_ec} \\\\")


def ec_distribution(dataset_type: Literal["full", "filtered"]):
    """Calculate relative EC class distribution (EC 1–7) for full vs filtered MBS datasets."""
    logger.info(f"Calculating EC distribution of the {dataset_type} MBS dataset...")

    path = config.directory.analysis / f"ec_distribution_{dataset_type}.npz"
    ec_counts: dict[int, int] = {k: 0 for k in range(1, 8)}

    def get_ec_class(ec: str) -> int | None:
        """Return the EC first-level class (1..7) from an EC string like '3.4.21.4'."""
        try:
            cls = int(ec.split(".", 1)[0])
            return cls if 1 <= cls <= 7 else None
        except Exception:
            return None

    with Session() as session:
        if dataset_type == "filtered":
            dataset = Dataset(config)
            dataset.get_representatives(session)
            mbss = dataset.mbss
        elif dataset_type == "full":
            mbss = session.execute(select(MetalBindingSite)).scalars().all()
        else:
            raise ValueError("Invalid dataset type. Expected 'full' or 'filtered'.")

        for site in tqdm(mbss):
            ec_str = getattr(site.peptide, "ec", None)
            if not ec_str:
                continue

            # Avoid double-counting duplicates in comma lists.
            for ec in set(part.strip() for part in ec_str.split(",") if part.strip()):
                ec_class = get_ec_class(ec)
                if ec_class is not None:
                    ec_counts[ec_class] += 1

    classes = np.arange(1, 8, dtype=int)
    counts = np.asarray([ec_counts[k] for k in classes], dtype=float)
    dist = counts / counts.sum() if counts.sum() > 0 else counts

    np.savez(path, classes=classes, dataset=dist)


def ligand_distribution(dataset_type: Literal["full", "filtered"]):
    """Calculate relative ligand distribution of the MBS dataset."""
    logger.info(f"Calculating ligand distribution of the {dataset_type} MBS dataset...")

    path = config.directory.analysis / f"ligand_distribution_{dataset_type}.npz"
    ligand_dict: dict[str, int] = {}

    with Session() as session:
        if dataset_type == "filtered":
            dataset = Dataset(config)
            dataset.get_representatives(session)
            mbss = dataset.mbss
        elif dataset_type == "full":
            mbss = session.execute(select(MetalBindingSite)).scalars().all()
        else:
            raise ValueError("Invalid dataset type.")

        for site in tqdm(mbss):
            for ligand in site.ligand_form.ligands:
                ligand_dict[ligand.pdb_id] = ligand_dict.get(ligand.pdb_id, 0) + 1

    dataset_ligands = [key.capitalize() for key in ligand_dict.keys()]
    dataset_distribution = np.asarray(list(ligand_dict.values()))
    dist = dataset_distribution / dataset_distribution.sum()

    np.savez(path, ligands=dataset_ligands, dataset=dist)


if __name__ == "__main__":
    dataset_summary()
    ligand_distribution("full")
    ligand_distribution("filtered")
    ec_distribution("full")
    ec_distribution("filtered")
