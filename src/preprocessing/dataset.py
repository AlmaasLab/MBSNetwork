from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Sequence

import orjson
from database import Session as _Session
from database.datamodel.models import Entry, MetalBindingSite, Peptide
from sqlalchemy import select
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from preprocessing.api import api_request
from preprocessing.structure import build_mbss, build_structure

if TYPE_CHECKING:
    from pathlib import Path

    from config import Config, DatasetConfig
    from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


def process_entry(entry: Entry, config: DatasetConfig):
    """Reconstruct MBSs for the given entry and add them to the database."""
    entry_mbss: list[MetalBindingSite] = []
    with _Session() as session:
        session.add(entry)

        # Filter out entries with resolution below threshold
        if entry.resolution > config.resolution:
            return

        for assembly in entry.assemblies:
            assembly_mbss: list[MetalBindingSite] = []
            if not assembly.metal_binding_sites:
                # Drop assembly if unable to parse structure
                try:
                    structure = build_structure(assembly)
                except Exception as e:
                    logger.error(
                        f"Error building structure for PDB entry {assembly.entry_pdb_id}, assembly {assembly.assembly_id}: {e}"
                    )
                    continue

                # Skip assembly if the number of atoms is less than the threshold
                n_atoms = len(list(structure.get_atoms()))
                if n_atoms < config.atoms:
                    continue

                for ligand_form in entry.ligand_forms:
                    try:
                        new_mbss = build_mbss(
                            assembly, structure, ligand_form, config.atoms
                        )
                        assembly_mbss += new_mbss
                    except Exception as e:
                        logger.error(
                            f"Error building MBSs for assembly {assembly.assembly_id} with ligand form {ligand_form}: {e}"
                        )
                        continue
                assembly.metal_binding_sites = assembly_mbss
                session.add(assembly)
            else:
                assembly_mbss = assembly.metal_binding_sites
            entry_mbss += assembly_mbss
        session.commit()


class Dataset:
    """Helper class to load and preprocess the MBS dataset."""

    def __init__(self, config: Config):
        self.config = config
        self.mbss: Sequence[MetalBindingSite] = []

    def populate(self, session: Session, entries: Sequence[Entry] | None) -> None:
        """Populate the dataset of MBSs.

        If `entries` is provided, the MBSs will be reconstructed for each entry only
        if no MBSs are present in the database. Otherwise, the existing MBSs will
        be used.
        """
        existing_mbss = session.execute(select(MetalBindingSite)).scalars().all()
        if existing_mbss:
            if entries:
                raise ValueError(
                    f"{len(existing_mbss)} MBSs already exist in the database. Wipe database if you want to repopulate."
                )

            self.mbss = existing_mbss
            return

        assert entries is not None
        with ProcessPoolExecutor(max_workers=self.config.alignment.n_jobs) as executor:
            [
                executor.submit(process_entry, entry, self.config.dataset)
                for entry in entries
            ]
        self.mbss = session.execute(select(MetalBindingSite)).scalars().all()

    def serialize(self, path: Path) -> None:
        """Serialize the MBS dataset to a JSON file."""
        data = {"mbss": [mbs.to_dict() for mbs in self.mbss]}
        with path.open("w") as f:
            json.dump(data, f, indent=4)

    def add_representatives(self, session: Session, path: Path) -> None:
        """Add representative/constituent attributes to the MBSs.

        The 'path' points to the directory where the filtered dataset is stored, containing
        the representative/constituent information.
        """
        if not self.mbss:
            self.populate(session, None)

        # Ensure that the MBSs have not already been filtered
        try:
            self.get_representatives(session)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Representatives/constituents already exist in the dataset"
            )

        # Load the filtered dataset
        with path.open("rb") as f:
            json_bytes = f.read()
        filtered_dataset = orjson.loads(json_bytes)

        # Add representative/constituent information
        full_dataset_dict = {site.id: site for site in self.mbss}
        filtered_dataset_dict = {
            data["db_id"]: data for data in filtered_dataset["mbss"]
        }
        for mbs in tqdm(self.mbss):
            if mbs.id in filtered_dataset_dict:
                filtered_data = filtered_dataset_dict[mbs.id]
                if filtered_data["constituents"]:
                    constituents = [
                        full_dataset_dict[const_id]
                        for const_id in filtered_data["constituents"]
                    ]
                    mbs.constituents = constituents
                    session.commit()

    def get_representatives(self, session: Session) -> None:
        """Filtering pipeline has been implemented in the compute/ subproject. This method
        simply returns the representative MBSs, raising an error if no
        representatives/constituents are found.
        """
        if not self.mbss:
            self.populate(session, None)

        datasize = len(self.mbss)

        self.mbss = [mbs for mbs in self.mbss if mbs.representative_id is None]
        logger.info(f"Preprocessed and filtered dataset size of {len(self.mbss)} MBSs.")

        if len(self.mbss) == datasize:
            raise ValueError("No representatives/constituents found in the dataset.")

    async def add_ec_numbers(self, session: Session) -> None:
        """Add EC numbers to peptides of the representative MBSs.

        First, it queries UniProt for missing EC numbers of peptides. Second, it checks
        if other peptides in the database with the same UniProt ID have an EC number and assigns it
        to the peptides.
        """
        self.get_representatives(session)

        # Query UniProt for missing EC numbers
        uniprot_peptides: dict[str, list[Peptide]] = defaultdict(list)
        uniprot_queries: dict[str, str] = {}

        for mbs in self.mbss:
            peptide = mbs.peptide

            if peptide.ec is not None:
                continue

            if peptide.uniprot is None:
                continue

            # Only need one query per unique UniProt ID
            if peptide.uniprot not in uniprot_peptides:
                query = f"https://rest.uniprot.org/uniprotkb/{peptide.uniprot}"
                uniprot_queries[peptide.uniprot] = query

            if peptide not in uniprot_peptides[peptide.uniprot]:
                uniprot_peptides[peptide.uniprot].append(peptide)

        uniprot_ids, query_list = map(list, zip(*uniprot_queries.items()))
        responses = await tqdm_asyncio.gather(
            *[api_request(query) for query in query_list],
            desc="Querying UniProt",
            total=len(query_list),
        )

        for response, uniprot_id in zip(responses, uniprot_ids):
            if response is None:
                continue

            try:
                ec_number = ""

                # Parse EC number from "Cataytic activity" subsection
                for comment in response.get("comments", []):
                    if comment.get("commentType") == "CATALYTIC ACTIVITY":
                        ec = comment.get("reaction", {}).get("ecNumber")
                        if ec is None:
                            continue

                        if ec_number:
                            ec_number += f",{ec}"
                        else:
                            ec_number = ec

                # Parse EC number from "Protein names" subsection
                if not ec_number:
                    for ec_entry in (
                        response.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("ecNumbers", [])
                    ):
                        ec = ec_entry.get("value")
                        if ec is None:
                            continue

                        if ec_number:
                            ec_number += f",{ec}"
                        else:
                            ec_number = ec

                # Update the database
                if ec_number:
                    for peptide in uniprot_peptides[uniprot_id]:
                        peptide.ec = ec_number
                        session.add(peptide)
            except Exception as e:
                logger.error(f"Error parsing EC number for peptide {peptide.id}: {e}")
                continue

        session.commit()

        # Add EC numbers from other peptides with the same UniProt ID
        peptides = {
            peptide
            for peptide_list in uniprot_peptides.values()
            for peptide in peptide_list
        }

        for peptide in tqdm(peptides):
            if peptide.ec is None:
                if peptide.uniprot is None:
                    continue

                # Check if other peptides have an EC number
                ecs = (
                    session.execute(
                        select(Peptide.ec).where(
                            Peptide.uniprot == peptide.uniprot,
                            Peptide.ec.is_not(None),
                        )
                    )
                    .scalars()
                    .all()
                )
                if ecs:
                    peptide.ec = ",".join(set(ecs))
                    session.add(peptide)

        session.commit()
