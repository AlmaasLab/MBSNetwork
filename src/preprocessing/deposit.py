from __future__ import annotations

import asyncio
import logging
import statistics
from typing import TYPE_CHECKING, Any, Iterable, TypeVar

from database.datamodel import Entry, Ligand, LigandForm
from database.datamodel.models import Assembly, Peptide
from database.queries import query_db, query_db_all
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from preprocessing.api import ENTRY_QUERY, LIGAND_QUERY, api_request, get_pdb_api
from preprocessing.util import download_cif, get_species

if TYPE_CHECKING:
    from pathlib import Path

    from config import DatasetConfig
    from sqlalchemy.orm import Session as _Session


logger = logging.getLogger(__name__)


LigandType = TypeVar("LigandType", Ligand, LigandForm)
PDBType = TypeVar("PDBType", Ligand, LigandForm, Entry, Assembly, Peptide)


class DatabaseConstructor:
    """Orchestrating constructor class for the database.

    Queries the PDB API for all protein structures containing any of the specified metal ligands
    and populates the database.
    """

    def __init__(self, session: _Session, config: DatasetConfig, structures_dir: Path):
        self.session = session
        self.config = config
        self.structures_dir = structures_dir

    async def construct_dataset(self) -> None:
        await self.retrieve_ligands()
        await self.retrieve_pdb_entries()

    async def retrieve_ligands(self) -> None:
        """Download and parse all metal ligands from PDB and store in database."""

        logger.info(
            f"Retrieving the following ligands: {', '.join(self.config.ligands)}..."
        )
        existing_ligands, lidang_json_list = await self._get_objs_or_json_data(
            [(l,) for l in self.config.ligands], Ligand, ("pdb_id",)
        )
        for ligand_id, ligand, ligand_json in zip(
            self.config.ligands, existing_ligands, lidang_json_list
        ):
            if ligand is None:
                if ligand_json is None:
                    logger.info(f"Failed to retrieve ligand {ligand_id} from PDB")
                    continue
                elif ligand_json.get("status") == 204:
                    logger.info(f"No ligand found for {ligand_id}")
                    continue
                ligand_obj = Ligand(
                    pdb_id=ligand_json["chem_comp"]["id"],
                    name=ligand_json["chem_comp"]["name"],
                )
                self.session.add(ligand_obj)
                self.session.commit()
            else:
                ligand_obj = ligand

            # Retrieve all associated ligand forms
            result = await api_request(LIGAND_QUERY.replace("<Ligand>", ligand_id))
            assert result is not None
            ligand_form_ids = [entry["identifier"] for entry in result["result_set"]]
            (
                _,
                ligand_form_json_list,
            ) = await self._get_objs_or_json_data(
                [(lf,) for lf in ligand_form_ids], LigandForm, ("pdb_id",)
            )
            ligand_forms: list[LigandForm] = []
            for ligand_form_id, ligand_form_json in zip(
                ligand_form_ids, ligand_form_json_list
            ):
                ligand_form = query_db(
                    self.session, LigandForm, (ligand_form_id,), ("pdb_id",)
                )
                if ligand_form is None:
                    if ligand_form_json is None:
                        logger.info(
                            f"Failed to retrieve ligand form {ligand_form_id} from PDB"
                        )
                        continue
                    elif ligand_form_json.get("status") == 204:
                        logger.info(f"No ligand form found for {ligand_form_id}")
                        continue
                    ligand_form_obj = LigandForm(
                        pdb_id=ligand_form_json["chem_comp"]["id"],
                        name=ligand_form_json["chem_comp"]["name"],
                    )
                    self.session.add(ligand_form_obj)
                    self.session.commit()
                else:
                    ligand_form_obj = ligand_form

                ligand_forms.append(ligand_form_obj)

            ligand_obj.ligand_forms = ligand_forms  # type: ignore[union-attr]
            self.session.commit()

    async def retrieve_pdb_entries(self) -> None:
        """Retrieve all PDB structure entries that binds any of the ligand forms in the database.

        Also retrieves and stores the associated assemblies and peptides."""

        logger.info("Retrieving entries...")
        ligand_forms = query_db_all(self.session, LigandForm)
        queries = [
            ENTRY_QUERY.replace("<Ligand>", form.pdb_id) for form in ligand_forms
        ]
        results = await asyncio.gather(*[api_request(query) for query in queries])

        # Concatenate all entry IDs associated with every ligand form
        all_ligand_form_entry_ids: list[tuple[LigandForm, str]] = []
        for ligand_form, ligand_form_json in zip(ligand_forms, results):
            if ligand_form_json.get("status") == 204:
                logger.info(
                    f"No entries found for ligand form {ligand_form.pdb_id}. Deleting it from the database..."
                )
                self.session.delete(ligand_form)
                self.session.commit()
                continue
            entry_ids: list[str] = [
                entry["identifier"] for entry in ligand_form_json["result_set"]
            ]
            all_ligand_form_entry_ids.extend(
                [(ligand_form, entry_id) for entry_id in entry_ids]
            )

        logger.info(f"Retrieving {len(all_ligand_form_entry_ids)} entries...")
        _, entry_json_list = await self._get_objs_or_json_data(
            [(entry_id,) for _, entry_id in all_ligand_form_entry_ids], Entry
        )
        logger.info("Processing...")

        # Add non-existing entries to the database (record associated assemblies and peptides)
        all_entry_assembly_ids: list[tuple[Entry, str]] = []
        all_entry_peptide_ids: list[tuple[Entry, str]] = []
        for (ligand_form, entry_id), entry_json in tqdm(
            zip(all_ligand_form_entry_ids, entry_json_list),
            total=len(all_ligand_form_entry_ids),
        ):
            entry = query_db(self.session, Entry, (entry_id,), ("pdb_id",))

            if entry is None:
                if entry_json is None:
                    continue
                elif entry_json.get("status") == 204:
                    continue
                elif (
                    entry_json["rcsb_entry_info"].get("polymer_entity_count_protein")
                    == 0
                ):
                    continue

                # Keep only crystallographic entries.
                exptl = entry_json.get("exptl") or []
                methods = {
                    m.get("method", "").strip().lower()
                    for m in exptl
                    if isinstance(m, dict)
                }

                if not any(
                    ("x-ray" in m) or ("electron diffraction" in m) for m in methods
                ):
                    continue

                resolution = entry_json["rcsb_entry_info"].get(
                    "resolution_combined", 0.0
                )
                if isinstance(resolution, Iterable):
                    resolution = statistics.mean(resolution)

                entry_obj = Entry(
                    name=entry_json.get("struct", {}).get("pdbx_descriptor"),
                    pdb_id=entry_id,
                    assembly_ids=entry_json["rcsb_entry_container_identifiers"][
                        "assembly_ids"
                    ],
                    entity_ids=entry_json["rcsb_entry_container_identifiers"][
                        "polymer_entity_ids"
                    ],
                    resolution=resolution,
                )
                self.session.add(entry_obj)
                self.session.commit()
            else:
                entry_obj = entry

            if ligand_form not in entry_obj.ligand_forms:
                entry_obj.ligand_forms.append(ligand_form)
            self.session.commit()

            # Concatenate all assesembly and entity IDs associated with every entry
            all_entry_assembly_ids.extend(
                [(entry_obj, assembly_id) for assembly_id in entry_obj.assembly_ids]
            )
            all_entry_peptide_ids.extend(
                [(entry_obj, entity_id) for entity_id in entry_obj.entity_ids]
            )

        # Retrieve or create assembly
        logger.info(f"Retrieving {len(all_entry_assembly_ids)} assemblies...")
        logger.info("Processing...")
        for entry, assembly_id in tqdm(
            all_entry_assembly_ids,
            total=len(all_entry_assembly_ids),
        ):
            assembly = query_db(
                self.session,
                Assembly,
                (entry.pdb_id, assembly_id),
                ("entry_pdb_id", "assembly_id"),
            )
            if assembly is None:
                gzipped_cif = (
                    self.structures_dir / f"{entry.pdb_id}_assembly{assembly_id}.cif.gz"
                )
                assembly_obj = Assembly(
                    assembly_id=assembly_id,
                    entry_pdb_id=entry.pdb_id,
                    cif_file=gzipped_cif.name,
                    entry=entry,
                )
                self.session.add(assembly_obj)
                self.session.commit()
            else:
                assembly_obj = assembly
            if assembly_obj not in entry.assemblies:
                entry.assemblies.append(assembly_obj)
            self.session.commit()

        # Retrieve or create peptide
        logger.info(f"Retrieving {len(all_entry_peptide_ids)} peptides...")
        _, peptide_json_list = await self._get_objs_or_json_data(
            [(entry.pdb_id, entity_id) for entry, entity_id in all_entry_peptide_ids],
            Peptide,
            (
                "entry_pdb_id",
                "entity_id",
            ),
        )
        logger.info("Processing...")
        for (entry, entity_id), peptide_json in tqdm(
            zip(all_entry_peptide_ids, peptide_json_list),
            total=len(all_entry_peptide_ids),
        ):
            # Re-query, as peptide may have been instantiated earlier in the loop
            peptide = query_db(
                self.session,
                Peptide,
                (entry.pdb_id, entity_id),
                ("entry_pdb_id", "entity_id"),
            )
            if peptide is None:
                if peptide_json is None:
                    continue
                elif peptide_json.get("status") == 204:
                    continue
                elif (
                    peptide_json["entity_poly"]["rcsb_entity_polymer_type"] != "Protein"
                ):
                    continue
                peptide_obj = Peptide(
                    entity_id=entity_id,
                    chain_ids=peptide_json["entity_poly"]["pdbx_strand_id"].split(","),
                    name=peptide_json["rcsb_polymer_entity"]["pdbx_description"],
                    uniprot=peptide_json[
                        "rcsb_polymer_entity_container_identifiers"
                    ].get("uniprot_ids", [None])[0],
                    ec=peptide_json["rcsb_polymer_entity"].get("pdbx_ec"),
                    species=get_species(peptide_json),
                    mutation=peptide_json["rcsb_polymer_entity"].get("pdbx_mutation"),
                    sequence=peptide_json["entity_poly"][
                        "pdbx_seq_one_letter_code_can"
                    ],
                    entry_pdb_id=entry.pdb_id,
                    entry=entry,
                )
                self.session.add(peptide_obj)
                self.session.commit()
            else:
                peptide_obj = peptide

            if peptide_obj not in entry.peptides:
                entry.peptides.append(peptide_obj)
            self.session.commit()

    async def _get_objs_or_json_data(
        self,
        id_collection: list[tuple[str | int, ...]],
        model: type[PDBType],
        columns: tuple[str, ...] | None = None,
    ) -> tuple[list[PDBType | None], list[dict[str, Any] | None]]:
        """Retrieve existing objects or JSON from API requests if not in database."""

        async def fetch_with_progress(
            identifiers: tuple[str | int, ...], target: str, pbar: tqdm_asyncio
        ):
            result = await get_pdb_api(identifiers, target)
            pbar.update(1)
            return result

        # Retrieve existing objects
        existing_objs: list[PDBType | None] = [None] * len(id_collection)
        if columns is not None:
            existing_objs = [
                query_db(self.session, model, identifiers, columns)
                for identifiers in id_collection
            ]

        # Retrieve JSON data for non-existing objects
        id_collection_rest = [
            ic for ic, obj in zip(id_collection, existing_objs) if obj is None
        ]
        with tqdm_asyncio(total=len(id_collection_rest)) as pbar:
            tasks = [
                fetch_with_progress(identifiers, model.__name__.lower(), pbar)
                for identifiers in id_collection_rest
            ]
            results = await asyncio.gather(*tasks)

        # Match lists of existing objects with JSON data
        json_data: list[dict[str, Any] | None] = []
        results_iter = iter(results)
        for obj in existing_objs:
            if obj is None:
                json_data.append(next(results_iter))
            else:
                json_data.append(None)
        return existing_objs, json_data

    async def download_cif_files(self) -> None:
        """Download CIF files for all assemblies in the database."""
        entries = query_db_all(self.session, Entry)
        args = [
            (entry.pdb_id, assembly.assembly_id)
            for entry in entries
            for assembly in entry.assemblies
        ]
        logger.info(f"Downloading {len(args)} CIF files...")

        async def download_with_progress(
            entry_id: str, assembly_id: str, pbar: tqdm_asyncio
        ):
            await download_cif(entry_id, assembly_id)
            pbar.update(1)

        with tqdm_asyncio(total=len(args)) as pbar:
            tasks = [
                download_with_progress(entry_id, assembly_id, pbar)
                for entry_id, assembly_id in args
            ]
            await asyncio.gather(*tasks)
