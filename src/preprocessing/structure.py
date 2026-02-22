from __future__ import annotations

import copy
import gzip
import shutil
import tempfile
from collections import Counter

import numpy as np
from Bio.PDB import FastMMCIFParser, Structure
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from config import config
from database.datamodel.models import Assembly, LigandForm, MetalBindingSite


def build_structure(assembly: Assembly) -> Structure.Structure:
    """Builds a structure from a biological assembly.

    Removes all heteroatoms except those belonging to metal ligand.
    Also removes any non-protein polymers (e.g., DNA/RNA).
    """
    parser = FastMMCIFParser(QUIET=True)
    cif_file = config.directory.structures / assembly.cif_file

    with tempfile.NamedTemporaryFile() as temp_file:
        with gzip.open(cif_file) as zip_file:
            shutil.copyfileobj(zip_file, temp_file)  # type: ignore[misc]
        structure = parser.get_structure(assembly.entry_pdb_id, temp_file.name)

        mmcif_dict = MMCIF2Dict(temp_file.name)

    non_protein_chains = get_non_protein_chains(mmcif_dict)
    ligand_ids = {ligand_form.pdb_id for ligand_form in assembly.entry.ligand_forms}

    residues_to_remove = set()
    for residue in structure.get_residues():
        chain = residue.get_parent()
        asym_id = chain.id
        is_het = residue.id[0] != " "
        if asym_id in non_protein_chains:
            residues_to_remove.add(residue.id)
        elif is_het and residue.get_resname() not in ligand_ids:
            residues_to_remove.add(residue.id)

    remove_residues(structure, residues_to_remove)
    return structure


def get_non_protein_chains(mmcif_dict: MMCIF2Dict) -> set[str]:
    """Returns the set of chain IDs (asym_ids) that are not protein chains
    (i.e., not 'polypeptide(L)') based on mmCIF entity and polymer type mapping."""

    def as_list(val):
        return val if isinstance(val, list) else [val]

    entity_poly_types = as_list(mmcif_dict.get("_entity_poly.type", []))
    entity_poly_ids = as_list(mmcif_dict.get("_entity_poly.entity_id", []))
    asym_entity_map = dict(
        zip(
            as_list(mmcif_dict.get("_struct_asym.id", [])),
            as_list(mmcif_dict.get("_struct_asym.entity_id", [])),
        )
    )

    entity_type_map = dict(zip(entity_poly_ids, entity_poly_types))

    # Any chain whose entity type isn't a protein gets flagged
    non_protein_chains = {
        asym_id
        for asym_id, entity_id in asym_entity_map.items()
        if entity_type_map.get(entity_id) != "polypeptide(L)"
    }

    return non_protein_chains


def remove_residues(
    structure: Structure.Structure, residues: set[tuple[str, int, str]]
) -> None:
    for model in structure:
        for chain in model:
            for residue_id in residues:
                if residue_id in chain.child_dict:
                    chain.detach_child(residue_id)


def get_atom_coordinates(model: Structure.Model):
    """Return the coordinates of all atoms in a protein structure model.

    Also returns the corresponding chain IDs of the atoms.
    """
    points = []
    chain_ids: list[str] = []
    for chain in model:
        for residue in chain:
            for atom in residue:
                # Skip non-heteroatom H-atoms
                if not atom.element == "H":
                    points.append(atom.get_coord())
                    chain_ids.append(chain.get_full_id()[2])
    return np.asarray(points), chain_ids


def get_mbs(
    assembly: Assembly,
    ligand_form: LigandForm,
    model: Structure.Model,
    ligand_coordinates: np.ndarray,
    n_atoms: int,
) -> MetalBindingSite:
    """Construct an MBS object associated with the target metal coordinates."""
    # Get MBS point cloud
    points, chain_ids = get_atom_coordinates(model)
    distances = np.linalg.norm(points - ligand_coordinates, axis=1)
    indices = np.argpartition(distances, n_atoms)[:n_atoms]
    mbs_points = points[indices]
    dominant_chain_id: str = Counter([chain_ids[i] for i in indices]).most_common(1)[0][
        0
    ]

    # Associate MBS with a single peptide chain; needed for downstream analysis of
    # MBS similarity and protein sequence alignment.
    found_peptide = False
    for peptide in assembly.entry.peptides:
        if (
            dominant_chain_id.split("-")[0] in peptide.chain_ids
        ):  # some PDB entries have repeated chain IDs with numerical suffixes (e.g. 'A-1')
            found_peptide = True
            break

    if not found_peptide:
        raise ValueError(
            f"Unable to find peptide with chain ID {dominant_chain_id} in assembly {assembly.assembly_id}."
        )

    return MetalBindingSite(
        peptide=peptide,
        assembly=assembly,
        ligand_form=ligand_form,
        ligand_coord=ligand_coordinates,
        x_coords=mbs_points[:, 0],
        y_coords=mbs_points[:, 1],
        z_coords=mbs_points[:, 2],
    )


def get_ligand_coordinates(
    structure: Structure.Structure, ligand_form: LigandForm, n_models: int
):
    """Also returns a set of residues to remove from the structure."""
    n_residues = len(list(structure.get_residues()))
    ligand_coords: list[  # unique ligand entity row-wise, model column-wise
        list[np.ndarray]
    ] = [[0] * n_models for _ in range(n_residues)]  # type: ignore[list-item]
    residues_to_remove = set()

    for model_idx, model in enumerate(structure):
        ligand_idx = 0
        for chain in model:
            for residue in chain:
                for atom in residue:
                    element = (getattr(atom, "element", "") or "").strip().upper()
                    name = atom.get_name().strip().upper()

                    # Store coordinate of metal
                    metal_ids = {ligand.pdb_id for ligand in ligand_form.ligands}
                    if element in metal_ids or name in metal_ids:
                        ligand_coords[ligand_idx][model_idx] = atom.get_coord()
                        ligand_idx += 1
                        residues_to_remove.add(residue.id)

                # Remove non-target ligands
                if residue.id[0] != " ":
                    residues_to_remove.add(residue.id)

    ligand_coords = ligand_coords[:ligand_idx]
    return ligand_coords, residues_to_remove


def build_mbss(
    assembly: Assembly,
    structure: Structure.Structure,
    ligand_form: LigandForm,
    n_atoms: int,
) -> list[MetalBindingSite]:
    """Builds MBSs from the structure of an assembly and ligand form."""
    structure_copy = copy.deepcopy(structure)
    n_models = len(list(structure_copy.get_models()))
    ligand_coords, residues_to_remove = get_ligand_coordinates(
        structure_copy, ligand_form, n_models
    )

    remove_residues(structure_copy, residues_to_remove)

    mbss: list[MetalBindingSite] = []
    for ligand_row in ligand_coords:
        selected_model_idx = 0
        model_mbss = [
            get_mbs(assembly, ligand_form, model, coordinates, n_atoms)
            for model, coordinates in zip(structure_copy, ligand_row)
        ]
        mbss.append(model_mbss[selected_model_idx])

    return mbss
