from __future__ import annotations

import numpy as np
from database import Session
from database.datamodel.models import Assembly
from sqlalchemy import select

from preprocessing.structure import (
    build_structure,
    get_atom_coordinates,
    get_ligand_coordinates,
)


def mbs_size(assembly_id: int, radius: float) -> int | None:
    with Session() as session:
        assembly = session.execute(
            select(Assembly).where(Assembly.id == assembly_id)
        ).scalar_one()

        try:
            structure = build_structure(assembly)
        except Exception:
            return None

        model = structure[0]
        points, _ = get_atom_coordinates(model)

        for ligand_form in assembly.entry.ligand_forms:
            ligand_coordinates = get_ligand_coordinates(structure, ligand_form, 1)
            try:
                ligand_coord = ligand_coordinates[0][0]
            except IndexError:
                continue
            try:
                distances = np.linalg.norm(points - ligand_coord, axis=1)
            except ValueError:
                continue
            return int(np.sum(distances <= radius))
    return None
