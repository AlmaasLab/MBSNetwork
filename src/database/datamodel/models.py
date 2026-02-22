from __future__ import annotations

from sqlalchemy import ARRAY, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.datamodel import Base


class EntryLigandForm(Base):
    """Association table for many-to-many relationship between Assembly and LigandForm objects."""

    __tablename__ = "entry_ligand_form"
    entry_id: Mapped[int | None] = mapped_column(
        ForeignKey("entry.id"), primary_key=True
    )
    ligand_form_id: Mapped[int | None] = mapped_column(
        ForeignKey("ligand_form.id"), primary_key=True
    )


class LigandLigandForm(Base):
    """Association table for many-to-many relationship between Ligand and LigandForm objects."""

    __tablename__ = "ligand_ligand_form"
    ligand_id: Mapped[int | None] = mapped_column(
        ForeignKey("ligand.id"), primary_key=True
    )
    ligand_form_id: Mapped[int | None] = mapped_column(
        ForeignKey("ligand_form.id"), primary_key=True
    )


class MetalBindingSite(Base):
    """Represents a metal binding site (MBS).

    A MBS is defined as the `N_ATOMS` closest protein residue atoms to the metal."""

    __tablename__ = "metal_binding_site"
    id: Mapped[int] = mapped_column(primary_key=True)
    peptide_id: Mapped[int] = mapped_column(ForeignKey("peptide.id"))
    peptide: Mapped[Peptide] = relationship(back_populates="metal_binding_sites")
    assembly_id: Mapped[int] = mapped_column(ForeignKey("assembly.id"))
    assembly: Mapped[Assembly] = relationship(back_populates="metal_binding_sites")
    ligand_form_id: Mapped[int] = mapped_column(ForeignKey("ligand_form.id"))
    ligand_form: Mapped[LigandForm] = relationship(back_populates="metal_binding_sites")
    representative_id: Mapped[int | None] = mapped_column(
        ForeignKey("metal_binding_site.id")
    )
    representative: Mapped[MetalBindingSite | None] = relationship(
        "MetalBindingSite", remote_side=[id], back_populates="constituents"
    )
    constituents: Mapped[list[MetalBindingSite] | None] = relationship(
        "MetalBindingSite", back_populates="representative"
    )
    ligand_coord: Mapped[list[float]] = mapped_column(ARRAY(Float))
    x_coords: Mapped[list[float]] = mapped_column(ARRAY(Float))
    y_coords: Mapped[list[float]] = mapped_column(ARRAY(Float))
    z_coords: Mapped[list[float]] = mapped_column(ARRAY(Float))

    def to_dict(self):
        return {
            "db_id": self.id,
            "ligand_form": {"db_id": self.ligand_form_id},
            "peptide": {"uniprot": self.peptide.uniprot},
            "representative_id": self.representative_id,
            "constituents": [site.id for site in self.constituents]
            if self.constituents
            else None,
            "x_coords": self.x_coords,
            "y_coords": self.y_coords,
            "z_coords": self.z_coords,
        }


class Assembly(Base):
    """Represents a biological assembly of a PDB entry.

    Main storage for structural information of the protein."""

    __tablename__ = "assembly"
    id: Mapped[int] = mapped_column(primary_key=True)
    assembly_id: Mapped[str]
    entry_pdb_id: Mapped[str]
    entry: Mapped[Entry] = relationship(back_populates="assemblies")
    entry_id: Mapped[int] = mapped_column(ForeignKey("entry.id"))
    cif_file: Mapped[str]
    metal_binding_sites: Mapped[list[MetalBindingSite] | None] = relationship(
        back_populates="assembly"
    )

    __table_args__ = (
        UniqueConstraint("entry_pdb_id", "assembly_id", name="unique_assembly"),
    )


class Entry(Base):
    """Represents a PDB entry."""

    __tablename__ = "entry"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str | None]
    pdb_id: Mapped[str] = mapped_column(unique=True)
    assembly_ids: Mapped[list[str]] = mapped_column(ARRAY(String))
    entity_ids: Mapped[list[str]] = mapped_column(ARRAY(String))
    assemblies: Mapped[list[Assembly]] = relationship(back_populates="entry")
    peptides: Mapped[list[Peptide]] = relationship(back_populates="entry")
    ligand_forms: Mapped[list[LigandForm]] = relationship(
        secondary=EntryLigandForm.__table__, back_populates="entries"
    )
    resolution: Mapped[float]


class Peptide(Base):
    """Represents a peptide chain.

    No structural information is stored in these entities. While one peptide (i.e. genetically encoded
    protein) can be present in multiple assemblies in multiple PDB entries, a chain as occuring in
    a single assembly is represented by a single Peptide object.
    """

    __tablename__ = "peptide"
    id: Mapped[int] = mapped_column(primary_key=True)
    entity_id: Mapped[str]
    chain_ids: Mapped[list[str]] = mapped_column(ARRAY(String))
    name: Mapped[str]
    uniprot: Mapped[str | None]
    ec: Mapped[str | None]
    species: Mapped[str | None]
    mutation: Mapped[str | None]
    sequence: Mapped[str]
    entry_pdb_id: Mapped[str]
    entry_id: Mapped[int] = mapped_column(ForeignKey("entry.id"))
    entry: Mapped[Entry] = relationship(back_populates="peptides")
    metal_binding_sites: Mapped[list[MetalBindingSite] | None] = relationship(
        back_populates="peptide"
    )

    __table_args__ = (
        UniqueConstraint("entry_pdb_id", "entity_id", name="unique_peptide"),
    )


class Ligand(Base):
    """Represents a metal ligand."""

    __tablename__ = "ligand"
    id: Mapped[int] = mapped_column(primary_key=True)
    pdb_id: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    ligand_forms: Mapped[list[LigandForm]] = relationship(
        secondary=LigandLigandForm.__table__, back_populates="ligands"
    )


class LigandForm(Base):
    """Represents a metal ligand form.

    E.g., metal-containing cofactors, prosthetic groups."""

    __tablename__ = "ligand_form"
    id: Mapped[int] = mapped_column(primary_key=True)
    pdb_id: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str]
    ligands: Mapped[list[Ligand]] = relationship(secondary=LigandLigandForm.__table__)
    entries: Mapped[list[Entry]] = relationship(
        secondary=EntryLigandForm.__table__, back_populates="ligand_forms"
    )
    metal_binding_sites: Mapped[list[MetalBindingSite] | None] = relationship(
        back_populates="ligand_form"
    )
