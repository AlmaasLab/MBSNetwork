from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


from .models import Assembly, Entry, Ligand, LigandForm, Peptide
