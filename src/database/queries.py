from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from sqlalchemy import select

if TYPE_CHECKING:
    from preprocessing.deposit import PDBType
    from sqlalchemy.orm import Session


def query_db(
    session: Session,
    model: type[PDBType],
    identifiers: tuple[str | int, ...],
    columns: tuple[str, ...],
) -> PDBType | None:
    """Generic query function used to filter on multiple columns."""
    return session.execute(
        select(model).where(
            *[
                model.__table__.columns[column] == identifier.upper()
                if isinstance(identifier, str)
                else model.__table__.columns[column] == identifier
                for identifier, column in zip(identifiers, columns)
            ]
        )
    ).scalar_one_or_none()


def query_db_all(session: Session, model: type[PDBType]) -> Sequence[PDBType]:
    return session.execute(select(model)).scalars().all()
