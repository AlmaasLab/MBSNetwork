from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click
from config import config
from database import Session
from database.datamodel.models import Entry
from preprocessing.dataset import Dataset
from preprocessing.deposit import DatabaseConstructor
from sqlalchemy import select

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


@cli.group()
def data():
    """Database and dataset management commands."""
    pass


@data.command()
def populate_db():
    """Build the database with entries from the PDB."""
    logger.info("Depositing PDB entries into the database...")
    with Session() as session:
        db_constructor = DatabaseConstructor(
            session, config.dataset, config.directory.structures
        )

        async def main():
            await db_constructor.construct_dataset()
            await db_constructor.download_cif_files()

        asyncio.run(main())


@data.command()
def build_dataset():
    """Build and preprocess the MBS dataset from a populated database."""
    logger.info("Building the MBS dataset...")
    with Session() as session:
        entries = session.execute(select(Entry)).scalars().all()
        if not entries:
            logger.error(
                "No entries found in the database. Run the build-database command first."
            )
            return

        dataset = Dataset(config)
        dataset.populate(session, entries)


@data.command()
@click.option(
    "--path",
    type=Path,
    help="Path to the filtered dataset JSON.",
)
async def add_representatives(path: Path):
    """Add representative information to the database.

    Also adds EC numbers from UniProt to the MBS dataset."""
    logger.info("Adding representative information to the MBS dataset...")

    async def main():
        with Session() as session:
            dataset = Dataset(config)
            dataset.add_representatives(session, path)
            await dataset.add_ec_numbers(session)

    asyncio.run(main())


@data.command()
@click.option(
    "--target-dir",
    type=Path,
    help="Path to target directory.",
)
def store_dataset(target_dir: Path) -> None:
    """Store the MBS dataset in a serialized format."""
    logger.info(f"Storing the MBS dataset in {target_dir}...")
    with Session() as session:
        dataset = Dataset(config)
        dataset.populate(session, None)
        dataset.serialize(target_dir / "datasets/metal_binding_sites.json")


if __name__ == "__main__":
    cli()
