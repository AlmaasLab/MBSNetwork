from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import aiofiles
import aiohttp
from config import config

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
SEMAPHORE = asyncio.Semaphore(20)


async def download_cif(entry_id: str, assembly_id: str) -> Path | None:
    """Download assembly CIF file from RCSB PDB."""
    url = f"https://files.rcsb.org/download/{entry_id.lower()}-assembly{assembly_id}.cif.gz"
    path = config.directory.structures / f"{entry_id}_assembly{assembly_id}.cif.gz"
    if path.exists():
        return path

    async with SEMAPHORE:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()

                    async with aiofiles.open(path, "wb") as f:
                        await f.write(await response.read())
                    return path
        except aiohttp.ClientResponseError as e:
            logger.error(
                f"Failed to download CIF file for entry {entry_id} assembly {assembly_id} ({e})."
            )
            return None


def get_species(peptide_json: dict[str, Any]) -> str:
    """Helper function to extract the species name of the organism encoding a peptide chain.

    Preferentially uses the scientific name, otherwise uses the common name.
    """

    def get_nested_value(d: dict[str, Any], keys: list[str | int]) -> Any:
        for key in keys:
            if isinstance(d, dict) and isinstance(key, str):
                d = d.get(key)  # type: ignore[assignment]
            elif isinstance(d, list) and isinstance(key, int):
                d = d[key]
            else:
                return None
            if d is None:
                return None
        return d

    species: str | None = get_nested_value(
        peptide_json, ["rcsb_entity_source_organism", 0, "scientific_name"]
    )

    if species is None:
        species = get_nested_value(
            peptide_json,
            ["rcsb_entity_source_organism", 0, "ncbi_scientific_name"],
        )

    if species is None:
        species = get_nested_value(
            peptide_json,
            ["entity_src_gen", 0, "pdbx_gene_src_scientific_name"],
        )

    if species is None:
        species = get_nested_value(
            peptide_json,
            ["rcsb_entity_source_organism", 0, "common_name"],
        )

    if species is None:
        species = get_nested_value(
            peptide_json,
            ["entity_src_gen", 0, "gene_src_common_name"],
        )

    return species or "Unknown species"
