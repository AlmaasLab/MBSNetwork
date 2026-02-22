from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiohttp import ClientResponseError, ClientSession

logger = logging.getLogger(__name__)

LIGAND_QUERY = 'https://search.rcsb.org/rcsbsearch/v2/query?json=%7B"query"%3A%7B"type"%3A"terminal"%2C"service"%3A"chemical"%2C"parameters"%3A%7B"type"%3A"descriptor"%2C"descriptor_type"%3A"SMILES"%2C"value"%3A"%5B<Ligand>%5D"%2C"match_type"%3A"sub-struct-graph-relaxed"%7D%2C"label"%3A"chemical"%7D%2C"return_type"%3A"mol_definition"%2C"request_options"%3A%7B"paginate"%3A%7B"start"%3A0%2C"rows"%3A10000%7D%2C"results_content_type"%3A%5B"experimental"%5D%2C"sort"%3A%5B%7B"sort_by"%3A"score"%2C"direction"%3A"desc"%7D%5D%2C"scoring_strategy"%3A"combined"%7D%7D'
ENTRY_QUERY = 'https://search.rcsb.org/rcsbsearch/v2/query?json={"query"%3A{"type"%3A"group"%2C"nodes"%3A[{"type"%3A"terminal"%2C"service"%3A"text"%2C"parameters"%3A{"attribute"%3A"rcsb_nonpolymer_instance_annotation.comp_id"%2C"operator"%3A"exact_match"%2C"value"%3A"<Ligand>"}}%2C{"type"%3A"terminal"%2C"service"%3A"text"%2C"parameters"%3A{"attribute"%3A"rcsb_nonpolymer_instance_annotation.type"%2C"operator"%3A"exact_match"%2C"value"%3A"HAS_NO_COVALENT_LINKAGE"}}]%2C"logical_operator"%3A"and"%2C"label"%3A"nested-attribute"}%2C"return_type"%3A"entry"%2C"request_options"%3A{"paginate"%3A{"start"%3A0%2C"rows"%3A10000}%2C"results_content_type"%3A["experimental"]%2C"sort"%3A[{"sort_by"%3A"score"%2C"direction"%3A"desc"}]%2C"scoring_strategy"%3A"combined"}}'
MAX_RETRIES = 3
COOLDOWN_PERIOD = 20
BACKOFF_FACTOR = 2
SEMAPHORE = asyncio.Semaphore(20)  # fine-tune (no issues with 5)


async def api_request(query: str) -> dict[str, Any] | None:
    async with SEMAPHORE:
        async with ClientSession() as session:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    async with session.get(query) as response:
                        if response.status == 204:
                            return {"status": 204}
                        response.raise_for_status()
                        return await response.json()
                except ClientResponseError as exception:
                    if exception.status == 429:
                        logger.info(
                            f"API rate limit reached. Cooling down for {COOLDOWN_PERIOD}..."
                        )
                        await asyncio.sleep(COOLDOWN_PERIOD)
                        retries += 1
                    elif 500 <= exception.status < 600:
                        logger.error(f"Server error: {exception.status}. Retrying...")
                        retries += 1
                    else:
                        logger.error(f"HTTP error: {exception.status}. Exiting...")
                        return None

                if retries < MAX_RETRIES:
                    wait_time = BACKOFF_FACTOR**retries
                    logger.info(
                        f"Retrying in {wait_time} seconds... (Attempt {retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
            return None


async def get_pdb_api(
    identifiers: tuple[str | int, ...], target: str
) -> dict[str, Any] | None:
    """Helper function for get requests to the PDB API."""
    if target not in ["entry", "peptide", "ligand", "ligandform"]:
        raise ValueError("Invalid target")

    if target == "entry":
        assert len(identifiers) == 1
        url = f"https://data.rcsb.org/rest/v1/core/entry/{identifiers[0]}"
    elif target == "peptide":
        assert len(identifiers) == 2
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{identifiers[0]}/{identifiers[1]}"
    elif target == "ligand" or target == "ligandform":
        assert len(identifiers) == 1
        url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{identifiers[0]}"
    return await api_request(url)
