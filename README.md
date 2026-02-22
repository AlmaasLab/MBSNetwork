# MBSNetwork

MBSNetwork - metal binding site network

Data and analysis associated with the manuscript:

**Metal binding site alignment enables network-driven discovery of recurrent geometries across sequence-divergent proteins and drug off-targets**

Preprint: bioRxiv.

## Repository structure

* `src/` — source code and analysis scripts
* `data/` — data files used and produced by the analysis
* `figures/` — generated figure output
* `backup/` — database dump chunks and restore script
* `docker-compose.yml` — local container stack
* `Dockerfile` — container definition
* `pyproject.toml` — Python dependencies

## Setup

This repository is intended to be run through the provided Docker Compose stack.

Build and start the containers:

```bash
docker compose up --build -d
```

Open a shell in the app container:

```bash
docker exec -it mbs bash
```

## Database restore

A PostgreSQL dump is provided in chunked form under `backup/`.

To populate an empty database, go to `backup/` and run:

```bash
./restore.sh
```

This restores the dump into the PostgreSQL container.

## Reproducing the figures

All manuscript figures can be regenerated from the scripts in `src/figures/figures.ipynb`.

## Please cite

If you use this repository, its data, or derived results in academic work, please cite the manuscript:

**Simensen V, Almaas E.**
*Metal binding site alignment enables network-driven discovery of recurrent geometries across sequence-divergent proteins and drug off-targets.*

## Contact

**Eivind Almaas**
[eivind.almaas@ntnu.no](mailto:eivind.almaas@ntnu.no)
Norwegian University of Science and Technology (NTNU), Trondheim, Norway
