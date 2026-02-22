from __future__ import annotations

from pathlib import Path  # noqa: TCH003
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class DirectoryConfig(BaseSettings):
    """Configuration for the directories used in the pipeline."""

    alignments: Path = Field(validation_alias="ALIGNMENTS_DIR")
    analysis: Path = Field(validation_alias="ANALYSIS_DIR")
    figures: Path = Field(validation_alias="FIGURES_DIR")
    networks: Path = Field(validation_alias="NETWORKS_DIR")
    logs: Path = Field(validation_alias="LOGS_DIR")
    structures: Path = Field(validation_alias="STRUCTURES_DIR")


class DatasetConfig(BaseSettings):
    """Configurations for the MBS dataset."""

    ligands: str | list[str]  # str necessary as workaround to avoid JSON parsing error
    atoms: int
    resolution: float
    uniprot_only: bool
    representative_threshold: float

    @field_validator("ligands", mode="before")
    @classmethod
    def parse_ligands(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",")]
        return v


class AlignmentConfig(BaseSettings):
    """Configuration for the pairwise alignment of MBSs."""

    n_jobs: int
    loss_function: Literal["tukey", "huber", "gm", "cauchy", "L1", "L2"]
    max_correspondence_distance: float
    configuration_parameter: float
    coarse_alignments: int
    max_coarse_iterations: int
    max_finetune_iterations: int
    top_ranking_fraction: float
    rmsd_ignore_threshold: float
    log_divisor: int
    save_divisor: int
    path: Path | None = None

    # Additional parameters for the ICP algorithm, all set during alignment setup
    transformation_estimation: Any | None = None
    criteria_coarse: Any | None = None
    criteria_finetune: Any | None = None
    finetune_number: int | None = None


class Config(BaseModel):
    alignment: AlignmentConfig
    dataset: DatasetConfig
    directory: DirectoryConfig


def setup_config():
    return Config.model_validate(
        {
            "alignment": AlignmentConfig(),  # type: ignore
            "dataset": DatasetConfig(),  # type: ignore
            "directory": DirectoryConfig(),  # type: ignore
        }
    )


config = setup_config()
