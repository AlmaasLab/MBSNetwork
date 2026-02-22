from __future__ import annotations

import numpy as np
import pandas as pd
from database import Session
from database.datamodel.models import Assembly
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from sqlalchemy import select
from tqdm import tqdm

from preprocessing.size_worker import mbs_size


def estimate_mbs_size_distribution(radius: float, sample_size: int) -> pd.DataFrame:
    """Estimate the distribution of number of atoms within a given radius of the metal ligand."""
    with Session() as session:
        assembly_id = session.execute(select(Assembly.id)).scalars().all()
        print(f"Total number of assemblies: {len(assembly_id)}")
        sampled_ids = np.random.choice(assembly_id, size=sample_size, replace=False)

    sizes_list = Parallel(n_jobs=-1)(
        delayed(mbs_size)(mbs_id, radius)
        for mbs_id in tqdm(sampled_ids, total=sample_size, desc="Estimating MBS sizes")
    )
    sizes = np.asarray([s for s in sizes_list if s is not None], dtype=int)
    df = pd.DataFrame({"size": sizes})

    print(
        f"Estimated MBS size (radius={radius} Å, n={len(df)}): "
        f"mean={df['size'].mean():.2f}, std={df['size'].std():.2f}"
    )

    return df


if __name__ == "__main__":
    df = estimate_mbs_size_distribution(radius=7.0, sample_size=10000)
