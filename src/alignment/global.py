from __future__ import annotations

import time
from multiprocessing import current_process

import numpy as np
from joblib import Parallel, delayed  # type: ignore[import-untyped]
from tqdm import tqdm


def _chunkify(
    pairs: list[tuple[int, int]], n_chunks: int
) -> list[list[tuple[int, int]]]:
    n_chunks = max(1, n_chunks)
    chunks = [pairs[i::n_chunks] for i in range(n_chunks)]
    return [c for c in chunks if c]


def _align_chunk_joblib(
    worker_idx: int,
    mbs_ids: list[int],
    pair_chunk: list[tuple[int, int]],
    num_trials: int,
    max_distance: float,
    log_every: int,
) -> np.ndarray:
    # Local imports avoid pickling SQLAlchemy/sessionmaker/Open3D-related globals
    import database
    import open3d as o3d  # type: ignore[import-untyped]
    from database.datamodel.models import MetalBindingSite
    from sqlalchemy import select

    from alignment.alignment import MBSPointCloud
    from alignment.util import evaluate_alignment

    def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float):
        radius_normal = voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        radius_feature = voxel_size * 3
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30),
        )

    def execute_fast_global_registration(
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        source_fpfh,
        target_fpfh,
        distance_threshold: float,
    ):
        return o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold
            ),
        )

    proc_name = current_process().name

    loss_function = o3d.pipelines.registration.TukeyLoss(1.8)
    estimation_method = (
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(
            1.0, loss_function
        )
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5)

    needed_ids = {mbs_ids[i] for i, _ in pair_chunk} | {
        mbs_ids[j] for _, j in pair_chunk
    }

    t_fetch0 = time.perf_counter()
    with database.Session() as session:
        rows = (
            session.execute(
                select(MetalBindingSite).where(MetalBindingSite.id.in_(needed_ids))
            )
            .scalars()
            .all()
        )
    by_id = {row.id: row for row in rows}
    t_fetch1 = time.perf_counter()

    print(
        f"[{proc_name} | worker {worker_idx}] chunk={len(pair_chunk)} sites={len(needed_ids)} "
        f"db_fetch={t_fetch1 - t_fetch0:.2f}s"
    )

    out = np.empty((len(pair_chunk), 5), dtype=np.float64)

    for k, (i, j) in enumerate(pair_chunk):
        if log_every and k and (k % log_every == 0):
            print(f"[{proc_name} | worker {worker_idx}] {k}/{len(pair_chunk)} pairs")

        src_id = mbs_ids[i]
        tgt_id = mbs_ids[j]

        as_source = by_id[src_id]
        as_target = by_id[tgt_id]

        base_target = MBSPointCloud.from_mbs(as_target)

        best_rmsd = float("inf")
        best_transformed_src: o3d.geometry.PointCloud | None = None

        t0 = time.perf_counter()

        for _ in range(num_trials):
            source = MBSPointCloud.from_mbs(as_source)
            target = MBSPointCloud.from_mbs(as_target)
            source.rotate_randomly()

            source_fpfh = preprocess_point_cloud(source.point_cloud, max_distance)
            target_fpfh = preprocess_point_cloud(target.point_cloud, max_distance)

            try:
                result = execute_fast_global_registration(
                    source.point_cloud,
                    target.point_cloud,
                    source_fpfh,
                    target_fpfh,
                    max_distance,
                )
            except RuntimeError:
                continue

            transformed = o3d.geometry.PointCloud(source.point_cloud)
            transformed.transform(result.transformation)

            rmsd = evaluate_alignment(transformed, target, 0.1)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_transformed_src = transformed

        t1 = time.perf_counter()

        if best_transformed_src is None:
            out[k] = (float(src_id), float(tgt_id), float("inf"), t1 - t0, 0.0)
            continue

        alignment = o3d.pipelines.registration.registration_generalized_icp(
            best_transformed_src,
            base_target.point_cloud,
            max_distance,
            estimation_method=estimation_method,
            criteria=criteria,
        )

        best_transformed_src.transform(alignment.transformation)
        final_rmsd = evaluate_alignment(best_transformed_src, base_target, 0.1)
        t2 = time.perf_counter()

        out[k] = (float(src_id), float(tgt_id), float(final_rmsd), t1 - t0, t2 - t1)

    print(f"[{proc_name} | worker {worker_idx}] done")
    return out


def run_global_registration_joblib(
    mbs_ids: list[int],
    pair_indices: list[tuple[int, int]],
    *,
    num_trials: int = 5,
    max_distance: float = 3.0,
    n_jobs: int = 8,
    log_every: int = 250,
) -> np.ndarray:
    chunks = _chunkify(pair_indices, n_jobs)

    tasks = [
        delayed(_align_chunk_joblib)(
            w, mbs_ids, chunk, num_trials, max_distance, log_every
        )
        for w, chunk in enumerate(chunks)
    ]

    parts = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        tqdm(tasks, desc="Submitting chunks")
    )

    return np.vstack(parts) if parts else np.empty((0, 5), dtype=np.float64)


if __name__ == "__main__":
    from analysis.preliminary import prepare_pair_indices
    from config import config

    alignment_path = config.directory.alignments / "preliminary/alignments_0_4.npy"
    mbs_ids, pair_indices = prepare_pair_indices(alignment_path)

    arr = run_global_registration_joblib(
        mbs_ids,
        pair_indices,
        num_trials=5,
        max_distance=3.0,
        n_jobs=8,
        log_every=200,
    )

    np.save(config.directory.alignments / "preliminary/global_registration.npy", arr)
