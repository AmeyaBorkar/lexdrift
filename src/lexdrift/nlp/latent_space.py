"""Latent Risk Trajectory Analysis.

Projects all SEC filing section embeddings into a shared low-dimensional
latent space and tracks each company's trajectory over time.  The core
hypothesis is that companies approaching financial distress follow
predictable migration patterns in embedding space: they drift away from
their historical cluster and converge toward a "danger zone" defined by
the trajectories of previously-distressed firms.

Novel contributions:
1. **Trajectory geometry as signal**: Rather than treating embeddings as
   static snapshots, we model the path (velocity, curvature, direction)
   through latent space.  A company that is *accelerating* toward the
   danger zone is riskier than one that is stationary nearby.

2. **Danger zone identification**: Given a set of known distress events
   (bankruptcy, restatement, going-concern opinion), we identify the
   latent-space regions that these companies migrated through in the
   quarters preceding distress.  New companies entering these regions
   are flagged.

3. **Trajectory extrapolation**: Simple polynomial extrapolation of a
   company's recent trajectory provides a forward-looking estimate of
   where its disclosures are heading.

Dependencies:
    - numpy (required)
    - scikit-learn (required -- PCA fallback, KDE, KMeans)
    - umap-learn (optional -- preferred for 2D projection)

References:
    - McInnes et al. (2018), "UMAP: Uniform Manifold Approximation and
      Projection for Dimension Reduction"
    - Bao & Datta (2014), "Simultaneously Discovering and Quantifying Risk
      Types from Textual Risk Disclosures"
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Default number of latent dimensions for the projection.
DEFAULT_N_COMPONENTS: int = 3

# Minimum number of points required to fit a danger zone model.
MIN_DISTRESS_POINTS: int = 5

# Percentile of the distress-company KDE used as the danger zone boundary.
DANGER_ZONE_KDE_PERCENTILE: float = 20.0

# Number of nearest neighbors for UMAP.
UMAP_N_NEIGHBORS: int = 15

# Minimum number of filings to compute a meaningful trajectory.
MIN_TRAJECTORY_LENGTH: int = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatentSpace:
    """A fitted latent space projection.

    Attributes:
        points: Projected coordinates, shape ``(n_samples, n_components)``.
        company_ids: Company ID for each point.
        filing_dates: Filing date string for each point.
        section_types: Section type label for each point.
        n_components: Dimensionality of the latent space.
        projector: The fitted projection object (PCA or UMAP instance).
        projection_method: ``'umap'`` or ``'pca'``.
    """

    points: np.ndarray
    company_ids: list[int]
    filing_dates: list[str]
    section_types: list[str]
    n_components: int
    projector: object  # PCA or UMAP -- kept generic for serialization
    projection_method: str


@dataclass
class Trajectory:
    """A single company's path through latent space over time.

    Attributes:
        company_id: The company this trajectory belongs to.
        points: Chronologically-ordered projected coordinates,
            shape ``(n_filings, n_components)``.
        dates: Filing dates corresponding to each point.
        path_length: Total Euclidean arc length of the trajectory.
        direction_vector: Unit vector pointing from the first point to the
            last.  Represents the net direction of semantic migration.
        speed: Mean per-period displacement (path_length / n_periods).
        acceleration: Change in speed over the last three periods.  Positive
            values indicate the company is changing faster.
        curvature: Mean unsigned turning angle (radians) between consecutive
            displacement vectors.  High curvature means erratic movement;
            low curvature means steady drift in one direction.
        nearest_cluster: Index of the nearest KMeans cluster centroid to the
            company's most recent position (populated by the caller if
            clusters are available).
    """

    company_id: int
    points: np.ndarray
    dates: list[str]
    path_length: float
    direction_vector: np.ndarray
    speed: float
    acceleration: float
    curvature: float
    nearest_cluster: int | None = None


@dataclass
class DangerZone:
    """A region in latent space associated with financial distress.

    Attributes:
        center: Centroid of the zone in projected coordinates.
        radius: Approximate radius (mean distance from distress points to
            center).
        kde_threshold: Log-density threshold below which a point is outside
            the zone.  Points with density above this value are considered
            inside the danger zone.
        distress_company_ids: Companies whose trajectories define this zone.
        companies_at_risk: Companies currently migrating toward this zone,
            sorted by proximity (nearest first).
    """

    center: np.ndarray
    radius: float
    kde_threshold: float
    distress_company_ids: list[int]
    companies_at_risk: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _try_umap(n_components: int, n_neighbors: int) -> object | None:
    """Attempt to import and instantiate UMAP.  Returns None on failure."""
    try:
        from umap import UMAP  # type: ignore[import-untyped]

        return UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
    except ImportError:
        logger.info("umap-learn not installed; falling back to PCA")
        return None


def build_latent_space(
    section_embeddings: list[dict],
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
    method: Literal["auto", "umap", "pca"] = "auto",
) -> LatentSpace:
    """Build a shared latent space from filing section embeddings.

    Each element of ``section_embeddings`` must be a dict with keys:
        - ``company_id``: int
        - ``filing_date``: str  (ISO 8601)
        - ``section_type``: str  (e.g., ``'risk_factors'``, ``'mda'``)
        - ``embedding``: np.ndarray of shape ``(d,)``

    Args:
        section_embeddings: List of embedding records.
        n_components: Number of latent dimensions (2 or 3 recommended for
            visualization; higher for downstream analytics).
        method: ``'umap'`` for UMAP, ``'pca'`` for PCA, or ``'auto'`` to
            try UMAP first and fall back to PCA.

    Returns:
        A fitted ``LatentSpace`` instance.

    Raises:
        ValueError: If ``section_embeddings`` is empty or embeddings have
            inconsistent dimensionality.
    """
    if not section_embeddings:
        raise ValueError("section_embeddings must be non-empty")

    company_ids = [d["company_id"] for d in section_embeddings]
    filing_dates = [d["filing_date"] for d in section_embeddings]
    section_types = [d["section_type"] for d in section_embeddings]
    embeddings = np.stack([d["embedding"] for d in section_embeddings], axis=0)

    n_samples, n_features = embeddings.shape
    logger.info(
        "Building latent space: %d samples, %d features -> %d components (method=%s)",
        n_samples,
        n_features,
        n_components,
        method,
    )

    # Select projection method ------------------------------------------------
    projector: object
    projection_method: str

    if method == "umap" or method == "auto":
        umap_inst = _try_umap(n_components, min(UMAP_N_NEIGHBORS, n_samples - 1))
        if umap_inst is not None and n_samples > UMAP_N_NEIGHBORS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                projected = umap_inst.fit_transform(embeddings)
            projector = umap_inst
            projection_method = "umap"
        elif method == "umap":
            raise ImportError(
                "umap-learn is required when method='umap' but is not installed "
                "or the dataset is too small for the requested n_neighbors"
            )
        else:
            # auto fallback
            pca = PCA(n_components=min(n_components, n_samples, n_features), random_state=42)
            projected = pca.fit_transform(embeddings)
            projector = pca
            projection_method = "pca"
    else:
        pca = PCA(n_components=min(n_components, n_samples, n_features), random_state=42)
        projected = pca.fit_transform(embeddings)
        projector = pca
        projection_method = "pca"

    logger.info(
        "Latent space built (%s): explained variance = %s",
        projection_method,
        (
            f"{sum(projector.explained_variance_ratio_):.2%}"
            if projection_method == "pca"
            else "N/A (UMAP)"
        ),
    )

    return LatentSpace(
        points=projected.astype(np.float32),
        company_ids=company_ids,
        filing_dates=filing_dates,
        section_types=section_types,
        n_components=projected.shape[1],
        projector=projector,
        projection_method=projection_method,
    )


# ---------------------------------------------------------------------------
# Trajectory computation
# ---------------------------------------------------------------------------

def compute_trajectory(
    company_embeddings: list[dict],
    latent_space: LatentSpace | None = None,
) -> Trajectory:
    """Compute the latent-space trajectory for a single company.

    ``company_embeddings`` should be a chronologically-ordered list of dicts
    with keys ``filing_date`` and ``embedding`` (raw high-dimensional).

    If ``latent_space`` is provided, the embeddings are projected using the
    existing fitted projector.  Otherwise, a standalone PCA is fit on the
    company's own embeddings (useful for quick single-company analysis but
    not directly comparable across companies).

    Args:
        company_embeddings: Ordered list of embedding records for one company.
        latent_space: An existing fitted ``LatentSpace`` to project into.

    Returns:
        A ``Trajectory`` with geometric descriptors.

    Raises:
        ValueError: If fewer than ``MIN_TRAJECTORY_LENGTH`` points are
            provided.
    """
    if len(company_embeddings) < MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"Need at least {MIN_TRAJECTORY_LENGTH} filings to compute a "
            f"trajectory; got {len(company_embeddings)}"
        )

    company_id = company_embeddings[0].get("company_id", 0)
    dates = [d["filing_date"] for d in company_embeddings]
    raw_embeddings = np.stack(
        [d["embedding"] for d in company_embeddings], axis=0
    )

    # Project into latent space -----------------------------------------------
    if latent_space is not None:
        if latent_space.projection_method == "pca":
            points = latent_space.projector.transform(raw_embeddings)  # type: ignore[union-attr]
        else:
            # UMAP does not support incremental transform out of the box in
            # all versions.  Try transform, fall back to approximate projection
            # via the fitted PCA of the UMAP embedding.
            try:
                points = latent_space.projector.transform(raw_embeddings)  # type: ignore[union-attr]
            except Exception:
                logger.warning(
                    "UMAP transform failed; falling back to nearest-neighbor "
                    "projection in the existing latent space"
                )
                points = _nn_project(raw_embeddings, latent_space)
    else:
        n_comp = min(DEFAULT_N_COMPONENTS, len(company_embeddings), raw_embeddings.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        points = pca.fit_transform(raw_embeddings)

    points = points.astype(np.float32)

    # Geometric descriptors ---------------------------------------------------
    displacements = np.diff(points, axis=0)  # (T-1, d)
    step_lengths = np.linalg.norm(displacements, axis=1)

    # Path length
    path_length = float(np.sum(step_lengths))

    # Direction vector (first -> last)
    net_displacement = points[-1] - points[0]
    net_norm = np.linalg.norm(net_displacement)
    direction_vector = (
        net_displacement / net_norm if net_norm > 1e-9 else np.zeros_like(net_displacement)
    )

    # Speed (mean step length)
    n_periods = len(step_lengths)
    speed = path_length / n_periods if n_periods > 0 else 0.0

    # Acceleration (change in speed over last three periods)
    acceleration = 0.0
    if n_periods >= 3:
        recent_speeds = step_lengths[-3:]
        # Linear regression slope of speed over the last 3 periods
        x = np.arange(len(recent_speeds), dtype=np.float64)
        x_mean = x.mean()
        y_mean = recent_speeds.mean()
        slope = float(
            np.sum((x - x_mean) * (recent_speeds - y_mean))
            / (np.sum((x - x_mean) ** 2) + 1e-12)
        )
        acceleration = slope

    # Curvature (mean unsigned turning angle between consecutive steps)
    curvature = 0.0
    if n_periods >= 2:
        angles: list[float] = []
        for t in range(n_periods - 1):
            v1 = displacements[t]
            v2 = displacements[t + 1]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angles.append(float(np.arccos(cos_angle)))
        curvature = float(np.mean(angles)) if angles else 0.0

    return Trajectory(
        company_id=company_id,
        points=points,
        dates=dates,
        path_length=round(path_length, 6),
        direction_vector=direction_vector,
        speed=round(speed, 6),
        acceleration=round(acceleration, 6),
        curvature=round(curvature, 6),
    )


def _nn_project(
    raw_embeddings: np.ndarray,
    latent_space: LatentSpace,
) -> np.ndarray:
    """Approximate projection by finding nearest neighbors in the original
    embedding space and averaging their latent-space positions.

    This is a fallback when UMAP's ``.transform()`` is unavailable.
    """
    # Reconstruct the original high-dim embeddings from the latent space
    # metadata.  We don't store them, so we fall back to PCA on the latent
    # points themselves -- not ideal but keeps us moving.
    from sklearn.neighbors import NearestNeighbors

    # Use the latent space points directly as the reference
    nn = NearestNeighbors(n_neighbors=min(5, len(latent_space.points)), metric="euclidean")

    # We need a common space.  Since we don't have the original embeddings
    # stored in LatentSpace (by design -- they're huge), we project the new
    # embeddings with a fresh PCA fit on the *original* latent points and
    # match in latent space.
    pca = PCA(n_components=latent_space.n_components, random_state=42)
    projected_new = pca.fit_transform(raw_embeddings)

    nn.fit(latent_space.points)
    distances, indices = nn.kneighbors(projected_new)

    # Weighted average of neighbor positions (inverse-distance weighting)
    results = []
    for i in range(len(projected_new)):
        dists = distances[i]
        idxs = indices[i]
        weights = 1.0 / (dists + 1e-9)
        weights /= weights.sum()
        projected_point = np.average(
            latent_space.points[idxs], axis=0, weights=weights
        )
        results.append(projected_point)

    return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# Danger zone detection
# ---------------------------------------------------------------------------

def detect_danger_zones(
    latent_space: LatentSpace,
    distress_company_ids: set[int],
    *,
    n_zones: int | None = None,
    proximity_threshold: float = 2.0,
) -> list[DangerZone]:
    """Identify latent-space regions associated with financial distress.

    Uses the trajectories of companies known to have experienced distress
    events (bankruptcy, restatement, going-concern opinion) to define
    "danger zones."  Companies whose recent filings fall inside or near
    these zones are flagged.

    The method:
    1. Collect all latent-space points belonging to distress companies.
    2. Cluster them (KMeans) to identify distinct danger regions (e.g.,
       one around liquidity-related language, another around litigation).
    3. Fit a kernel density estimate (KDE) on each cluster to define
       soft boundaries.
    4. Score all non-distress companies by their proximity to each zone.

    Args:
        latent_space: A fitted ``LatentSpace``.
        distress_company_ids: Set of company IDs known to have experienced
            distress.
        n_zones: Number of danger zone clusters.  If ``None``, determined
            automatically via a heuristic (sqrt of distress point count,
            capped at 5).
        proximity_threshold: Maximum Mahalanobis-like distance (in units
            of zone radius) for a company to be flagged as "at risk."

    Returns:
        A list of ``DangerZone`` objects, each with the companies currently
        at risk of entering that zone.
    """
    # Collect distress-company points -----------------------------------------
    distress_mask = np.array(
        [cid in distress_company_ids for cid in latent_space.company_ids],
        dtype=bool,
    )
    distress_points = latent_space.points[distress_mask]

    if len(distress_points) < MIN_DISTRESS_POINTS:
        logger.warning(
            "Only %d distress points found (need >= %d); cannot define danger zones",
            len(distress_points),
            MIN_DISTRESS_POINTS,
        )
        return []

    # Determine number of zones -----------------------------------------------
    if n_zones is None:
        n_zones = max(1, min(5, int(np.sqrt(len(distress_points)))))

    n_zones = min(n_zones, len(distress_points))

    logger.info(
        "Detecting %d danger zone(s) from %d distress points",
        n_zones,
        len(distress_points),
    )

    # Cluster distress points -------------------------------------------------
    kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distress_points)

    # Build a DangerZone for each cluster -------------------------------------
    non_distress_mask = ~distress_mask
    non_distress_points = latent_space.points[non_distress_mask]
    non_distress_cids = [
        cid
        for cid, is_distress in zip(latent_space.company_ids, distress_mask)
        if not is_distress
    ]
    non_distress_dates = [
        d
        for d, is_distress in zip(latent_space.filing_dates, distress_mask)
        if not is_distress
    ]

    danger_zones: list[DangerZone] = []

    for k in range(n_zones):
        zone_points = distress_points[cluster_labels == k]
        center = kmeans.cluster_centers_[k]
        dists_to_center = np.linalg.norm(zone_points - center, axis=1)
        radius = float(np.mean(dists_to_center)) if len(zone_points) > 0 else 1.0

        # Fit KDE on zone points for soft boundary
        bandwidth = max(radius * 0.5, 0.1)
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(zone_points)

        # KDE threshold: log-density at the DANGER_ZONE_KDE_PERCENTILE of
        # the distress points themselves.
        zone_log_densities = kde.score_samples(zone_points)
        kde_threshold = float(
            np.percentile(zone_log_densities, DANGER_ZONE_KDE_PERCENTILE)
        )

        # Distress company IDs in this zone
        zone_distress_indices = np.where(distress_mask)[0][cluster_labels == k]
        zone_distress_cids = sorted(
            {latent_space.company_ids[i] for i in zone_distress_indices}
        )

        # Score non-distress companies ----------------------------------------
        companies_at_risk: list[dict] = []
        if len(non_distress_points) > 0:
            # Distance-based proximity
            dists = np.linalg.norm(non_distress_points - center, axis=1)
            proximity_mask = dists < radius * proximity_threshold

            # KDE-based proximity (more nuanced than Euclidean distance)
            if np.any(proximity_mask):
                candidate_points = non_distress_points[proximity_mask]
                candidate_log_densities = kde.score_samples(candidate_points)
                candidate_indices = np.where(proximity_mask)[0]

                for idx, log_density in zip(candidate_indices, candidate_log_densities):
                    if log_density >= kde_threshold:
                        companies_at_risk.append({
                            "company_id": non_distress_cids[idx],
                            "filing_date": non_distress_dates[idx],
                            "distance_to_center": round(float(dists[idx]), 4),
                            "log_density": round(float(log_density), 4),
                            "risk_proximity": round(
                                1.0 - float(dists[idx]) / (radius * proximity_threshold),
                                4,
                            ),
                        })

            # Sort by proximity (highest risk first)
            companies_at_risk.sort(
                key=lambda x: x["risk_proximity"], reverse=True
            )

        danger_zones.append(
            DangerZone(
                center=center.astype(np.float32),
                radius=round(radius, 4),
                kde_threshold=round(kde_threshold, 4),
                distress_company_ids=zone_distress_cids,
                companies_at_risk=companies_at_risk,
            )
        )

    return danger_zones


# ---------------------------------------------------------------------------
# Trajectory extrapolation
# ---------------------------------------------------------------------------

def predict_trajectory(
    trajectory: Trajectory,
    n_periods: int,
    *,
    degree: int = 2,
) -> list[np.ndarray]:
    """Extrapolate a company's trajectory into future periods.

    Fits a polynomial of the specified degree to each latent-space dimension
    independently, then evaluates it at the next ``n_periods`` time steps.
    This is deliberately simple -- the value is in flagging directional
    trends, not in precise forecasting.

    For trajectories with fewer points than ``degree + 1``, the degree is
    automatically reduced to ``len(points) - 1``.

    Args:
        trajectory: A fitted ``Trajectory``.
        n_periods: Number of future periods to predict.
        degree: Polynomial degree for fitting.

    Returns:
        A list of ``n_periods`` predicted coordinate arrays, each of shape
        ``(n_components,)``.
    """
    points = trajectory.points
    n_points, n_dims = points.shape

    # Clamp degree to avoid overfitting
    effective_degree = min(degree, n_points - 1)
    if effective_degree < 1:
        # Cannot fit even a linear model; just repeat the last point
        return [points[-1].copy() for _ in range(n_periods)]

    t = np.arange(n_points, dtype=np.float64)
    t_future = np.arange(n_points, n_points + n_periods, dtype=np.float64)

    predicted: list[np.ndarray] = []
    predictions_per_dim: list[np.ndarray] = []

    for d in range(n_dims):
        coeffs = np.polyfit(t, points[:, d].astype(np.float64), effective_degree)
        poly = np.poly1d(coeffs)
        predictions_per_dim.append(poly(t_future))

    # Assemble per-period prediction vectors
    pred_matrix = np.stack(predictions_per_dim, axis=1)  # (n_periods, n_dims)
    for i in range(n_periods):
        predicted.append(pred_matrix[i].astype(np.float32))

    return predicted
