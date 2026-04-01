"""Cross-Company Risk Contagion Graph Analysis.

A novel approach to systemic risk monitoring in SEC filings. This module
models the inter-company propagation of risk disclosure language as a
contagion process on a graph, enabling:

1. **Risk contagion detection**: When Company A introduces novel risk
   language and peer companies adopt similar language 1-2 quarters later,
   we identify the propagation chain and source company.

2. **Systemic risk quantification**: Graph-theoretic measures (betweenness
   centrality, clustering coefficient, spectral properties) identify
   companies that act as "risk language hubs" -- firms whose disclosure
   changes predict sector-wide shifts.

3. **First-mover identification**: Companies that originate new risk
   narratives before they spread to peers are flagged as "first movers."
   These are often the first to experience the underlying risk event.

The key insight is that boilerplate risk language is stable and widespread,
so when genuinely new risk phrases appear and then propagate through a
company graph, the propagation pattern itself carries signal about the
nature and severity of the emerging risk.

References:
    - Hoberg & Phillips (2016), "Text-Based Network Industries and
      Endogenous Product Differentiation"
    - Hanley & Hoberg (2019), "Dynamic Interpretation of Emerging Risks
      in the Financial Sector"
    - Brown & Tucker (2011), "Large-Sample Evidence on Firms' Year-over-Year
      MD&A Modifications"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import networkx as nx
import numpy as np

from lexdrift.nlp.embeddings import cosine_similarity, encode_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Minimum cosine similarity to form an edge in the risk graph.
# Empirically, risk factor sections within the same industry cluster above
# 0.6, while cross-sector similarity sits around 0.3-0.45.  We set the
# threshold between these regimes to capture meaningful connections without
# drowning in noise.
EDGE_SIMILARITY_THRESHOLD: float = 0.50

# When checking whether language "contagion" occurred, we require at least
# this similarity between the new phrases and text that appeared in
# connected companies' subsequent filings.
CONTAGION_PHRASE_THRESHOLD: float = 0.70

# Maximum number of quarters to look back for contagion source evidence.
MAX_CONTAGION_LAG_QUARTERS: int = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContagionPath:
    """A single propagation path from source to the queried company."""

    source_company_id: int
    intermediary_ids: list[int]
    phrase_similarity: float
    lag_quarters: int

    @property
    def path_length(self) -> int:
        return len(self.intermediary_ids) + 1


@dataclass
class ContagionResult:
    """Results of contagion analysis for a single company's new risk language.

    Attributes:
        company_id: The company whose new language was analyzed.
        contagion_score: Aggregate score in [0, 1] summarizing how much of
            this company's new risk language appears to have propagated from
            connected peers.  0 = entirely novel, 1 = pure follower.
        source_companies: Company IDs identified as likely originators of the
            language this company adopted.
        propagation_paths: Detailed paths showing how language reached this
            company through the graph.
        propagation_lag: Weighted-average number of quarters between the
            first appearance in source companies and adoption here.
        is_first_mover: True if this company's new language does NOT appear
            in any connected company's prior filings -- i.e., this company
            is originating a new risk narrative.
        novel_phrases: Phrases that appear genuinely novel (no contagion
            source found).
    """

    company_id: int
    contagion_score: float
    source_companies: list[int]
    propagation_paths: list[ContagionPath]
    propagation_lag: float
    is_first_mover: bool
    novel_phrases: list[str] = field(default_factory=list)


@dataclass
class SystemicRiskMetrics:
    """Graph-level metrics characterizing systemic risk structure.

    Attributes:
        betweenness_centrality: Per-node betweenness centrality.  High values
            indicate companies that bridge otherwise-disconnected clusters --
            potential vectors for cross-sector contagion.
        clustering_coefficients: Per-node clustering coefficient.  Companies
            with high clustering are embedded in tightly-coupled peer groups
            where risk language spreads rapidly.
        risk_hubs: Company IDs with betweenness centrality above the 90th
            percentile -- these are the key propagation nodes.
        connected_components: Number of disconnected subgraphs.  A single
            large component suggests pervasive interconnection.
        largest_component_fraction: Fraction of total nodes in the largest
            connected component.
        average_path_length: Mean shortest path length within the largest
            connected component.  Shorter paths = faster contagion.
        spectral_gap: Difference between the two largest eigenvalues of the
            normalized graph Laplacian.  A large spectral gap indicates
            rapid mixing / fast information diffusion.
        density: Graph density (actual edges / possible edges).
        modularity_communities: List of sets, each containing company IDs
            that form a densely-connected community.  Risk contagion tends
            to stay within communities before crossing boundaries.
    """

    betweenness_centrality: dict[int, float]
    clustering_coefficients: dict[int, float]
    risk_hubs: list[int]
    connected_components: int
    largest_component_fraction: float
    average_path_length: float | None
    spectral_gap: float | None
    density: float
    modularity_communities: list[set[int]]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_risk_graph(
    company_sections: dict[int, str],
    *,
    similarity_threshold: float = EDGE_SIMILARITY_THRESHOLD,
    batch_size: int = 64,
) -> nx.Graph:
    """Build a weighted graph of inter-company risk-disclosure similarity.

    Each node is a company; each edge weight is the cosine similarity
    between the two companies' risk factor section embeddings.  Only edges
    above ``similarity_threshold`` are retained to keep the graph sparse
    and interpretable.

    Args:
        company_sections: Mapping of company_id to risk factor section text.
        similarity_threshold: Minimum cosine similarity to create an edge.
        batch_size: Number of texts to encode in a single batch for
            efficiency.  Large corpora benefit from higher values.

    Returns:
        A networkx Graph with:
            - Node attribute ``'embedding'``: the section embedding vector.
            - Edge attribute ``'weight'``: cosine similarity in
              [similarity_threshold, 1.0].
    """
    if not company_sections:
        return nx.Graph()

    company_ids = list(company_sections.keys())
    texts = [company_sections[cid] for cid in company_ids]

    logger.info(
        "Building risk graph for %d companies (threshold=%.2f)",
        len(company_ids),
        similarity_threshold,
    )

    # Encode all risk sections ------------------------------------------------
    embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            embeddings.append(encode_text(text))

    embedding_matrix = np.stack(embeddings, axis=0)

    # Compute pairwise cosine similarity in bulk ------------------------------
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = embedding_matrix / norms
    sim_matrix = normed @ normed.T  # (N, N) symmetric

    # Build graph -------------------------------------------------------------
    graph = nx.Graph()

    for idx, cid in enumerate(company_ids):
        graph.add_node(cid, embedding=embeddings[idx])

    n = len(company_ids)
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sim_matrix[i, j])
            if sim >= similarity_threshold:
                graph.add_edge(
                    company_ids[i],
                    company_ids[j],
                    weight=sim,
                )
                edge_count += 1

    logger.info(
        "Risk graph: %d nodes, %d edges (density=%.4f)",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        nx.density(graph) if graph.number_of_nodes() > 1 else 0.0,
    )

    return graph


# ---------------------------------------------------------------------------
# Contagion detection
# ---------------------------------------------------------------------------

def _embed_phrases(phrases: list[str]) -> np.ndarray:
    """Encode a list of short phrases into an embedding matrix."""
    vecs = [encode_text(p) for p in phrases]
    return np.stack(vecs, axis=0)


def _quarter_label(date_str: str) -> str:
    """Normalize a date string to a 'YYYY-QN' quarter label."""
    dt = datetime.fromisoformat(date_str)
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}-Q{quarter}"


def _quarter_distance(q_a: str, q_b: str) -> int:
    """Compute the number of quarters between two quarter labels.

    Returns a non-negative integer.  If q_b is earlier than q_a the
    result is still positive (absolute distance).
    """
    def _to_ordinal(q: str) -> int:
        year, qn = q.split("-Q")
        return int(year) * 4 + int(qn)

    return abs(_to_ordinal(q_a) - _to_ordinal(q_b))


def detect_contagion(
    graph: nx.Graph,
    company_id: int,
    new_phrases: list[str],
    historical_graphs: list[tuple[str, nx.Graph]],
    *,
    phrase_threshold: float = CONTAGION_PHRASE_THRESHOLD,
    max_lag: int = MAX_CONTAGION_LAG_QUARTERS,
) -> ContagionResult:
    """Detect whether a company's new risk language was adopted from peers.

    For each new phrase introduced by ``company_id``, we check whether
    semantically similar language already existed in the risk disclosures
    of graph-connected companies in prior quarters.

    The historical graph sequence must be ordered chronologically (earliest
    first).  Each entry is ``(date_string, graph_at_that_date)``.

    Args:
        graph: The current-period risk graph.
        company_id: The company to analyze.
        new_phrases: New risk phrases/sentences this company added in the
            latest filing.
        historical_graphs: Chronologically-ordered list of
            ``(date_str, nx.Graph)`` representing the risk graph at prior
            filing dates.  Nodes must carry an ``'embedding'`` attribute.
        phrase_threshold: Minimum cosine similarity to consider a phrase
            match as evidence of contagion.
        max_lag: Maximum number of quarters to look back.

    Returns:
        A ``ContagionResult`` summarizing the contagion analysis.
    """
    if not new_phrases:
        return ContagionResult(
            company_id=company_id,
            contagion_score=0.0,
            source_companies=[],
            propagation_paths=[],
            propagation_lag=0.0,
            is_first_mover=True,
            novel_phrases=[],
        )

    if company_id not in graph:
        logger.warning("Company %d not in current risk graph", company_id)
        return ContagionResult(
            company_id=company_id,
            contagion_score=0.0,
            source_companies=[],
            propagation_paths=[],
            propagation_lag=0.0,
            is_first_mover=True,
            novel_phrases=list(new_phrases),
        )

    # Identify the queried company's neighbors (up to distance 2)
    # to capture both direct peers and one-hop-away companies.
    neighbors_d1 = set(graph.neighbors(company_id))
    neighbors_d2: set[int] = set()
    for n1 in neighbors_d1:
        for n2 in graph.neighbors(n1):
            if n2 != company_id:
                neighbors_d2.add(n2)
    reachable_peers = neighbors_d1 | neighbors_d2

    if not reachable_peers:
        return ContagionResult(
            company_id=company_id,
            contagion_score=0.0,
            source_companies=[],
            propagation_paths=[],
            propagation_lag=0.0,
            is_first_mover=True,
            novel_phrases=list(new_phrases),
        )

    # Embed the new phrases
    phrase_embeddings = _embed_phrases(new_phrases)

    # Determine the current quarter for lag computation
    current_quarter: str | None = None
    if historical_graphs:
        # Assume the current period is one quarter after the last historical
        last_date = historical_graphs[-1][0]
        last_q = _quarter_label(last_date)
        year, qn = last_q.split("-Q")
        next_q = int(qn) % 4 + 1
        next_year = int(year) + (1 if next_q == 1 else 0)
        current_quarter = f"{next_year}-Q{next_q}"

    # Search historical graphs for prior occurrences in peer companies --------
    propagation_paths: list[ContagionPath] = []
    phrase_matched = [False] * len(new_phrases)
    source_set: set[int] = set()

    for hist_date, hist_graph in reversed(historical_graphs):
        hist_quarter = _quarter_label(hist_date)

        if current_quarter is not None:
            lag = _quarter_distance(current_quarter, hist_quarter)
            if lag > max_lag:
                break
        else:
            lag = 0

        for peer_id in reachable_peers:
            if peer_id not in hist_graph.nodes:
                continue

            peer_data = hist_graph.nodes[peer_id]
            peer_embedding = peer_data.get("embedding")
            if peer_embedding is None:
                continue

            # Compare each new phrase against the peer's historical embedding
            for pi, phrase_emb in enumerate(phrase_embeddings):
                if phrase_matched[pi]:
                    continue

                sim = cosine_similarity(phrase_emb, peer_embedding)
                if sim >= phrase_threshold:
                    phrase_matched[pi] = True
                    source_set.add(peer_id)

                    # Determine intermediary path if the peer is distance-2
                    intermediaries: list[int] = []
                    if peer_id not in neighbors_d1:
                        # Find the bridging node
                        for bridge in neighbors_d1:
                            if hist_graph.has_edge(bridge, peer_id):
                                intermediaries = [bridge]
                                break

                    propagation_paths.append(
                        ContagionPath(
                            source_company_id=peer_id,
                            intermediary_ids=intermediaries,
                            phrase_similarity=round(sim, 4),
                            lag_quarters=lag,
                        )
                    )

    # Aggregate metrics -------------------------------------------------------
    n_matched = sum(phrase_matched)
    n_total = len(new_phrases)
    contagion_score = n_matched / n_total if n_total > 0 else 0.0

    avg_lag = 0.0
    if propagation_paths:
        avg_lag = sum(p.lag_quarters for p in propagation_paths) / len(
            propagation_paths
        )

    is_first_mover = n_matched == 0

    novel_phrases = [
        new_phrases[i] for i in range(n_total) if not phrase_matched[i]
    ]

    return ContagionResult(
        company_id=company_id,
        contagion_score=round(contagion_score, 4),
        source_companies=sorted(source_set),
        propagation_paths=propagation_paths,
        propagation_lag=round(avg_lag, 2),
        is_first_mover=is_first_mover,
        novel_phrases=novel_phrases,
    )


# ---------------------------------------------------------------------------
# Systemic risk computation
# ---------------------------------------------------------------------------

def compute_systemic_risk(
    graph: nx.Graph,
    *,
    hub_percentile: float = 90.0,
) -> SystemicRiskMetrics:
    """Compute graph-theoretic systemic risk metrics.

    Identifies structural properties of the inter-company risk graph that
    characterize how rapidly and broadly risk language can propagate.

    Args:
        graph: The risk similarity graph (from ``build_risk_graph``).
        hub_percentile: Percentile threshold above which a node is
            classified as a risk hub by betweenness centrality.

    Returns:
        A ``SystemicRiskMetrics`` instance with per-node and global metrics.
    """
    n_nodes = graph.number_of_nodes()

    if n_nodes == 0:
        return SystemicRiskMetrics(
            betweenness_centrality={},
            clustering_coefficients={},
            risk_hubs=[],
            connected_components=0,
            largest_component_fraction=0.0,
            average_path_length=None,
            spectral_gap=None,
            density=0.0,
            modularity_communities=[],
        )

    # Per-node centrality measures -------------------------------------------
    betweenness = nx.betweenness_centrality(graph, weight="weight")
    clustering = nx.clustering(graph, weight="weight")

    # Identify risk hubs (top percentile of betweenness centrality) ----------
    if betweenness:
        bc_values = np.array(list(betweenness.values()))
        threshold = float(np.percentile(bc_values, hub_percentile))
        risk_hubs = sorted(
            cid for cid, bc in betweenness.items() if bc >= threshold
        )
    else:
        risk_hubs = []

    # Connected components ---------------------------------------------------
    components = list(nx.connected_components(graph))
    n_components = len(components)
    largest_component = max(components, key=len) if components else set()
    largest_frac = len(largest_component) / n_nodes if n_nodes > 0 else 0.0

    # Average shortest path length (within largest component) ----------------
    avg_path: float | None = None
    if len(largest_component) > 1:
        subgraph = graph.subgraph(largest_component)
        try:
            avg_path = nx.average_shortest_path_length(
                subgraph, weight=None
            )
        except nx.NetworkXError:
            avg_path = None

    # Spectral gap -----------------------------------------------------------
    spectral_gap: float | None = None
    if n_nodes >= 3 and len(largest_component) >= 3:
        try:
            subgraph = graph.subgraph(largest_component)
            laplacian = nx.normalized_laplacian_matrix(subgraph).toarray()
            eigenvalues = np.sort(np.real(np.linalg.eigvalsh(laplacian)))
            # Spectral gap = lambda_2 - lambda_1 (lambda_1 is always 0 for
            # the normalized Laplacian of a connected graph).
            if len(eigenvalues) >= 2:
                spectral_gap = float(eigenvalues[1] - eigenvalues[0])
        except Exception:
            logger.debug("Spectral gap computation failed", exc_info=True)
            spectral_gap = None

    # Community detection (Louvain-style greedy modularity) -------------------
    try:
        communities = list(
            nx.community.greedy_modularity_communities(graph, weight="weight")
        )
        modularity_communities = [set(c) for c in communities]
    except Exception:
        logger.debug("Community detection failed", exc_info=True)
        modularity_communities = []

    density = nx.density(graph)

    return SystemicRiskMetrics(
        betweenness_centrality={k: round(v, 6) for k, v in betweenness.items()},
        clustering_coefficients={k: round(v, 6) for k, v in clustering.items()},
        risk_hubs=risk_hubs,
        connected_components=n_components,
        largest_component_fraction=round(largest_frac, 4),
        average_path_length=round(avg_path, 4) if avg_path is not None else None,
        spectral_gap=round(spectral_gap, 6) if spectral_gap is not None else None,
        density=round(density, 6),
        modularity_communities=modularity_communities,
    )
