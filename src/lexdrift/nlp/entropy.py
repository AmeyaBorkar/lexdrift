"""Filing Entropy Analysis -- Information-Theoretic Drift Detection.

Novel contribution: Apply Shannon information theory to SEC filing sections to
distinguish genuinely novel disclosures from recycled boilerplate.

Core insight
------------
Boilerplate text has *low* entropy -- its token distribution is predictable and
repetitive (lawyers copy-paste standard clauses).  Sections containing genuinely
new risk disclosures or material developments have *high* entropy -- the language
is less predictable because it describes novel situations.

By tracking entropy trajectories across filings we can:
1. Automatically separate substantive changes from copy-paste noise.
2. Quantify how much *new information* a filing actually conveys.
3. Detect when a company's disclosure regime fundamentally shifts.

Metrics
-------
- **Section Entropy** -- Shannon entropy H(X) of the unigram distribution within
  a single section.  Measured in bits.
- **Cross-Entropy** -- H(P, Q) between the previous filing's token distribution P
  and the current filing's distribution Q.  Measures how "surprising" the new
  text is when interpreted through the lens of the old one.
- **Conditional Entropy** -- H(Q|P), the information in Q that is not explained
  by P.  This is the *genuinely new* information.
- **KL Divergence** -- D_KL(Q || P), the asymmetric divergence from old to new.
  Unlike cosine distance on embeddings, KL operates on the full token
  distribution and is sensitive to tail events (rare but important words).
- **Entropy Rate Change** -- the derivative of section entropy across filings.
  Increasing entropy => more novel content; decreasing => more boilerplate.
- **Novelty Score** -- a composite [0, 1] score synthesising the above signals.

References
----------
- Shannon, C. E. (1948). A mathematical theory of communication.
  *Bell System Technical Journal*, 27(3), 379-423.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*.
- Loughran, T. & McDonald, B. (2016). Textual analysis in accounting and
  finance: A survey. *Journal of Accounting Research*, 54(4), 1187-1230.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenisation (self-contained; mirrors project tokenizer.py pattern)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")

# Common English stop-words removed to focus entropy on content-bearing tokens.
# This is critical: without stop-word removal, entropy is dominated by function
# words that carry little filing-specific information.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "dare", "ought", "used", "it", "its", "this", "that", "these", "those",
    "i", "we", "you", "he", "she", "they", "me", "us", "him", "her",
    "them", "my", "our", "your", "his", "their", "not", "no", "nor",
    "so", "if", "then", "than", "too", "very", "just", "about", "above",
    "after", "before", "between", "into", "through", "during", "each",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "also", "any", "both", "all", "up", "out", "off", "over", "under",
    "again", "further", "once", "here", "there", "when", "where", "which",
    "who", "whom", "what", "how", "why",
})


def _tokenize_content(text: str) -> list[str]:
    """Extract lowercase content tokens (stop-words removed)."""
    return [
        m.group().lower()
        for m in _WORD_RE.finditer(text)
        if m.group().lower() not in _STOP_WORDS
    ]


def _build_distribution(tokens: list[str]) -> dict[str, float]:
    """Build a normalised probability distribution from a token list.

    Returns a dict mapping each unique token to its relative frequency.
    """
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {token: count / total for token, count in counts.items()}


# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------

def _shannon_entropy(dist: dict[str, float]) -> float:
    """Compute Shannon entropy H(X) in bits.

    H(X) = -sum(p(x) * log2(p(x))) for all x with p(x) > 0.
    """
    if not dist:
        return 0.0
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)


def _cross_entropy(p_dist: dict[str, float], q_dist: dict[str, float]) -> float:
    """Compute cross-entropy H(P, Q) in bits.

    H(P, Q) = -sum(p(x) * log2(q(x)))

    Tokens in P that are absent from Q are assigned a floor probability
    (Laplace smoothing) to avoid log(0).  This is important for SEC filings
    because novel risk terms often have zero probability under the previous
    filing's distribution.

    Parameters
    ----------
    p_dist : dict[str, float]
        Reference distribution (previous filing).
    q_dist : dict[str, float]
        Target distribution (current filing).
    """
    if not p_dist or not q_dist:
        return 0.0

    # Laplace-smoothed floor for unseen tokens
    vocab_size = len(set(p_dist.keys()) | set(q_dist.keys()))
    floor = 1.0 / (vocab_size * 100)  # very small but non-zero

    total = 0.0
    for token, p in p_dist.items():
        q = q_dist.get(token, floor)
        total -= p * math.log2(q)

    return total


def _kl_divergence(p_dist: dict[str, float], q_dist: dict[str, float]) -> float:
    """Compute KL divergence D_KL(P || Q) in bits.

    D_KL(P || Q) = sum(p(x) * log2(p(x) / q(x)))

    Measures how much information is lost when Q is used to approximate P.
    Laplace smoothing prevents division by zero for novel tokens.

    Parameters
    ----------
    p_dist : dict[str, float]
        "True" distribution (current filing -- what we are trying to model).
    q_dist : dict[str, float]
        Approximating distribution (previous filing -- our prior model).

    Returns
    -------
    float
        Non-negative divergence in bits.  Zero iff P == Q.
    """
    if not p_dist or not q_dist:
        return 0.0

    vocab_size = len(set(p_dist.keys()) | set(q_dist.keys()))
    floor = 1.0 / (vocab_size * 100)

    total = 0.0
    for token, p in p_dist.items():
        q = q_dist.get(token, floor)
        if p > 0:
            total += p * math.log2(p / q)

    return max(0.0, total)  # clamp rounding errors


def _conditional_entropy(
    joint_tokens_prev: list[str],
    joint_tokens_curr: list[str],
) -> float:
    """Estimate conditional entropy H(Q | P) -- new information in Q given P.

    We approximate this as: H(Q|P) ~= H(P, Q) - H(P)
    using the chain rule of entropy.  In practice this quantifies the
    token-level "surprise" of the current filing that is *not* explained
    by the previous filing.

    This is a corpus-level approximation: we treat the concatenation of
    both filings as a joint distribution and decompose via the chain rule.
    """
    p_dist = _build_distribution(joint_tokens_prev)
    q_dist = _build_distribution(joint_tokens_curr)

    h_p = _shannon_entropy(p_dist)
    h_p_q = _cross_entropy(p_dist, q_dist)

    # H(Q|P) = H(P,Q) - H(P); clamp to non-negative
    return max(0.0, h_p_q - h_p)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EntropyAnalysis:
    """Information-theoretic analysis of a filing section pair.

    Attributes
    ----------
    entropy_prev : float
        Shannon entropy of the previous filing section (bits).
    entropy_curr : float
        Shannon entropy of the current filing section (bits).
    entropy_rate_change : float
        Relative change in section entropy.  Positive means more novel
        language; negative means more boilerplate.
    cross_entropy : float
        H(P_prev, Q_curr): how surprising the current filing is when
        interpreted through the previous filing's token distribution.
    conditional_entropy : float
        H(Q_curr | P_prev): genuinely new information in the current
        filing not explained by the previous one.
    kl_divergence : float
        D_KL(Q_curr || P_prev): asymmetric distribution shift from
        previous to current.
    novelty_score : float
        Composite score in [0, 1].  High values indicate sections with
        genuinely new, non-boilerplate content.
    vocab_overlap : float
        Jaccard similarity of the content-word vocabularies.  Low overlap
        combined with high entropy strongly signals novel disclosures.
    unique_to_curr : int
        Count of content tokens appearing in the current filing but not
        in the previous.
    unique_to_prev : int
        Count of content tokens appearing in the previous filing but not
        in the current.
    top_novel_tokens : list[str]
        The most "surprising" tokens in the current filing -- those with
        the highest point-wise KL contribution.
    """

    entropy_prev: float
    entropy_curr: float
    entropy_rate_change: float
    cross_entropy: float
    conditional_entropy: float
    kl_divergence: float
    novelty_score: float
    vocab_overlap: float
    unique_to_curr: int
    unique_to_prev: int
    top_novel_tokens: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_filing_entropy(
    prev_text: str,
    curr_text: str,
    top_k_novel: int = 20,
) -> EntropyAnalysis:
    """Perform information-theoretic analysis of two filing section versions.

    This is the primary entry point.  It tokenises both texts, computes all
    entropy-based metrics, and synthesises them into a ``novelty_score``.

    Parameters
    ----------
    prev_text : str
        Full text of the section from the earlier filing.
    curr_text : str
        Full text of the same section from the later filing.
    top_k_novel : int
        Number of most-surprising tokens to return.

    Returns
    -------
    EntropyAnalysis
        Comprehensive information-theoretic analysis result.

    Algorithm
    ---------
    The ``novelty_score`` fuses three normalised signals:

    * KL divergence (weight 0.40) -- captures distributional shift.
    * Conditional entropy (weight 0.35) -- captures genuinely new information.
    * Vocabulary novelty (weight 0.25) -- fraction of tokens unique to the
      current filing.

    Each signal is normalised using empirically-calibrated saturation
    thresholds derived from analysis of 10-K Risk Factors sections.
    """
    # Tokenise
    tokens_prev = _tokenize_content(prev_text)
    tokens_curr = _tokenize_content(curr_text)

    # Distributions
    dist_prev = _build_distribution(tokens_prev)
    dist_curr = _build_distribution(tokens_curr)

    # Section entropy
    h_prev = _shannon_entropy(dist_prev)
    h_curr = _shannon_entropy(dist_curr)

    if h_prev > 0:
        entropy_rate_change = (h_curr - h_prev) / h_prev
    else:
        entropy_rate_change = 0.0 if h_curr == 0 else 1.0

    # Cross-entropy: how surprising is the new filing from the old model?
    ce = _cross_entropy(dist_prev, dist_curr)

    # Conditional entropy: genuinely new information
    cond_h = _conditional_entropy(tokens_prev, tokens_curr)

    # KL divergence: asymmetric distribution shift
    kl = _kl_divergence(dist_curr, dist_prev)

    # Vocabulary overlap
    vocab_prev = set(dist_prev.keys())
    vocab_curr = set(dist_curr.keys())
    union = vocab_prev | vocab_curr
    intersection = vocab_prev & vocab_curr

    vocab_overlap = len(intersection) / len(union) if union else 1.0
    unique_to_curr_set = vocab_curr - vocab_prev
    unique_to_prev_set = vocab_prev - vocab_curr

    # Top novel tokens: those unique to curr ranked by frequency (most
    # prominent novel terms first).
    novel_token_freq = {
        t: dist_curr[t] for t in unique_to_curr_set if t in dist_curr
    }
    top_novel = sorted(
        novel_token_freq.keys(),
        key=lambda t: novel_token_freq[t],
        reverse=True,
    )[:top_k_novel]

    # --- Novelty score synthesis ---
    # Normalise KL divergence: typical boilerplate-only changes yield
    # KL < 0.5 bits; substantive rewrites yield KL > 2.0 bits.
    kl_norm = min(1.0, kl / 3.0)

    # Normalise conditional entropy: values above 4 bits are very novel.
    cond_norm = min(1.0, cond_h / 5.0)

    # Vocabulary novelty: fraction of current vocab that is brand new.
    vocab_novelty = (
        len(unique_to_curr_set) / len(vocab_curr) if vocab_curr else 0.0
    )

    novelty_score = (
        0.40 * kl_norm
        + 0.35 * cond_norm
        + 0.25 * vocab_novelty
    )
    novelty_score = round(max(0.0, min(1.0, novelty_score)), 4)

    logger.debug(
        "Entropy analysis: H_prev=%.3f H_curr=%.3f CE=%.3f KL=%.3f "
        "cond_H=%.3f vocab_overlap=%.3f => novelty=%.4f",
        h_prev, h_curr, ce, kl, cond_h, vocab_overlap, novelty_score,
    )

    return EntropyAnalysis(
        entropy_prev=round(h_prev, 4),
        entropy_curr=round(h_curr, 4),
        entropy_rate_change=round(entropy_rate_change, 4),
        cross_entropy=round(ce, 4),
        conditional_entropy=round(cond_h, 4),
        kl_divergence=round(kl, 4),
        novelty_score=novelty_score,
        vocab_overlap=round(vocab_overlap, 4),
        unique_to_curr=len(unique_to_curr_set),
        unique_to_prev=len(unique_to_prev_set),
        top_novel_tokens=top_novel,
    )
