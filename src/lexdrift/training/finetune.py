"""Self-Supervised Embedding Fine-Tuning Pipeline for SEC Filings.

The key insight: LexDrift's own sentence-level diff pipeline generates
high-quality training signal *automatically*.  When we compare consecutive
filings from the same company and the same section:

    - **Unchanged sentences** form **positive pairs**: two sentences that
      should be close in embedding space because they convey the same
      disclosure content across filing periods.

    - **Added or removed sentences** form **hard negatives**: sentences
      that are topically related (same company, same section) but carry
      genuinely different information.  These are far more informative
      negatives than random sentence pairs.

This creates a self-supervised contrastive learning setup without any
human annotation.  Fine-tuning on this signal teaches the embedding model
to:
    1. Distinguish meaningful semantic changes from paraphrasing.
    2. Place SEC-specific language (boilerplate, risk factors, MD&A) into
       tighter, more discriminative clusters.
    3. Better capture the domain-specific notion of "drift."

Usage:
    python -m lexdrift.training.finetune \\
        --model-name all-MiniLM-L6-v2 \\
        --output-path models/lexdrift-finetuned \\
        --epochs 3 \\
        --batch-size 32

Dependencies:
    - sentence-transformers (required)
    - torch (required, pulled in by sentence-transformers)
    - sqlalchemy (required, for DB access)

References:
    - Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using
      Siamese BERT-Networks"
    - Gao et al. (2021), "SimCSE: Simple Contrastive Learning of Sentence
      Embeddings"
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of positive/negative pairs to sample per (company, section)
# to avoid data imbalance toward prolific filers.
MAX_PAIRS_PER_SECTION: int = 200

# Minimum sentence length (chars) to include in training data.  Filters out
# headers, numbering artifacts, etc.
MIN_SENTENCE_LENGTH: int = 30

# Train/validation split ratio.
VALIDATION_SPLIT: float = 0.1

# Default training hyperparameters.
DEFAULT_MODEL_NAME: str = "all-MiniLM-L6-v2"
DEFAULT_OUTPUT_PATH: str = "models/lexdrift-finetuned"
DEFAULT_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 2e-5
DEFAULT_WARMUP_FRACTION: float = 0.1


# ---------------------------------------------------------------------------
# Training pair generation
# ---------------------------------------------------------------------------

def generate_training_pairs(
    db_session: Session,
    *,
    max_pairs_per_section: int = MAX_PAIRS_PER_SECTION,
    min_sentence_length: int = MIN_SENTENCE_LENGTH,
) -> list:
    """Generate contrastive training pairs from consecutive SEC filings.

    Queries the database for all ``SentenceChange`` records (produced by the
    existing diff pipeline) and constructs:

    - **Positive pairs** (label ~1.0): Unchanged or very-similar sentences
      across consecutive filings.  These teach the model that paraphrases
      and minor edits should map to nearby embeddings.

    - **Hard negative pairs** (label ~0.0): Added sentences paired with
      removed sentences from the same (company, section).  These share
      topical context but differ in meaning, producing informative negatives.

    - **Soft negative pairs** (label ~0.3): Semantically changed sentences
      (the "changed" category from the diff pipeline, similarity 0.55-0.85).
      These occupy the middle ground and teach the model fine-grained
      discrimination.

    Args:
        db_session: A synchronous SQLAlchemy session.
        max_pairs_per_section: Cap on pairs per (company, section_type) to
            prevent data imbalance.
        min_sentence_length: Minimum character length for included sentences.

    Returns:
        A list of ``sentence_transformers.InputExample`` objects ready for
        training.
    """
    from sentence_transformers import InputExample

    from lexdrift.db.models import (
        Company,
        DriftScore,
        Filing,
        SentenceChange,
    )

    logger.info("Generating training pairs from sentence changes in the database")

    # Query all sentence changes with their drift score context ---------------
    stmt = (
        select(
            SentenceChange.change_type,
            SentenceChange.sentence_text,
            SentenceChange.matched_text,
            SentenceChange.similarity_score,
            DriftScore.company_id,
            DriftScore.section_type,
        )
        .join(DriftScore, SentenceChange.drift_score_id == DriftScore.id)
        .order_by(DriftScore.company_id, DriftScore.section_type)
    )

    rows = db_session.execute(stmt).all()
    logger.info("Fetched %d sentence change records", len(rows))

    # Organize by (company_id, section_type) ---------------------------------
    buckets: dict[tuple[int, str], dict[str, list]] = {}

    for row in rows:
        change_type = row.change_type
        sentence = row.sentence_text
        matched = row.matched_text
        similarity = row.similarity_score
        company_id = row.company_id
        section_type = row.section_type

        key = (company_id, section_type)
        if key not in buckets:
            buckets[key] = {"unchanged": [], "added": [], "removed": [], "changed": []}

        if len(sentence) < min_sentence_length:
            continue

        if change_type == "changed" and matched and len(matched) >= min_sentence_length:
            buckets[key]["changed"].append((sentence, matched, similarity or 0.5))
        elif change_type == "added":
            buckets[key]["added"].append(sentence)
        elif change_type == "removed":
            buckets[key]["removed"].append(sentence)

    # Build InputExample instances -------------------------------------------
    examples: list[InputExample] = []

    for (company_id, section_type), data in buckets.items():
        section_examples: list[InputExample] = []

        # --- Positive pairs from "changed" with high similarity (>= 0.80) ---
        # These are near-paraphrases that should be close in embedding space.
        for sent_a, sent_b, sim in data["changed"]:
            if sim >= 0.80:
                section_examples.append(
                    InputExample(texts=[sent_a, sent_b], label=float(sim))
                )

        # --- Soft negatives from "changed" with moderate similarity ----------
        for sent_a, sent_b, sim in data["changed"]:
            if 0.55 <= sim < 0.80:
                # Map similarity to a label that reflects the partial overlap.
                # A sentence pair at 0.55 similarity gets label ~0.2;
                # at 0.79 it gets label ~0.65.
                label = max(0.0, (sim - 0.55) / (0.80 - 0.55) * 0.65)
                section_examples.append(
                    InputExample(texts=[sent_a, sent_b], label=round(label, 3))
                )

        # --- Hard negatives: added x removed --------------------------------
        added = data["added"]
        removed = data["removed"]
        if added and removed:
            # Pair each added sentence with a random removed sentence.
            # These are topically related but semantically different.
            n_neg_pairs = min(len(added), len(removed), max_pairs_per_section // 3)
            sampled_added = random.sample(added, n_neg_pairs)
            sampled_removed = random.sample(removed, n_neg_pairs)
            for sa, sr in zip(sampled_added, sampled_removed):
                section_examples.append(
                    InputExample(texts=[sa, sr], label=0.0)
                )

        # Cap per section
        if len(section_examples) > max_pairs_per_section:
            section_examples = random.sample(section_examples, max_pairs_per_section)

        examples.extend(section_examples)

    # Shuffle globally
    random.shuffle(examples)

    logger.info(
        "Generated %d training pairs from %d (company, section) buckets",
        len(examples),
        len(buckets),
    )
    return examples


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def finetune_embeddings(
    training_pairs: list,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    output_path: str = DEFAULT_OUTPUT_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION,
    use_mnrl: bool = False,
) -> str:
    """Fine-tune a sentence-transformer on SEC filing training pairs.

    Two loss function options:

    - **CosineSimilarityLoss** (default): Suitable when training pairs have
      continuous similarity labels (which ours do, from the diff pipeline's
      similarity scores).

    - **MultipleNegativesRankingLoss** (``use_mnrl=True``): Treats each
      positive pair's in-batch negatives as hard negatives.  More
      sample-efficient but requires pairs with binary labels (similar /
      dissimilar).  We filter to only high-similarity and zero-similarity
      pairs when this mode is selected.

    Args:
        training_pairs: List of ``InputExample`` objects from
            ``generate_training_pairs``.
        model_name: HuggingFace model name or local path to start from.
        output_path: Directory to save the fine-tuned model.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: AdamW learning rate.
        warmup_fraction: Fraction of total steps used for LR warmup.
        use_mnrl: If True, use MultipleNegativesRankingLoss instead of
            CosineSimilarityLoss.

    Returns:
        The output path where the model was saved.

    Raises:
        ValueError: If ``training_pairs`` is empty.
    """
    if not training_pairs:
        raise ValueError("No training pairs provided; cannot fine-tune")

    from sentence_transformers import SentenceTransformer, losses
    from torch.utils.data import DataLoader

    logger.info(
        "Fine-tuning '%s' on %d pairs (epochs=%d, batch_size=%d, lr=%.1e, loss=%s)",
        model_name,
        len(training_pairs),
        epochs,
        batch_size,
        learning_rate,
        "MNRL" if use_mnrl else "CosineSimilarityLoss",
    )

    # Load base model ---------------------------------------------------------
    model = SentenceTransformer(model_name)

    # Prepare data loader -----------------------------------------------------
    if use_mnrl:
        # MNRL works best with (anchor, positive) pairs where in-batch
        # elements serve as negatives.  Filter to high-similarity pairs only.
        mnrl_pairs = [
            p for p in training_pairs if p.label >= 0.8
        ]
        if len(mnrl_pairs) < batch_size:
            logger.warning(
                "Only %d high-similarity pairs available for MNRL "
                "(need >= batch_size=%d); falling back to CosineSimilarityLoss",
                len(mnrl_pairs),
                batch_size,
            )
            use_mnrl = False
        else:
            training_pairs = mnrl_pairs

    # Split off validation set ------------------------------------------------
    n_val = max(1, int(len(training_pairs) * VALIDATION_SPLIT))
    val_pairs = training_pairs[:n_val]
    train_pairs = training_pairs[n_val:]

    train_loader = DataLoader(
        train_pairs,
        shuffle=True,
        batch_size=batch_size,
    )

    # Select loss function ----------------------------------------------------
    if use_mnrl:
        loss = losses.MultipleNegativesRankingLoss(model=model)
    else:
        loss = losses.CosineSimilarityLoss(model=model)

    # Warmup steps
    total_steps = math.ceil(len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_fraction)

    # Train -------------------------------------------------------------------
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    final_output_path = str(Path(output_path) / f"lexdrift-{timestamp}")

    logger.info(
        "Training: %d batches/epoch, %d total steps, %d warmup steps",
        len(train_loader),
        total_steps,
        warmup_steps,
    )

    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=final_output_path,
        optimizer_params={"lr": learning_rate},
        show_progress_bar=True,
    )

    logger.info("Fine-tuned model saved to: %s", final_output_path)
    return final_output_path


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune sentence-transformer embeddings on SEC filing diffs",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=f"Base model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output directory for the fine-tuned model (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--use-mnrl",
        action="store_true",
        help="Use MultipleNegativesRankingLoss instead of CosineSimilarityLoss",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override the database URL from config",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=MAX_PAIRS_PER_SECTION,
        help=f"Max pairs per (company, section) (default: {MAX_PAIRS_PER_SECTION})",
    )
    parser.add_argument(
        "--elite",
        action="store_true",
        help=(
            "Use elite training data generator (text-level signals only, "
            "no circular model dependency). Requires raw section text in the "
            "database but does NOT require prior analysis runs."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full fine-tuning pipeline: generate pairs, then train."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args(argv)

    # Set up a synchronous DB session for pair generation ---------------------
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from lexdrift.config import settings

    db_url = args.database_url or settings.database_url

    # Convert async URL to sync if needed (aiosqlite -> pysqlite / plain)
    sync_url = db_url.replace("+aiosqlite", "").replace("+aiomysql", "+pymysql")

    logger.info("Connecting to database: %s", sync_url.split("@")[-1])
    engine = create_engine(sync_url, echo=False)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        if args.elite:
            from lexdrift.training.data_quality import generate_elite_pairs

            logger.info("Using elite data generator (text-level signals, no model labels)")
            pairs = generate_elite_pairs(session, max_pairs=args.max_pairs)
        else:
            pairs = generate_training_pairs(
                session,
                max_pairs_per_section=args.max_pairs,
            )

    if not pairs:
        logger.error(
            "No training pairs generated.  %s",
            "Ensure raw section text exists in the database (run backfill first)."
            if args.elite
            else "Ensure the diff pipeline has been run (sentence_changes table must be populated).",
        )
        sys.exit(1)

    logger.info("Total training pairs: %d", len(pairs))

    # Fine-tune ---------------------------------------------------------------
    saved_path = finetune_embeddings(
        pairs,
        model_name=args.model_name,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_mnrl=args.use_mnrl,
    )

    logger.info("Done. Model saved to: %s", saved_path)


if __name__ == "__main__":
    main()
