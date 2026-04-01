"""Boilerplate Classifier Training Pipeline.

Replaces the in-memory cross-company index (lexdrift.nlp.boilerplate) with
a trained binary classifier on sentence embeddings.

Architecture:
    sentence-transformers encodes each sentence -> 384-dim vector
    -> BoilerplateClassifier (2-layer MLP) -> sigmoid -> P(boilerplate)

Training data is self-generated from the database:
    - Sentences added by 3+ different companies = boilerplate (label 1)
    - Sentences unique to 1 company = substantive (label 0)

Usage:
    python -m lexdrift.training.boilerplate_classifier \\
        --model-path models/boilerplate_classifier.pt \\
        --epochs 10 \\
        --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_MODEL_PATH = "models/boilerplate_classifier.pt"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
EMBEDDING_DIM = 384

# Minimum number of distinct companies adding a sentence for it to
# be considered boilerplate.
BOILERPLATE_COMPANY_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Label generation (self-supervised from cross-company frequency)
# ---------------------------------------------------------------------------

def generate_boilerplate_labels(db_session: Session) -> list[dict]:
    """Query all 'added' SentenceChange records and label based on cross-company
    frequency.

    Sentences added by 3+ different companies are labeled as boilerplate (1).
    Sentences unique to exactly 1 company are labeled as substantive (0).

    Returns a list of dicts with keys: sentence_text, label, company_count.
    """
    from lexdrift.db.models import DriftScore, SentenceChange

    logger.info("Generating boilerplate labels from sentence changes in the database")

    stmt = (
        select(
            SentenceChange.sentence_text,
            DriftScore.company_id,
        )
        .join(DriftScore, SentenceChange.drift_score_id == DriftScore.id)
        .where(SentenceChange.change_type == "added")
    )
    rows = db_session.execute(stmt).all()
    logger.info("Fetched %d 'added' sentence change records", len(rows))

    # Group by normalized sentence text -> set of company IDs
    sentence_companies: dict[str, set[int]] = defaultdict(set)
    sentence_examples: dict[str, str] = {}  # normalized -> original text

    for row in rows:
        text = row.sentence_text
        if not text or len(text.strip()) < 20:
            continue

        # Normalize: lowercase + strip for grouping
        normalized = text.strip().lower()
        sentence_companies[normalized].add(row.company_id)
        # Keep the first original-cased version seen
        if normalized not in sentence_examples:
            sentence_examples[normalized] = text

    labeled_data: list[dict] = []

    for normalized, company_ids in sentence_companies.items():
        n_companies = len(company_ids)
        original_text = sentence_examples[normalized]

        if n_companies >= BOILERPLATE_COMPANY_THRESHOLD:
            # Boilerplate: added by many companies
            labeled_data.append({
                "sentence_text": original_text,
                "label": 1,
                "company_count": n_companies,
            })
        elif n_companies == 1:
            # Substantive: unique to one company
            labeled_data.append({
                "sentence_text": original_text,
                "label": 0,
                "company_count": n_companies,
            })
        # Sentences appearing in exactly 2 companies are ambiguous; skip them.

    # Log distribution
    n_boilerplate = sum(1 for d in labeled_data if d["label"] == 1)
    n_substantive = sum(1 for d in labeled_data if d["label"] == 0)
    logger.info(
        "Label distribution: boilerplate=%d, substantive=%d",
        n_boilerplate, n_substantive,
    )

    return labeled_data


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class BoilerplateClassifier(nn.Module):
    """Simple 2-layer MLP for binary boilerplate classification.

    Takes 384-dim sentence embeddings as input.
    Outputs a single probability: P(boilerplate).
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_boilerplate_classifier(
    training_data: list[dict],
    model_path: str = DEFAULT_MODEL_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> str:
    """Train the boilerplate classifier MLP on sentence embeddings.

    Steps:
        1. Encode all sentences with the sentence-transformer.
        2. Split into train/test sets.
        3. Train with BCELoss.
        4. Print classification report.
        5. Save model to model_path.

    Returns the path where the model was saved.
    """
    if not training_data:
        raise ValueError("No training data provided; cannot train classifier")

    from lexdrift.nlp.embeddings import _get_model

    # Encode sentences
    logger.info("Encoding %d sentences with sentence-transformer...", len(training_data))
    st_model = _get_model()
    sentences = [d["sentence_text"] for d in training_data]
    labels = [d["label"] for d in training_data]

    embeddings = st_model.encode(sentences, show_progress_bar=True, batch_size=batch_size)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels,
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Build model
    input_dim = X_train.shape[1]
    model = BoilerplateClassifier(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(
        "Training BoilerplateClassifier: %d train / %d test, %d epochs, batch_size=%d",
        len(X_train), len(X_test), epochs, batch_size,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)
        y_test_np = y_test_t.numpy().flatten().astype(int)

    report = classification_report(
        y_test_np, test_preds,
        target_names=["substantive", "boilerplate"],
        zero_division=0,
    )
    logger.info("Classification Report:\n%s", report)

    # Save model
    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
    }, str(out_path))
    logger.info("Model saved to %s", out_path)

    return str(out_path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_boilerplate_classifier(
    model_path: str = DEFAULT_MODEL_PATH,
) -> BoilerplateClassifier:
    """Load a trained BoilerplateClassifier from disk."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = BoilerplateClassifier(
        input_dim=checkpoint.get("input_dim", EMBEDDING_DIM),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded BoilerplateClassifier from %s", model_path)
    return model


def predict_boilerplate(
    sentences: list[str],
    model_path: str = DEFAULT_MODEL_PATH,
) -> list[dict]:
    """Encode sentences and predict boilerplate probability.

    Returns a list of dicts with keys: sentence, boilerplate_probability,
    is_boilerplate.
    """
    if not sentences:
        return []

    from lexdrift.nlp.embeddings import _get_model

    # Encode
    st_model = _get_model()
    embeddings = st_model.encode(sentences, show_progress_bar=False, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)
    X = torch.tensor(embeddings, dtype=torch.float32)

    # Load classifier and predict
    classifier = load_boilerplate_classifier(model_path)
    with torch.no_grad():
        probs = classifier(X).numpy().flatten()

    results = []
    for i, sentence in enumerate(sentences):
        prob = float(probs[i])
        results.append({
            "sentence": sentence,
            "boilerplate_probability": round(prob, 4),
            "is_boilerplate": prob >= 0.5,
        })

    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a boilerplate classifier on sentence embeddings",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to save the trained model (default: {DEFAULT_MODEL_PATH})",
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
        "--database-url",
        default=None,
        help="Override the database URL from config",
    )
    parser.add_argument(
        "--company-threshold",
        type=int,
        default=BOILERPLATE_COMPANY_THRESHOLD,
        help=f"Min companies for boilerplate label (default: {BOILERPLATE_COMPANY_THRESHOLD})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full training pipeline: generate labels, train, save."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args(argv)

    # Allow overriding the threshold from CLI
    global BOILERPLATE_COMPANY_THRESHOLD
    BOILERPLATE_COMPANY_THRESHOLD = args.company_threshold

    # Set up synchronous DB session
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from lexdrift.config import settings

    db_url = args.database_url or settings.database_url
    sync_url = db_url.replace("+aiosqlite", "").replace("+aiomysql", "+pymysql")

    logger.info("Connecting to database: %s", sync_url.split("@")[-1])
    engine = create_engine(sync_url, echo=False)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        training_data = generate_boilerplate_labels(session)

    if not training_data:
        logger.error(
            "No training data generated. Ensure the analysis pipeline has been "
            "run with multiple companies (sentence_changes table must be populated)."
        )
        sys.exit(1)

    logger.info("Total training samples: %d", len(training_data))

    saved_path = train_boilerplate_classifier(
        training_data,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    logger.info("Done. Model saved to: %s", saved_path)


if __name__ == "__main__":
    main()
