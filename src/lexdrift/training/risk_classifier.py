"""Risk Severity Classifier Training Pipeline.

Replaces the keyword-based risk scoring (lexdrift.nlp.risk) with a trained
MLP classifier that operates on sentence embeddings.

Architecture:
    sentence-transformers encodes each sentence -> 384-dim vector
    -> RiskClassifier (2-layer MLP) -> 4 classes: critical, high, medium, low

The training data is bootstrapped from the existing keyword-based system:
we run score_sentence_risk() on every SentenceChange in the database and
use the resulting labels to train the classifier.

Usage:
    python -m lexdrift.training.risk_classifier \\
        --model-path models/risk_classifier.pt \\
        --epochs 10 \\
        --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Risk level -> integer label mapping
RISK_LABELS = {"critical": 0, "high": 1, "medium": 2, "low": 3}
LABEL_NAMES = {v: k for k, v in RISK_LABELS.items()}
NUM_CLASSES = len(RISK_LABELS)

# Defaults
DEFAULT_MODEL_PATH = "models/risk_classifier.pt"
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Label generation (bootstrap from keyword-based system)
# ---------------------------------------------------------------------------

def generate_risk_labels(db_session: Session) -> list[dict]:
    """Query SentenceChange table and label each sentence using the existing
    keyword-based score_sentence_risk() function.

    Returns a list of dicts with keys: sentence_text, label, risk_level, risk_score.
    """
    from lexdrift.db.models import SentenceChange
    from lexdrift.nlp.risk import score_sentence_risk

    logger.info("Generating risk labels from sentence changes in the database")

    stmt = select(
        SentenceChange.sentence_text,
        SentenceChange.change_type,
    )
    rows = db_session.execute(stmt).all()
    logger.info("Fetched %d sentence change records", len(rows))

    labeled_data: list[dict] = []
    for row in rows:
        text = row.sentence_text
        if not text or len(text.strip()) < 20:
            continue

        risk = score_sentence_risk(text)

        # Map the keyword-based level to our 4-class scheme.
        # "boilerplate" from the keyword system maps to "low" for training.
        level = risk.level if risk.level in RISK_LABELS else "low"
        label = RISK_LABELS[level]

        labeled_data.append({
            "sentence_text": text,
            "label": label,
            "risk_level": level,
            "risk_score": risk.score,
        })

    # Log class distribution
    from collections import Counter
    dist = Counter(d["risk_level"] for d in labeled_data)
    logger.info("Label distribution: %s", dict(dist))

    return labeled_data


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class RiskClassifier(nn.Module):
    """Simple 2-layer MLP for risk severity classification.

    Takes 384-dim sentence embeddings as input.
    Outputs probabilities over 4 classes: critical, high, medium, low.
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_risk_classifier(
    training_data: list[dict],
    model_path: str = DEFAULT_MODEL_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> str:
    """Train the risk classifier MLP on sentence embeddings.

    Steps:
        1. Encode all sentences with the sentence-transformer.
        2. Split into train/test sets.
        3. Train with CrossEntropyLoss.
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
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    # Build model
    input_dim = X_train.shape[1]
    model = RiskClassifier(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(
        "Training RiskClassifier: %d train / %d test, %d epochs, batch_size=%d",
        len(X_train), len(X_test), epochs, batch_size,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            # Forward pass: model returns softmax probs, but CrossEntropyLoss
            # expects logits. We extract the logits before softmax.
            logits = model.net(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_logits = model.net(X_test_t)
        test_preds = torch.argmax(test_logits, dim=1).numpy()
        y_test_np = y_test_t.numpy()

    label_names = [LABEL_NAMES[i] for i in range(NUM_CLASSES)]
    report = classification_report(y_test_np, test_preds, target_names=label_names, zero_division=0)
    logger.info("Classification Report:\n%s", report)

    # Save model
    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_classes": NUM_CLASSES,
        "label_map": RISK_LABELS,
    }, str(out_path))
    logger.info("Model saved to %s", out_path)

    return str(out_path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_risk_classifier(model_path: str = DEFAULT_MODEL_PATH) -> RiskClassifier:
    """Load a trained RiskClassifier from disk."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = RiskClassifier(
        input_dim=checkpoint.get("input_dim", EMBEDDING_DIM),
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded RiskClassifier from %s", model_path)
    return model


def predict_risk(
    sentences: list[str],
    model_path: str = DEFAULT_MODEL_PATH,
) -> list[dict]:
    """Encode sentences and predict risk severity using the trained classifier.

    Returns a list of dicts with keys: sentence, predicted_level,
    confidence, probabilities.
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
    classifier = load_risk_classifier(model_path)
    with torch.no_grad():
        probs = classifier(X).numpy()

    results = []
    for i, sentence in enumerate(sentences):
        pred_label = int(np.argmax(probs[i]))
        results.append({
            "sentence": sentence,
            "predicted_level": LABEL_NAMES[pred_label],
            "confidence": round(float(probs[i][pred_label]), 4),
            "probabilities": {
                LABEL_NAMES[j]: round(float(probs[i][j]), 4)
                for j in range(NUM_CLASSES)
            },
        })

    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a risk severity classifier on sentence embeddings",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full training pipeline: generate labels, train, save."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args(argv)

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
        training_data = generate_risk_labels(session)

    if not training_data:
        logger.error(
            "No training data generated. Ensure the analysis pipeline has been "
            "run (sentence_changes table must be populated)."
        )
        sys.exit(1)

    logger.info("Total training samples: %d", len(training_data))

    saved_path = train_risk_classifier(
        training_data,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    logger.info("Done. Model saved to: %s", saved_path)


if __name__ == "__main__":
    main()
