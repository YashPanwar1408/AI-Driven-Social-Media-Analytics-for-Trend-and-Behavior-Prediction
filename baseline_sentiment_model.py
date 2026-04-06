"""Baseline sentiment analysis model (TF-IDF + Logistic Regression).

What this script does:
1) Load the CSV dataset with pandas
2) (Optional) Create a cleaned text column called `clean_text`
3) Split data into train/test
4) Convert text to TF-IDF features
5) Train a Logistic Regression classifier
6) Evaluate with accuracy, precision, recall, F1 + classification report

Notes for your dataset:
- The file in this workspace is Sentiment140-style with NO header and 6 columns.
- Sentiment labels are typically 0=negative and 4=positive.

You can start small by setting NROWS to e.g. 50_000 for faster iteration.
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

# Scikit-learn for TF-IDF, model training, and evaluation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# NLTK is used only if you choose to create a cleaned text column.
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --- Configuration (edit as needed) ---
CSV_PATH = Path("data") / "training.1600000.processed.noemoticon.csv"

# The Sentiment140 training file is typically ordered:
# - first ~800,000 rows are negative (label 0)
# - next ~800,000 rows are positive (label 4)
#
# If you load only the "first N rows", you may get ONLY negatives.
# To avoid that, we load a small balanced sample by default.
SAMPLE_SIZE: int | None = 100_000  # set to None to load the full dataset (slow/big)
SENTIMENT140_POSITIVE_START_ROW = 800_000

# Dataset column names (Sentiment140-style)
COL_SENTIMENT = "sentiment"
COL_TEXT = "text"

# Whether to create and use a cleaned text column (clean_text).
# If False, the model will train directly on the raw `text` column.
CREATE_CLEAN_TEXT = True
CLEAN_TEXT_COL = "clean_text"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF settings (keep it simple for a baseline)
MAX_FEATURES = 50_000

# Logistic Regression settings
# - 'liblinear' is a good simple default for binary text classification.
# - Increase max_iter if you see a ConvergenceWarning.
LOGREG_MAX_ITER = 1000


def _read_sentiment140_rows(
    csv_path: Path,
    *,
    nrows: int | None = None,
    skiprows: int = 0,
) -> pd.DataFrame:
    """Read rows from a Sentiment140-style CSV (no header, 6 columns)."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    names = ["sentiment", "id", "timestamp", "query", "user", "text"]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=names,
        encoding="latin-1",
        nrows=nrows,
        skiprows=skiprows,
        low_memory=False,
    )

    return df


def load_sentiment140_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the dataset.

    By default this returns a *balanced sample* (half negative + half positive)
    so you can train a baseline model quickly.

    Set SAMPLE_SIZE=None to load the full dataset.
    """

    if SAMPLE_SIZE is None:
        return _read_sentiment140_rows(csv_path)

    # Balanced sample: take half negatives from the start
    # and half positives from the known positive section.
    neg_n = SAMPLE_SIZE // 2
    pos_n = SAMPLE_SIZE - neg_n

    df_neg = _read_sentiment140_rows(csv_path, nrows=neg_n, skiprows=0)
    df_pos = _read_sentiment140_rows(
        csv_path,
        nrows=pos_n,
        skiprows=SENTIMENT140_POSITIVE_START_ROW,
    )

    df = pd.concat([df_neg, df_pos], ignore_index=True)

    # Shuffle so train/test split isn't impacted by original ordering.
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Basic missing-value handling for modeling."""

    # Fill missing text so vectorizers/tokenizers don't crash.
    df[COL_TEXT] = df[COL_TEXT].fillna("")

    # Drop rows with missing sentiment.
    df = df.dropna(subset=[COL_SENTIMENT])

    return df


def prepare_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare labels for training.

    For Sentiment140, labels are often {0, 4}:
    - 0 -> 0 (negative)
    - 4 -> 1 (positive)

    If other labels exist, we keep them as-is (multiclass).
    """

    df[COL_SENTIMENT] = pd.to_numeric(df[COL_SENTIMENT], errors="coerce")
    df = df.dropna(subset=[COL_SENTIMENT]).copy()

    unique_labels = set(df[COL_SENTIMENT].unique().tolist())

    if unique_labels.issubset({0, 4}):
        y = (df[COL_SENTIMENT] == 4).astype(int)
    else:
        # Multiclass case (e.g., 0/2/4). Keep numeric labels.
        y = df[COL_SENTIMENT].astype(int)

    return df, y


def _ensure_nltk_data() -> None:
    """Download/check the NLTK datasets needed for preprocessing."""

    required = [
        ("tokenizers/punkt", "punkt"),
        # Newer NLTK versions may require this extra resource.
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, package_name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package_name, quiet=True)


# --- Text cleaning helpers (used only when CREATE_CLEAN_TEXT=True) ---
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# Emoji/unicode pictograph blocks commonly found in social media posts.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "]+",
    flags=re.UNICODE,
 )

# Keep only letters/numbers/whitespace; replace the rest with spaces.
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+", flags=re.IGNORECASE)


def preprocess_text(
    text: str,
    *,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
) -> str:
    """Basic NLP preprocessing for a single string.

    Each step is intentionally simple and beginner-friendly:

    1) Lowercasing
       - Normalizes text so "Happy" and "happy" are treated the same.

    2) Remove URLs
       - Links usually don't add much semantic value for a baseline model.

    3) Remove emojis
       - Requested; emojis can be informative, but we remove them here.

    4) Remove punctuation
       - Simplifies tokens (e.g., removes commas, periods, etc.).

    5) Tokenization
       - Split into word tokens.

    6) Stopword removal
       - Remove very common words like "the", "and", "is".

    7) Lemmatization
       - Convert words to their base form (e.g., "cars" -> "car").
    """

    # 1) Lowercase
    text = str(text).lower()

    # 2) Remove URLs
    text = _URL_RE.sub(" ", text)

    # 3) Remove emojis
    text = _EMOJI_RE.sub(" ", text)

    # 4) Remove punctuation
    text = _PUNCT_RE.sub(" ", text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 5) Tokenize
    tokens = word_tokenize(text)

    # 6) Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # 7) Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def add_clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a new `clean_text` column by applying preprocessing to `text`."""

    _ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df[COL_TEXT] = df[COL_TEXT].fillna("")

    print(f"Creating '{CLEAN_TEXT_COL}'...")
    df[CLEAN_TEXT_COL] = df[COL_TEXT].apply(
        lambda x: preprocess_text(x, stop_words=stop_words, lemmatizer=lemmatizer)
    )

    return df


def build_model() -> Pipeline:
    """Build a simple TF-IDF + Logistic Regression pipeline."""

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=MAX_FEATURES,
                    # Since we may already lowercase in preprocessing,
                    # leaving lowercase=True is fine either way.
                    lowercase=True,
                ),
            ),
            (
                "logreg",
                LogisticRegression(
                    max_iter=LOGREG_MAX_ITER,
                    # 'saga' works well with sparse TF-IDF features and supports multiclass.
                    solver="saga",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


def evaluate(y_true, y_pred) -> None:
    """Print required evaluation metrics."""

    accuracy = accuracy_score(y_true, y_pred)

    # Choose averaging strategy depending on binary vs multiclass.
    unique = sorted(set(pd.Series(y_true).unique().tolist()))
    is_binary = len(unique) == 2

    if is_binary:
        precision = precision_score(y_true, y_pred, average="binary", pos_label=1)
        recall = recall_score(y_true, y_pred, average="binary", pos_label=1)
        f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    else:
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n--- Evaluation ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n--- Classification report ---")
    print(classification_report(y_true, y_pred, zero_division=0))


def main() -> None:
    # 1) Load dataset using pandas
    df = load_sentiment140_dataset(CSV_PATH)
    print("Loaded rows:", len(df))

    # 2) Handle missing values
    df = handle_missing_values(df)

    # 3) Prepare labels (convert Sentiment140 {0,4} -> {0,1})
    df, y = prepare_labels(df)

    # Sanity check: you need at least 2 classes to train a classifier.
    label_counts = pd.Series(y).value_counts()
    print("\nLabel distribution:")
    print(label_counts)
    if label_counts.shape[0] < 2:
        raise ValueError(
            "Only one class found in the loaded data. "
            "If you set SAMPLE_SIZE, increase it or set SAMPLE_SIZE=None to load more rows."
        )

    # 4) Choose which text column to use
    if CREATE_CLEAN_TEXT:
        df = add_clean_text_column(df)
        X = df[CLEAN_TEXT_COL]
    else:
        X = df[COL_TEXT]

    # 5) Split data (train/test)
    # Stratify keeps the same class distribution in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # 6) Convert text to TF-IDF + train Logistic Regression
    model = build_model()
    model.fit(X_train, y_train)

    # 7) Predict and evaluate
    y_pred = model.predict(X_test)
    evaluate(y_test, y_pred)


if __name__ == "__main__":
    main()
