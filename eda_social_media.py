"""Beginner-friendly EDA for a social media sentiment CSV.

This workspace contains the Sentiment140-style file:
  data/training.1600000.processed.noemoticon.csv

That file has *no header row* and 6 columns:
  sentiment, id, timestamp, query, user, text

If your CSV already has headers like: text, sentiment, timestamp
you can set HAS_HEADER=True and adjust the COLUMN_* names below.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd

# NLTK is used for stopwords, tokenization, and lemmatization.
# If you get an NLTK LookupError, the script will try to download the needed data.
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# --- Configuration (edit these if your file/columns differ) ---
CSV_PATH = Path("data") / "training.1600000.processed.noemoticon.csv"

# Set to a number like 50_000 to load faster while experimenting.
# Keep as None to load the full dataset.
NROWS: int | None = None

# This specific dataset has NO header row.
HAS_HEADER = False

# Column names used by this dataset when HAS_HEADER=False
COLUMN_SENTIMENT = "sentiment"
COLUMN_ID = "id"
COLUMN_TIMESTAMP = "timestamp"
COLUMN_QUERY = "query"
COLUMN_USER = "user"
COLUMN_TEXT = "text"

# NLP preprocessing output column
COLUMN_CLEAN_TEXT = "clean_text"

# Set to False if you only want EDA (no text cleaning).
RUN_NLP_PREPROCESSING = True


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load the dataset from disk into a pandas DataFrame."""

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    if HAS_HEADER:
        # If your CSV already has column names in the first row.
        df = pd.read_csv(csv_path, nrows=NROWS)
        return df

    # Sentiment140 uses latin-1 and has quoted, comma-separated fields.
    # We pass column names since the file has no header row.
    names = [
        COLUMN_SENTIMENT,
        COLUMN_ID,
        COLUMN_TIMESTAMP,
        COLUMN_QUERY,
        COLUMN_USER,
        COLUMN_TEXT,
    ]

    df = pd.read_csv(
        csv_path,
        header=None,
        names=names,
        encoding="latin-1",
        nrows=NROWS,
        low_memory=False,
    )

    return df


def _ensure_nltk_data() -> None:
    """Ensure required NLTK datasets are available.

    NLTK ships code separately from its language datasets.
    This helper checks and downloads the minimum we need:
    - Tokenizer models (punkt)
    - English stopwords list
    - WordNet (for lemmatization)
    """

    required = [
        ("tokenizers/punkt", "punkt"),
        # Newer NLTK versions may also require 'punkt_tab' for sentence tokenization.
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, package_name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            # Download quietly; if you are offline, this may fail.
            nltk.download(package_name, quiet=True)


# Regex patterns used during cleaning
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# This covers most emoji/unicode pictographs used on social media.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-c
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols, etc.
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "\U00002700-\U000027BF"  # dingbats
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "]+",
    flags=re.UNICODE,
)

# Replace punctuation with spaces, but keep letters/numbers/whitespace.
_PUNCT_RE = re.compile(r"[^a-z0-9\s]+", flags=re.IGNORECASE)


def preprocess_text(
    text: str,
    *,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
) -> str:
    """Clean a single text string and return a normalized version.

    Steps (kept simple for beginners):
    1) Lowercase
    2) Remove URLs
    3) Remove emojis
    4) Remove punctuation
    5) Tokenize into words
    6) Remove stopwords
    7) Lemmatize (convert words to their base form)
    """

    # 1) Lowercasing
    text = str(text).lower()

    # 2) Remove URLs (links usually aren't helpful for basic sentiment models)
    text = _URL_RE.sub(" ", text)

    # 3) Remove emojis (optional, but requested)
    text = _EMOJI_RE.sub(" ", text)

    # 4) Remove punctuation by keeping only letters/numbers/whitespace
    text = _PUNCT_RE.sub(" ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 5) Tokenization: split text into word tokens
    tokens = word_tokenize(text)

    # 6) Stopword removal: drop common words like "the", "and", "is"
    tokens = [t for t in tokens if t not in stop_words]

    # 7) Lemmatization: normalize tokens to their base form
    # Note: WordNet lemmatizer is more accurate with POS tags; this keeps it simple.
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def add_clean_text_column(
    df: pd.DataFrame,
    *,
    text_col: str,
    output_col: str = COLUMN_CLEAN_TEXT,
) -> pd.DataFrame:
    """Apply NLP preprocessing to `text_col` and create `output_col`."""

    if text_col not in df.columns:
        raise KeyError(f"Text column not found: {text_col!r}. Available: {list(df.columns)}")

    _ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Fill missing values so `.apply()` doesn't crash
    df[text_col] = df[text_col].fillna("")

    print(f"\nCreating '{output_col}' with NLP preprocessing...")
    df[output_col] = df[text_col].apply(
        lambda x: preprocess_text(x, stop_words=stop_words, lemmatizer=lemmatizer)
    )

    print("\n--- Clean text examples (original -> clean_text) ---")
    preview = df[[text_col, output_col]].head(3)
    for i, row in preview.iterrows():
        print(f"\nRow {i}:")
        print("ORIGINAL:", row[text_col])
        print("CLEAN   :", row[output_col])

    return df


def show_basic_info(df: pd.DataFrame) -> None:
    """Print basic information about the dataset."""

    print("\n--- Head (first 5 rows) ---")
    print(df.head())

    print("\n--- Shape (rows, columns) ---")
    print(df.shape)

    print("\n--- Columns ---")
    print(list(df.columns))

    print("\n--- Info (dtypes + non-null counts) ---")
    df.info()


def handle_missing_values(
    df: pd.DataFrame,
    *,
    text_col: str,
    sentiment_col: str,
) -> pd.DataFrame:
    """Simple missing-value handling.

    Strategy (beginner-friendly):
    - Fill missing text with an empty string
    - Drop rows where sentiment is missing
    """

    print("\n--- Missing values per column ---")
    print(df.isna().sum().sort_values(ascending=False))

    # Fill missing text with empty strings (safe for simple text analysis)
    if text_col in df.columns:
        df[text_col] = df[text_col].fillna("")

    # Drop rows with missing sentiment (you can't analyze sentiment without labels)
    if sentiment_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[sentiment_col])
        after = len(df)
        print(f"\nDropped {before - after:,} rows with missing sentiment")

    return df


def add_useful_columns(
    df: pd.DataFrame,
    *,
    text_col: str,
    sentiment_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Add a few helper columns for EDA."""

    # Convert sentiment to numeric if possible (helps sorting/plotting)
    if sentiment_col in df.columns:
        df[sentiment_col] = pd.to_numeric(df[sentiment_col], errors="coerce")

        # Optional: map common Sentiment140 labels (0=negative, 4=positive)
        sentiment_map = {0: "negative", 2: "neutral", 4: "positive"}
        df["sentiment_label"] = df[sentiment_col].map(sentiment_map).fillna(
            df[sentiment_col].astype("Int64").astype(str)
        )

    # Parse timestamp if present.
    # Sentiment140 timestamps look like: "Mon Apr 06 22:19:45 PDT 2009".
    # Pandas may not recognize timezone abbreviations (e.g., PDT), so we remove them.
    if timestamp_col in df.columns:
        raw_ts = df[timestamp_col].astype(str).str.strip()

        # Detect Sentiment140-style strings and parse them with a known format.
        sent140_like = raw_ts.str.match(
            r"^[A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} [A-Za-z]{2,5} \d{4}$"
        )

        if sent140_like.any():
            # Drop the timezone token before the year ("... PDT 2009" -> "... 2009")
            cleaned = raw_ts.where(
                ~sent140_like,
                raw_ts.str.replace(r"\s[A-Za-z]{2,5}\s(\d{4})$", r" \1", regex=True),
            )
            parsed = pd.to_datetime(
                cleaned,
                format="%a %b %d %H:%M:%S %Y",
                errors="coerce",
            )
        else:
            # Generic fallback for other timestamp formats (ISO, etc.)
            parsed = pd.to_datetime(raw_ts, errors="coerce")

        df["timestamp_parsed"] = parsed

    # Text length is a simple, useful numeric feature
    if text_col in df.columns:
        df["text_length"] = df[text_col].astype(str).str.len()

    return df


def sentiment_distribution(df: pd.DataFrame, *, label_col: str) -> pd.Series:
    """Return counts for each sentiment label."""

    counts = df[label_col].value_counts(dropna=False)

    print("\n--- Sentiment distribution ---")
    print(counts)

    return counts


def plot_sentiment_counts(counts: pd.Series) -> None:
    """Bar chart of sentiment counts."""

    plt.figure(figsize=(8, 4))
    counts.sort_index().plot(kind="bar")
    plt.title("Sentiment distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of posts")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_text_length_histogram(df: pd.DataFrame, *, length_col: str) -> None:
    """Histogram of text lengths."""

    if length_col not in df.columns:
        return

    plt.figure(figsize=(8, 4))
    plt.hist(df[length_col], bins=50)
    plt.title("Text length distribution")
    plt.xlabel("Characters per post")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_posts_over_time(df: pd.DataFrame, *, ts_col: str) -> None:
    """Optional: line plot of post volume over time (if timestamps parse)."""

    if ts_col not in df.columns:
        return

    ts = df[ts_col].dropna()
    if ts.empty:
        print("\nTimestamp parsing produced no valid dates; skipping time plot.")
        return

    # Group by day and plot counts
    per_day = ts.dt.floor("D").value_counts().sort_index()

    plt.figure(figsize=(10, 4))
    plt.plot(per_day.index, per_day.values)
    plt.title("Posts per day")
    plt.xlabel("Date")
    plt.ylabel("Number of posts")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # 1) Load dataset using pandas
    df = load_dataset(CSV_PATH)

    # 2) Show basic info (head, shape, columns)
    show_basic_info(df)

    # 3) Handle missing values
    df = handle_missing_values(df, text_col=COLUMN_TEXT, sentiment_col=COLUMN_SENTIMENT)

    # NLP preprocessing: create a new column named "clean_text" from the text column
    if RUN_NLP_PREPROCESSING:
        df = add_clean_text_column(df, text_col=COLUMN_TEXT, output_col=COLUMN_CLEAN_TEXT)

    # Add a few helper columns for EDA
    df = add_useful_columns(
        df,
        text_col=COLUMN_TEXT,
        sentiment_col=COLUMN_SENTIMENT,
        timestamp_col=COLUMN_TIMESTAMP,
    )

    # 4) Perform basic EDA (distribution of sentiment)
    label_col = "sentiment_label" if "sentiment_label" in df.columns else COLUMN_SENTIMENT
    counts = sentiment_distribution(df, label_col=label_col)

    # 5) Plot graphs using matplotlib
    plot_sentiment_counts(counts)
    plot_text_length_histogram(df, length_col="text_length")

    # Optional time plot (only if timestamps parse successfully)
    plot_posts_over_time(df, ts_col="timestamp_parsed")


if __name__ == "__main__":
    main()
