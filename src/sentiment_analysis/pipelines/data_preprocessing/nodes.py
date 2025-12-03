import re
import pandas as pd
from sklearn.model_selection import train_test_split


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # quitar URLs
    text = re.sub(r"[^a-záéíóúñü0-9\s]", "", text)  # dejar letras y números
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].astype(str).apply(_clean_text)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float,
    random_state: int
):
    X = df["text_clean"].tolist()
    y = df["sentiment"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
