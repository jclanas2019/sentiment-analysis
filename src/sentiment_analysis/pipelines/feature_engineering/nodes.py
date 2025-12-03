from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer


def _normalize_ngram_range(ngram_range: Any) -> tuple[int, int]:
    """
    Asegura que ngram_range sea una tupla (min_n, max_n),
    venga como lista, tupla o string tipo "(1, 2)" o "1,2".
    """
    # Si ya es tupla de dos enteros, lo usamos tal cual
    if isinstance(ngram_range, tuple) and len(ngram_range) == 2:
        return ngram_range

    # Si es lista, la convertimos a tupla
    if isinstance(ngram_range, list) and len(ngram_range) == 2:
        return tuple(int(x) for x in ngram_range)

    # Si viene como string, lo parseamos
    if isinstance(ngram_range, str):
        cleaned = (
            ngram_range.replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .strip()
        )
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"ngram_range string inválido: {ngram_range!r}")
        return tuple(int(x) for x in parts)

    # Cualquier otra cosa: error explícito
    raise TypeError(
        f"ngram_range debe ser lista, tupla o string parseable, pero llegó {type(ngram_range)} = {ngram_range!r}"
    )


def vectorize_text(X_train, X_test, max_features, ngram_range):
    # Normalizamos siempre a tupla (min_n, max_n)
    ngram_range_tuple = _normalize_ngram_range(ngram_range)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range_tuple,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer
