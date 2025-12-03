import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def train_model(X_train, y_train, C: float, max_iter: int):
    """Entrena el modelo base de sentimiento."""
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluación simple (la que ya tenías).
    Devuelve métricas globales.
    """
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1_weighted": f1,
        "classification_report": report,
    }
    return metrics


def build_detailed_report(model, X_test_tfidf, X_test, y_test):
    """
    Nodo de reporte muy completo:
    - Predicciones por fila
    - Marca aciertos/errores
    - Matriz de confusión
    - Métricas por clase
    - Resumen global

    Salidas:
    - detailed_predictions (DataFrame)
    - detailed_report (dict -> JSON)
    """
    # Predicciones
    y_pred = model.predict(X_test_tfidf)

    # DataFrame con detalle por fila
    df = pd.DataFrame(
        {
            "text": list(X_test),
            "true_sentiment": list(y_test),
            "predicted_sentiment": list(y_pred),
        }
    )
    df["is_correct"] = df["true_sentiment"] == df["predicted_sentiment"]

    # Labels ordenados
    labels = sorted(list({*df["true_sentiment"].unique(), *df["predicted_sentiment"].unique()}))

    # Métricas
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    detailed_report = {
        "summary": {
            "accuracy": acc,
            "f1_weighted": f1,
            "num_samples": int(len(df)),
            "num_classes": len(labels),
        },
        "labels": labels,
        "confusion_matrix": cm.tolist(),  # para poder serializar a JSON
        "classification_report": cls_report,
        "examples": {
            "first_5": df.head(5).to_dict(orient="records"),
            "errors_sample": df[~df["is_correct"]].head(5).to_dict(orient="records"),
        },
    }

    return df, detailed_report
