import pickle
import sys
from pathlib import Path
import numpy as np
from flask import Flask, render_template, request

# ------------------------------------------------------------------
# CONFIG: acceso a src/ para importar paquetes del proyecto Kedro
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# IMPORT DEL NODE DE PREPROCESAMIENTO
from sentiment_analysis.pipelines.data_preprocessing.nodes import _clean_text  # noqa: E402


# ------------------------------------------------------------------
# CARGA DE MODELO Y VECTORIZADOR
# ------------------------------------------------------------------
MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "data" / "04_feature" / "tfidf_vectorizer.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


# ------------------------------------------------------------------
# REFUERZO PROBABILÍSTICO CONTEXTUAL
# ------------------------------------------------------------------
NEGATIVE_POLARITY = {
    "rabia", "odio", "asco", "asqueroso", "horrible", "estafa",
    "molesto", "molesta", "maltratado", "peor",
    "defectuoso", "pésimo", "pesimo",
    "terrible", "vergonzoso", "nunca", "mal",
    "romper", "arruinado", "dañado", "lento", "malo",
}

POSITIVE_POLARITY = {
    "excelente", "increíble", "amable", "agradable", "solucionaron",
    "rápido", "fantástico", "perfecto", "funciona bien",
    "satisfecho", "maravilloso", "recomendable",
    "me encantó", "encanto", "buenísimo", "estupendo",
}


def contextual_reforce(raw_text: str, base_pred: str, proba: np.ndarray):
    """
    Ajuste probabilístico contextual:
    - No es binario
    - Mide densidad de polaridad en el texto
    - Mezcla contexto y predicción ML
    """

    txt = raw_text.lower()

    neg_hits = sum(1 for w in NEGATIVE_POLARITY if w in txt)
    pos_hits = sum(1 for w in POSITIVE_POLARITY if w in txt)
    total_hits = neg_hits + pos_hits

    # Si no aparecen señales, dejamos modelo intacto
    if total_hits == 0:
        return base_pred, float(max(proba))

    ctx_neg = neg_hits / total_hits
    ctx_pos = pos_hits / total_hits

    p_neg, p_neu, p_pos = proba

    # Peso del refuerzo contextual
    w = 0.30
    p_neg2 = (p_neg * (1 - w)) + (ctx_neg * w)
    p_pos2 = (p_pos * (1 - w)) + (ctx_pos * w)

    adjusted = {
        "negative": p_neg2,
        "neutral": p_neu,
        "positive": p_pos2,
    }

    best = max(adjusted, key=adjusted.get)
    confidence = adjusted[best]

    return best, float(confidence)


# ------------------------------------------------------------------
# Explicación simple al usuario
# ------------------------------------------------------------------
def build_explanation(prediction: str) -> str:
    if prediction == "positive":
        return "El texto expresa señales positivas o satisfacción."
    if prediction == "negative":
        return "El lenguaje expresa quejas, frustración o emociones negativas predominantes."
    return "No se detectan señales emocionales fuertes hacia positivo o negativo."


# ------------------------------------------------------------------
# Flask APP
# ------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    cleaned_text = None
    explanation = None
    raw_text = ""

    if request.method == "POST":
        raw_text = request.form.get("text", "")

        # limpieza consistente con el pipeline
        cleaned_text = _clean_text(raw_text)

        # vectorización
        X = vectorizer.transform([cleaned_text])

        proba = model.predict_proba(X)[0]
        base_pred = model.predict(X)[0]

        # refuerzo contextual
        prediction, confidence = contextual_reforce(
            raw_text, base_pred, proba
        )

        explanation = build_explanation(prediction)

    return render_template(
        "sentiment_form.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        raw_text=raw_text,
        cleaned_text=cleaned_text,
    )


if __name__ == "__main__":
    print("🔥 App de inferencia iniciada en http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
