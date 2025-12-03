# 🧠 Sentiment Analysis — Proyecto Kedro

Este proyecto implementa un flujo completo de análisis de sentimientos basado en Machine Learning usando Kedro.
Incluye tres pipelines independientes: preprocesamiento, extracción de características y modelado.

---

## 📁 Dataset

Formato esperado (reviews.csv):

id,text,sentiment
1,"Me encantó el producto","positive"
2,"Es terrible, nunca más compro","negative"
3,"Está bien, podría mejorar","neutral"

Ubicación del archivo:

data/01_raw/reviews.csv

---

## 🚀 Instalación

### Crear entorno virtual
Linux/Mac:
python -m venv venv
source venv/bin/activate

Windows:
python -m venv venv
venv\Scripts\activate

### Instalar dependencias
pip install kedro==0.19.1 pandas scikit-learn

---

# 🔧 Crear proyecto Kedro

kedro new

Valores sugeridos:
- Project Name: Sentiment Analysis
- Repository Name: sentiment-analysis
- Python Package: sentiment_analysis
- Example pipeline: n

Luego:
cd sentiment-analysis
kedro install

---

```
## Comandos útiles de Kedro para el proyecto `sentiment-analysis`

### 1. Inicializar entorno del proyecto
```bash
cd /Users/juancarloslanasocampo/Documents/2-lab-ai/workshop-kedro/sentiment-analysis
```

### 2. Ejecutar **todo** el pipeline

```bash
kedro run
```

### 3. Ejecutar solo desde un nodo específico

(útil para re-probar una parte)

```bash
kedro run --from-nodes tfidf_vectorization_node
kedro run --from-nodes train_model_node
kedro run --from-nodes interactive_html_node
```

### 4. Ver los pipelines y nodos disponibles

```bash
kedro pipeline list
kedro run --node-names
```

### 5. Limpiar datos generados (outputs)

```bash
kedro run --tags reset  # solo si definiste tags de reseteo
# o manualmente:
rm -rf data/03_primary data/04_feature data/06_models data/08_reporting data/09_ui
```

### 6. Ejecutar tests del proyecto

```bash
pytest
```

### 7. Mostrar versión de Kedro y diagnosticar entorno

```bash
kedro info
```

```
::contentReference[oaicite:0]{index=0}
```


# ⚙️ Configuración

## conf/base/parameters.yml
test_size: 0.2
random_state: 42
max_features: 5000
ngram_range: [1, 2]
C: 1.0
max_iter: 1000

---

## conf/base/catalog.yml
raw_reviews:
  type: pandas.CSVDataSet
  filepath: data/01_raw/reviews.csv

clean_reviews:
  type: pandas.CSVDataSet
  filepath: data/02_intermediate/clean_reviews.csv

X_train:
  type: pickle.PickleDataSet
  filepath: data/03_primary/X_train.pkl

X_test:
  type: pickle.PickleDataSet
  filepath: data/03_primary/X_test.pkl

y_train:
  type: pickle.PickleDataSet
  filepath: data/03_primary/y_train.pkl

y_test:
  type: pickle.PickleDataSet
  filepath: data/03_primary/y_test.pkl

tfidf_vectorizer:
  type: pickle.PickleDataSet
  filepath: data/04_feature/tfidf_vectorizer.pkl

X_train_tfidf:
  type: pickle.PickleDataSet
  filepath: data/04_feature/X_train_tfidf.pkl

X_test_tfidf:
  type: pickle.PickleDataSet
  filepath: data/04_feature/X_test_tfidf.pkl

sentiment_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/sentiment_model.pkl

metrics:
  type: json.JSONDataSet
  filepath: data/08_reporting/metrics.json

---

# 🔁 Pipeline 1 — data_preprocessing

src/sentiment_analysis/pipelines/data_preprocessing/nodes.py

import re
import pandas as pd
from sklearn.model_selection import train_test_split

def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-záéíóúñü0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].astype(str).apply(_clean_text)
    return df

def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    X = df["text_clean"].tolist()
    y = df["sentiment"].tolist()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

---

## Pipeline 2 — feature_engineering

src/sentiment_analysis/pipelines/feature_engineering/nodes.py

from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_text(X_train, X_test, max_features, ngram_range):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    return vectorizer.fit_transform(X_train), vectorizer.transform(X_test), vectorizer

---

## Pipeline 3 — modeling

src/sentiment_analysis/pipelines/modeling/nodes.py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

def train_model(X_train, y_train, C, max_iter):
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test,preds),
        "f1_weighted": f1_score(y_test,preds,average="weighted"),
        "report": classification_report(y_test,preds,output_dict=True)
    }

---

# ▶️ Pipeline completo

kedro run

---

# 🧩 Ejecución por pipeline

kedro run --pipeline data_preprocessing
kedro run --pipeline feature_engineering
kedro run --pipeline modeling

---

# 📊 Visualización

pip install kedro-viz
kedro viz

---

# 📦 Artefactos

Modelo:
data/06_models/sentiment_model.pkl

Métricas:
data/08_reporting/metrics.json

Datos limpios:
data/02_intermediate/clean_reviews.csv

---

# 📚 Expansión futura

- Pipeline de inferencia
- MLflow / W&B
- Transformers
- API REST o batch jobs

---

# 📄 Licencia
MIT o Apache 2.0
