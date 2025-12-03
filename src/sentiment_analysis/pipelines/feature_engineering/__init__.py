from kedro.pipeline import Pipeline, node, pipeline
from .nodes import vectorize_text


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=vectorize_text,
                inputs=["X_train", "X_test", "params:max_features", "params:ngram_range"],
                outputs=["X_train_tfidf", "X_test_tfidf", "tfidf_vectorizer"],
                name="tfidf_vectorization_node",
            )
        ]
    )
