from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_texts, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_texts,
                inputs="raw_reviews",
                outputs="clean_reviews",
                name="clean_texts_node",
            ),
            node(
                func=split_data,
                inputs=["clean_reviews", "params:test_size", "params:random_state"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="train_test_split_node",
            ),
        ]
    )
