from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, evaluate_model, build_detailed_report
from .nodes_html import build_interactive_html


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["X_train_tfidf", "y_train", "params:C", "params:max_iter"],
                outputs="sentiment_model",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["sentiment_model", "X_test_tfidf", "y_test"],
                outputs="metrics",
                name="evaluate_model_node",
            ),
            node(
                func=build_detailed_report,
                inputs=["sentiment_model", "X_test_tfidf", "X_test", "y_test"],
                outputs=["detailed_predictions", "detailed_report"],
                name="build_detailed_report_node",
            ),
            node(
			    func=build_interactive_html,
			    inputs="detailed_predictions",
			    outputs="sentiment_report_html",
			    name="interactive_html_node",
			),
        ]
    )
