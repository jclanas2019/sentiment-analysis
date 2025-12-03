from typing import Dict
from kedro.pipeline import Pipeline

from sentiment_analysis.pipelines import (
    data_preprocessing as dp,
    feature_engineering as fe,
    modeling as mdl,
)


def register_pipelines() -> Dict[str, Pipeline]:
    data_preprocessing = dp.create_pipeline()
    feature_engineering = fe.create_pipeline()
    modeling = mdl.create_pipeline()

    # pipeline completo
    full = data_preprocessing + feature_engineering + modeling

    return {
        "__default__": full,
        "data_preprocessing": data_preprocessing,
        "feature_engineering": feature_engineering,
        "modeling": modeling,
    }
