import numpy as np

from ptx_classification.evaluation.evaluation_on_ptx_cohorts import (
    PTX_NEGATIVE_FILTERS,
    PTX_POSITIVE_FILTERS,
    DataPoint,
    perform_analyses,
)

rng = np.random.RandomState(seed=0)


def random_data_point() -> DataPoint:
    is_ptx = rng.random() < 0.2
    return DataPoint(
        ptx_groundtruth=is_ptx,
        ct_groundtruth=rng.random() < 0.1,
        ptx_size=f"{rng.randint(1, 3)}cm",
        predicted_ptx_probabilities={
            "model_a": (0.3 + rng.random() * 0.7) if is_ptx else (rng.random() * 0.7)
        },
    )


dps = [random_data_point() for _ in range(100)]


def test_perform_analyses() -> None:
    perform_analyses(
        data=dps,
        filters_down=PTX_NEGATIVE_FILTERS,
        filters_right=PTX_POSITIVE_FILTERS,
        show=True,
    )
