from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from anomaly_detection.visualization.run_comparison import plot_run_comparison


def test_one_panel_per_metric_with_missing_values() -> None:
    summary = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "recall": [0.8, 1.0],
            "precision": [0.9, np.nan],
            "f1": [0.85, 0.9],
            "threshold": [38.8, 41.0],
        }
    )
    figure = plot_run_comparison(summary)
    assert isinstance(figure, Figure)
    assert [ax.get_title(loc="left") for ax in figure.axes] == [
        "recall",
        "precision",
        "f1",
        "threshold",
    ]
    precision_labels = [t.get_text() for t in figure.axes[1].texts]
    assert "N/A" in precision_labels
    assert figure.axes[0].get_xlim()[1] == pytest.approx(1.18)  # 率の指標は0〜1で揃える


def test_empty_summary() -> None:
    with pytest.raises(ValueError, match="Run"):
        plot_run_comparison(pd.DataFrame(columns=["run_id", "recall"]), metrics=("recall",))
