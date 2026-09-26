from __future__ import annotations

import random

import numpy as np
import torch

from anomaly_detection.utils.reproducibility import collect_environment, set_seed


def draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_gives_same_random_values() -> None:
    set_seed(123)
    first = draw()
    set_seed(123)
    assert draw() == first


def test_collect_environment_contains_versions() -> None:
    env = collect_environment()
    assert env["python"]
    for package in ("torch", "anomalib", "opencv-python-headless"):
        assert env["packages"][package]
