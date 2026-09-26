from __future__ import annotations

import logging

import pytest
import torch

from anomaly_detection.utils.device import resolve_device


def test_cpu_is_used_when_requested() -> None:
    assert resolve_device("cpu") == "cpu"


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_cuda_request_without_cuda_warns_and_falls_back(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # setup_logging()済みでもcaplogが捕捉できるよう、Root Loggerへの伝播を有効にする
    monkeypatch.setattr(logging.getLogger("anomaly_detection"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="anomaly_detection"):
        assert resolve_device("cuda") == "cpu"
    assert "CUDA を利用できません" in caplog.text


def test_invalid_device() -> None:
    with pytest.raises(ValueError, match="runtime.device"):
        resolve_device("gpu")
