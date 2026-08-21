"""Baseline model helpers for S03."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier


def build_dummy_classifier(strategy: str = "prior", random_seed: int = 42) -> DummyClassifier:
    """Build the required DummyClassifier baseline."""

    return DummyClassifier(strategy=strategy, random_state=random_seed)
