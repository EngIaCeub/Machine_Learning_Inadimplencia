from types import SimpleNamespace

import pandas as pd
import pytest

from credit_default.data.load import load_uci_dataset, normalize_binary_target


def test_load_uci_dataset_uses_default_uci_id(monkeypatch):
    calls: list[int] = []

    def fake_fetch_ucirepo(id: int):
        calls.append(id)
        return SimpleNamespace(
            data=SimpleNamespace(
                features=pd.DataFrame(
                    {
                        "ID": range(30_000),
                        "LIMIT_BAL": [20_000] * 30_000,
                        **{f"X{i}": [i] * 30_000 for i in range(1, 22)},
                    }
                ),
                targets=pd.DataFrame({"default payment next month": [0, 1] * 15_000}),
            )
        )

    monkeypatch.setattr("credit_default.data.load.fetch_ucirepo", fake_fetch_ucirepo)

    features, target = load_uci_dataset()

    assert calls == [350]
    assert features.shape == (30_000, 23)
    assert target.name == "default payment next month"
    assert set(target.unique()) == {0, 1}


def test_normalize_binary_target_rejects_non_binary_values():
    target = pd.Series([0, 1, 2], name="Y")

    with pytest.raises(ValueError, match="binary"):
        normalize_binary_target(target)
