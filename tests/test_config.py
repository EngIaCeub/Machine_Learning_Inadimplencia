from credit_default.config import get_project_config


def test_split_fractions_sum_to_one():
    cfg = get_project_config()
    assert cfg.train_size + cfg.validation_size + cfg.test_size == 1.0


def test_default_seed_is_fixed():
    assert get_project_config().random_seed == 42
