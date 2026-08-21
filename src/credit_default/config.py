"""Central deterministic project configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectConfig:
    random_seed: int = 42
    dataset_id: int = 350
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15

    def validate(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Train/validation/test fractions must sum to 1.0.")


def get_project_config() -> ProjectConfig:
    config = ProjectConfig()
    config.validate()
    return config
