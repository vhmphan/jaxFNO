from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FNOConfig:
    data_path: str = "synthetic"
    dataset_layout: dict = field(default_factory=dict)
    width: int = 16
    modes: tuple[int, int, int] = (8, 8, 8)
    input_channels: int = 4
    output_channels: int = 1
    padding: int = 4
    learning_rate: float = 1e-3
    batch_size: int = 1
    epochs: int = 200
    patience: int = 20
    min_delta: float = 0.0
    loss_floor: float = 1e-8
    weight_decay: float = 1e-4
    seed: int = 0
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    checkpoint_dir: str = "checkpoints"
    synthetic_samples: int = 32
    synthetic_grid: tuple[int, int, int] = (16, 16, 12)

    def __post_init__(self):
        self.modes = tuple(self.modes)
        self.synthetic_grid = tuple(self.synthetic_grid)
        if len(self.modes) != 3 or any(m < 1 for m in self.modes):
            raise ValueError("modes must contain three positive integers")
        if len(self.synthetic_grid) != 3 or min(self.synthetic_grid) < 2:
            raise ValueError("synthetic_grid must contain three sizes >= 2")
        if min(self.width, self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("width, batch_size, epochs, and patience must be positive")
        if self.padding < 0 or self.learning_rate <= 0 or self.loss_floor <= 0:
            raise ValueError("Invalid padding, learning rate, or loss floor")
        if self.min_delta < 0 or self.weight_decay < 0:
            raise ValueError("min_delta and weight_decay must be nonnegative")
        if self.input_channels != 4 or self.output_channels != 1:
            raise ValueError("Source encoding requires four inputs and one output")
        if not (0 < self.validation_fraction and 0 < self.test_fraction
                and self.validation_fraction + self.test_fraction < 1):
            raise ValueError("Validation/test fractions must be positive and sum to < 1")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "FNOConfig":
        return cls(**json.loads(Path(path).read_text()))
