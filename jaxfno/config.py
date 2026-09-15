from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FNOConfig:
    data_path: str = "uxyz_data.npz"
    dataset_layout: dict = field(default_factory=dict)
    width: int = 16
    modes: tuple[int, int, int] = (8, 8, 8)
    input_channels: int = 4
    output_channels: int = 1
    padding: int = 4
    normalization: str = "source_mean"
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
    checkpoint_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parent.parent / "model"))

    def __post_init__(self):
        self.modes = tuple(self.modes)
        if self.normalization != "source_mean":
            raise ValueError("Only source_mean normalization is supported")
        if len(self.modes) != 3 or any(m < 1 for m in self.modes):
            raise ValueError("modes must contain three positive integers")
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
    def from_dict(cls, values: dict) -> "FNOConfig":
        # Old physical checkpoints/configs included these unused demo fields.
        # Discard only those obsolete fields; keep rejecting other unknown keys.
        values = dict(values)
        values.pop("synthetic_samples", None)
        values.pop("synthetic_grid", None)
        return cls(**values)

    @classmethod
    def load(cls, path: str | Path) -> "FNOConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
