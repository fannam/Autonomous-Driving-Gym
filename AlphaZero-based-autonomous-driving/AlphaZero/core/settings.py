from dataclasses import dataclass, field

try:
    from environment.config import DEFAULT_GRID_SIZE, DEFAULT_GRID_STEP, compute_grid_shape
except ModuleNotFoundError as exc:
    if exc.name != "environment":
        raise
    from ..environment.config import (
        DEFAULT_GRID_SIZE,
        DEFAULT_GRID_STEP,
        compute_grid_shape,
    )

DEFAULT_GRID_SHAPE = compute_grid_shape(DEFAULT_GRID_SIZE, DEFAULT_GRID_STEP)
DEFAULT_EGO_POSITION = (
    DEFAULT_GRID_SHAPE[0] // 2,
    DEFAULT_GRID_SHAPE[1] // 2,
)


@dataclass(frozen=True)
class StackConfig:
    grid_size: tuple[int, int] = DEFAULT_GRID_SHAPE
    ego_position: tuple[int, int] | None = DEFAULT_EGO_POSITION
    history_length: int = 5
    include_absolute_speed: bool = False
    mark_ego_in_init: bool = True
    presence_feature_index: int = 0
    lane_feature_index: int = 1
    presence_feature_name: str = "presence"
    lane_feature_name: str = "on_lane"
    lane_feature_fallback_name: str = "on_road"

    @property
    def plane_count(self) -> int:
        # history + lane + speed_max + speed_min (+ abs_speed optional)
        return self.history_length + 3 + int(self.include_absolute_speed)

    @property
    def network_input_shape(self) -> tuple[int, int, int]:
        width, height = self.grid_size
        return (width, height, self.plane_count)


@dataclass(frozen=True)
class AlphaZeroConfig:
    stack: StackConfig = field(default_factory=StackConfig)
    n_actions: int = 25
    n_residual_layers: int = 10
    c_puct: float = 2.5
    n_simulations: int = 5
    temperature: float = 1.0
    temperature_drop_step: int | None = None
    root_dirichlet_alpha: float = 0.3
    root_exploration_fraction: float = 0.25
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    model_path: str = "alphazero_model (19).pth"

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self.stack.network_input_shape


SELF_PLAY_CONFIG = AlphaZeroConfig(
    stack=StackConfig(include_absolute_speed=False),
    c_puct=2.5,
    n_simulations=5,
    temperature=1.0,
    temperature_drop_step=None,
    root_dirichlet_alpha=0.3,
    root_exploration_fraction=0.25,
    learning_rate=0.001,
    weight_decay=1e-4,
    batch_size=32,
    epochs=10,
)

INFERENCE_CONFIG = AlphaZeroConfig(
    stack=StackConfig(include_absolute_speed=True),
    c_puct=3.5,
    n_simulations=15,
    temperature=0.0,
    learning_rate=0.001,
    weight_decay=1e-4,
    batch_size=64,
    epochs=30,
    model_path="alphazero_model (19).pth",
)

EVALUATION_CONFIG = AlphaZeroConfig(
    stack=StackConfig(include_absolute_speed=True),
    c_puct=3.5,
    n_simulations=15,
    temperature=0.0,
    learning_rate=0.001,
    weight_decay=1e-4,
    batch_size=64,
    epochs=30,
    model_path="alphazero_model (19).pth",
)
