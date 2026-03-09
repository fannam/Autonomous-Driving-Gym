from dataclasses import dataclass, field


@dataclass(frozen=True)
class StackConfig:
    grid_size: tuple[int, int] = (21, 5)
    ego_position: tuple[int, int] = (4, 2)
    history_length: int = 5
    include_absolute_speed: bool = False

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
    n_actions: int = 5
    n_residual_layers: int = 10
    c_puct: float = 2.5
    n_simulations: int = 5
    learning_rate: float = 0.001
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
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
)

INFERENCE_CONFIG = AlphaZeroConfig(
    stack=StackConfig(include_absolute_speed=True),
    c_puct=3.5,
    n_simulations=15,
    learning_rate=0.001,
    batch_size=64,
    epochs=30,
    model_path="alphazero_model (19).pth",
)

EVALUATION_CONFIG = AlphaZeroConfig(
    stack=StackConfig(include_absolute_speed=True),
    c_puct=3.5,
    n_simulations=15,
    learning_rate=0.001,
    batch_size=64,
    epochs=30,
    model_path="alphazero_model (19).pth",
)
