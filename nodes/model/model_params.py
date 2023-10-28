from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    model: str
    pretrained: bool = field(default=True)
