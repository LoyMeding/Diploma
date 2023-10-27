from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    pretrained: bool = field(default=True)
