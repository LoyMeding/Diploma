from dataclasses import dataclass, field


@dataclass()
class TrainParams:
    epochs: int = field(default=1)
    pretrained_path: str = field(default="None")
    model_name: str = field(default="None")
    batch_size: int = field(default=32)
    cuda: bool = field(default=False)
