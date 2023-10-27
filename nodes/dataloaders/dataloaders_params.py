from dataclasses import dataclass, field
from typing import List


@dataclass()
class DataloadersParams:
    image_shape: List[int]
    classes: List[str]
    batch_size: int = field(default=32)
    shuffle: bool = field(default=False)
    train_data_dir: str = field(default="/content/animals10/train/")
    test_data_dir: str = field(default="/content/animals10/test/")
    valid_data_dir: str = field(default="/content/animals10/valid/")


