from dataclasses import dataclass


@dataclass()
class TrainingParams:
    model: str
    word: str