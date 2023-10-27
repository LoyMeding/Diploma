import argparse
import yaml
import logging
import sys

from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from entities.train_params import TrainingParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


PATH = "C:\\Users\\sharn\\Desktop\\prod\\diploma\\config.yaml"


@dataclass()
class PipelineParams:
    train_params: TrainingParams


PipelineParamsSchema = class_schema(PipelineParams)


def read_pipeline_params(path: str) -> PipelineParams:
    # Получение словаря путей к конфигурационным файлам
    with open(path, "r") as input_stream:
        configs_path_dict = yaml.safe_load(input_stream)
    logger.info(configs_path_dict)
    all_configs_dict = dict()
    # Получение параметров из каждого конфигурационного файла
    for config in configs_path_dict:
        path = configs_path_dict[config]
        with open(path, "r") as input_stream:
            configs_dict = yaml.safe_load(input_stream)
        all_configs_dict[config] = configs_dict

    logger.info("All params check: %s", all_configs_dict)
    schema = PipelineParamsSchema().load(all_configs_dict)
    logger.info("Check schema: %s", schema)
    return schema


if __name__ == "__main__":
    read_pipeline_params(PATH)


