import yaml
import logging
import sys

from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from nodes.train.train_params import TrainParams
from nodes.dataloaders.dataloader_params import DataloadersParams
from nodes.model.model_params import ModelParams
from nodes.predict.predict_params import PredictParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


@dataclass()
class PipelineParams:
    model_params: ModelParams
    dataloader_params: DataloadersParams
    train_params: TrainParams
    predict_params: PredictParams


PipelineParamsSchema = class_schema(PipelineParams)

"""
Информация о подключенных узлах хранится в config.yaml файле. Получаем информацию из данного файла
и для каждого узла также считываем его конфигурационный файл.
    
Параметры:
- path: путь к конфигурационному файлу, полученному в результате работы узлов
"""


def read_pipeline_params(path: str) -> PipelineParams:
    # Получение словаря путей к конфигурационным файлам
    with open(path, "r") as input_stream:
        configs_path_dict = yaml.safe_load(input_stream)
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
    logger.info("Successful read")
    return schema


