import sys
import logging

from entities.get_params import read_pipeline_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Указать путь к config.yaml с перечнем узлов
PATH = "C:\\Users\\sharn\\Desktop\\prod\\diploma\\animals_pipeline\\config.yaml"

params = read_pipeline_params(PATH)
try:
    # Импортируем функцию для создания загрузчиков данных
    module_path = params.dataloader_params.dataloader
    sys.path.append(module_path)
    module = __import__("dataloaders")
    create_dataloaders = getattr(module, 'dataloaders')
    logger.info("Successful dataload import")
    # Создаём загрузчики данных
    train_loader, test_loader = create_dataloaders(
        params.dataloader_params.batch_size,
        params.dataloader_params.shuffle,
        params.dataloader_params.train_dir,
        params.dataloader_params.test_dir,
        params.dataloader_params.valid_dir
    )
    logger.info("Successful dataloaders creating")
except:
    logger.info("Create dataload error")
try:
    # Импортируем функцию для инициализации модели
    module_path = params.model_params.model
    sys.path.append(module_path)
    module = __import__("model")
    create_model = getattr(module, 'model')
    logger.info("Successful model import")
    # Инициализируем модель
    model = create_model(
        params.dataloader_params.classes
    )
    logger.info("Successful model creating")
except:
    logger.info("Create model error")

try:
    # Импортируем функцию для обучения модели
    module_path = params.train_params.train
    sys.path.append(module_path)
    module = __import__("train")
    train = getattr(module, 'train')
    logger.info("Successful train import")
    # Обучаем модель
    train(
        model,
        params.dataloader_params.image_shape,
        params.dataloader_params.classes,
        train_loader,
        params.train_params.epochs,
        params.train_params.pretrained_path,
        params.train_params.model_name,
        params.train_params.batch_size,
        params.train_params.cuda
    )
    logger.info("Successful training")
except:
    logger.info("Training error")
