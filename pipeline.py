import importlib
import sys

from get_params import read_pipeline_params

PATH = "C:\\Users\\sharn\\Desktop\\prod\\diploma\\config.yaml"

params = read_pipeline_params(PATH)

module_path = params.train_params.model
sys.path.append(module_path)
module = __import__("train")
train = getattr(module, 'train')

code = train(params.train_params.word)
print(code)
