import sys
import pickle
from train import adaboost, haar_features

# 关键：将 train 下的模块强行映射到全局，让 pickle 以为它们在根目录下
sys.modules['adaboost'] = adaboost
sys.modules['haar_features'] = haar_features

class CascadeClassifier:
    def __init__(self, model_path: str):
        print(f"[Cascade] 正在加载模型: {model_path} ...")
        with open(model_path, "rb") as f:
            # 现在 pickle 寻找 adaboost 时会从 sys.modules 中找到我们刚才手动注入的模块
            self.stages = pickle.load(f)