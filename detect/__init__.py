"""
detect/__init__.py
==================
detect 包初始化模块。

职责：
    1. 确保 pickle 在反序列化级联模型时能正确找到 train 模块中的类。
    2. 将 train.adaboost 和 train.haar_features 注入 sys.modules，
       使得 pickle.load() 在寻找 'adaboost' 和 'haar_features' 时能找到它们。

注意：
    CascadeClassifier 类定义在 detect/cascade_classifier.py 中，
    不要在此文件中重复定义，以免造成导入冲突。
"""

import sys
import pickle
from train import adaboost, haar_features

# 关键：将 train 下的模块强行映射到全局，让 pickle 以为它们在根目录下
sys.modules['adaboost'] = adaboost
sys.modules['haar_features'] = haar_features
