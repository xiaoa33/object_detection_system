import sys
import os
# 把根目录加入搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 预注册 adaboost 模块，防止 pickle 报错
import train.adaboost
sys.modules['adaboost'] = train.adaboost

# 现在可以安全运行你的测试逻辑了
from detect.test import test_single_image
# 把路径引号里的空格去掉，并确保路径拼写完全正确
image_path = r"image.jpg" 
# 注意最前面的 r，代表原始字符串，防止路径里的反斜杠被转义
test_single_image(image_path, "models/cascade_model.pkl")
