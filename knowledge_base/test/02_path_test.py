"""
    测试：Path
"""
import os
import sys
from pathlib import Path

# 获取项目的根路径
# print(Path(__file__).parent.parent)
# print(Path(__file__).parents[1])

# 获取文件的标题(不用path对象)
# print(Path(__file__).stem)
# print(os.path.basename(__file__).split(".")[0])

def test_fun():
    # 动态获取当前的函数名
    print(sys._getframe().f_code.co_name)
test_fun()