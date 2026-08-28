"""
    测试：Path
"""
from pathlib import Path

# 获取项目的根路径
print(Path(__file__).parent.parent)
print(Path(__file__).parents[1])
