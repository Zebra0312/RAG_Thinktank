import re
import sys
from pathlib import Path

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# 判断md文件中所引用的图片是否是IMAGE_EXTENSIONS中所支持的图片
def is_supported_image(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in IMAGE_EXTENSIONS

@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    # 获取md_path并判断是否为空
    md_path = state["md_path"]
    if not md_path:
        raise ValueError("md_path为空，请检查流程")
    # 获取md_path所对应的Path对象
    md_path_obj = Path(md_path)
    # 判断md_path_obj是否存在
    if not md_path_obj.exists():
        raise FileNotFoundError(f"{md_path}文件不存在")
    # 判断md_content是否为空
    # 若为空，说明当前流程中，上传的文件是md，节点的执行node_entry-->node_md_img
    # 若不为空，说明前流程中，是将上传的pdf转换为md文件
    if not state["md_content"]:
        state["md_content"] = md_path_obj.read_text(encoding="utf-8")
    # 获取存储md文件所对应的图片的目录路径
    images_dir_obj = md_path_obj.parent / "images"
    return state["md_content"], md_path_obj, images_dir_obj

@step_log("step_2_scan_images")
def step_2_scan_images(md_content: str, images_dir_obj: Path):
    # 创建存储最终结果的列表，list[tuple(图片名,图片路径,tuple(上文, 下文))]
    image_targets = []
    # 遍历images_dir_obj中所有的图片文件
    for image_file in images_dir_obj.iterdir():
        # 获取每个图片文件的文件名
        image_name = image_file.name
        # 判断文件名是否是支持的图片类型
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}不是支持的图片类型")
            continue
        # 创建匹配md_content中图片的正则表达式
        # re.escape()将字符串中所有的特殊符号进行转义
        # re.escape(a.jpg)==>a\.jpg
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) +".*?\)")
        # 在md_content中匹配正则
        match_results = list(pattern.finditer(md_content))
        # 判断match_results是否为空
        if not match_results:
            logger.warning(f"图片{image_name}未在md文件中引用")
            continue
        # 获取image_name在md_content中第一次出现的开始索引和结束索引
        start, end = match_results[0].span()
        # 分别获取开始索引前100个字符和结束索引后100个字符作为上下文
        pre_text = md_content[max(0, start - 100):start]
        post_text = md_content[end:min(len(md_content), end + 100)]
        # 将上文和下文组合成上下文
        context = (pre_text, post_text)
        # 将遍历的图片的信息追加到image_targets中
        image_targets.append((image_name, str(image_file), context))
    return image_targets


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    # 记录任务状态为运行中
    add_running_task(state["task_id"], "node_md_img")
    # 步骤1：进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # 步骤2：提取md文件中的图片
    image_targets = step_2_scan_images(md_content, images_dir_obj)
    # 记录任务状态为已完成
    add_done_task(state["task_id"], "node_md_img")
    return state
