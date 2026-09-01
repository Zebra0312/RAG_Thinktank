import re
import sys

from app.core.logger import logger, node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200 # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20

@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState):
    # 获取md_content
    md_content = state["md_content"]
    # 判断md_content是否为空
    if not md_content:
        logger.error("md文档内容获取失败，无法完成切分")
        raise RuntimeError("md文档内容获取失败，无法完成切分")
    # 统一将md_content中的\r\n和\r替换为\n
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 获取file_title
    file_title = state["file_title"]
    return md_content, file_title

@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content, file_title):
    # 创建匹配标题的正则表达式
    pattern = re.compile(r"^#{1,6}\s+.+")
    # 将md_content通过\n分割（列表中一行一行的数据）
    lines = md_content.split("\n")
    # 准备列表存储初切的切片
    sections = []
    # 准备存储当前的标题行
    current_title = ""
    # 准备存储当前标题下的行内容
    current_lines = []
    # 是否是代码块
    is_code_block = False
    # 遍历每一行数据
    for line in lines:
        # 去掉每一行数据两边的空格
        line = line.strip()
        # 判断当前行是否在代码块中
        if line.startswith("```") or line.startswith("~~~"):
            # 更新is_code_block的值，代码块的格式：第一行和最后一行以```或~~~开头
            # 当前行为代码块的开始行，将is_code_block=True
            # 当前行为代码块的结束行，将is_code_block=False
            # 因此直接将is_code_block直接取反
            is_code_block = not is_code_block
            # 存储此行数据
            current_lines.append(line)
            continue
        # 判断当前行不是代码块，是标题的情况
        if not is_code_block and pattern.match(line):
            # 判断current_title是否为空，若不为空，则记录上一个标题所对应的切片
            if current_title:
                sections.append(
                    {
                        "title": current_title,
                        "content": "\n".join(current_lines),
                        "file_title": file_title,
                    }
                )
            # 遇到新的标题，记录新的标题相关的数据
            current_title = line
            current_lines = [line]
        else:
            # 表示是代码块或者不是标题，直接保存当前行
            current_lines.append(line)
    # 保存md_content中最后以一个标题以及后续的行
    if current_title:
        sections.append(
            {
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title,
            }
        )
    # 若遍历了md_content的每一行，sections仍然为空，表示md_content中没有任何一个标题
    # 此时直接将md_content作为一个切片
    if not sections:
        sections.append(
            {
                "title": "无主题",
                "content": md_content,
                "file_title": file_title,
            }
        )
    return sections


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    # 记录节点的状态为运行中
    add_running_task(state["task_id"], "node_document_split")
    # 步骤1：获取与清洗内容，进行state中数据清晰(md_content / file_title (做标题兜底))
    md_content, file_title = step_1_get_content(state)
    # 步骤2：通过标题进行初切，保证语义的完整
    sections = step_2_split_by_title(md_content, file_title)
    # 记录节点的状态为已完成
    add_done_task(state["task_id"], "node_document_split")
    return state