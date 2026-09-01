import sys
from pathlib import Path
from app.core.logger import logger, node_log
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task

"""
1.  **接收状态**: 获取 `local_file_path`。
2.  **判断类型**: 检查文件后缀是 `.pdf` 还是 `.md`。
3.  **设置标记**: 更新 state 中的 `is_pdf_read_enabled` 或 `is_md_read_enabled`，供主图路由使用。
4.  **提取标题**: 从文件名中提取 `file_title`，后续作为元数据。
"""

@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
        1. 进行任务状态记录,开始和结束列表记录
        2. 根据state中 local_file_path属性判断数据类型进而修改
           相关参数, is_md_read_enabled 或者 is_pdf_read_enabled
                    md_path 或者 pdf_path
        3. 不可解析结果类型不可用,直接输出对应警告日志! 逻辑路由节点会自动处理
        4. 获取file_tile标识,用于后期识别pdf对应的主体(item_name)进行兜底
    """

    # 记录节点的运行状态
    add_running_task(state["task_id"], "node_entry")
    # 获取文件上传的路径
    local_file_path = state.get("local_file_path")

    # 判断local_file_path是否唯恐字符串或者None
    if not local_file_path:
        # 说明state里没有local_file_path,进行健壮性处理,
        logger.warning(f"当前状态中没有local_file_path,请检查初始状态")
        # 返回状态
        return state
    if local_file_path.endswith(".pdf"):
        # 更新state里is_pdf_read_enable,pdf_path
        state["is_pdf_read_enabled"] = True
        state["pdf_file_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_file_path"] = local_file_path
    else:
        # 说明文件不是pdf或者md
        logger.warning(f"当前上传文件路径{local_file_path},文件格式不是系统支持的格式")
        # 记录节点的状态为已完成
        add_done_task(state["task_id"], "node_entry")
        # 返回状态
        return state

    # 获取文件的标题(文件名去掉后缀的结果)
    file_title = Path(local_file_path).stem
    # 更新file_title
    state["file_title"] = file_title
    # 记录节点的完成状态
    add_done_task(state["task_id"], "node_entry")
    return state





if __name__ == '__main__':

    # 单元测试：覆盖不支持类型、MD、PDF三种场景
    logger.info("===== 开始node_entry节点单元测试 =====")

    # 测试1: 不支持的TXT文件
    test_state1 = create_default_state(
        task_id="test_task_001",
        local_file_path="联想海豚用户手册.txt"
    )
    print(node_entry(test_state1))

    # 测试2: MD文件
    test_state2 = create_default_state(
        task_id="test_task_002",
        local_file_path="小米用户手册.md"
    )
    print(node_entry(test_state2))

    # 测试3: PDF文件
    test_state3 = create_default_state(
        task_id="test_task_003",
        local_file_path="万用表的使用.pdf"
    )
    print(node_entry(test_state3))

    logger.info("===== 结束node_entry节点单元测试 =====")