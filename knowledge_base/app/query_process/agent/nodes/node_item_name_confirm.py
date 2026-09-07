from app.core.logger import node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState):
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_item_name_confirm", state["is_stream"])

    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_item_name_confirm", state["is_stream"])
    return state