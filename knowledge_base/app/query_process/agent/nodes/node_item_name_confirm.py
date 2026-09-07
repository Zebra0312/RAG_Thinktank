import sys

from app.clients.mongo_history_utils import save_chat_message
from app.core.logger import node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


@node_log("node_item_name_confirm")
def node_item_name_confirm(state):
    print(f"---node_item_name_confirm 处理")

    add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

    save_chat_message(state['session_id'], "user", state['original_query'], "", state.get("item_names", []))

    print(f"---已保存对话 处理完成")