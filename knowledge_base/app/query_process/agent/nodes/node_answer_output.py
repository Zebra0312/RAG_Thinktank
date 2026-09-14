import re

from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import logger, node_log, step_log
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import push_to_session, SSEEvent
from app.utils.task_utils import add_running_task, add_done_task, set_task_result

# 上下文最大字符数
MAX_CONTEXT_CHARS = 12000

@step_log("step_1_check_answer")
def step_1_check_answer(state: QueryGraphState):
    # 分别获取状态中answer和is_stream
    answer = state.get("answer")
    is_stream = state.get("is_stream")
    # 判断answer是否为空
    if answer:
        # 判断is_stream是否为空
        if is_stream:
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)
        return True
    else:
        return False

@step_log("step_2_construct_prompt")
def step_2_construct_prompt(state: QueryGraphState):
    # 从状态中获取所需要的数据
    original_query = state.get("original_query")
    rewritten_query = state.get("rewritten_query")
    question = rewritten_query if rewritten_query else original_query
    reranked_docs = state.get("reranked_docs")
    item_names = state.get("item_names")
    history_list = state.get("history")
    """
    将reranked_docs中的数据转换为以下格式：
    "[1] [local] [chunk_id=123] [score=0.95] [title=操作手册]
     这里是文档的正文内容..."
    """
    # 处理上下文，创建存储处理之后的结果的列表
    docs = []
    # 创建记录最大字符数的变量
    used = 0
    # 对reranked_docs进行遍历
    for num, chunk in enumerate(reranked_docs, start=1):
        # 从chunk中获取所需要的数据，并存储到列表中
        text = chunk.get("text")
        if not text:
            continue
        data_list = [f"[{num}]"]
        source = chunk.get("source")
        if source:
            data_list.append(f"[{source}]")
        chunk_id = chunk.get("chunk_id")
        if chunk_id:
            data_list.append(f"[chunk_id={chunk_id}]")
        score = chunk.get("score")
        if score is not None:
            data_list.append(f"[score={float(score):.4f}]")
        title = chunk.get("title")
        if title:
            data_list.append(f"[title={title}]")
        # 拼接各个数据为指定格式
        doc = " ".join(data_list) + "\n" + text
        # 判断当前字符数是否超过最大字符数的阈值
        if used + len(doc) > MAX_CONTEXT_CHARS:
            break
        # 存储每条数据转换的结果
        docs.append(doc)
        # 记录本次循环的字节数
        used += len(doc) + 2
    # 将每条数据转换的结果拼接为字符串
    context = "\n\n".join(docs) if docs else "无参考内容"
    """
    将历史对话转换为以下格式：
    用户: xxx
    助手: xxx
    """
    # 处理历史记录
    history_str = ""
    # 判断历史记录是否为空
    if history_list:
        # 对history_list进行遍历
        for history in history_list:
            # 分别获取历史记录中role和text
            role = history.get("role")
            text = history.get("text")
            # 判断role是否为user，若为user，则拼接"用户: xxx"
            history_text = ""
            if role == "user" and text:
                history_text += f"用户: {text}\n"
            elif role == "assistant" and text:
                history_text += f"助手: {text}\n"
            if used + len(history_text) > MAX_CONTEXT_CHARS:
                break
            history_str += history_text
            # 记录拼接之后的字符数
            used += len(history_text)
    else:
        history_str = "无历史记录"
    """
    将产品主体转换为以下格式：
    产品主体1, 产品主体2, ...
    """
    item_names_str = ", ".join(item_names)
    # 读取提示词
    prompt = load_prompt(
        "answer_out",
        context=context,
        history=history_str,
        item_names=item_names_str,
        question=question,
    )
    return prompt

@step_log("step_3_generate_response")
def step_3_generate_response(state: QueryGraphState, prompt: str):
    # 从状态中获取session_id和is_stream
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")
    # 获取大模型对象
    llm = get_llm_client()
    # 判断是否为流式调用
    if is_stream:
        # 创建存储最终answer的变量
        answer = ""
        try:
            # 以流式方式调用大模型
            for chunk in llm.stream(prompt):
                # 当使用流式调用大模型时，每次返回封装了token的对象
                # 每次返回的具体的token
                content = chunk.content
                # 将每次返回的token添加到队列中
                push_to_session(session_id, SSEEvent.DELTA, {"delta": content})
                # 将每次的token拼接，获取最终的answer
                answer += content
        except Exception as e:
            push_to_session(session_id, SSEEvent.ERROR, {"error": e})
        # 更新状态
        state["answer"] = answer
    else:
        try:
            # 表示非流式，则直接调用大模型
            response = llm.invoke(prompt)
            # 获取最终的answer
            answer = response.content
            # 设置当前任务的状态
            set_task_result(session_id, "answer", answer)
            # 更新状态
            state["answer"] = answer
        except Exception as e:
            state["answer"] = "抱歉，生成回答时出现错误。"
    return state

# 从最终的文档中提取图片的url
def _extract_images_from_docs(reranked_docs):
    # 创建存储提取的图片url的列表
    image_urls = []
    # 创建正则表达式
    pattern = r"!\[.*?\]\((.*?)\)"
    # 对文档进行遍历
    for doc in reranked_docs:
        # 获取文档中的url
        url = doc.get("url")
        # 判断url是否为空
        if url:
            if url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')):
                if url not in image_urls:
                    image_urls.append(url)
        # 获取文档中的text
        text = doc.get("text")
        if text:
            # 提供正则表达式提取文档中图片的url
            matches = re.findall(pattern, text)
            # 判断matches是否为空
            if matches:
                for match in matches:
                    if match not in image_urls:
                        image_urls.append(match)
    return image_urls


# 图片 URL：惰性匹配到“图片扩展名”为止，避免把后面的中文说明文字一起吞掉
_IMG_URL_RE = re.compile(r"https?://[^\s)\]<>\"'】]*?\.(?:png|jpe?g|gif|webp|bmp|svg)", re.IGNORECASE)
# 扩展名后面紧跟的 ?query / #hash（用于原样保留，例如预签名地址）
_URL_TAIL_RE = re.compile(r"[?#][^\s)\]<>\"'】]*")
# 正文里的 URL 候选（贪婪，交给 _extract_image_url 再裁剪）
_URL_BARE_RE = re.compile(r"https?://[^\s)\]<>\"'】]+")
# Markdown 图片语法 ![alt](url)
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
# 【图片】 标记（兼容 [图片] 写法）
_IMG_MARKER_RE = re.compile(r"【\s*图片\s*】|\[\s*图片\s*\]")


def _extract_image_url(text: str):
    """从一段文本里抽取第一个图片 URL；不是图片地址则返回 None。保留 ?query/#hash。"""
    m = _IMG_URL_RE.search(text or "")
    if not m:
        return None
    url = m.group(0)
    rest = (text or "")[m.end():]
    if rest[:1] in ("?", "#"):
        tail = _URL_TAIL_RE.match(rest)
        if tail:
            url += tail.group(0)
    return url


def _norm_img_url(u: str) -> str:
    """归一化图片 URL 用于比对：去包裹符号、去 query/fragment、去结尾标点。"""
    s = str(u or "").strip().strip("<>").strip().rstrip(".,;，。、")
    return s.split("?")[0].split("#")[0]


def _sanitize_answer_images(answer: str, allowed_urls) -> str:
    """
    清洗答案里的图片地址，防止大模型编造/幻觉出参考内容中不存在的图片链接
    （典型表现：参考资料无图时编出 https://example.com/xxx.jpg 这类占位地址）。

    规则：
    1) 白名单 = 参考切片里真实存在的图片 URL，只有白名单内的地址才允许保留；
    2) Markdown 图片 ![](url) 中 url 不在白名单 → 整段删除；
    3) 正文中游离的、非白名单图片 URL → 删除该 URL（普通网页链接不受影响）；
    4) 【图片】区块只保留白名单内的 URL；若一个都不剩 → 连【图片】标题一起删除。
    """
    if not answer:
        return answer
    allowed = {_norm_img_url(u) for u in (allowed_urls or []) if u}

    def _keep(url: str) -> bool:
        return _norm_img_url(url) in allowed

    # 1) 去掉参考内容里没有的 Markdown 图片
    answer = _MD_IMG_RE.sub(lambda m: m.group(0) if _keep(m.group(1)) else "", answer)

    def _clean_prose(text: str) -> str:
        """正文里的图片 URL 白名单过滤（普通网页链接原样保留）。"""
        def _bare(m):
            raw = m.group(0)
            url = _extract_image_url(raw)
            if url is None or _keep(url):
                return raw
            return raw.replace(url, "", 1)
        return _URL_BARE_RE.sub(_bare, text)

    # 2) 以最后一个【图片】标记为界，拆分“正文”与“图片区块”
    last_marker = None
    for m in _IMG_MARKER_RE.finditer(answer):
        last_marker = m

    if last_marker is None:
        return _clean_prose(answer).strip()

    head = _clean_prose(answer[:last_marker.start()]).rstrip()
    tail = answer[last_marker.end():]
    kept = []
    for line in tail.splitlines():
        url = _extract_image_url(line)
        if url and _keep(url) and url not in kept:
            kept.append(url)
    # 一张真实图片都没有 → 整个区块丢掉，避免前端把编造地址当图片展示
    return (f"{head}\n\n【图片】\n" + "\n".join(kept) if kept else head).strip()


@step_log("step_4_write_history")
def step_4_write_history(state: QueryGraphState, image_urls):
    # 保存历史记录
    session_id = state.get("session_id")
    item_names = state.get("item_names")
    answer = state.get("answer")
    # 判断answer是否有值
    if answer:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            rewritten_query="",
            item_names=item_names,
            image_urls=image_urls,
            message_id=None
        )



@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_answer_output", state["is_stream"])
    # 阶段一：检查answer是否存在,如果存在直接输出answer中的答案
    answer_exists = step_1_check_answer(state)
    # 判断answer是否存在
    if not answer_exists:
        prompt = step_2_construct_prompt(state)
        state["prompt"] = prompt
        state = step_3_generate_response(state, prompt)
    # 提取图片URL（白名单：仅参考切片里真实存在的图片地址，用于历史记录和前端展示）
    image_urls = _extract_images_from_docs(state.get("reranked_docs") or [])
    if state.get("answer"):
        # 清洗答案中的图片地址：丢弃模型编造的 URL（如参考资料无图时幻觉出的 example.com 占位地址）
        cleaned = _sanitize_answer_images(state["answer"], image_urls)
        if cleaned != state["answer"]:
            logger.warning(
                f"{state.get('session_id')} 答案中的图片地址已被清洗（疑似模型编造，已按参考内容白名单过滤）"
            )
        state["answer"] = cleaned
        step_4_write_history(state, image_urls=image_urls)
    # 将图片和最终answer推送到浏览器端
    if state.get("is_stream"):
        push_to_session(
            state['session_id'],
            SSEEvent.FINAL,
            {
                "answer": state["answer"],
                "image_urls": image_urls  # 发送图片URL给前端
            }
        )
    # 记录当前任务的状态为已完成
    add_done_task(state["session_id"], "node_answer_output", state["is_stream"])
    return state