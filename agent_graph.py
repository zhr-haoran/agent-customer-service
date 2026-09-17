from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mcp_adapters.client import MultiServerMCPClient
from vector_store import VectorStore
from langchain_ollama import ChatOllama

store = VectorStore(collection_name="knowledge_base")

llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)

_mcp_client = None
_mcp_tools = None


async def get_mcp_tools():
    global _mcp_client, _mcp_tools
    if _mcp_tools is None:
        _mcp_client = MultiServerMCPClient({
            "ticket": {
                "command": r"E:\Minconda\envs\test\python.exe",
                "args": [r"E:\python\agent_project\mcp_server.py"],
                "transport": "stdio"
            }
        })
        _mcp_tools = await _mcp_client.get_tools()
        print(f"【MCP】已加载 {len(_mcp_tools)} 个工具：{[t.name for t in _mcp_tools]}")
    return _mcp_tools


class RAGState(TypedDict):
    query: str
    rewritten_query: str
    docs: List[str]
    relevant: bool
    answer: str
    retry_count: int
    chat_history: List[dict]
    need_retrieval: bool
    need_tool: bool
    tool_results: List[dict]


async def intent_node(state: RAGState):
    """节点0：判断问题类型（工具 / 检索 / 闲聊），结合历史对话"""
    query = state["query"]
    history = state.get("chat_history", [])

    # 规则层：问历史的明显特征，直接判为"闲聊"
    history_keywords = ["刚刚", "刚才", "之前说", "我说过", "我说了什么",
                        "我问了什么", "我上一句", "我前面说的", "重复一下我说"]
    if any(kw in query for kw in history_keywords):
        print(f"【意图节点】规则命中历史关键词，判为闲聊")
        return {"need_tool": False, "need_retrieval": False}

    # LLM 层：结合历史对话判断
    history_text = ""
    for msg in history[-4:]:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}：{msg['content']}\n"

    prompt = ChatPromptTemplate.from_template("""你是意图判断助手。请根据历史对话和用户最新输入，判断用户问题类型。

类型定义：
- "工具"：涉及工单查询、创建、列表等操作。如果上一轮助手在追问工单信息，用户正在补充，也必须判为"工具"。
- "检索"：询问具体的知识、规定、制度、文档内容。例如"财务管理制度是什么"、"年假规定"。
- "闲聊"：问候、追问历史、让你重复之前的话、闲聊。例如"你好"、"我刚刚问了什么"、"继续"。

<历史对话>
{history}
</历史对话>

用户最新输入：{query}

只回复"工具"、"检索"或"闲聊"。""")

    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"query": query, "history": history_text})
    result = result.strip()

    # 修复：精确匹配，避免"不是工具"这类否定句被误判
    need_tool = result == "工具"
    need_retrieval = result == "检索"
    print(f"【意图节点】{result}，需工具：{need_tool}，需检索：{need_retrieval}")
    return {"need_tool": need_tool, "need_retrieval": need_retrieval}


async def tool_node(state: RAGState):
    query = state["query"]
    history = state.get("chat_history", [])

    # 工具调用时也带上历史，让 LLM 知道用户在补充什么信息
    if history:
        last_assistant = next(
            (m["content"] for m in reversed(history) if m["role"] == "assistant"),
            ""
        )
        full_query = f"上一轮助手说：{last_assistant}\n用户补充：{query}"
    else:
        full_query = query

    tools = await get_mcp_tools()
    llm_with_tools = llm.bind_tools(tools)
    response = await llm_with_tools.ainvoke(full_query)

    results = []
    if response.tool_calls:
        for tc in response.tool_calls:
            tool = next((t for t in tools if t.name == tc["name"]), None)
            if tool:
                try:
                    result = await tool.ainvoke(tc["args"])
                    results.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                        "result": result
                    })
                    print(f"【工具节点】调用 {tc['name']}({tc['args']})")
                except Exception as e:
                    results.append({"tool": tc["name"], "error": str(e)})
    else:
        print(f"【工具节点】LLM 未决定调用任何工具")

    return {"tool_results": results}


async def tool_answer_node(state: RAGState):
    query = state["query"]
    tool_results = state.get("tool_results", [])

    if not tool_results:
        prompt = ChatPromptTemplate.from_template("""用户想使用工单系统，但提供的信息不足以调用工具。
请友好地追问用户，让他补充必要的信息。

例如：
- 查询工单：需要工单号
- 创建工单：需要提交人姓名和问题描述

用户说：{query}

请用自然语言回复用户。""")
        chain = prompt | llm | StrOutputParser()
        answer = await chain.ainvoke({"query": query})
        print(f"【工具回答节点】信息不足，追问用户")
        return {"answer": answer}

    results_text = "\n".join([
        f"调用 {r['tool']}({r.get('args', '')})：\n{r.get('result', r.get('error'))}"
        for r in tool_results
    ])

    prompt = ChatPromptTemplate.from_template("""根据工具调用结果，用自然语言回答用户问题。

<工具结果>
{results}
</工具结果>

问题：{query}""")
    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"results": results_text, "query": query})
    print(f"【工具回答节点】已生成回答")
    return {"answer": answer}


async def direct_answer_node(state: RAGState):
    query = state["query"]
    history = state.get("chat_history", [])
    history_text = ""
    for msg in history[-6:]:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}：{msg['content']}\n"

    prompt = ChatPromptTemplate.from_template("""请根据历史对话回答用户的问题。如果历史对话中没有相关信息，请如实说明。

<历史对话>
{history}
</历史对话>

问题：{query}""")
    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"history": history_text, "query": query})
    print(f"【直接回答节点】已生成回答")
    return {"answer": answer}


async def retrieve_node(state: RAGState):
    query = state.get("rewritten_query") or state["query"]
    docs = store.search(query, n_results=3)
    print(f"【检索节点】使用查询：{query}")
    print(f"【检索节点】找到 {len(docs)} 条文档")
    return {"docs": docs}


async def grade_node(state: RAGState):
    query = state.get("rewritten_query") or state["query"]
    docs = state["docs"]
    context = "\n".join(docs)

    prompt = ChatPromptTemplate.from_template("""你是一个评估员。请判断以下文档是否与用户问题相关。
如果文档能回答问题，回复"相关"；如果完全无关，回复"不相关"。

文档：
{context}

问题：{query}

只回复"相关"或"不相关"，不要解释。""")

    chain = prompt | llm | StrOutputParser()
    result = await chain.ainvoke({"context": context, "query": query})
    result = result.strip()

    is_relevant = "相关" in result and "不相关" not in result
    print(f"【评估节点】判断结果：{result}，是否相关：{is_relevant}")
    return {"relevant": is_relevant}


async def rewrite_node(state: RAGState):
    original = state.get("rewritten_query") or state["query"]
    prompt = ChatPromptTemplate.from_template("""你是一个查询改写助手。
用户的问题在知识库中没有检索到相关内容。请将问题改写得更通用、更简洁，以提高检索命中率。

原始问题：{query}

只输出改写后的问题，不要解释。""")
    chain = prompt | llm | StrOutputParser()
    new_query = await chain.ainvoke({"query": original})
    new_query = new_query.strip()
    print(f"【重写节点】{original} → {new_query}")
    return {"rewritten_query": new_query, "retry_count": state["retry_count"] + 1}


async def generate_node(state: RAGState):
    query = state.get("rewritten_query") or state["query"]
    docs = state["docs"]
    context = "\n".join(docs)
    history = state.get("chat_history", [])
    history_text = ""
    for msg in history[-6:]:
        role = "用户" if msg["role"] == "user" else "助手"
        history_text += f"{role}：{msg['content']}\n"

    prompt = ChatPromptTemplate.from_template("""仅根据以下上下文回答问题。

<上下文>
{context}
</上下文>

<历史对话>
{history}
</历史对话>

问题：{query}""")

    chain = prompt | llm | StrOutputParser()
    answer = await chain.ainvoke({"context": context, "query": query, "history": history_text})
    print(f"【生成节点】已生成回答")
    return {"answer": answer}


async def no_answer_node(state: RAGState):
    print(f"【无答案节点】知识库中没有相关信息")
    return {"answer": "抱歉，知识库中没有找到相关信息。", "docs": []}


def route_after_intent(state: RAGState):
    if state.get("need_tool"):
        return "tool"
    if state.get("need_retrieval"):
        return "retrieve"
    return "direct_answer"


def route_after_grade(state: RAGState):
    if state["relevant"]:
        return "generate"
    if state["retry_count"] < 2:
        return "rewrite"
    return "no_answer"


workflow = StateGraph(RAGState)

workflow.add_node("intent", intent_node)
workflow.add_node("tool", tool_node)
workflow.add_node("tool_answer", tool_answer_node)
workflow.add_node("direct_answer", direct_answer_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("generate", generate_node)
workflow.add_node("no_answer", no_answer_node)

workflow.set_entry_point("intent")

workflow.add_conditional_edges(
    "intent",
    route_after_intent,
    {
        "tool": "tool",
        "retrieve": "retrieve",
        "direct_answer": "direct_answer"
    }
)

workflow.add_edge("tool", "tool_answer")
workflow.add_edge("tool_answer", END)

workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    route_after_grade,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "no_answer": "no_answer"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)
workflow.add_edge("no_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()