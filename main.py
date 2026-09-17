import os
import shutil
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pdf_loader import load_pdf
from vector_store import VectorStore
from agent_graph import app_graph

app = FastAPI(
    title="企业智能客服Agent系统",
    description="基于LangGraph的Agentic RAG工单系统",
    version="0.1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

store = VectorStore()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

sessions = {}


class Question(BaseModel):
    query: str
    session_id: str = "default"


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "服务正常运行"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = load_pdf(save_path)
    store.add_documents(chunks)

    return {
        "message": f"已上传并处理 {len(chunks)} 个文本块",
        "filename": file.filename
    }


@app.post("/ask")
async def ask(question: Question):
    history = sessions.get(question.session_id, [])

    result = await app_graph.ainvoke({
        "query": question.query,
        "rewritten_query": "",
        "docs": [],
        "relevant": False,
        "answer": "",
        "retry_count": 0,
        "chat_history": history,
        "need_retrieval": False,
        "need_tool": False,
        "tool_results": []
    })

    history.append({"role": "user", "content": question.query})
    history.append({"role": "assistant", "content": result["answer"]})
    sessions[question.session_id] = history

    return {
        "query": question.query,
        "answer": result["answer"],
        "sources": result.get("docs", [])
    }


@app.post("/ask/stream")
async def ask_stream(question: Question):
    history = sessions.get(question.session_id, [])

    async def event_generator():
        full_answer = ""
        final_answer = ""
        sent_status = set()   # 修复：状态消息去重

        async for mode, data in app_graph.astream(
            {
                "query": question.query,
                "rewritten_query": "",
                "docs": [],
                "relevant": False,
                "answer": "",
                "retry_count": 0,
                "chat_history": history,
                "need_retrieval": False,
                "need_tool": False,
                "tool_results": []
            },
            stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                msg_chunk, metadata = data
                node_name = metadata.get("langgraph_node", "")
                if node_name in ("generate", "direct_answer", "tool_answer"):
                    content = msg_chunk.content
                    if content:
                        full_answer += content
                        payload = {"type": "content", "content": content}
                        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            elif mode == "updates":
                for node_name, node_output in data.items():
                    if not isinstance(node_output, dict):
                        continue

                    if node_name in ("generate", "direct_answer", "tool_answer", "no_answer"):
                        if node_output.get("answer"):
                            final_answer = node_output["answer"]

                    if node_name == "intent":
                        if node_output.get("need_tool"):
                            msg = "🤔 检测到工具调用意图，准备调用工单系统..."
                            if msg not in sent_status:
                                sent_status.add(msg)
                                payload = {"type": "status", "message": msg}
                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        elif node_output.get("need_retrieval"):
                            msg = "🔍 正在检索知识库..."
                            if msg not in sent_status:
                                sent_status.add(msg)
                                payload = {"type": "status", "message": msg}
                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    elif node_name == "tool":
                        tool_results = node_output.get("tool_results", [])
                        for r in tool_results:
                            tool_name = r.get("tool", "unknown")
                            args = r.get("args", {})
                            msg = f"🔧 已调用工具：{tool_name}，参数：{args}"
                            if msg not in sent_status:
                                sent_status.add(msg)
                                payload = {"type": "status", "message": msg}
                                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    elif node_name == "retrieve":
                        docs = node_output.get("docs", [])
                        msg = f"📚 检索到 {len(docs)} 条相关资料"
                        if msg not in sent_status:
                            sent_status.add(msg)
                            payload = {"type": "status", "message": msg}
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 流式没拿到内容时，用捕获的 final_answer，不重新执行图
        if not full_answer and final_answer:
            payload = {"type": "content", "content": final_answer}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            full_answer = final_answer

        history.append({"role": "user", "content": question.query})
        history.append({"role": "assistant", "content": full_answer})
        sessions[question.session_id] = history

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )