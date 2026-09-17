# 企业级智能客服Agent系统

基于 LangGraph 的 Agentic RAG 工作流，支持 PDF 知识库上传、语义检索、相关性评估、自动拒答和 MCP 工具调用。

## 功能

- Agentic RAG：检索 → 评估相关性 → 不相关时重写查询 → 最多重试2次 → 拒答
- MCP 工具调用：查工单、建工单、列工单
- 多轮对话记忆
- 意图路由：工具 / 检索 / 闲聊
- 流式响应
- 全本地运行

## 技术栈

- FastAPI
- LangGraph
- BAAI/bge-large-zh-v1.5（本地）
- ChromaDB
- Ollama (qwen2.5:7b)
- MCP

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

### 下载对话模型
ollama pull qwen2.5:7b(大语言模型即可)

### 配置模型路径
# 编辑 vector_store.py，修改 model_path 为本地向量模型路径

## 启动
uvicorn main:app --reload
访问 http://127.0.0.1:8000/

# 项目结构
agent_project/
├── static/index.html
├── agent_graph.py
├── main.py
├── vector_store.py
├── pdf_loader.py
├── mock_ticket_system.py
├── mcp_server.py
├── config.py
└── requirements.txt