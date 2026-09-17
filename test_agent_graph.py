from agent_graph import app_graph

# 测试1：相关问题
print("=" * 50)
print("测试1：相关问题")
result = app_graph.invoke({
    "query": "年假有几天？",
    "rewritten_query": "",
    "docs": [],
    "relevant": False,
    "answer": "",
    "retry_count": 0
})
print(f"最终回答：{result['answer']}")

# 测试2：不相关问题（观察是否走重写流程）
print("=" * 50)
print("测试2：不相关问题")
result = app_graph.invoke({
    "query": "今天北京天气怎么样？",
    "rewritten_query": "",
    "docs": [],
    "relevant": False,
    "answer": "",
    "retry_count": 0
})
print(f"最终回答：{result['answer']}")