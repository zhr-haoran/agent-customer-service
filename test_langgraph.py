from langgraph.graph import StateGraph, END
from typing import TypedDict


# 1. 定义状态（共享文件夹）
class MyState(TypedDict):
    number: int
    message: str


# 2. 定义节点函数（每个节点接收状态，返回要更新的部分）
def add_one(state: MyState):
    print(f"【节点1】收到数字：{state['number']}")
    return {"number": state["number"] + 1}


def multiply_two(state: MyState):
    print(f"【节点2】收到数字：{state['number']}")
    return {"number": state["number"] * 2, "message": "处理完成"}


# 3. 创建状态图
workflow = StateGraph(MyState)

# 4. 添加节点
workflow.add_node("add_one", add_one)
workflow.add_node("multiply_two", multiply_two)

# 5. 设置起点和边
workflow.set_entry_point("add_one")           # 从 add_one 开始
workflow.add_edge("add_one", "multiply_two")  # add_one 走完走 multiply_two
workflow.add_edge("multiply_two", END)        # multiply_two 走完结束

# 6. 编译成可执行的图
app = workflow.compile()

# 7. 运行
result = app.invoke({"number": 5, "message": ""})
print(f"最终结果：{result}")