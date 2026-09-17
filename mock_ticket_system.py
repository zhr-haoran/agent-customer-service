# 模拟工单数据库
tickets = {
    "1001": {"user": "张三", "issue": "电脑坏了", "status": "处理中"},
    "1002": {"user": "李四", "issue": "打印机没墨", "status": "已完成"},
    "1003": {"user": "王五", "issue": "网络连不上", "status": "待处理"},
}


def query_ticket(ticket_id: str) -> dict:
    """查询工单"""
    if ticket_id in tickets:
        return {"success": True, "data": tickets[ticket_id]}
    return {"success": False, "message": f"工单 {ticket_id} 不存在"}


def create_ticket(user: str, issue: str) -> dict:
    """创建工单"""
    new_id = str(max(int(k) for k in tickets.keys()) + 1)
    tickets[new_id] = {"user": user, "issue": issue, "status": "待处理"}
    return {"success": True, "ticket_id": new_id, "data": tickets[new_id]}


def list_tickets() -> list:
    """列出所有工单"""
    return [{"id": k, **v} for k, v in tickets.items()]

# 测试
if __name__ == "__main__":
    print("查询工单 1001：", query_ticket("1001"))
    print("创建新工单：", create_ticket("赵六", "键盘失灵"))
    print("所有工单：", list_tickets())
