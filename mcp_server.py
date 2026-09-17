from mcp.server.fastmcp import FastMCP
from mock_ticket_system import query_ticket, create_ticket, list_tickets

# 创建 MCP 服务器实例
mcp = FastMCP("工单管理系统")


@mcp.tool()
def tool_query_ticket(ticket_id: str) -> dict:
    """根据工单ID查询工单详情。参数 ticket_id 是工单编号，例如 '1001'。"""
    return query_ticket(ticket_id)


@mcp.tool()
def tool_create_ticket(user: str, issue: str) -> dict:
    """创建一个新工单。参数 user 是提交人姓名，issue 是问题描述。"""
    return create_ticket(user, issue)


@mcp.tool()
def tool_list_tickets() -> list:
    """列出所有工单。不需要参数。"""
    return list_tickets()


if __name__ == "__main__":
    mcp.run(transport="stdio")