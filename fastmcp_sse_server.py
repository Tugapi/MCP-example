from fastmcp import FastMCP

mcp = FastMCP("sse_server")


@mcp.resource("mcp:server_info")
def get_server_info() -> str:
    """
    Get information about the server.
    :return: Information about the server.
    """
    return "MCP Server Info: This is a MCP Server."


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """
    Get a greeting for a given name.
    :param name: The name to get a greeting for.
    :return: A greeting for the given name.
    """
    return f"Hello, {name}!"


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    :param a: The first number.
    :param b: The second number.
    :return: The sum of the two numbers.
    """
    return a + b


@mcp.tool()
def substract(a: float, b: float) -> float:
    """
    Subtract the second number from the first number.
    :param a: The first number.
    :param b: The second number.
    :return: The difference between the two numbers.
    """
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    :param a: The first number.
    :param b: The second number.
    :return: The product of the two numbers.
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide the first number by the second number.
    :param a: The first number.
    :param b: The second number.
    :return: The quotient of the two numbers.
    """
    return a / b


if __name__ == '__main__':
    mcp.run(transport="sse", host="127.0.0.1", port=3001)