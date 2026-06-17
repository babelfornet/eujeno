from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test")


@mcp.tool()
def echo(text: str) -> str:
    """Ripete il testo ricevuto."""
    return f"echo: {text}"


if __name__ == "__main__":
    mcp.run()
