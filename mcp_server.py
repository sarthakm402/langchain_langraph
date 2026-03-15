from mcp.server.fastmcp import FastMCP
server=FastMCP("mcp server")
@server.tool
def calculator(expression: str) -> str:
    return str(eval(expression))
@server.tool
def say(txt:str)->str:
    res=print(f"hey chulo u said {txt}")
    return res
if __name__=="__main__":
    server.run()