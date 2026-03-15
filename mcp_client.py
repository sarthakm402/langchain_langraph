from mcp.client import ClientSession
from mcp.client.stdio import stdio_client

async def main():
    async with stdio_client("python mcp_server.py") as (read, write):
        async with ClientSession(read, write) as session:

            tools = await session.list_tools()
            print("Available tools:", tools)

            result = await session.call_tool(
                "calculator",
                {"expression": "12*18"}
            )

            print("Result:", result)

import asyncio
asyncio.run(main())