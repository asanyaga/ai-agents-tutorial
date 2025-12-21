from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
import mcp.server.stdio
import asyncio
import os
# Create the server instance
server = Server("code-review-server")

# Define a resource template for files

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available file resources in the current directory"""
    resources = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                resources.append(
                    Resource(uri=f"file://{os.path.abspath(file_path)}",
                             name="file",
                             description=f"Python file: {file_path}",
                             mimeType="text/x-python")
                )
    return resources
@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read the contents of a file resource"""
    # Extract the file path from the uri
    if uri.startswith("file://"):
        file_path = uri[7:] # Remove "file//:" prefix

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        with open(file_path,"r") as f:
            content = f.read()
        
        return TextContent(
            type="text",
            text=content
        )

    raise ValueError(f"Unsupported URI {uri}")

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare available tools"""
    return [
        Tool(
            name="analyze_code",
            description="Analyze Python code and provide imrovement suggestions",
            inputSchema={
                "type":"object",
                "properties": {
                    "code" : {
                        "type":"string",
                        "description": "The Python code to analyze"
                    }
                },
                "require": ["code"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "analyze_code":
        code = arguments["code"]
        result = analyze_code_impl(code)

        return [TextContent(type="text",text=result)]
        
# Implementation functions (existing code)
def analyze_code_impl(code: str) -> str:
    """Implementation of code analysis"""
    import openai
    prompt = f"""
    You are a helpful code review assistant.
    Analyze the following Python code and suggest one improvement.

    Code:
    {code}
    """
    response = openai.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": prompt}]
    )
    return response.output_text

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
    
if __name__ == "__main__":
    asyncio.run(main())