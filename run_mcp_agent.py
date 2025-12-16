"""
MCP Code Review Agent - Standalone Script
This script works on Windows and doesn't have Jupyter limitations
Run: python run_mcp_agent.py
"""

import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Optional
import json
import openai
import os

class MCPClientManager:
    """Official MCP Client Manager using AsyncExitStack"""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.available_tools: dict[str, dict] = {}
    
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        command = "python" if is_python else "node"

        server_env = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}


        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=server_env
        )
        
        print(f"Connecting to MCP server: {server_script_path}")
        
        # Use AsyncExitStack (official pattern)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        await self.session.initialize()
        
        # Discover tools
        response = await self.session.list_tools()
        for tool in response.tools:
            self.available_tools[tool.name] = {"tool": tool}
            print(f"Discovered tool: {tool.name}")
        
        print(f"Connected successfully!\n")
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool through MCP"""
        if not self.session:
            raise RuntimeError("Not connected to a server")
        
        if tool_name not in self.available_tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        result = await self.session.call_tool(tool_name, arguments)
        
        if result.content and len(result.content) > 0:
            return result.content[0].text
        
        return "Tool executed successfully (no output)"
    
    def get_tool_descriptions(self) -> list[dict]:
        """Get tool descriptions for the LLM"""
        return [
            {
                "name": tool_name,
                "description": tool_info["tool"].description,
                "input_schema": tool_info["tool"].inputSchema
            }
            for tool_name, tool_info in self.available_tools.items()
        ]
    
    async def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        await self.exit_stack.aclose()
        print("Cleanup complete!")


class CodeReviewAgentMCP:
    """Code review agent powered by MCP"""
    
    def __init__(self, mcp_manager: MCPClientManager, model="gpt-4o-mini"):
        self.mcp = mcp_manager
        self.model = model
    
    async def think(self, user_input: str) -> str:
        """LLM decides which tool to use"""
        tool_descriptions = self.mcp.get_tool_descriptions()
        tools_list = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in tool_descriptions
        ])
        
        prompt = f"""
        You are a code assistant with access to these tools:
        
        {tools_list}
        
        Based on the user request, decide which tool to use.
        Reply ONLY with JSON: {{"tool": "tool_name", "args": {{"param": "value"}}}}
        
        Example: {{"tool": "analyze_code", "args": {{"code": "def foo(): pass"}}}}
        
        User request: {user_input}
        """
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    async def act(self, decision: str) -> str:
        """Execute the chosen tool"""
        try:
            parsed = json.loads(decision)
            tool_name = parsed["tool"]
            args = parsed.get("args", {})
            
            result = await self.mcp.call_tool(tool_name, args)
            return result
        except Exception as e:
            return f"Error executing tool: {e}"
    
    async def run(self, user_input: str) -> str:
        """Complete think-act loop"""
        print(f"Thinking about: {user_input}")
        decision = await self.think(user_input)
        print(f"Decision: {decision}")
        
        print(f"\nExecuting...")
        result = await self.act(decision)
        print(f"\nResult:\n{result}")
        
        return result


async def main():
    """Main execution function"""
    print("=" * 60)
    print("MCP Code Review Agent")
    print("=" * 60)
    print()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    mcp_manager = MCPClientManager()
    
    try:
        # Connect to MCP server
        await mcp_manager.connect_to_server("code_review_mcp_server.py")
        
        # Create agent
        agent = CodeReviewAgentMCP(mcp_manager)
        
        # Test with a code snippet
        code_snippet = """
def divide(a, b):
    return a / b
        """
        
        user_request = f"Please analyze this code: {code_snippet}"
        result = await agent.run(user_request)
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        
    except FileNotFoundError:
        print("Error: code_review_mcp_server.py not found")
        print("Make sure the MCP server file is in the same directory")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await mcp_manager.cleanup()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())