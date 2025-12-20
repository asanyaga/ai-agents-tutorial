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
        
        self.sessions: dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.available_tools: dict[str, dict] = {}
    
    async def connect_to_server(self, server_name: str,server_script_path: str):
        """Connect to an MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        is_npm_package = server_script_path.startswith("npx")

        if not (is_python or is_js or is_npm_package):
            raise ValueError("Server script must be a .py or .js file")
        
        if is_npm_package:
            parts = server_script_path.split()
            command = parts[0]
            args = parts[1:]
        else:
            command = "python" if is_python else "node"
            args = [server_script_path]

        server_env = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}


        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=server_env
        )
        
        print(f"Connecting to MCP server: {server_script_path}")
        
        # Use AsyncExitStack (official pattern)
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        
        session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        await session.initialize()
        
        self.sessions[server_name] = session

        # Discover tools from this server
        response = await session.list_tools()
        for tool in response.tools:
            tool_key = f"{server_name}.{tool.name}"
            self.available_tools[tool_key] = {
                "server_name":server_name,
                "session":session,
                "tool": tool}
            #print(f"Discovered tool: {tool.name}")
            #print(f"Tool Spec: {tool}")
        print(f"Connected successfully!\n")
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool through MCP"""

        if tool_name not in self.available_tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool_info = self.available_tools[tool_name]
        session = tool_info["session"]

        actual_tool_name = tool_info["tool"].name

        print(f"In MCPClientManager Calling tool {actual_tool_name} with arguments {arguments}")

        result = await session.call_tool(actual_tool_name, arguments)
        
        if result.content and len(result.content) > 0:
            return result.content[0].text
        
        return "Tool executed successfully (no output)"
    
    def get_tool_descriptions(self) -> list[dict]:
        """Get tool descriptions for the LLM"""
        return [
            {
                "name": tool_key,
                "description": tool_info["tool"].description,
                "inputSchema": tool_info["tool"].inputSchema
            }
            for tool_key, tool_info in self.available_tools.items()
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
        self.conversation_history = []
    
    async def think(self, user_input: str) -> str:
        """LLM decides which tool to use"""
        tool_descriptions = self.mcp.get_tool_descriptions()

        tools_list = "\n".join([
            f"tool:{tool['name']},\ndescription:{tool['description']}\n,parameters:{tool.get('inputSchema', {}).get('properties', {})}"
            for tool in tool_descriptions
        ])
        
        prompt = f"""
        You are a code assistant with access to these tools via MCP:
        
        {tools_list}
        
        Based on the user request and conversation history, decide which tool to use next.
        Reply ONLY with JSON: {{"tool": "tool_name", "args": {{...}}}}
        
        IMPORTANT: The "args" object must use the exact parameter names shown in each tool's Parameters schema above.
        
        If the task is complete, reply ONLY with: {{"done": true, "summary": "what was accomplished"}}

        Conversation history:
        {json.dumps(self.conversation_history, indent=2) if self.conversation_history else "No history yet"}

        User request: {user_input}
        """ 

        response = openai.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}]
        )

        print(f"PROMPT=========={prompt}\nRESPONSE=========={response.output_text}")

        return response.output_text
    
    async def act(self, decision: str) -> tuple[str, bool]:
        """Execute the chosen tool"""
        try:
            print(f"In act() about to parse decision")
            parsed = json.loads(decision)

            print(f"In act() parsed decision {parsed}")
            if parsed.get("done"):
                return parsed.get("summary") , True

            tool_name = parsed["tool"]
            args = parsed.get("args", {})
            
            print(f"In act() about to make tool call")
            result = await self.mcp.call_tool(tool_name, args)

            self.conversation_history.append(f"I used tool {tool_name} with args {args}. Result: {result}")

            return result, False
        except Exception as e:
            error_msg  = f"Error executing tool {e}"
            print(error_msg)
            return error_msg, False
    
    async def run(self, user_input: str,max_steps: int=10) -> str:
        """Complete think-act loop"""
        print(f"Thinking about task: {user_input}")
        self.conversation_history = []
        current_request = user_input

        for step in range(1, max_steps+1):
            print(f"Step: {step}")
            print("-"* 40)
            decision = await self.think(current_request)
            print(f"Decision: {decision}")
            
            print(f"\nExecuting...")
            result, is_done = await self.act(decision)
            print(f"\nResult:\n{result}")

            if is_done:
                print("Task complete")
                return result
            current_request = f"Previous result: {result}\nContinue with the original task {user_input}"
            
        summary = f"Workflow incomplete after {max_steps} steps. Last result: {result}"
        print(f"\n{summary}")
        return summary


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
        await mcp_manager.connect_to_server(
            server_name="filesystem",
            server_script_path=f"npx -y @modelcontextprotocol/server-filesystem {os.getcwd()}"
        )
        # Create agent
        agent = CodeReviewAgentMCP(mcp_manager)

        user_request = f"Review the code in sample.py and report any issues."

        result = await agent.run(user_request,max_steps=5)
        
        print("\n" + "=" * 60)
        print(f"Task Complete:\nTask: {user_request}\nResult: {result}")
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