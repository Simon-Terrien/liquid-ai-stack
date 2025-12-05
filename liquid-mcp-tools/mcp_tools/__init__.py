# mcp_tools/__init__.py
"""
Liquid MCP Tools - Model Context Protocol servers for the LiquidAI stack.

Provides MCP servers for:
- RAG retrieval and search
- Filesystem operations (sandboxed)
- PDF extraction (coming soon)

Usage with Pydantic AI:
    from pydantic_ai import Agent
    from pydantic_ai.mcp import MCPServerStdio
    
    rag_server = MCPServerStdio(
        'python', args=['-m', 'mcp_tools.servers.rag_server']
    )
    
    agent = Agent('model', toolsets=[rag_server])
"""

__version__ = "0.1.0"
