# mcp_tools/servers/filesystem_server.py
"""
MCP Server for filesystem operations.

Provides secure, sandboxed file operations for agents.
"""
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Liquid Filesystem Tools")

# Sandboxed root directory
SANDBOX_ROOT = Path(os.environ.get("SANDBOX_ROOT", "/data"))


def resolve_path(path: str) -> Path:
    """
    Resolve a path within the sandbox.
    
    Raises:
        ValueError: If path escapes sandbox
    """
    # Resolve to absolute path within sandbox
    resolved = (SANDBOX_ROOT / path).resolve()

    # Ensure it's within sandbox
    if not str(resolved).startswith(str(SANDBOX_ROOT.resolve())):
        raise ValueError(f"Path escapes sandbox: {path}")

    return resolved


@mcp.tool()
async def read_file(
    ctx: Context,
    path: str,
    encoding: str = "utf-8",
) -> str:
    """
    Read contents of a file.
    
    Args:
        path: Path relative to sandbox root
        encoding: Text encoding (default utf-8)
        
    Returns:
        File contents as string
    """
    logger.info(f"MCP read_file: {path}")

    try:
        resolved = resolve_path(path)

        if not resolved.exists():
            return f"File not found: {path}"

        if not resolved.is_file():
            return f"Not a file: {path}"

        # Limit file size
        if resolved.stat().st_size > 1_000_000:  # 1MB limit
            return f"File too large (>1MB): {path}"

        return resolved.read_text(encoding=encoding, errors="replace")

    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {e}"


@mcp.tool()
async def write_file(
    ctx: Context,
    path: str,
    content: str,
    encoding: str = "utf-8",
) -> str:
    """
    Write content to a file.
    
    Args:
        path: Path relative to sandbox root
        content: Content to write
        encoding: Text encoding (default utf-8)
        
    Returns:
        Success message or error
    """
    logger.info(f"MCP write_file: {path}")

    try:
        resolved = resolve_path(path)

        # Create parent directories
        resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(content, encoding=encoding)

        return f"Successfully wrote {len(content)} bytes to {path}"

    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return f"Error writing file: {e}"


@mcp.tool()
async def list_directory(
    ctx: Context,
    path: str = ".",
    recursive: bool = False,
) -> str:
    """
    List contents of a directory.
    
    Args:
        path: Path relative to sandbox root (default: root)
        recursive: Whether to list recursively
        
    Returns:
        Directory listing
    """
    logger.info(f"MCP list_directory: {path}")

    try:
        resolved = resolve_path(path)

        if not resolved.exists():
            return f"Directory not found: {path}"

        if not resolved.is_dir():
            return f"Not a directory: {path}"

        items = []
        if recursive:
            for item in resolved.rglob("*"):
                rel = item.relative_to(resolved)
                prefix = "📁 " if item.is_dir() else "📄 "
                items.append(f"{prefix}{rel}")
        else:
            for item in resolved.iterdir():
                prefix = "📁 " if item.is_dir() else "📄 "
                size = ""
                if item.is_file():
                    size = f" ({item.stat().st_size} bytes)"
                items.append(f"{prefix}{item.name}{size}")

        if not items:
            return f"Directory is empty: {path}"

        return "\n".join(sorted(items))

    except ValueError as e:
        return str(e)
    except Exception as e:
        logger.error(f"Error listing directory: {e}")
        return f"Error listing directory: {e}"


@mcp.tool()
async def file_info(
    ctx: Context,
    path: str,
) -> str:
    """
    Get information about a file or directory.
    
    Args:
        path: Path relative to sandbox root
        
    Returns:
        File/directory metadata
    """
    logger.info(f"MCP file_info: {path}")

    try:
        resolved = resolve_path(path)

        if not resolved.exists():
            return f"Path not found: {path}"

        stat = resolved.stat()

        info = f"""Path: {path}
Type: {'Directory' if resolved.is_dir() else 'File'}
Size: {stat.st_size} bytes
Modified: {stat.st_mtime}
"""

        if resolved.is_file():
            info += f"Extension: {resolved.suffix or 'none'}\n"

        return info

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error getting file info: {e}"


@mcp.tool()
async def delete_file(
    ctx: Context,
    path: str,
) -> str:
    """
    Delete a file.
    
    Args:
        path: Path relative to sandbox root
        
    Returns:
        Success message or error
    """
    logger.info(f"MCP delete_file: {path}")

    try:
        resolved = resolve_path(path)

        if not resolved.exists():
            return f"File not found: {path}"

        if resolved.is_dir():
            return f"Cannot delete directory with this tool: {path}"

        resolved.unlink()

        return f"Successfully deleted: {path}"

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error deleting file: {e}"


@mcp.tool()
async def search_files(
    ctx: Context,
    pattern: str,
    path: str = ".",
) -> str:
    """
    Search for files matching a pattern.
    
    Args:
        pattern: Glob pattern (e.g., "*.pdf", "**/*.txt")
        path: Starting directory relative to sandbox root
        
    Returns:
        List of matching files
    """
    logger.info(f"MCP search_files: {pattern} in {path}")

    try:
        resolved = resolve_path(path)

        if not resolved.exists():
            return f"Directory not found: {path}"

        matches = list(resolved.glob(pattern))[:100]  # Limit results

        if not matches:
            return f"No files matching '{pattern}' in {path}"

        results = []
        for m in matches:
            rel = m.relative_to(resolved)
            size = m.stat().st_size if m.is_file() else 0
            results.append(f"{rel} ({size} bytes)")

        return "\n".join(results)

    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error searching files: {e}"


def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Filesystem MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument(
        "--sandbox-root",
        type=str,
        default="/data",
        help="Root directory for sandboxed operations"
    )

    args = parser.parse_args()

    global SANDBOX_ROOT
    SANDBOX_ROOT = Path(args.sandbox_root)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", port=args.port)


if __name__ == "__main__":
    main()
