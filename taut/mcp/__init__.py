"""MCP (Model Context Protocol) stdio server for Taut (optional extra).

Install the extra to use it:

    pip install "taut[mcp]"

Then run the server:

    taut-mcp-server            # console script
    python -m taut.mcp.server  # equivalent module form

The submodules ``device`` and ``tools`` are importable without the ``mcp``
dependency; ``server`` requires the extra.
"""

__all__ = ["device", "server", "tools"]
