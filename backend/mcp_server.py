"""Bundled stdio MCP server; stdout is reserved for the SDK protocol."""
from mcp.server import MCPServer
from backend.storage import Store
from backend.research import Providers, retrieve_context, context_chunks

server = MCPServer("Splash research", version="2.0.0")
providers = Providers()


@server.tool()
async def search_context(project_id: str, query: str, limit: int = 4) -> dict:
    """Retrieve passages only from the host-scoped project."""
    return retrieve_context(Store().get(project_id), query, max(1, min(limit, 5)))


@server.tool()
async def read_context(project_id: str, passage_id: str) -> dict:
    """Read a passage belonging to the host-scoped project."""
    return next(c for c in context_chunks(Store().get(project_id)) if c["id"] == passage_id)


@server.tool()
async def search_scholarly(query: str, provider: str = "semantic_scholar") -> dict:
    """Search Semantic Scholar, Crossref, or optional Google Scholar."""
    return {"sources": await providers.scholarly(query, provider)}


@server.tool()
async def search_official_docs(query: str) -> dict:
    """Find supported official documentation, retaining its public URLs."""
    return {"sources": await providers.official(query)}


@server.tool()
async def fetch_source(url: str) -> dict:
    """Fetch a public web source, never local/private URLs."""
    return {"sources": [await providers.fetch(url)]}


if __name__ == "__main__":
    server.run(transport="stdio")
