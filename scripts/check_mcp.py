import asyncio
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.storage import Store
from backend.mcp_host import ToolHost
from backend.research import parse_document

async def main():
    with tempfile.TemporaryDirectory() as d:
        store = Store(d)
        project = store.create("splash", "Example prompt")
        other = store.create("splash", "Private project")
        with store.edit(other["id"]) as p:
            p["documents"].append(parse_document("private.txt", b"Not part of the first project"))
        host = ToolHost(d)
        try:
            result = await host.call(project, "search_context", {"query": "example", "project_id":other["id"]})
            print(result)
            assert result["passages"] == []
            try:
                await host.call(project, "search_context", {"query": 123})
                raise AssertionError("Malformed arguments were accepted")
            except ValueError:
                pass
            assert (await host.call(project, "search_context", {"query":"example"}))["passages"] == []
            # Kill only the subprocess launched by this test to exercise recovery.
            import psutil
            owned = [p for p in psutil.Process().children() if "backend.mcp_server" in p.cmdline()]
            assert len(owned) == 1
            owned[0].terminate()
            try:
                await host.call(project, "search_context", {"query":"after interruption"})
            except Exception:
                pass  # The first in-flight request may fail with the interrupted server.
            await asyncio.sleep(2.5)
            assert (await host.call(project, "search_context", {"query":"after recovery"}))["passages"] == []
            print("Persistent MCP SDK connection, scope, malformed-call isolation, and restart passed.")
        finally:
            print("Tool host status:", host.state, host.error)
            await host.close()

if __name__ == "__main__":
    asyncio.run(main())
