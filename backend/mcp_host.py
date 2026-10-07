"""A persistent SDK connection owned by one asyncio task, with scoped dispatch."""
import asyncio
import os
from pathlib import Path
import sys
import json
import jsonschema

LOCAL_TOOLS = {"search_context", "read_context"}
RESEARCH_TOOLS = {"search_scholarly", "search_official_docs", "fetch_source"}


class ToolHost:
    def __init__(self, directory):
        self.directory = str(directory)
        self.queue = asyncio.Queue()
        self.task = None
        self.schemas = {}
        self.state = "starting"
        self.error = None

    def start(self):
        if not self.task:
            self.task = asyncio.create_task(self.run())

    async def run(self):
        from mcp import Client
        from mcp.client.stdio import StdioServerParameters
        while True:
            request = None
            try:
                params = StdioServerParameters(command=sys.executable, args=["-m", "backend.mcp_server"],
                    cwd=str(Path(__file__).resolve().parents[1]), env={**os.environ, "SPLASH_DATA_DIR": self.directory})
                async with Client(params, read_timeout_seconds=25) as client:
                    discovered = await client.list_tools()
                    self.schemas = {t.name: t.input_schema for t in discovered.tools}
                    self.state, self.error = "ready", None
                    while True:
                        request = await self.queue.get()
                        name, args, future = request
                        if future.cancelled():
                            continue
                        try:
                            jsonschema.validate(args, self.schemas[name])
                        except (jsonschema.ValidationError, KeyError) as exc:
                            future.set_exception(ValueError(f"Invalid tool arguments: {exc}"))
                            request = None
                            continue
                        result = await client.call_tool(name, args, read_timeout_seconds=25)
                        if not future.done():
                            if result.is_error:
                                future.set_exception(RuntimeError(" ".join(getattr(c, "text", "") for c in result.content)))
                            else:
                                value = result.structured_content
                                if value is None:
                                    text = "\n".join(getattr(c, "text", "") for c in result.content)
                                    value = json.loads(text)
                                if not isinstance(value, dict):
                                    raise ValueError("Tool returned an unexpected result shape.")
                                future.set_result(value)
                        request = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state, self.error = "unavailable", str(exc)
                if request and not request[2].done():
                    request[2].set_exception(RuntimeError(f"Research tool unavailable: {exc}"))
                await asyncio.sleep(2)

    async def call(self, project, name, arguments):
        allowed = LOCAL_TOOLS | (RESEARCH_TOOLS if project["agent"] != "splash" and project["settings"]["research"] else set())
        if name not in allowed:
            raise ValueError("This agent cannot use that tool.")
        args = dict(arguments)
        if name in LOCAL_TOOLS:
            args["project_id"] = project["id"]
        elif "project_id" in args:
            raise ValueError("Public tools do not accept project scope.")
        if name == "search_scholarly" and args.get("provider") == "google_scholar" and not project["settings"]["scholar"]:
            raise ValueError("Enable experimental Scholar scraping in project settings first.")
        self.start()
        future = asyncio.get_running_loop().create_future()
        await self.queue.put((name, args, future))
        try:
            return await asyncio.wait_for(future, 30)
        finally:
            if not future.done():
                future.cancel()

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
