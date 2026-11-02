"""Project-scoped local Chroma index orchestrated by a small LangGraph."""
from pathlib import Path
from typing import TypedDict

from backend.ollama import EMBEDDING_MODEL


class IndexState(TypedDict, total=False):
    project_id: str
    directory: str
    documents: list
    query: str
    passages: list
    recursive: bool
    indexed: int


SUPPORTED = {".txt", ".md", ".markdown", ".rst", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".csv", ".html", ".css", ".pdf"}


def _collection(root, project_id):
    import chromadb
    client = chromadb.PersistentClient(path=str(root / "vector-index"))
    return client.get_or_create_collection("project_" + project_id, metadata={"hnsw:space": "cosine"})


async def _embed(models, texts):
    import psutil
    if psutil.virtual_memory().available < 2 * 2**30:
        raise ValueError("Local embeddings need at least 2 GiB of available system memory.")
    vectors = []
    for offset in range(0, len(texts), 32):
        response = await models.request("POST", "/api/embed", json={"model": EMBEDDING_MODEL,
            "input": texts[offset:offset+32], "options": {"num_gpu": 0}, "keep_alive": 0})
        vectors.extend(response.json()["embeddings"])
    if len(vectors) != len(texts):
        raise ValueError("Ollama returned an incomplete embedding batch.")
    return vectors


def _graph(index_node):
    from langgraph.graph import END, START, StateGraph
    graph = StateGraph(IndexState)
    graph.add_node("index", index_node)
    graph.add_edge(START, "index")
    graph.add_edge("index", END)
    return graph.compile()


async def index_documents(root, project_id, documents, models):
    async def node(state):
        import chromadb
        client = chromadb.PersistentClient(path=str(root / "vector-index"))
        name = "project_" + project_id
        chunks = []
        for document in documents:
            for page in document["pages"]:
                text = page["text"]
                for offset in range(0, len(text), 1200):
                    piece = text[offset:offset + 1200].strip()
                    if piece:
                        chunks.append({"id": f"{document['id']}:{page['page']}:{offset}",
                            "text": piece, "name": document["name"], "page": page["page"],
                            "document_id": document["id"]})
        vectors = await _embed(models, [item["text"] for item in chunks]) if chunks else []
        try:
            client.delete_collection(name)
        except Exception:
            pass
        collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        if chunks:
            collection.upsert(ids=[x["id"] for x in chunks], documents=[x["text"] for x in chunks],
                embeddings=vectors, metadatas=[{k: x[k] for k in ("name", "page", "document_id")} for x in chunks])
        return {"project_id": project_id, "indexed": len(chunks)}
    app = _graph(node)
    return await app.ainvoke({"project_id": project_id, "documents": documents})


async def retrieve(root, project_id, query, models, limit=4):
    async def node(state):
        collection = _collection(root, project_id)
        if collection.count() == 0:
            return {"project_id": project_id, "query": query, "passages": []}
        vectors = await _embed(models, [query])
        result = collection.query(query_embeddings=vectors, n_results=min(limit, collection.count()))
        passages = []
        for text, metadata in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0]):
            passages.append({"text": text, **metadata})
        return {"project_id": project_id, "query": query, "passages": passages}
    # Separate graph entry path for retrieval, sharing the same graph-managed state.
    from langgraph.graph import END, START, StateGraph
    graph = StateGraph(IndexState)
    graph.add_node("search", node)
    graph.add_edge(START, "search")
    graph.add_edge("search", END)
    result = await graph.compile().ainvoke({"project_id": project_id, "query": query})
    return result["passages"]


def directory_documents(directory, recursive=False):
    from backend.research import parse_document
    try:
        root = Path(directory).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError("The selected directory does not exist or cannot be read.") from exc
    if not root.is_dir():
        raise ValueError("Choose a local directory.")
    paths = root.rglob("*") if recursive else root.iterdir()
    found = []
    total = 0
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if len(found) >= 200:
            raise ValueError("A directory index is limited to 200 supported files.")
        try:
            size = path.stat().st_size
        except OSError:
            continue
        total += size
        if size > 8_000_000 or total > 100_000_000:
            raise ValueError("Directory indexing is limited to 8 MB per file and 100 MB total.")
        try:
            found.append(parse_document(str(path.relative_to(root)), path.read_bytes()))
        except (OSError, ValueError):
            continue
    if not found:
        raise ValueError("No supported text, code, Markdown, or text PDF files were found.")
    return found


def remove_index(root, project_id):
    if not (root / "vector-index").exists():
        return False
    try:
        import chromadb
    except ImportError:
        return False
    client = chromadb.PersistentClient(path=str(root / "vector-index"))
    try:
        client.delete_collection("project_" + project_id)
    except Exception:
        return False
    return True


def index_status(root, project_id):
    if not (root / "vector-index").exists():
        return {"ready": False, "chunks": 0}
    try:
        import chromadb
    except ImportError:
        return {"ready": False, "chunks": 0, "error": "Install the backend requirements to use local indexing."}
    client = chromadb.PersistentClient(path=str(root / "vector-index"))
    try:
        collection = client.get_collection("project_" + project_id)
        return {"ready": collection.count() > 0, "chunks": collection.count()}
    except Exception:
        return {"ready": False, "chunks": 0}
