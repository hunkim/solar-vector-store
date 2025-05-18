import uuid
import logging
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, HttpUrl

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# ─── Logging Setup ───────────────────────────────────────────────────────────
logger = logging.getLogger("vector_stores_app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ─── Settings ────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    upstage_api_key: str = Field(..., env="UPSTAGE_API_KEY")
    upstage_dp_url: HttpUrl = Field(
        "https://api.upstage.ai/v1/document-digitization",
        env="UPSTAGE_DP_URL"
    )
    upstage_embed_url: HttpUrl = Field(
        "https://api.upstage.ai/v1/solar/embeddings",
        env="UPSTAGE_EMBED_URL"
    )
    embed_model_passage: str = Field(
        "solar-embedding-1-large-passage",
        env="UPSTAGE_EMBED_MODEL_PASSAGE"
    )
    embed_model_query: str = Field(
        "solar-embedding-1-large-query",
        env="UPSTAGE_EMBED_MODEL_QUERY"
    )
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    request_timeout: float = Field(300.0, env="REQUEST_TIMEOUT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# ─── HTTP & Qdrant Clients ───────────────────────────────────────────────────
http_client = httpx.Client(timeout=settings.request_timeout)
qdrant = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    prefer_grpc=True
)

# ─── In-memory Vector Store Registry ─────────────────────────────────────────
VECTOR_STORES: Dict[str, Dict[str, Any]] = {}

# ─── Pydantic Models ─────────────────────────────────────────────────────────
class CreateVectorStoreRequest(BaseModel):
    name: str
    dimension: int
    distance: rest.Distance = rest.Distance.COSINE

class UpdateVectorStoreRequest(BaseModel):
    name: Optional[str] = None
    distance: Optional[rest.Distance] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

# ─── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(title="Upstage + Qdrant Vector Stores")

@app.post("/vector_stores", status_code=201)
def create_vector_store(req: CreateVectorStoreRequest):
    store_id = str(uuid.uuid4())
    coll_name = f"vs_{store_id}"
    try:
        qdrant.recreate_collection(
            collection_name=coll_name,
            vectors_config=rest.VectorParams(
                size=req.dimension, distance=req.distance
            ),
        )
        # Index 'file' payload for keyword filtering
        qdrant.create_payload_index(
            collection_name=coll_name,
            field_name="file",
            field_schema="keyword",
        )
        VECTOR_STORES[store_id] = {
            "name": req.name,
            "collection": coll_name,
            "dimension": req.dimension,
            "distance": req.distance,
            "files": {}
        }
        logger.info(f"Created vector store {store_id}")
        return {"id": store_id, "name": req.name}
    except Exception as e:
        logger.error(f"Error creating Qdrant collection {coll_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create vector store")

@app.get("/vector_stores")
def list_vector_stores():
    return [
        {"id": sid, "name": m["name"], "dimension": m["dimension"], "distance": m["distance"]}
        for sid, m in VECTOR_STORES.items()
    ]

@app.get("/vector_stores/{store_id}")
def get_vector_store(store_id: str):
    meta = VECTOR_STORES.get(store_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Vector store not found")
    return {"id": store_id, **meta}

@app.patch("/vector_stores/{store_id}")
def update_vector_store(store_id: str, req: UpdateVectorStoreRequest):
    meta = VECTOR_STORES.get(store_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Vector store not found")
    try:
        if req.name:
            meta["name"] = req.name
        if req.distance and req.distance != meta["distance"]:
            qdrant.recreate_collection(
                collection_name=meta["collection"],
                vectors_config=rest.VectorParams(
                    size=meta["dimension"], distance=req.distance
                ),
            )
            # Re-index 'file' payload after recreation
            qdrant.create_payload_index(
                collection_name=meta["collection"],
                field_name="file",
                field_schema="keyword",
            )
            meta["distance"] = req.distance
        logger.info(f"Updated vector store {store_id}")
        return {"id": store_id, **meta}
    except Exception as e:
        logger.error(f"Error updating vector store {store_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update vector store")

@app.delete("/vector_stores/{store_id}")
def delete_vector_store(store_id: str):
    meta = VECTOR_STORES.pop(store_id, None)
    if not meta:
        raise HTTPException(status_code=404, detail="Vector store not found")
    try:
        qdrant.delete_collection(collection_name=meta["collection"])
        logger.info(f"Deleted vector store {store_id}")
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Error deleting vector store {store_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete vector store")

@app.post("/vector_stores/{store_id}/files", status_code=201)
def upload_file(store_id: str, file: UploadFile = File(...)):
    meta = VECTOR_STORES.get(store_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Vector store not found")
    try:
        content = file.file.read()
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Invalid file upload")

    # Document parsing
    try:
        dp_resp = http_client.post(
            str(settings.upstage_dp_url),
            headers={"Authorization": f"Bearer {settings.upstage_api_key}"},
            files={"document": (file.filename, content)},
            data={
                "ocr": "force",
                "base64_encoding": "['table']",
                "model": "document-parse"
            }
        )
        dp_resp.raise_for_status()
        json_resp = dp_resp.json()
        elements = json_resp.get("elements")
        if isinstance(elements, list) and elements:
            pages_map: Dict[int, List[str]] = {}
            for el in elements:
                pg = el.get("page", 1)
                html = el.get("content", {}).get("html", "")
                if html:
                    pages_map.setdefault(pg, []).append(html)
            pages = ["\n".join(pages_map[p]) for p in sorted(pages_map)]
        else:
            content_obj = json_resp.get("content", {})
            html = content_obj.get("html") or content_obj.get("text") or ""
            pages = [html] if html else []
        if not pages:
            raise ValueError("Invalid parse response structure: no pages extracted")
    except (httpx.HTTPError, ValueError) as e:
        logger.error(f"Document parse failed: {e}")
        raise HTTPException(status_code=500, detail="Document parsing failed")

    points: List[rest.PointStruct] = []
    for idx, text in enumerate(pages):
        try:
            emb_resp = http_client.post(
                str(settings.upstage_embed_url),
                headers={"Authorization": f"Bearer {settings.upstage_api_key}"},
                json={"model": settings.embed_model_passage, "input": [text]}
            )
            emb_resp.raise_for_status()
            vector = emb_resp.json()["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding failed for page {idx}: {e}")
            continue
        points.append(
            rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"file": file.filename, "page": idx}
            )
        )

    if not points:
        raise HTTPException(status_code=500, detail="No embeddings generated")

    try:
        qdrant.upsert(collection_name=meta["collection"], points=points)
    except Exception as e:
        logger.error(f"Qdrant upsert failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store embeddings")

    file_id = str(uuid.uuid4())
    meta["files"][file_id] = {"filename": file.filename, "pages": len(pages)}
    logger.info(f"Ingested file {file.filename} as {file_id} ({len(pages)} pages)")
    return {"file_id": file_id, "pages": len(pages)}

@app.get("/vector_stores/{store_id}/files")
def list_files(store_id: str):
    meta = VECTOR_STORES.get(store_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Vector store not found")
    return meta["files"]

@app.get("/vector_stores/{store_id}/files/{file_id}")
def get_file(store_id: str, file_id: str):
    meta = VECTOR_STORES.get(store_id)
    file_meta = meta["files"].get(file_id) if meta else None
    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
    return file_meta

@app.delete("/vector_stores/{store_id}/files/{file_id}")
def delete_file(store_id: str, file_id: str):
    meta = VECTOR_STORES.get(store_id)
    file_meta = meta["files"].get(file_id) if meta else None
    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        qdrant.delete(
            collection_name=meta["collection"],
            points_selector=rest.Filter(must=[rest.FieldCondition(
                key="file", match=rest.MatchValue(value=file_meta["filename"]) )])
        )
        del meta["files"][file_id]
        logger.info(f"Deleted file {file_id} from store {store_id}")
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Failed to delete file vectors: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete file vectors")

@app.post("/vector_stores/{store_id}/query")
def query_vectors(store_id: str, req: QueryRequest):
    meta = VECTOR_STORES.get(store_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Vector store not found")
    try:
        emb_resp = http_client.post(
            str(settings.upstage_embed_url),
            headers={"Authorization": f"Bearer {settings.upstage_api_key}"},
            json={"model": settings.embed_model_query, "input": [req.query]}
        )
        emb_resp.raise_for_status()
        q_vec = emb_resp.json()["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to embed query")

    try:
        hits = qdrant.search(collection_name=meta["collection"], query_vector=q_vec, limit=req.top_k)
        return [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits]
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        raise HTTPException(status_code=500, detail="Search operation failed")
