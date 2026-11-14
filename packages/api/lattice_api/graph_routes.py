"""Knowledge graph queries and knowledge base ingestion."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool

from lattice_api.models import (
    CommunitiesResponse,
    Edge,
    EntityDetailResponse,
    GraphResponse,
    GraphStatsResponse,
    IngestRequest,
    IngestResponse,
    Node,
)


def get_knowledge_base(request: Request):
    """Return the app's knowledge base, or report unavailable startup."""
    kb = getattr(request.app.state, "knowledge_base", None)
    if kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base unavailable")
    return kb


KnowledgeBaseDependency = Annotated[Any, Depends(get_knowledge_base)]

router = APIRouter(prefix="/kg", tags=["knowledge graph"])


def _node(entity) -> Node:
    return Node.model_validate(entity, from_attributes=True)


def _edge(relation) -> Edge:
    return Edge(
        id=relation.id,
        source=relation.source_id,
        target=relation.target_id,
        relation_type=relation.relation_type,
        provenance=relation.provenance,
        confidence=relation.confidence,
    )


@router.get("/graph", response_model=GraphResponse)
def get_graph(kb: KnowledgeBaseDependency, limit: int = Query(500, ge=1, le=5000)):
    entities = kb.graph.get_all_entities()
    nodes = [_node(entity) for entity in entities[:limit]]
    ids = {node.id for node in nodes}
    edges = [
        _edge(relation) for relation in kb.graph.get_all_relations()
        if relation.source_id in ids and relation.target_id in ids
    ]
    return GraphResponse(nodes=nodes, edges=edges, truncated=len(entities) > limit)


@router.get("/stats", response_model=GraphStatsResponse)
def get_stats(kb: KnowledgeBaseDependency):
    return {**kb.stats, "backend": type(kb.graph).__name__}


@router.get("/entities", response_model=list[Node])
def get_entities(
    kb: KnowledgeBaseDependency,
    q: str = "",
    entity_type: str | None = None,
    limit: int = Query(50, ge=1, le=5000),
):
    if q:
        entities = kb.graph.search_entities(q, entity_type=entity_type)
    elif entity_type:
        entities = kb.graph.get_entities_by_type(entity_type)
    else:
        entities = kb.graph.get_all_entities()
    return [_node(entity) for entity in entities[:limit]]


@router.get("/entities/{entity_id}", response_model=EntityDetailResponse)
def get_entity(entity_id: str, kb: KnowledgeBaseDependency):
    entity = kb.graph.get_entity(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    relations = [
        relation for relation in kb.graph.get_all_relations()
        if entity_id in (relation.source_id, relation.target_id)
    ]
    neighbor_ids = {
        endpoint for relation in relations
        for endpoint in (relation.source_id, relation.target_id)
    } - {entity_id}
    neighbors = [kb.graph.get_entity(node_id) for node_id in sorted(neighbor_ids)]
    return EntityDetailResponse(
        entity=_node(entity),
        neighbors=[_node(node) for node in neighbors if node is not None],
        relations=[_edge(relation) for relation in relations],
    )


@router.get("/communities", response_model=CommunitiesResponse)
def get_communities(kb: KnowledgeBaseDependency):
    from lattice.graph.clustering import cluster_graph

    result = cluster_graph(kb.graph)
    entities = {entity.id: entity for entity in kb.graph.get_all_entities()}

    def reference(node_id):
        return {"id": node_id, "name": entities[node_id].name}

    return {
        "communities": [
            {"id": community_id, "size": len(members),
             "members": [reference(node_id) for node_id in members]}
            for community_id, members in result.communities.items()
        ],
        "god_nodes": [reference(node_id) for node_id in result.god_nodes],
        "surprising_connections": [
            {"source": source, "target": target, "explanation": explanation}
            for source, target, explanation in result.surprising_connections
        ],
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest, kb: KnowledgeBaseDependency):
    def ingest_and_stats():
        chunks = kb.ingest_text(request.text, source=request.source)
        return IngestResponse(chunks=chunks, stats=kb.stats)

    return await run_in_threadpool(ingest_and_stats)
