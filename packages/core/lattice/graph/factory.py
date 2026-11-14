"""
Knowledge graph backend factory.

Selects the backend from settings (LATTICE_GRAPH_BACKEND) so callers never
construct a concrete backend directly.
"""

from __future__ import annotations

from lattice.config import GraphBackend, Settings, get_settings
from lattice.graph.engine import KnowledgeGraphBackend, NetworkXBackend


def create_graph_backend(settings: Settings | None = None) -> KnowledgeGraphBackend:
    settings = settings or get_settings()

    if settings.graph_backend == GraphBackend.NEO4J:
        from lattice.graph.neo4j_backend import Neo4jBackend

        return Neo4jBackend(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )

    return NetworkXBackend()
