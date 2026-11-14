#!/usr/bin/env python3
"""
Lattice Local Demo — Run the full agent team on a task using Ollama.

Showcases: agent orchestration, knowledge graph with provenance & community
detection, code AST parsing, ingestion cache, guardrails, and tracing.

Prerequisites:
    1. Ollama running:     ollama serve
    2. Model pulled:       ollama pull llama3.2
    3. Lattice installed:  pip install -e packages/core --no-deps

Usage:
    python demo.py "Write a technical overview of fan blade erosion"
    python demo.py  # Uses a default demo task
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "packages" / "core"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "api"))


async def main():
    from lattice.agents.orchestrator import OrchestratorAgent
    from lattice.agents.researcher import ResearcherAgent
    from lattice.agents.state import AgentState, TaskStatus
    from lattice.graph.clustering import cluster_graph
    from lattice.graph.engine import NetworkXBackend
    from lattice.graph.extractor import CodeExtractor
    from lattice.graph.models import Entity, Relation
    from lattice.guardrails import GuardrailRunner
    from lattice.ingestion.cache import IngestionCache
    from lattice.ingestion.code_parser import parse_code
    from lattice.ingestion.chunker import ChunkingEngine, ChunkConfig
    from lattice.ingestion.models import DocumentMetadata, DocumentType, Chunk
    from lattice.inference.provider import OllamaProvider
    from lattice.observability.tracing import TracingProvider
    from lattice.retrieval.vector_store import VectorStore

    # --- Config ---
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Write a concise technical summary of turbofan engine blade erosion: "
        "causes, detection methods, and recommended maintenance intervals."
    )

    print("\n" + "=" * 60)
    print("  LATTICE — Local Demo (with Graphify Enhancements)")
    print("=" * 60)
    print(f"\n  Task: {task}")
    print(f"  Model: llama3.2 via Ollama (localhost:11434)")
    print("=" * 60 + "\n")

    # ================================================================
    # 1. KNOWLEDGE GRAPH WITH PROVENANCE LABELS
    # ================================================================
    print("[1/7] Building knowledge graph (with edge provenance)...")
    kg = NetworkXBackend()

    entities = [
        Entity(name="TurboFan Engine", entity_type="component", description="High-bypass turbofan engine used in commercial aviation"),
        Entity(name="Fan Blade", entity_type="component", description="First-stage rotating airfoil in the fan section"),
        Entity(name="Compressor", entity_type="component", description="Multi-stage axial compressor for air pressurization"),
        Entity(name="Blade Erosion", entity_type="failure_mode", description="Progressive material loss from fan blade leading edges due to particulate ingestion"),
        Entity(name="Thermal Fatigue", entity_type="failure_mode", description="Cyclic thermal stress causing crack initiation in hot-section components"),
        Entity(name="Borescope Inspection", entity_type="procedure", description="Visual inspection of internal engine components using fiber-optic borescope at 500 flight-hour intervals"),
        Entity(name="Eddy Current Testing", entity_type="procedure", description="Non-destructive electromagnetic testing to detect surface and near-surface cracks"),
        Entity(name="Ti-6Al-4V", entity_type="material", description="Titanium alloy used for fan blades — excellent strength-to-weight ratio"),
    ]

    for e in entities:
        kg.add_entity(e)

    relations_data = [
        ("TurboFan Engine", "Fan Blade", "HAS_COMPONENT", "extracted"),
        ("TurboFan Engine", "Compressor", "HAS_COMPONENT", "extracted"),
        ("Fan Blade", "Blade Erosion", "HAS_FAILURE_MODE", "extracted"),
        ("Fan Blade", "Ti-6Al-4V", "MADE_OF", "extracted"),
        ("Blade Erosion", "Borescope Inspection", "DETECTED_BY", "extracted"),
        ("Blade Erosion", "Eddy Current Testing", "DETECTED_BY", "extracted"),
        ("Compressor", "Thermal Fatigue", "HAS_FAILURE_MODE", "inferred"),
        ("Thermal Fatigue", "Borescope Inspection", "DETECTED_BY", "inferred"),
    ]

    entity_map = {e.name: e for e in entities}
    for src, tgt, rel_type, prov in relations_data:
        kg.add_relation(Relation(
            source_id=entity_map[src].id,
            target_id=entity_map[tgt].id,
            relation_type=rel_type,
            provenance=prov,
        ))

    all_rels = kg.get_all_relations()
    prov_counts = {}
    for r in all_rels:
        prov_counts[r.provenance] = prov_counts.get(r.provenance, 0) + 1
    print(f"    KG: {len(entities)} entities, {len(all_rels)} relations")
    print(f"    Provenance: {prov_counts}")

    # ================================================================
    # 2. COMMUNITY DETECTION (Leiden)
    # ================================================================
    print("\n[2/7] Running community detection...")
    cluster_result = cluster_graph(kg, top_k_gods=3)
    print(f"    Communities: {cluster_result.num_communities}")
    for cid, members in cluster_result.communities.items():
        names = [kg.get_entity(eid).name for eid in members if kg.get_entity(eid)]
        print(f"      Community {cid}: {', '.join(names)}")
    if cluster_result.god_nodes:
        god_names = [kg.get_entity(gid).name for gid in cluster_result.god_nodes if kg.get_entity(gid)]
        print(f"    God nodes (highest degree): {', '.join(god_names)}")
    if cluster_result.surprising_connections:
        print(f"    Surprising connections: {len(cluster_result.surprising_connections)}")
        for _, _, explanation in cluster_result.surprising_connections[:3]:
            print(f"      {explanation}")

    # ================================================================
    # 3. CODE AST PARSING (Tree-sitter)
    # ================================================================
    print("\n[3/7] Demonstrating code AST parsing...")
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write('''\
import numpy as np
from dataclasses import dataclass

@dataclass
class BladeMetrics:
    erosion_depth: float
    tip_clearance: float

    def is_critical(self) -> bool:
        return self.erosion_depth > 1.2

def analyze_blade(metrics: BladeMetrics) -> str:
    if metrics.is_critical():
        return "REPLACE"
    return "MONITOR"

def main():
    m = BladeMetrics(erosion_depth=0.8, tip_clearance=2.1)
    result = analyze_blade(m)
    print(result)
''')
        code_path = Path(f.name)

    doc = parse_code(code_path)
    ast_nodes = doc.metadata.extra.get("ast_nodes", [])
    print(f"    Parsed: {doc.metadata.title} ({doc.metadata.extra['language']})")
    print(f"    AST nodes found: {len(ast_nodes)}")

    chunker = ChunkingEngine(ChunkConfig(max_tokens=500, min_chunk_tokens=10))
    chunks = chunker.chunk_document(doc)
    for c in chunks:
        c.metadata = doc.metadata

    code_extractor = CodeExtractor()
    code_result = code_extractor.extract(chunks)
    print(f"    Code entities: {len(code_result.entities)}")
    for e in code_result.entities:
        print(f"      {e.entity_type}: {e.name}")
    print(f"    Code relations: {len(code_result.relations)}")
    for r in code_result.relations:
        src = next((e.name for e in code_result.entities if e.id == r.source_id), r.source_id[:8])
        tgt = next((e.name for e in code_result.entities if e.id == r.target_id), r.target_id[:8])
        print(f"      {src} --{r.relation_type}--> {tgt} [{r.provenance}]")

    code_path.unlink(missing_ok=True)

    # ================================================================
    # 4. INGESTION CACHE
    # ================================================================
    print("\n[4/7] Demonstrating ingestion cache...")
    cache_dir = Path("data/demo_cache")
    cache = IngestionCache(cache_dir)

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, dir=".") as f:
        f.write("Fan blade erosion is a critical failure mode in turbofan engines. "
                "Regular inspection at 500-hour intervals is recommended. "
                "Ti-6Al-4V alloy provides excellent erosion resistance.")
        cache_test_path = Path(f.name)

    print(f"    Cache hit (first time): {cache.is_cached(cache_test_path)}")
    dummy_chunks = [Chunk(
        document_id="demo", content="cached content", index=0,
        metadata=DocumentMetadata(source=str(cache_test_path), doc_type=DocumentType.TEXT),
    )]
    cache.save(cache_test_path, dummy_chunks)
    print(f"    Cache hit (after save): {cache.is_cached(cache_test_path)}")
    loaded = cache.load(cache_test_path)
    print(f"    Loaded from cache: {len(loaded)} chunks")
    cache_test_path.unlink(missing_ok=True)

    # ================================================================
    # 5. VECTOR STORE
    # ================================================================
    print("\n[5/7] Setting up vector store...")
    vs = VectorStore(persist_dir="data/demo_chroma")

    demo_chunks = [
        "Fan blade erosion occurs when airborne particulates (sand, volcanic ash, ice crystals) impact the leading edge of rotating fan blades at high velocity. The erosion rate depends on particle size, concentration, and blade tip speed. Ti-6Al-4V blades show 0.002mm/1000hr erosion rate under normal conditions.",
        "Detection of blade erosion relies on three primary methods: (1) borescope inspection at 500 flight-hour intervals for visual assessment, (2) eddy current testing for sub-surface damage mapping, and (3) blade tip clearance monitoring via capacitive probes for real-time wear tracking.",
        "Recommended maintenance intervals for fan blade erosion: Visual inspection every 500 flight hours. Detailed measurement every 2,000 hours. Blend repair when erosion exceeds 0.5mm depth. Blade replacement at 1.2mm depth or when leading edge radius falls below specification.",
        "Root causes of accelerated blade erosion include: operations in high-particulate environments (desert, volcanic), improper engine wash procedures, degraded inlet filtration, and operation above rated thrust settings causing increased relative airspeed at blade tips.",
    ]

    vs._collection.upsert(
        ids=[f"demo_chunk_{i}" for i in range(len(demo_chunks))],
        documents=demo_chunks,
        metadatas=[{"source": "demo_knowledge_base", "doc_type": "text", "document_id": "demo", "index": i, "token_count": 50} for i in range(len(demo_chunks))],
    )
    print(f"    Vector store: {vs.count} chunks indexed")

    # ================================================================
    # 6. CONNECT & RUN AGENT TEAM
    # ================================================================
    print("\n[6/7] Connecting to Ollama (llama3.2)...")
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.2")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any("llama3.2" in m for m in models):
                print(f"    WARNING: llama3.2 not found. Available: {models}")
                print("    Run: ollama pull llama3.2")
                return
            print(f"    Connected. Available models: {models}")
    except Exception as e:
        print(f"    ERROR: Cannot connect to Ollama at localhost:11434")
        print(f"    Make sure Ollama is running: ollama serve")
        print(f"    Error: {e}")
        return

    tracing = TracingProvider()

    print("\n    Running agent team on task...")
    print()

    researcher = ResearcherAgent(provider=provider, vector_store=vs, kg_backend=kg)
    orchestrator = OrchestratorAgent(provider=provider, researcher=researcher)

    start = time.time()

    with tracing.trace("demo_task", task_id="demo-001", tags=["demo", "local"]):
        result = await orchestrator.run(task=task, max_iterations=2)

    elapsed = time.time() - start

    # ================================================================
    # 7. RESULTS & GUARDRAILS
    # ================================================================
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\n  Status: {result.status.value}")
    print(f"  Iterations: {result.iteration_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Artifacts: {len(result.artifacts)}")

    for i, artifact in enumerate(result.artifacts):
        print(f"\n  --- Artifact {i+1} ({artifact.artifact_type} by {artifact.source_agent.value}) ---")
        print(f"  Confidence: {artifact.confidence:.0%}")
        if artifact.citations:
            print(f"  Citations: {artifact.citations}")
        print()
        for line in artifact.content.split("\n")[:30]:
            print(f"    {line}")
        if len(artifact.content.split("\n")) > 30:
            print("    ... (truncated)")

    print("\n\n[7/7] Running guardrails on output...")
    runner = GuardrailRunner.default(kg_backend=kg)

    for artifact in result.artifacts:
        report = runner.run(artifact)
        print(f"\n  Guardrail Report ({artifact.artifact_type}):")
        print(f"    Overall: {report.overall_verdict.value.upper()}")
        for check in report.results:
            print(f"    - {check.guardrail_name}: {check.verdict.value} (score: {check.score:.2f})")
            if check.flagged_claims:
                print(f"      Flagged: {check.flagged_claims[0][:60]}...")

    # Trace summary
    print("\n\n  Trace Summary:")
    for t in tracing.traces:
        s = t.summary()
        print(f"    Trace: {s['name']} | Duration: {s['duration_ms']:.0f}ms | "
              f"Generations: {s['generations']} | Tokens: {s['total_tokens']}")

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
