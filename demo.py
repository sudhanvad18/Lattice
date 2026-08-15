#!/usr/bin/env python3
"""
Lattice Local Demo — Run the full agent team on a task using Ollama.

Prerequisites:
    1. Ollama running:     ollama serve
    2. Model pulled:       ollama pull llama3.2
    3. Lattice installed:  pip install -e . --no-deps && pip install <deps>

Usage:
    python demo.py "Write a technical overview of fan blade erosion"
    python demo.py  # Uses a default demo task
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

# Ensure packages are importable
sys.path.insert(0, str(Path(__file__).parent / "packages" / "core"))
sys.path.insert(0, str(Path(__file__).parent / "packages" / "api"))


async def main():
    from lattice.agents.orchestrator import OrchestratorAgent
    from lattice.agents.researcher import ResearcherAgent
    from lattice.agents.state import AgentState, TaskStatus
    from lattice.graph.engine import NetworkXBackend
    from lattice.graph.models import Entity, Relation
    from lattice.guardrails import GuardrailRunner
    from lattice.inference.provider import OllamaProvider
    from lattice.observability.tracing import TracingProvider
    from lattice.retrieval.vector_store import VectorStore

    # --- Config ---
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Write a concise technical summary of turbofan engine blade erosion: "
        "causes, detection methods, and recommended maintenance intervals."
    )

    print("\n" + "=" * 60)
    print("  LATTICE — Local Demo")
    print("=" * 60)
    print(f"\n  Task: {task}")
    print(f"  Model: llama3.2 via Ollama (localhost:11434)")
    print("=" * 60 + "\n")

    # --- Setup Knowledge Base with demo content ---
    print("[1/5] Setting up knowledge base...")
    kg = NetworkXBackend()

    # Add demo entities
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

    # Add relations
    relations = [
        ("TurboFan Engine", "Fan Blade", "HAS_COMPONENT"),
        ("TurboFan Engine", "Compressor", "HAS_COMPONENT"),
        ("Fan Blade", "Blade Erosion", "HAS_FAILURE_MODE"),
        ("Fan Blade", "Ti-6Al-4V", "MADE_OF"),
        ("Blade Erosion", "Borescope Inspection", "DETECTED_BY"),
        ("Blade Erosion", "Eddy Current Testing", "DETECTED_BY"),
        ("Compressor", "Thermal Fatigue", "HAS_FAILURE_MODE"),
    ]

    entity_map = {e.name: e for e in entities}
    for src, tgt, rel_type in relations:
        kg.add_relation(Relation(
            source_id=entity_map[src].id,
            target_id=entity_map[tgt].id,
            relation_type=rel_type,
        ))

    print(f"    KG loaded: {len(entities)} entities, {len(relations)} relations")

    # --- Setup Vector Store with demo chunks ---
    print("[2/5] Setting up vector store...")
    vs = VectorStore(persist_dir="data/demo_chroma")

    demo_chunks = [
        "Fan blade erosion occurs when airborne particulates (sand, volcanic ash, ice crystals) impact the leading edge of rotating fan blades at high velocity. The erosion rate depends on particle size, concentration, and blade tip speed. Ti-6Al-4V blades show 0.002mm/1000hr erosion rate under normal conditions.",
        "Detection of blade erosion relies on three primary methods: (1) borescope inspection at 500 flight-hour intervals for visual assessment, (2) eddy current testing for sub-surface damage mapping, and (3) blade tip clearance monitoring via capacitive probes for real-time wear tracking.",
        "Recommended maintenance intervals for fan blade erosion: Visual inspection every 500 flight hours. Detailed measurement every 2,000 hours. Blend repair when erosion exceeds 0.5mm depth. Blade replacement at 1.2mm depth or when leading edge radius falls below specification.",
        "Root causes of accelerated blade erosion include: operations in high-particulate environments (desert, volcanic), improper engine wash procedures, degraded inlet filtration, and operation above rated thrust settings causing increased relative airspeed at blade tips.",
    ]

    # Add directly via ChromaDB collection (bypassing Chunk model for demo simplicity)
    vs._collection.upsert(
        ids=[f"demo_chunk_{i}" for i in range(len(demo_chunks))],
        documents=demo_chunks,
        metadatas=[{"source": "demo_knowledge_base", "doc_type": "text", "document_id": "demo", "index": i, "token_count": 50} for i in range(len(demo_chunks))],
    )
    print(f"    Vector store loaded: {vs.count} chunks")

    # --- Setup Inference ---
    print("[3/5] Connecting to Ollama (llama3.2)...")
    provider = OllamaProvider(base_url="http://localhost:11434", model="llama3.2")

    # Quick connectivity check
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

    # --- Setup Tracing ---
    tracing = TracingProvider()

    # --- Run Orchestrator ---
    print("[4/5] Running agent team...")
    print()

    researcher = ResearcherAgent(provider=provider, vector_store=vs, kg_backend=kg)
    orchestrator = OrchestratorAgent(
        provider=provider,
        researcher=researcher,
    )

    start = time.time()

    with tracing.trace("demo_task", task_id="demo-001", tags=["demo", "local"]):
        result = await orchestrator.run(task=task, max_iterations=2)

    elapsed = time.time() - start

    # --- Display Results ---
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
        # Print content with indentation
        for line in artifact.content.split("\n")[:30]:
            print(f"    {line}")
        if len(artifact.content.split("\n")) > 30:
            print("    ... (truncated)")

    # --- Run Guardrails ---
    print("\n\n[5/5] Running guardrails on output...")
    runner = GuardrailRunner.default(kg_backend=kg)

    for artifact in result.artifacts:
        report = runner.run(artifact)
        print(f"\n  Guardrail Report ({artifact.artifact_type}):")
        print(f"    Overall: {report.overall_verdict.value.upper()}")
        for check in report.results:
            print(f"    - {check.guardrail_name}: {check.verdict.value} (score: {check.score:.2f})")
            if check.flagged_claims:
                print(f"      Flagged: {check.flagged_claims[0][:60]}...")

    # --- Trace Summary ---
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
