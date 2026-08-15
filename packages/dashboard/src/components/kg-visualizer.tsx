"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Network, ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { getKnowledgeGraph, KGNode, KGEdge } from "@/lib/api";

interface NodePosition {
  x: number;
  y: number;
  vx: number;
  vy: number;
  node: KGNode;
}

export function KnowledgeGraphView() {
  const [nodes, setNodes] = useState<KGNode[]>([]);
  const [edges, setEdges] = useState<KGEdge[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<KGNode | null>(null);
  const [zoom, setZoom] = useState(1);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const positionsRef = useRef<NodePosition[]>([]);
  const animRef = useRef<number>(0);

  useEffect(() => {
    loadGraph();
  }, []);

  const loadGraph = async () => {
    try {
      const data = await getKnowledgeGraph();
      setNodes(data.nodes);
      setEdges(data.edges);
      initPositions(data.nodes);
    } catch {
      setError("Could not connect to API. Using demo data.");
      const demo = getDemoData();
      setNodes(demo.nodes);
      setEdges(demo.edges);
      initPositions(demo.nodes);
    } finally {
      setLoading(false);
    }
  };

  const initPositions = (nodes: KGNode[]) => {
    const width = 800;
    const height = 600;
    positionsRef.current = nodes.map((node) => ({
      x: width / 2 + (Math.random() - 0.5) * 400,
      y: height / 2 + (Math.random() - 0.5) * 300,
      vx: 0,
      vy: 0,
      node,
    }));
    startSimulation();
  };

  const startSimulation = useCallback(() => {
    let ticks = 0;
    const maxTicks = 200;

    const tick = () => {
      if (ticks >= maxTicks) return;
      ticks++;

      const positions = positionsRef.current;
      const width = 800;
      const height = 600;

      // Repulsion between all nodes
      for (let i = 0; i < positions.length; i++) {
        for (let j = i + 1; j < positions.length; j++) {
          const dx = positions[j].x - positions[i].x;
          const dy = positions[j].y - positions[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          const force = 3000 / (dist * dist);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          positions[i].vx -= fx;
          positions[i].vy -= fy;
          positions[j].vx += fx;
          positions[j].vy += fy;
        }
      }

      // Attraction along edges
      for (const edge of edges) {
        const si = positions.findIndex((p) => p.node.id === edge.source);
        const ti = positions.findIndex((p) => p.node.id === edge.target);
        if (si === -1 || ti === -1) continue;
        const dx = positions[ti].x - positions[si].x;
        const dy = positions[ti].y - positions[si].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - 120) * 0.01;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        positions[si].vx += fx;
        positions[si].vy += fy;
        positions[ti].vx -= fx;
        positions[ti].vy -= fy;
      }

      // Center gravity
      for (const p of positions) {
        p.vx += (width / 2 - p.x) * 0.001;
        p.vy += (height / 2 - p.y) * 0.001;
        p.vx *= 0.85;
        p.vy *= 0.85;
        p.x += p.vx;
        p.y += p.vy;
        p.x = Math.max(40, Math.min(width - 40, p.x));
        p.y = Math.max(40, Math.min(height - 40, p.y));
      }

      draw();
      animRef.current = requestAnimationFrame(tick);
    };

    cancelAnimationFrame(animRef.current);
    tick();
  }, [edges]);

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const positions = positionsRef.current;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(zoom, zoom);

    // Draw edges
    ctx.strokeStyle = "rgba(63, 63, 70, 0.6)";
    ctx.lineWidth = 1;
    for (const edge of edges) {
      const s = positions.find((p) => p.node.id === edge.source);
      const t = positions.find((p) => p.node.id === edge.target);
      if (!s || !t) continue;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();

      // Edge label
      const mx = (s.x + t.x) / 2;
      const my = (s.y + t.y) / 2;
      ctx.fillStyle = "rgba(161, 161, 170, 0.5)";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(edge.relation_type, mx, my - 4);
    }

    // Draw nodes
    const typeColors: Record<string, string> = {
      component: "#34d399",
      service: "#60a5fa",
      failure_mode: "#f87171",
      procedure: "#a78bfa",
      person: "#fbbf24",
    };

    for (const p of positions) {
      const color = typeColors[p.node.entity_type] || "#a1a1aa";
      const isSelected = selected?.id === p.node.id;
      const radius = isSelected ? 10 : 7;

      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? color : `${color}80`;
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 2 : 1;
      ctx.stroke();

      ctx.fillStyle = "#e4e4e7";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(p.node.name, p.x, p.y + radius + 14);
    }

    ctx.restore();
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / zoom;
    const y = (e.clientY - rect.top) / zoom;

    const clicked = positionsRef.current.find((p) => {
      const dx = p.x - x;
      const dy = p.y - y;
      return dx * dx + dy * dy < 200;
    });

    setSelected(clicked?.node ?? null);
    draw();
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-zinc-500 text-sm">Loading knowledge graph...</div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold">Knowledge Graph</h1>
          <p className="text-zinc-400 text-sm mt-1">
            {nodes.length} entities, {edges.length} relations
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => { setZoom(z => Math.min(z + 0.2, 3)); draw(); }} className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button onClick={() => { setZoom(z => Math.max(z - 0.2, 0.4)); draw(); }} className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button onClick={() => { setZoom(1); draw(); }} className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-3 px-3 py-2 bg-amber-950/30 border border-amber-800 rounded-lg text-xs text-amber-300">
          {error}
        </div>
      )}

      <div className="flex-1 flex gap-4 min-h-0">
        <div className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="w-full h-full cursor-crosshair"
            onClick={handleCanvasClick}
          />
        </div>

        {selected && (
          <div className="w-64 bg-zinc-900 border border-zinc-800 rounded-xl p-4">
            <h3 className="font-medium text-sm">{selected.name}</h3>
            <span className="inline-block mt-1 text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">
              {selected.entity_type}
            </span>
            {selected.description && (
              <p className="text-xs text-zinc-400 mt-3">{selected.description}</p>
            )}
            <div className="mt-3 pt-3 border-t border-zinc-800">
              <p className="text-xs text-zinc-500">
                Connections: {edges.filter(e => e.source === selected.id || e.target === selected.id).length}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function getDemoData(): { nodes: KGNode[]; edges: KGEdge[] } {
  return {
    nodes: [
      { id: "1", name: "TurboFan", entity_type: "component", description: "High-bypass turbofan engine" },
      { id: "2", name: "Compressor", entity_type: "component", description: "Multi-stage air compressor" },
      { id: "3", name: "Combustion Chamber", entity_type: "component" },
      { id: "4", name: "Blade Erosion", entity_type: "failure_mode", description: "Wear of fan blade surfaces" },
      { id: "5", name: "Thermal Fatigue", entity_type: "failure_mode" },
      { id: "6", name: "Inspection Proc", entity_type: "procedure", description: "Quarterly visual inspection" },
      { id: "7", name: "AuthService", entity_type: "service", description: "OAuth2 authentication" },
      { id: "8", name: "DataPipeline", entity_type: "service" },
    ],
    edges: [
      { source: "1", target: "2", relation_type: "HAS_COMPONENT" },
      { source: "1", target: "3", relation_type: "HAS_COMPONENT" },
      { source: "2", target: "4", relation_type: "HAS_FAILURE_MODE" },
      { source: "3", target: "5", relation_type: "HAS_FAILURE_MODE" },
      { source: "4", target: "6", relation_type: "MITIGATED_BY" },
      { source: "7", target: "8", relation_type: "DEPENDS_ON" },
    ],
  };
}
