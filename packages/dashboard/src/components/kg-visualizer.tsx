"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Search, ZoomIn, ZoomOut, Maximize2, X } from "lucide-react";
import { getKnowledgeGraph, getEntityDetail, ingestText, EntityDetail, KGNode, KGEdge } from "@/lib/api";

const COMMUNITY_PALETTE = [
  "#34d399", "#60a5fa", "#f87171", "#a78bfa", "#fbbf24",
  "#f472b6", "#2dd4bf", "#fb923c", "#818cf8", "#4ade80",
  "#e879f9", "#38bdf8", "#facc15", "#a3e635", "#f43f5e",
];

const PROVENANCE_STYLES: Record<string, { dash: number[]; opacity: number; label: string }> = {
  extracted: { dash: [], opacity: 0.7, label: "Extracted" },
  inferred: { dash: [6, 4], opacity: 0.5, label: "Inferred" },
  ambiguous: { dash: [2, 3], opacity: 0.35, label: "Ambiguous" },
};

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
  const [truncated, setTruncated] = useState(false);
  const [demo, setDemo] = useState(false);
  const [text, setText] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [detail, setDetail] = useState<EntityDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [selected, setSelected] = useState<KGNode | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const positionsRef = useRef<NodePosition[]>([]);
  const animRef = useRef<number>(0);
  const isDragging = useRef(false);
  const dragStart = useRef({ x: 0, y: 0 });

  const loadGraph = useCallback(() => {
    return getKnowledgeGraph().then((data) => {
      setNodes(data.nodes);
      setEdges(data.edges);
      setTruncated(data.truncated);
      setDemo(false);
      setError(null);
    }).catch((err: unknown) => {
      const unreachable = err instanceof TypeError;
      setDemo(unreachable);
      setTruncated(false);
      const data = unreachable ? getDemoData() : { nodes: [], edges: [] };
      setNodes(data.nodes);
      setEdges(data.edges);
      setError(unreachable ? "Could not connect to API. Using demo data." : "Could not load knowledge graph. Please try again.");
    }).finally(() => {
      setLoading(false);
    });
  }, []);

  useEffect(() => {
    void loadGraph();
  }, [loadGraph]);

  useEffect(() => {
    let cancelled = false;
    if (selected && !demo) {
      getEntityDetail(selected.id).then((data) => {
        if (!cancelled) {
          setDetail(data);
          setDetailError(null);
        }
      }).catch(() => {
        if (!cancelled) setDetailError("Could not load entity details.");
      });
    }
    return () => { cancelled = true; };
  }, [selected, demo]);

  const handleIngest = async () => {
    setIngesting(true);
    try {
      await ingestText(text);
      setText("");
      await loadGraph();
    } catch {
      setError("Could not ingest text. Please try again.");
    } finally {
      setIngesting(false);
    }
  };

  useEffect(() => {
    positionsRef.current = nodes.map((node) => ({
      x: 400 + (Math.random() - 0.5) * 400,
      y: 300 + (Math.random() - 0.5) * 300,
      vx: 0, vy: 0, node,
    }));
  }, [nodes]);

  const nodeColor = (node: KGNode): string => {
    if (node.community != null) {
      return COMMUNITY_PALETTE[node.community % COMMUNITY_PALETTE.length];
    }
    const fallback: Record<string, string> = {
      component: "#34d399", service: "#60a5fa", failure_mode: "#f87171",
      procedure: "#a78bfa", person: "#fbbf24", Function: "#38bdf8",
      Class: "#f472b6", Module: "#2dd4bf",
    };
    return fallback[node.entity_type] || "#a1a1aa";
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const positions = positionsRef.current;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(panOffset.x, panOffset.y);
    ctx.scale(zoom, zoom);

    const searchLower = searchQuery.toLowerCase();
    const matchIds = new Set(
      searchLower
        ? positions
            .filter((p) => p.node.name.toLowerCase().includes(searchLower) || p.node.entity_type.toLowerCase().includes(searchLower))
            .map((p) => p.node.id)
        : []
    );

    for (const edge of edges) {
      const s = positions.find((p) => p.node.id === edge.source);
      const t = positions.find((p) => p.node.id === edge.target);
      if (!s || !t) continue;

      const prov = PROVENANCE_STYLES[edge.provenance || "extracted"] || PROVENANCE_STYLES.extracted;
      const dimmed = searchLower && !matchIds.has(edge.source) && !matchIds.has(edge.target);

      ctx.beginPath();
      ctx.setLineDash(prov.dash);
      ctx.strokeStyle = dimmed ? "rgba(63,63,70,0.15)" : `rgba(161,161,170,${prov.opacity})`;
      ctx.lineWidth = 1.2;
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Arrow head
      const angle = Math.atan2(t.y - s.y, t.x - s.x);
      const arrowLen = 8;
      const mx = (s.x + t.x) / 2;
      const my = (s.y + t.y) / 2;
      if (!dimmed) {
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx - arrowLen * Math.cos(angle - 0.4), my - arrowLen * Math.sin(angle - 0.4));
        ctx.moveTo(mx, my);
        ctx.lineTo(mx - arrowLen * Math.cos(angle + 0.4), my - arrowLen * Math.sin(angle + 0.4));
        ctx.strokeStyle = `rgba(161,161,170,${prov.opacity})`;
        ctx.stroke();
      }

      // Edge label
      if (!dimmed) {
        ctx.fillStyle = `rgba(161,161,170,${prov.opacity * 0.7})`;
        ctx.font = "9px monospace";
        ctx.textAlign = "center";
        ctx.fillText(edge.relation_type, mx, my - 6);
      }
    }

    for (const p of positions) {
      const color = nodeColor(p.node);
      const isSelected = selected?.id === p.node.id;
      const isMatch = matchIds.has(p.node.id);
      const dimmed = searchLower && !isMatch && !isSelected;
      const radius = isSelected ? 12 : isMatch ? 10 : 7;

      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = dimmed ? `${color}20` : isSelected || isMatch ? color : `${color}80`;
      ctx.fill();
      ctx.strokeStyle = dimmed ? `${color}30` : color;
      ctx.lineWidth = isSelected ? 2.5 : 1;
      ctx.stroke();

      if (!dimmed) {
        ctx.fillStyle = dimmed ? "rgba(228,228,231,0.2)" : "#e4e4e7";
        ctx.font = isSelected ? "bold 12px sans-serif" : "11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(p.node.name, p.x, p.y + radius + 14);
      }
    }

    ctx.restore();
  }, [edges, zoom, panOffset, searchQuery, selected]);

  useEffect(() => {
    const simEdges = edges;
    let ticks = 0;
    const maxTicks = 250;

    const tick = () => {
      if (ticks >= maxTicks) return;
      ticks++;

      const positions = positionsRef.current;
      const width = 800;
      const height = 600;

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

      for (const edge of simEdges) {
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
    return () => cancelAnimationFrame(animRef.current);
  }, [draw, edges, nodes]);



  const selectNode = (node: KGNode | null) => {
    setSelected(node);
    setDetail(null);
    setDetailError(null);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - panOffset.x) / zoom;
    const y = (e.clientY - rect.top - panOffset.y) / zoom;

    const clicked = positionsRef.current.find((p) => {
      const dx = p.x - x;
      const dy = p.y - y;
      return dx * dx + dy * dy < 200;
    });

    selectNode(clicked?.node ?? null);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    isDragging.current = false;
    dragStart.current = { x: e.clientX - panOffset.x, y: e.clientY - panOffset.y };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (e.buttons !== 1) return;
    isDragging.current = true;
    setPanOffset({ x: e.clientX - dragStart.current.x, y: e.clientY - dragStart.current.y });
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom((z) => Math.max(0.2, Math.min(4, z - e.deltaY * 0.001)));
  };

  const communitySet = new Set(nodes.map((n) => n.community).filter((c): c is number => c != null));
  const communities = Array.from(communitySet).sort((a, b) => a - b);

  const neighborIds = selected
    ? new Set(
        edges
          .filter((e) => e.source === selected.id || e.target === selected.id)
          .flatMap((e) => [e.source, e.target])
          .filter((id) => id !== selected.id)
      )
    : new Set<string>();
  const liveDetail = !demo && detail?.entity.id === selected?.id ? detail : null;
  const panelEntity = liveDetail?.entity ?? selected;
  const neighborNodes = liveDetail?.neighbors ?? nodes.filter((n) => neighborIds.has(n.id));
  const selectedEdges = liveDetail?.relations ?? (selected ? edges.filter((e) => e.source === selected.id || e.target === selected.id) : []);

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
            {nodes.length} entities · {edges.length} relations
            {communities.length > 0 && ` · ${communities.length} communities`}
            {truncated && " · truncated"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="w-4 h-4 absolute left-2.5 top-2.5 text-zinc-500" />
            <input
              type="text"
              placeholder="Search entities..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-8 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-zinc-200 placeholder-zinc-500 w-56 focus:outline-none focus:border-zinc-500"
            />
            {searchQuery && (
              <button onClick={() => setSearchQuery("")} className="absolute right-2 top-2.5 text-zinc-500 hover:text-zinc-300">
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
          <button onClick={() => { setZoom((z) => Math.min(z + 0.2, 4)); }} className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button onClick={() => { setZoom((z) => Math.max(z - 0.2, 0.2)); }} className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button onClick={() => { setZoom(1); setPanOffset({ x: 0, y: 0 }); }} className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-3 px-3 py-2 bg-amber-950/30 border border-amber-800 rounded-lg text-xs text-amber-300">
          {error}
        </div>
      )}

      {!demo && nodes.length === 0 && (
        <div className="mb-4 p-4 bg-zinc-900 border border-zinc-800 rounded-xl text-sm text-zinc-400">
          <p>Knowledge graph is empty. Ingest documents with <code>lattice ingest &lt;dir&gt;</code> or paste text below.</p>
          <textarea aria-label="Text to ingest" value={text} onChange={(e) => setText(e.target.value)} rows={3}
            className="mt-3 w-full p-2 bg-zinc-800 border border-zinc-700 rounded-lg" />
          <button onClick={handleIngest} disabled={ingesting || !text.trim()}
            className="mt-2 px-3 py-1.5 bg-zinc-800 rounded-lg disabled:opacity-50">
            {ingesting ? "Ingesting..." : "Ingest"}
          </button>
        </div>
      )}

      {/* Legends */}
      <div className="flex gap-6 mb-3 flex-wrap">
        {communities.length > 0 && (
          <div className="flex items-center gap-3 text-xs text-zinc-400">
            <span className="font-medium text-zinc-300">Communities:</span>
            {communities.slice(0, 8).map((c) => (
              <span key={c} className="flex items-center gap-1">
                <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COMMUNITY_PALETTE[c % COMMUNITY_PALETTE.length] }} />
                {c}
              </span>
            ))}
            {communities.length > 8 && <span>+{communities.length - 8}</span>}
          </div>
        )}
        <div className="flex items-center gap-3 text-xs text-zinc-400">
          <span className="font-medium text-zinc-300">Edge style:</span>
          {Object.entries(PROVENANCE_STYLES).map(([key, val]) => (
            <span key={key} className="flex items-center gap-1">
              <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke="#a1a1aa" strokeWidth="1.5" strokeDasharray={val.dash.join(",")} opacity={val.opacity} /></svg>
              {val.label}
            </span>
          ))}
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        <div
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden"
          onWheel={handleWheel}
        >
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="w-full h-full cursor-grab active:cursor-grabbing"
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
          />
        </div>

        {selected && (
          <div className="w-72 bg-zinc-900 border border-zinc-800 rounded-xl p-4 overflow-y-auto">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm truncate">{panelEntity?.name}</h3>
              <button onClick={() => selectNode(null)} className="text-zinc-500 hover:text-zinc-300">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex gap-1.5 mt-2">
              <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">
                {selected.entity_type}
              </span>
              {selected.community != null && (
                <span
                  className="text-xs px-2 py-0.5 rounded-full text-white"
                  style={{ backgroundColor: COMMUNITY_PALETTE[selected.community % COMMUNITY_PALETTE.length] + "cc" }}
                >
                  community {selected.community}
                </span>
              )}
            </div>
            {panelEntity?.description && (
              <p className="text-xs text-zinc-400 mt-3">{panelEntity.description}</p>
            )}

            {detailError && <p className="mt-3 text-xs text-amber-300">{detailError}</p>}
            <div className="mt-4 pt-3 border-t border-zinc-800">
              <p className="text-xs font-medium text-zinc-300 mb-2">
                Connections ({selectedEdges.length})
              </p>
              {selectedEdges.map((e, i) => {
                const otherId = e.source === selected.id ? e.target : e.source;
                const otherNode = neighborNodes.find((n) => n.id === otherId) ?? nodes.find((n) => n.id === otherId);
                const direction = e.source === selected.id ? "→" : "←";
                const prov = e.provenance || "extracted";
                return (
                  <div
                    key={i}
                    className="text-xs text-zinc-400 py-1 flex items-center gap-1 cursor-pointer hover:text-zinc-200"
                    onClick={() => otherNode && selectNode(otherNode)}
                  >
                    <span>{direction}</span>
                    <span className="font-medium text-zinc-300">{otherNode?.name || otherId}</span>
                    <span className="text-zinc-600">({e.relation_type})</span>
                    {prov !== "extracted" && (
                      <span className="text-zinc-600 italic">[{prov}]</span>
                    )}
                  </div>
                );
              })}
            </div>

            {neighborNodes.length > 0 && (
              <div className="mt-3 pt-3 border-t border-zinc-800">
                <p className="text-xs font-medium text-zinc-300 mb-2">Neighbors</p>
                <div className="flex flex-wrap gap-1">
                  {neighborNodes.map((n) => (
                    <button
                      key={n.id}
                      onClick={() => selectNode(n)}
                      className="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200 truncate max-w-[120px]"
                    >
                      {n.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function getDemoData(): { nodes: KGNode[]; edges: KGEdge[] } {
  return {
    nodes: [
      { id: "1", name: "TurboFan", entity_type: "component", description: "High-bypass turbofan engine", community: 0 },
      { id: "2", name: "Compressor", entity_type: "component", description: "Multi-stage air compressor", community: 0 },
      { id: "3", name: "Combustion Chamber", entity_type: "component", community: 0 },
      { id: "4", name: "Blade Erosion", entity_type: "failure_mode", description: "Wear of fan blade surfaces", community: 0 },
      { id: "5", name: "Thermal Fatigue", entity_type: "failure_mode", community: 0 },
      { id: "6", name: "Inspection Proc", entity_type: "procedure", description: "Quarterly visual inspection", community: 1 },
      { id: "7", name: "AuthService", entity_type: "service", description: "OAuth2 authentication", community: 1 },
      { id: "8", name: "DataPipeline", entity_type: "service", community: 1 },
    ],
    edges: [
      { id: "demo-1", confidence: 1, source: "1", target: "2", relation_type: "HAS_COMPONENT", provenance: "extracted" },
      { id: "demo-2", confidence: 1, source: "1", target: "3", relation_type: "HAS_COMPONENT", provenance: "extracted" },
      { id: "demo-3", confidence: 1, source: "2", target: "4", relation_type: "HAS_FAILURE_MODE", provenance: "extracted" },
      { id: "demo-4", confidence: 1, source: "3", target: "5", relation_type: "HAS_FAILURE_MODE", provenance: "inferred" },
      { id: "demo-5", confidence: 1, source: "4", target: "6", relation_type: "MITIGATED_BY", provenance: "inferred" },
      { id: "demo-6", confidence: 1, source: "7", target: "8", relation_type: "DEPENDS_ON", provenance: "ambiguous" },
    ],
  };
}
