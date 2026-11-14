"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import dynamic from "next/dynamic";
import { Search, Maximize2, X, Box, Square, Sparkles } from "lucide-react";
import {
  getKnowledgeGraph, getEntityDetail, getCommunities, ingestText,
  Communities, EntityDetail, KGNode, KGEdge,
} from "@/lib/api";
import { COMMUNITY_PALETTE, PROVENANCE_STYLES, communityColor, type GraphMode } from "./kg-style";

// three.js and the force engines need `window`, so the renderer is client-only.
const KGForceGraph = dynamic(() => import("./kg-force-graph").then((m) => m.KGForceGraph), {
  ssr: false,
  loading: () => <div className="absolute inset-0 flex items-center justify-center text-sm text-zinc-500">Loading renderer...</div>,
});

const MAX_COMMUNITY_CHIPS = 12;

export function KnowledgeGraphView() {
  const [nodes, setNodes] = useState<KGNode[]>([]);
  const [edges, setEdges] = useState<KGEdge[]>([]);
  const [insights, setInsights] = useState<Communities | null>(null);
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
  const [mode, setMode] = useState<GraphMode>("3d");
  const [hiddenCommunities, setHiddenCommunities] = useState<Set<number>>(new Set());
  const [fitSignal, setFitSignal] = useState(0);

  const loadGraph = useCallback(() => {
    return getKnowledgeGraph().then((data) => {
      setNodes(data.nodes);
      setEdges(data.edges);
      setTruncated(data.truncated);
      setDemo(false);
      setError(null);
      getCommunities().then(setInsights).catch(() => setInsights(null));
    }).catch((err: unknown) => {
      const unreachable = err instanceof TypeError;
      setDemo(unreachable);
      setTruncated(false);
      setInsights(null);
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

  const selectNode = useCallback((node: KGNode | null) => {
    setSelected(node);
    setDetail(null);
    setDetailError(null);
  }, []);

  const selectById = (id: string) => {
    const node = nodes.find((n) => n.id === id);
    if (node) selectNode(node);
  };

  const toggleCommunity = (c: number) => {
    setHiddenCommunities((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c); else next.add(c);
      return next;
    });
  };

  const matchIds = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return new Set<string>();
    return new Set(nodes
      .filter((n) => n.name.toLowerCase().includes(q) || n.entity_type.toLowerCase().includes(q))
      .map((n) => n.id));
  }, [nodes, searchQuery]);

  const godNodeIds = useMemo(() => new Set(insights?.god_nodes.map((g) => g.id) ?? []), [insights]);

  const communities = useMemo(() => {
    const sizes = new Map<number, number>();
    for (const n of nodes) {
      if (n.community != null) sizes.set(n.community, (sizes.get(n.community) ?? 0) + 1);
    }
    return Array.from(sizes.entries()).sort((a, b) => b[1] - a[1]);
  }, [nodes]);

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
          <div className="flex bg-zinc-800 rounded-lg p-0.5" role="group" aria-label="Render mode">
            {(["3d", "2d"] as const).map((m) => (
              <button key={m} onClick={() => setMode(m)} aria-pressed={mode === m}
                className={`flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs ${mode === m ? "bg-zinc-700 text-emerald-400" : "text-zinc-400 hover:text-zinc-200"}`}>
                {m === "3d" ? <Box className="w-3.5 h-3.5" /> : <Square className="w-3.5 h-3.5" />}
                {m.toUpperCase()}
              </button>
            ))}
          </div>
          <button onClick={() => setFitSignal((s) => s + 1)} title="Fit graph to view"
            className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
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

      <div className="flex gap-6 mb-3 flex-wrap items-center">
        {communities.length > 0 && (
          <div className="flex items-center gap-1.5 text-xs text-zinc-400 flex-wrap">
            <span className="font-medium text-zinc-300 mr-1">Communities:</span>
            {communities.slice(0, MAX_COMMUNITY_CHIPS).map(([c, size]) => {
              const hidden = hiddenCommunities.has(c);
              return (
                <button key={c} onClick={() => toggleCommunity(c)} aria-pressed={!hidden}
                  title={hidden ? "Show community" : "Hide community"}
                  className={`flex items-center gap-1 px-1.5 py-0.5 rounded ${hidden ? "opacity-35 line-through" : "hover:bg-zinc-800"}`}>
                  <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COMMUNITY_PALETTE[c % COMMUNITY_PALETTE.length] }} />
                  {c}<span className="text-zinc-600">({size})</span>
                </button>
              );
            })}
            {communities.length > MAX_COMMUNITY_CHIPS && <span>+{communities.length - MAX_COMMUNITY_CHIPS}</span>}
            {hiddenCommunities.size > 0 && (
              <button onClick={() => setHiddenCommunities(new Set())} className="ml-1 text-emerald-400 hover:underline">show all</button>
            )}
          </div>
        )}
        <div className="flex items-center gap-3 text-xs text-zinc-400">
          <span className="font-medium text-zinc-300">Edges:</span>
          {Object.entries(PROVENANCE_STYLES).map(([key, val]) => (
            <span key={key} className="flex items-center gap-1">
              <svg width="20" height="8"><line x1="0" y1="4" x2="20" y2="4" stroke="#a1a1aa" strokeWidth="1.5" strokeDasharray={val.dash?.join(",") ?? ""} opacity={val.opacity + 0.3} /></svg>
              {val.label}
            </span>
          ))}
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        <div className="relative flex-1 bg-[#0c0c0e] border border-zinc-800 rounded-xl overflow-hidden">
          {nodes.length > 0 && (
            <KGForceGraph
              mode={mode}
              nodes={nodes}
              edges={edges}
              selectedId={selected?.id ?? null}
              matchIds={matchIds}
              godNodeIds={godNodeIds}
              hiddenCommunities={hiddenCommunities}
              fitSignal={fitSignal}
              onSelect={selectNode}
            />
          )}
          <p className="absolute bottom-2 left-3 text-[11px] text-zinc-600 pointer-events-none">
            {mode === "3d" ? "Drag to rotate · right-drag to pan · scroll to zoom" : "Drag to pan · scroll to zoom"} · click a node to focus
          </p>
        </div>

        {selected ? (
          <div className="w-72 bg-zinc-900 border border-zinc-800 rounded-xl p-4 overflow-y-auto">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-sm truncate">{panelEntity?.name}</h3>
              <button onClick={() => selectNode(null)} className="text-zinc-500 hover:text-zinc-300">
                <X className="w-4 h-4" />
              </button>
            </div>
            <div className="flex gap-1.5 mt-2 flex-wrap">
              <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400">
                {selected.entity_type}
              </span>
              {selected.community != null && (
                <span className="text-xs px-2 py-0.5 rounded-full text-white"
                  style={{ backgroundColor: communityColor(selected) + "cc" }}>
                  community {selected.community}
                </span>
              )}
              {godNodeIds.has(selected.id) && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-amber-900/50 text-amber-300">hub</span>
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
              {selectedEdges.map((e) => {
                const otherId = e.source === selected.id ? e.target : e.source;
                const otherNode = neighborNodes.find((n) => n.id === otherId) ?? nodes.find((n) => n.id === otherId);
                const direction = e.source === selected.id ? "→" : "←";
                const prov = e.provenance || "extracted";
                return (
                  <div
                    key={e.id}
                    className="text-xs text-zinc-400 py-1 flex items-center gap-1 cursor-pointer hover:text-zinc-200"
                    onClick={() => otherNode && selectNode(otherNode)}
                  >
                    <span>{direction}</span>
                    <span className="font-medium text-zinc-300 truncate">{otherNode?.name || otherId}</span>
                    <span className="text-zinc-600 shrink-0">({e.relation_type})</span>
                    {prov !== "extracted" && (
                      <span className="text-zinc-600 italic">[{prov}]</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : insights && (insights.god_nodes.length > 0 || insights.surprising_connections.length > 0) ? (
          <div className="w-72 bg-zinc-900 border border-zinc-800 rounded-xl p-4 overflow-y-auto text-xs">
            <p className="flex items-center gap-1.5 font-medium text-zinc-300">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Graph insights
            </p>
            {insights.god_nodes.length > 0 && (
              <div className="mt-3">
                <p className="text-zinc-500 mb-1.5">Hub entities</p>
                <div className="flex flex-wrap gap-1">
                  {insights.god_nodes.map((g) => (
                    <button key={g.id} onClick={() => selectById(g.id)}
                      className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 hover:bg-zinc-700 truncate max-w-[200px]">
                      {g.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {insights.surprising_connections.length > 0 && (
              <div className="mt-4">
                <p className="text-zinc-500 mb-1.5">Surprising connections</p>
                <ul className="space-y-1.5">
                  {insights.surprising_connections.map((s, i) => (
                    <li key={i}>
                      <button onClick={() => selectById(s.source)} className="text-left text-zinc-400 hover:text-zinc-200">
                        {s.explanation}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : null}
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
