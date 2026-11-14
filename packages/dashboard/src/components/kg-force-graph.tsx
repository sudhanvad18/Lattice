"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import ForceGraph3D from "react-force-graph-3d";
import SpriteText from "three-spritetext";
import type { KGEdge, KGNode } from "@/lib/api";
import { PROVENANCE_STYLES, communityColor, type GraphMode } from "./kg-style";

interface GNode extends KGNode {
  degree: number;
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
}

interface GLink {
  id: string;
  source: string | GNode;
  target: string | GNode;
  relation_type: string;
  provenance?: string;
  confidence: number;
}

interface Props {
  mode: GraphMode;
  nodes: KGNode[];
  edges: KGEdge[];
  selectedId: string | null;
  matchIds: Set<string>;
  godNodeIds: Set<string>;
  hiddenCommunities: Set<number>;
  fitSignal: number;
  onSelect: (node: KGNode | null) => void;
}

const DIM = "#3f3f46";

function endpointId(end: string | GNode): string {
  return typeof end === "string" ? end : end.id;
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

/** Pulls nodes toward their community centroid so clusters read as regions. */
function communityForce(strength: number) {
  let nodes: GNode[] = [];
  const force = (alpha: number) => {
    const sums = new Map<number, { x: number; y: number; z: number; n: number }>();
    for (const node of nodes) {
      if (node.community == null) continue;
      const s = sums.get(node.community) ?? { x: 0, y: 0, z: 0, n: 0 };
      s.x += node.x ?? 0; s.y += node.y ?? 0; s.z += node.z ?? 0; s.n += 1;
      sums.set(node.community, s);
    }
    const k = strength * alpha;
    for (const node of nodes) {
      const s = node.community != null ? sums.get(node.community) : undefined;
      if (!s || s.n < 2) continue;
      node.vx = (node.vx ?? 0) + (s.x / s.n - (node.x ?? 0)) * k;
      node.vy = (node.vy ?? 0) + (s.y / s.n - (node.y ?? 0)) * k;
      if (node.z !== undefined) node.vz = (node.vz ?? 0) + (s.z / s.n - node.z) * k;
    }
  };
  force.initialize = (n: GNode[]) => { nodes = n; };
  return force;
}

export function KGForceGraph({
  mode, nodes, edges, selectedId, matchIds, godNodeIds, hiddenCommunities, fitSignal, onSelect,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  // Both renderers expose the same imperative subset used here.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(undefined);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [hoverId, setHoverId] = useState<string | null>(null);
  const fittedRef = useRef(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // The force engine mutates nodes/links in place, so hand it fresh copies per mode.
  const graphData = useMemo(() => {
    const visible = nodes.filter((n) => n.community == null || !hiddenCommunities.has(n.community));
    const ids = new Set(visible.map((n) => n.id));
    const links: GLink[] = edges
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .map((e) => ({ ...e }));
    const degree = new Map<string, number>();
    for (const link of links) {
      degree.set(link.source as string, (degree.get(link.source as string) ?? 0) + 1);
      degree.set(link.target as string, (degree.get(link.target as string) ?? 0) + 1);
    }
    const gnodes: GNode[] = visible.map((n) => ({ ...n, degree: degree.get(n.id) ?? 0 }));
    return { nodes: gnodes, links };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, edges, hiddenCommunities, mode]);

  const neighbors = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const link of graphData.links) {
      const s = endpointId(link.source);
      const t = endpointId(link.target);
      if (!map.has(s)) map.set(s, new Set());
      if (!map.has(t)) map.set(t, new Set());
      map.get(s)!.add(t);
      map.get(t)!.add(s);
    }
    return map;
  }, [graphData]);

  // Label the hubs: god nodes plus the top decile by degree, always at least a handful.
  const labeledIds = useMemo(() => {
    const ranked = [...graphData.nodes].sort((a, b) => b.degree - a.degree);
    const cutoff = Math.max(8, Math.ceil(ranked.length * 0.1));
    const ids = new Set(ranked.slice(0, cutoff).filter((n) => n.degree > 0).map((n) => n.id));
    godNodeIds.forEach((id) => ids.add(id));
    matchIds.forEach((id) => ids.add(id));
    if (selectedId) ids.add(selectedId);
    return ids;
  }, [graphData, godNodeIds, matchIds, selectedId]);

  const focusId = hoverId ?? selectedId;
  const focusIds = useMemo(() => {
    if (focusId) return new Set([focusId, ...(neighbors.get(focusId) ?? [])]);
    if (matchIds.size) return matchIds;
    return null;
  }, [focusId, neighbors, matchIds]);

  const isLinkFocused = useCallback((link: GLink) => {
    if (!focusId) return false;
    return endpointId(link.source) === focusId || endpointId(link.target) === focusId;
  }, [focusId]);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    fg.d3Force("community", communityForce(mode === "3d" ? 0.08 : 0.12));
    fg.d3Force("charge")?.strength(mode === "3d" ? -45 : -35).distanceMax(250);
    fg.d3Force("link")?.distance((link: GLink) => {
      const source = link.source as GNode;
      const target = link.target as GNode;
      return source.community != null && source.community === target.community ? 30 : 70;
    });
    fg.d3ReheatSimulation?.();
    fittedRef.current = false;
  }, [graphData, mode]);

  useEffect(() => {
    if (fitSignal > 0) fgRef.current?.zoomToFit(600, 40);
  }, [fitSignal]);

  useEffect(() => {
    if (matchIds.size) fgRef.current?.zoomToFit(600, 80, (n: GNode) => matchIds.has(n.id));
  }, [matchIds]);

  useEffect(() => {
    const fg = fgRef.current;
    const node = graphData.nodes.find((n) => n.id === selectedId);
    if (!fg || !node || node.x === undefined) return;
    if (mode === "3d") {
      const distance = 200;
      const { x = 0, y = 0, z = 0 } = node;
      const ratio = 1 + distance / Math.max(1, Math.hypot(x, y, z));
      fg.cameraPosition({ x: x * ratio, y: y * ratio, z: z * ratio }, { x, y, z }, 900);
    } else {
      fg.centerAt(node.x, node.y, 700);
      fg.zoom(3, 700);
    }
  }, [selectedId, graphData, mode]);

  const nodeColor = useCallback((n: GNode) => {
    if (focusIds && !focusIds.has(n.id)) return DIM;
    return communityColor(n);
  }, [focusIds]);

  const nodeVal = useCallback((n: GNode) => {
    const base = 1 + n.degree * 1.6;
    return godNodeIds.has(n.id) ? base * 2 : base;
  }, [godNodeIds]);

  const nodeLabel = useCallback((n: GNode) =>
    `<div style="font:12px sans-serif;padding:4px 6px;background:#18181b;border:1px solid #3f3f46;border-radius:6px;color:#e4e4e7">
      <b>${escapeHtml(n.name)}</b><br/>
      <span style="color:#a1a1aa">${escapeHtml(n.entity_type)}${n.community != null ? ` · community ${n.community}` : ""} · ${n.degree} links</span>
    </div>`, []);

  const linkColor = useCallback((l: GLink) => {
    if (isLinkFocused(l)) return "rgba(52, 211, 153, 0.9)";
    const style = PROVENANCE_STYLES[l.provenance ?? "extracted"] ?? PROVENANCE_STYLES.extracted;
    const alpha = focusIds ? style.opacity * 0.3 : style.opacity;
    return `rgba(161, 161, 170, ${alpha})`;
  }, [focusIds, isLinkFocused]);

  const linkLabel = useCallback((l: GLink) =>
    `${escapeHtml(l.relation_type)} · ${escapeHtml(l.provenance ?? "extracted")} · ${(l.confidence * 100).toFixed(0)}%`, []);

  const handleClick = useCallback((n: GNode) => onSelect(n), [onSelect]);
  const handleHover = useCallback((n: GNode | null) => setHoverId(n?.id ?? null), []);
  const handleStop = useCallback(() => {
    if (!fittedRef.current) {
      fittedRef.current = true;
      fgRef.current?.zoomToFit(600, 40);
    }
  }, []);

  const nodeThreeObject = useCallback((n: GNode) => {
    if (!labeledIds.has(n.id)) return null;
    const sprite = new SpriteText(n.name, godNodeIds.has(n.id) ? 7 : 5, "#e4e4e7");
    sprite.fontFace = "sans-serif";
    sprite.backgroundColor = "rgba(9, 9, 11, 0.55)";
    sprite.padding = 1;
    sprite.borderRadius = 2;
    sprite.center.set(0.5, -0.9);
    return sprite;
  }, [labeledIds, godNodeIds]);

  const drawLabel2D = useCallback((n: GNode, ctx: CanvasRenderingContext2D, scale: number) => {
    const show = labeledIds.has(n.id) || n.id === hoverId || scale > 2.5;
    if (!show || (focusIds && !focusIds.has(n.id))) return;
    const fontSize = Math.max(10 / scale, 2);
    ctx.font = `${godNodeIds.has(n.id) ? "600 " : ""}${fontSize}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = "rgba(228, 228, 231, 0.9)";
    const radius = Math.sqrt(nodeVal(n)) * 4;
    ctx.fillText(n.name, n.x ?? 0, (n.y ?? 0) + radius + 1);
  }, [labeledIds, hoverId, focusIds, godNodeIds, nodeVal]);

  const common = {
    width: size.width,
    height: size.height,
    graphData,
    backgroundColor: "#0c0c0e",
    nodeId: "id",
    nodeVal,
    nodeColor,
    nodeLabel,
    linkColor,
    linkLabel,
    linkWidth: (l: GLink) => (isLinkFocused(l) ? 1.6 : 0.4),
    linkDirectionalArrowLength: 3,
    linkDirectionalArrowRelPos: 0.92,
    linkDirectionalArrowColor: linkColor,
    linkDirectionalParticles: (l: GLink) => (isLinkFocused(l) ? 2 : 0),
    linkDirectionalParticleWidth: 1.6,
    linkDirectionalParticleColor: () => "#34d399",
    cooldownTicks: 200,
    onNodeClick: handleClick,
    onNodeHover: handleHover,
    onBackgroundClick: () => onSelect(null),
    onEngineStop: handleStop,
  };

  return (
    <div ref={containerRef} className="absolute inset-0">
      {size.width > 0 && (mode === "3d" ? (
        <ForceGraph3D
          {...common}
          ref={fgRef}
          nodeOpacity={0.92}
          nodeResolution={12}
          linkOpacity={0.6}
          // three-forcegraph falls back to the default sphere when this returns null.
          nodeThreeObject={nodeThreeObject as (n: GNode) => SpriteText}
          nodeThreeObjectExtend
          showNavInfo={false}
        />
      ) : (
        <ForceGraph2D
          {...common}
          ref={fgRef}
          nodeRelSize={4}
          linkLineDash={(l: GLink) => (PROVENANCE_STYLES[l.provenance ?? "extracted"] ?? PROVENANCE_STYLES.extracted).dash}
          nodeCanvasObjectMode={() => "after"}
          nodeCanvasObject={drawLabel2D}
        />
      ))}
    </div>
  );
}
