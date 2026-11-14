import type { KGNode } from "@/lib/api";

export type GraphMode = "2d" | "3d";

export const COMMUNITY_PALETTE = [
  "#34d399", "#60a5fa", "#f87171", "#a78bfa", "#fbbf24",
  "#f472b6", "#2dd4bf", "#fb923c", "#818cf8", "#4ade80",
  "#e879f9", "#38bdf8", "#facc15", "#a3e635", "#f43f5e",
];

export const PROVENANCE_STYLES: Record<string, { dash: number[] | null; opacity: number; label: string }> = {
  extracted: { dash: null, opacity: 0.55, label: "Extracted" },
  inferred: { dash: [4, 3], opacity: 0.4, label: "Inferred" },
  ambiguous: { dash: [1, 3], opacity: 0.25, label: "Ambiguous" },
};

const TYPE_COLORS: Record<string, string> = {
  Function: "#38bdf8", Class: "#f472b6", Module: "#2dd4bf",
};

export function communityColor(node: Pick<KGNode, "community" | "entity_type">): string {
  if (node.community != null) return COMMUNITY_PALETTE[node.community % COMMUNITY_PALETTE.length];
  return TYPE_COLORS[node.entity_type] ?? "#a1a1aa";
}
