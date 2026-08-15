"use client";

import { useEffect, useState } from "react";
import { BarChart3, TrendingUp, Award, Clock } from "lucide-react";
import { getEvalResults, EvalResult } from "@/lib/api";

export function EvalDashboard() {
  const [results, setResults] = useState<EvalResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    try {
      const data = await getEvalResults();
      setResults(data);
    } catch {
      setResults(getDemoResults());
    } finally {
      setLoading(false);
    }
  };

  const avgQuality = results.length ? results.reduce((s, r) => s + r.quality_score, 0) / results.length : 0;
  const avgCitation = results.length ? results.reduce((s, r) => s + r.citation_accuracy, 0) / results.length : 0;
  const avgLatency = results.length ? results.reduce((s, r) => s + r.latency_seconds, 0) / results.length : 0;
  const completionRate = results.length ? results.filter((r) => r.completed).length / results.length : 0;

  if (loading) {
    return <div className="h-full flex items-center justify-center text-zinc-500">Loading evaluations...</div>;
  }

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto scrollbar-thin">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Evaluation Results</h1>
        <p className="text-zinc-400 text-sm mt-1">
          Benchmark performance across {results.length} evaluation tasks
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard
          icon={<Award className="w-5 h-5 text-emerald-400" />}
          label="Avg Quality"
          value={`${(avgQuality * 100).toFixed(0)}%`}
          color="emerald"
        />
        <MetricCard
          icon={<TrendingUp className="w-5 h-5 text-blue-400" />}
          label="Citation Accuracy"
          value={`${(avgCitation * 100).toFixed(0)}%`}
          color="blue"
        />
        <MetricCard
          icon={<Clock className="w-5 h-5 text-amber-400" />}
          label="Avg Latency"
          value={`${avgLatency.toFixed(1)}s`}
          color="amber"
        />
        <MetricCard
          icon={<BarChart3 className="w-5 h-5 text-purple-400" />}
          label="Completion"
          value={`${(completionRate * 100).toFixed(0)}%`}
          color="purple"
        />
      </div>

      {/* Results Table */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-zinc-500">
              <th className="px-4 py-3 text-left font-medium">Task</th>
              <th className="px-4 py-3 text-center font-medium">Status</th>
              <th className="px-4 py-3 text-center font-medium">Quality</th>
              <th className="px-4 py-3 text-center font-medium">Citations</th>
              <th className="px-4 py-3 text-center font-medium">Latency</th>
              <th className="px-4 py-3 text-center font-medium">Iterations</th>
            </tr>
          </thead>
          <tbody>
            {results.map((result, i) => (
              <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                <td className="px-4 py-3 text-zinc-200">{result.task_name}</td>
                <td className="px-4 py-3 text-center">
                  {result.completed ? (
                    <span className="text-emerald-400 text-xs font-medium">PASS</span>
                  ) : (
                    <span className="text-red-400 text-xs font-medium">FAIL</span>
                  )}
                </td>
                <td className="px-4 py-3 text-center">
                  <ScoreBar value={result.quality_score} />
                </td>
                <td className="px-4 py-3 text-center">
                  <ScoreBar value={result.citation_accuracy} />
                </td>
                <td className="px-4 py-3 text-center text-zinc-400">
                  {result.latency_seconds.toFixed(1)}s
                </td>
                <td className="px-4 py-3 text-center text-zinc-400">
                  {result.iterations}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-xs text-zinc-500">{label}</span>
      </div>
      <span className="text-2xl font-bold">{value}</span>
    </div>
  );
}

function ScoreBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "bg-emerald-500" : pct >= 60 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-zinc-400">{pct}%</span>
    </div>
  );
}

function getDemoResults(): EvalResult[] {
  return [
    { task_name: "Generate turbine maintenance doc", completed: true, quality_score: 0.88, citation_accuracy: 0.92, latency_seconds: 12.3, iterations: 2 },
    { task_name: "Summarize blade inspection report", completed: true, quality_score: 0.82, citation_accuracy: 0.85, latency_seconds: 8.1, iterations: 1 },
    { task_name: "Code review: auth middleware", completed: true, quality_score: 0.91, citation_accuracy: 0.78, latency_seconds: 15.7, iterations: 3 },
    { task_name: "Draft API migration plan", completed: true, quality_score: 0.76, citation_accuracy: 0.70, latency_seconds: 22.4, iterations: 2 },
    { task_name: "Explain thermal fatigue process", completed: true, quality_score: 0.94, citation_accuracy: 0.96, latency_seconds: 6.8, iterations: 1 },
    { task_name: "Generate unit tests for pipeline", completed: false, quality_score: 0.45, citation_accuracy: 0.40, latency_seconds: 30.0, iterations: 3 },
  ];
}
