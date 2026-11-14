"use client";

import { useCallback, useEffect, useState } from "react";
import { BarChart3, TrendingUp, Award, Clock, RefreshCw } from "lucide-react";
import { getEvalResults, EvalRun } from "@/lib/api";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; run: EvalRun; demo: boolean }
  | { kind: "error"; message: string };

export function EvalDashboard() {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  const load = useCallback(() => {
    return getEvalResults().then((run) => {
      setState({ kind: "ready", run, demo: false });
    }).catch((err: unknown) => {
      if (err instanceof TypeError) {
        setState({ kind: "ready", run: getDemoRun(), demo: true });
      } else {
        setState({ kind: "error", message: err instanceof Error ? err.message : "Unknown error" });
      }
    });
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (state.kind === "loading") {
    return <div className="h-full flex items-center justify-center text-zinc-500">Loading evaluations...</div>;
  }

  if (state.kind === "error") {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 text-sm text-zinc-400">
        <p>Could not load evaluation results: {state.message}</p>
        <button onClick={() => { setState({ kind: "loading" }); void load(); }}
          className="px-3 py-1.5 bg-zinc-800 rounded-lg hover:bg-zinc-700">
          Retry
        </button>
      </div>
    );
  }

  const { run, demo } = state;
  const results = run.results;
  const summary = run.summary;

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto scrollbar-thin">
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">Evaluation Results</h1>
          <p className="text-zinc-400 text-sm mt-1">
            {run.run_at
              ? `Benchmark run ${new Date(run.run_at).toLocaleString()} · ${run.provider ?? "unknown provider"}${run.model ? ` / ${run.model}` : ""} · ${results.length} tasks`
              : demo ? "Demo results" : "No benchmark runs yet"}
          </p>
        </div>
        <button onClick={() => void load()} title="Reload results"
          className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {demo && (
        <div className="mb-4 px-3 py-2 bg-amber-950/30 border border-amber-800 rounded-lg text-xs text-amber-300">
          Could not connect to API. Showing demo data.
        </div>
      )}

      {results.length === 0 ? (
        <div className="p-6 bg-zinc-900 border border-zinc-800 rounded-xl text-sm text-zinc-400">
          <p>No evaluation results yet. Run the benchmark from the project root:</p>
          <pre className="mt-3 p-3 bg-zinc-950 border border-zinc-800 rounded-lg text-zinc-300 text-xs">lattice eval --provider ollama</pre>
          <p className="mt-3 text-xs text-zinc-500">
            Use <code>--provider mock</code> for an offline smoke run, then reload this page.
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-4 gap-4 mb-6">
            <MetricCard icon={<Award className="w-5 h-5 text-emerald-400" />} label="Pass Rate"
              value={pct(summary?.pass_rate ?? average(results.map((r) => (r.passed ? 1 : 0))))} />
            <MetricCard icon={<TrendingUp className="w-5 h-5 text-blue-400" />} label="Keyword Recall"
              value={pct(summary?.avg_keyword_recall ?? average(results.map((r) => r.keyword_recall)))} />
            <MetricCard icon={<Clock className="w-5 h-5 text-amber-400" />} label="Avg Latency"
              value={`${(summary?.avg_latency_seconds ?? average(results.map((r) => r.latency_seconds))).toFixed(1)}s`} />
            <MetricCard icon={<BarChart3 className="w-5 h-5 text-purple-400" />} label="Completion"
              value={pct(summary?.completion_rate ?? average(results.map((r) => (r.completed ? 1 : 0))))} />
          </div>

          <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-500">
                  <th className="px-4 py-3 text-left font-medium">Task</th>
                  <th className="px-4 py-3 text-center font-medium">Result</th>
                  <th className="px-4 py-3 text-center font-medium">Review confidence</th>
                  <th className="px-4 py-3 text-center font-medium">Keyword recall</th>
                  <th className="px-4 py-3 text-center font-medium">Latency</th>
                  <th className="px-4 py-3 text-center font-medium">Iterations</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <tr key={result.task_id} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 align-top">
                    <td className="px-4 py-3 text-zinc-200">
                      {result.task_name}
                      {result.errors.length > 0 && (
                        <p className="mt-1 text-xs text-red-400/80 truncate max-w-md" title={result.errors.join("\n")}>
                          {result.errors[0]}
                        </p>
                      )}
                      {result.keywords_missing.length > 0 && (
                        <p className="mt-1 text-xs text-zinc-500">Missing: {result.keywords_missing.join(", ")}</p>
                      )}
                    </td>
                    <td className="px-4 py-3 text-center">
                      {result.passed ? (
                        <span className="text-emerald-400 text-xs font-medium">PASS</span>
                      ) : (
                        <span className="text-red-400 text-xs font-medium" title={result.status}>FAIL</span>
                      )}
                    </td>
                    <td className="px-4 py-3"><ScoreBar value={result.quality_score} /></td>
                    <td className="px-4 py-3"><ScoreBar value={result.keyword_recall} /></td>
                    <td className="px-4 py-3 text-center text-zinc-400">{result.latency_seconds.toFixed(1)}s</td>
                    <td className="px-4 py-3 text-center text-zinc-400">{result.iterations}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function average(values: number[]): number {
  return values.length ? values.reduce((s, v) => s + v, 0) / values.length : 0;
}

function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`;
}

function MetricCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
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
  const percent = Math.round(value * 100);
  const color = percent >= 80 ? "bg-emerald-500" : percent >= 60 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center justify-center gap-2">
      <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${percent}%` }} />
      </div>
      <span className="text-xs text-zinc-400">{percent}%</span>
    </div>
  );
}

function getDemoRun(): EvalRun {
  const row = (task_id: string, task_name: string, passed: boolean, quality_score: number,
    keyword_recall: number, latency_seconds: number, iterations: number) => ({
    task_id, task_name, status: passed ? "completed" : "failed", completed: passed, passed,
    quality_score, keyword_recall, keywords_missing: [], latency_seconds, iterations,
    artifact_count: 2, errors: [],
  });
  return {
    run_at: null,
    provider: null,
    model: null,
    summary: null,
    results: [
      row("demo-1", "Generate turbine maintenance doc", true, 0.88, 0.92, 12.3, 2),
      row("demo-2", "Summarize blade inspection report", true, 0.82, 0.85, 8.1, 1),
      row("demo-3", "Code review: auth middleware", true, 0.91, 0.78, 15.7, 3),
      row("demo-4", "Draft API migration plan", true, 0.76, 0.7, 22.4, 2),
      row("demo-5", "Generate unit tests for pipeline", false, 0.45, 0.4, 30.0, 3),
    ],
  };
}
