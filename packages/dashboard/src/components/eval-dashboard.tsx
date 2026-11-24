"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { BarChart3, TrendingUp, Award, Clock, RefreshCw, Play, Loader2, AlertTriangle } from "lucide-react";
import { getEvalResults, getEvalStatus, runEval, EvalRun } from "@/lib/api";
import { useKB } from "@/lib/kb-context";

type LoadState =
  | { kind: "loading" }
  | { kind: "ready"; run: EvalRun }
  | { kind: "empty" }
  | { kind: "error"; message: string };

export function EvalDashboard() {
  const { kb, health, apiReachable } = useKB();
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [running, setRunning] = useState(false);
  const [runProgress, setRunProgress] = useState<{ completed: number; total: number } | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const kbReady = kb?.ready ?? false;
  const modelReachable = apiReachable && !!health && !health.provider.detail;

  const load = useCallback(() => {
    return getEvalResults()
      .then((run) => {
        if (!run.run_at && run.results.length === 0) {
          setState({ kind: "empty" });
        } else {
          setState({ kind: "ready", run });
        }
      })
      .catch((err: unknown) => {
        if (!apiReachable) {
          setState({ kind: "error", message: "API unreachable" });
        } else {
          setState({
            kind: "error",
            message: err instanceof Error ? err.message : "Unknown error",
          });
        }
      });
  }, [apiReachable]);

  useEffect(() => {
    void load();
  }, [load]);

  const stopPoll = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPoll = useCallback(() => {
    stopPoll();
    pollRef.current = setInterval(async () => {
      try {
        const status = await getEvalStatus();
        setRunProgress({ completed: status.completed, total: status.total });
        if (!status.running) {
          stopPoll();
          setRunning(false);
          setRunProgress(null);
          await load();
        }
      } catch {
        stopPoll();
        setRunning(false);
        setRunProgress(null);
      }
    }, 2000);
  }, [load, stopPoll]);

  useEffect(() => () => stopPoll(), [stopPoll]);

  const handleRun = useCallback(async () => {
    setRunError(null);
    setRunning(true);
    setRunProgress(null);
    try {
      await runEval();
      startPoll();
    } catch (err) {
      setRunning(false);
      setRunError(err instanceof Error ? err.message : "Failed to start benchmark");
    }
  }, [startPoll]);

  const runDisabledReason = !kbReady
    ? "Knowledge base is empty"
    : !modelReachable
      ? (health?.provider.detail ?? "Model unavailable")
      : running
        ? "Benchmark already running"
        : null;

  if (state.kind === "loading") {
    return (
      <div className="h-full flex items-center justify-center text-zinc-500">
        Loading evaluations...
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-3 text-sm text-zinc-400">
        <p>Could not load evaluation results: {state.message}</p>
        <button
          onClick={() => { setState({ kind: "loading" }); void load(); }}
          className="px-3 py-1.5 bg-zinc-800 rounded-lg hover:bg-zinc-700"
        >
          Retry
        </button>
      </div>
    );
  }

  const run = state.kind === "ready" ? state.run : null;
  const results = run?.results ?? [];
  const summary = run?.summary ?? null;

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto scrollbar-thin">
      <div className="mb-6 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">Evaluation Results</h1>
          <p className="text-zinc-400 text-sm mt-1">
            {run?.run_at
              ? `Benchmark run ${new Date(run.run_at).toLocaleString()} · ${run.provider ?? "unknown provider"}${run.model ? ` / ${run.model}` : ""}${run.kb ? ` · ${run.kb.chunks} chunks / ${run.kb.entities} entities` : ""} · ${results.length} tasks`
              : "No benchmark runs yet"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => { setState({ kind: "loading" }); void load(); }}
            title="Reload results"
            className="p-2 bg-zinc-800 rounded-lg hover:bg-zinc-700"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={handleRun}
            disabled={!!runDisabledReason}
            title={runDisabledReason ?? undefined}
            className="flex items-center gap-2 px-3 py-2 bg-emerald-700 hover:bg-emerald-600 disabled:bg-zinc-700 disabled:text-zinc-500 text-white rounded-lg text-sm font-medium transition-colors"
          >
            {running ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {running ? (runProgress ? `${runProgress.completed}/${runProgress.total}` : "Starting...") : "Run benchmark"}
          </button>
        </div>
      </div>

      {runError && (
        <div className="mb-4 px-3 py-2 bg-red-950/30 border border-red-800 rounded-lg text-xs text-red-300 flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5" />
          {runError}
        </div>
      )}

      {running && (
        <div className="mb-4 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-xs text-zinc-400 flex items-center gap-2">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          Benchmark running against the current knowledge base
          {runProgress && ` — ${runProgress.completed} of ${runProgress.total} tasks complete`}
          ...
        </div>
      )}

      {state.kind === "empty" ? (
        <div className="p-6 bg-zinc-900 border border-zinc-800 rounded-xl text-sm text-zinc-400">
          <p>No runs yet. Run the benchmark to score answers against your knowledge base.</p>
          {!kbReady && (
            <p className="mt-2 text-xs text-amber-300">
              Ingest documents into the knowledge base first to run a benchmark.
            </p>
          )}
        </div>
      ) : results.length === 0 ? (
        <div className="p-6 bg-zinc-900 border border-zinc-800 rounded-xl text-sm text-zinc-400">
          <p>No results in this run.</p>
        </div>
      ) : (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-5 gap-4 mb-6">
            <MetricCard
              icon={<Award className="w-5 h-5 text-emerald-400" />}
              label="Pass Rate"
              value={pct(summary?.pass_rate ?? average(results.map((r) => (r.passed ? 1 : 0))))}
            />
            <MetricCard
              icon={<TrendingUp className="w-5 h-5 text-blue-400" />}
              label="Keyword Recall"
              value={pct(
                summary?.avg_keyword_recall ?? average(results.map((r) => r.keyword_recall)),
              )}
            />
            <MetricCard
              icon={<BarChart3 className="w-5 h-5 text-purple-400" />}
              label="Groundedness"
              value={pct(
                summary?.avg_groundedness ?? average(results.map((r) => r.groundedness ?? 0)),
              )}
            />
            <MetricCard
              icon={<Clock className="w-5 h-5 text-amber-400" />}
              label="Avg Latency"
              value={`${(summary?.avg_latency_seconds ?? average(results.map((r) => r.latency_seconds))).toFixed(1)}s`}
            />
            <MetricCard
              icon={<BarChart3 className="w-5 h-5 text-zinc-400" />}
              label="Completion"
              value={pct(
                summary?.completion_rate ?? average(results.map((r) => (r.completed ? 1 : 0))),
              )}
            />
          </div>

          <div className="bg-zinc-900 border border-zinc-800 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-500">
                  <th className="px-4 py-3 text-left font-medium">Task</th>
                  {results.some((r) => r.category) && (
                    <th className="px-4 py-3 text-center font-medium">Category</th>
                  )}
                  <th className="px-4 py-3 text-center font-medium">Result</th>
                  <th className="px-4 py-3 text-center font-medium">Groundedness</th>
                  <th className="px-4 py-3 text-center font-medium">Review confidence</th>
                  <th className="px-4 py-3 text-center font-medium">Keyword recall</th>
                  <th className="px-4 py-3 text-center font-medium">Latency</th>
                  <th className="px-4 py-3 text-center font-medium">Iterations</th>
                </tr>
              </thead>
              <tbody>
                {results.map((result) => (
                  <tr
                    key={result.task_id}
                    className="border-b border-zinc-800/50 hover:bg-zinc-800/30 align-top"
                  >
                    <td className="px-4 py-3 text-zinc-200">
                      {result.task_name}
                      {result.errors.length > 0 && (
                        <p
                          className="mt-1 text-xs text-red-400/80 truncate max-w-md"
                          title={result.errors.join("\n")}
                        >
                          {result.errors[0]}
                        </p>
                      )}
                      {result.keywords_missing.length > 0 && (
                        <p className="mt-1 text-xs text-zinc-500">
                          Missing: {result.keywords_missing.join(", ")}
                        </p>
                      )}
                    </td>
                    {results.some((r) => r.category) && (
                      <td className="px-4 py-3 text-center text-xs text-zinc-500">
                        {result.category ?? "—"}
                      </td>
                    )}
                    <td className="px-4 py-3 text-center">
                      {result.passed ? (
                        <span className="text-emerald-400 text-xs font-medium">PASS</span>
                      ) : (
                        <span
                          className="text-red-400 text-xs font-medium"
                          title={result.status}
                        >
                          FAIL
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <ScoreBar value={result.groundedness ?? 0} />
                    </td>
                    <td className="px-4 py-3">
                      <ScoreBar value={result.quality_score} />
                    </td>
                    <td className="px-4 py-3">
                      <ScoreBar value={result.keyword_recall} />
                    </td>
                    <td className="px-4 py-3 text-center text-zinc-400">
                      {result.latency_seconds.toFixed(1)}s
                    </td>
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

function MetricCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
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
  const percent = Math.round(value * 100);
  const color =
    percent >= 80 ? "bg-emerald-500" : percent >= 60 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center justify-center gap-2">
      <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${percent}%` }} />
      </div>
      <span className="text-xs text-zinc-400">{percent}%</span>
    </div>
  );
}
