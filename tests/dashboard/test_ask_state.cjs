const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const dashboard = path.resolve(__dirname, "../../packages/dashboard");
const ts = require(path.join(dashboard, "node_modules/typescript"));

// Exercise the same storage and monitoring functions used by Ask, without a DOM.
function compile(file, extra = "") {
  const source = readFileSync(path.join(dashboard, "src", file), "utf8") + extra;
  return ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX,
  } }).outputText;
}
const apiCode = compile("lib/api.ts");
const askCode = compile("components/ask.tsx", "\nexport { readConversation, saveConversation, monitorTask, groupArtifacts };");
function runtime(options = {}) {
  const api = {};
  vm.runInNewContext(apiCode, { exports: api, process: { env: {} } });
  const apiOverrides = options.api;
  const exports = {};
  vm.runInNewContext(askCode, {
    exports, require: (name) => name === "@/lib/api" ? { ...api, ...apiOverrides } : {},
    Error, AbortController, AbortSignal, setTimeout, clearTimeout, ...options,
  });
  return exports;
}

test("round-trip restores partial Answer, citations, mode, draft and task ID", () => {
  let raw;
  const api = runtime({ localStorage: { getItem: () => raw, setItem: (_, value) => { raw = value; } } });
  const citation = { index: 1, kind: "chunk", id: "src", source: "README", snippet: "contents" };
  const message = { role: "assistant", timestamp: new Date(), streaming: true,
    content: "Partial [1]", citations: [citation, null] };
  api.saveConversation({ mode: "agent", input: "draft", approvalMode: "always", messages: [
    message, { ...message, streaming: false, agentRun: { taskId: "run-1", status: "running", events: [] } },
  ] });
  const restored = api.readConversation();
  assert.equal(restored.mode, "agent");
  assert.equal(restored.approvalMode, "always");
  assert.equal(restored.input, "draft");
  assert.equal(restored.messages[0].content, message.content);
  assert.equal(restored.messages[0].citations.length, 1);
  assert.equal(restored.messages[0].citations[0].source, citation.source);
  assert.equal(restored.messages[0].timestamp.toISOString(), message.timestamp.toISOString());
  assert.match(restored.messages[0].notice, /interrupted/i);
  assert.ok(!restored.messages[0].streaming);
  assert.equal(restored.messages[1].agentRun.taskId, "run-1");
  assert.equal(restored.messages[1].agentRun.status, "running");
});

test("corrupt and unavailable storage degrade safely", () => {
  for (const raw of ["{", "null", '{"messages":[null,{},{"role":"user","timestamp":false}]}']) {
    assert.equal(runtime({ localStorage: { getItem: () => raw } }).readConversation().messages.length, 0);
  }
  const api = runtime();
  const state = api.readConversation();
  assert.equal(state.mode, "answer");
  assert.doesNotThrow(() => api.saveConversation(state));
});

function monitor(status, result = () => ({ artifacts: [{ content: "draft" }] })) {
  const timers = new Map();
  let index = 0;
  let resultCalls = 0;
  let statusCalls = 0;
  let socket;
  let paused;
  let run = { taskId: "task", events: [], status: "running" };
  const api = runtime({
    setTimeout: (callback) => { timers.set(++index, callback); return index; },
    clearTimeout: (id) => timers.delete(id),
    api: {
      createWebSocket: () => (socket = { close() {} }),
      getTask: async () => { statusCalls++; return status(); },
      getTaskResult: () => { resultCalls++; return result(); },
    },
  });
  const stop = api.monitorTask(run, (_, change) => { run = change(run); }, (err) => { paused = err; });
  return {
    get run() { return run; }, get paused() { return paused; },
    get resultCalls() { return resultCalls; }, get statusCalls() { return statusCalls; },
    tick() {
      assert.equal(timers.size, 1);
      const [key, callback] = timers.entries().next().value;
      timers.delete(key);
      callback();
      return new Promise((resolve) => setImmediate(resolve));
    },
    event(event, message) {
      socket.onmessage({ data: JSON.stringify({ event, timestamp: "2026-11-01T10:00:00Z", data: { message } }) });
    },
    close() { socket.onclose(); },
    stop, timers,
  };
}

test("reconnect replay and intermediate completions keep a single running task", async () => {
  let status = "in_progress";
  const m = monitor(() => ({ status }));
  await m.tick();
  m.event("agent_completed", "Retrieved KB passages");
  m.close();
  await m.tick();
  m.event("agent_completed", "Retrieved KB passages");
  assert.equal(m.run.events.length, 1);
  assert.equal(m.run.status, "running");
  assert.equal(m.resultCalls, 0);
  status = "completed";
  await m.tick();
  assert.equal(m.run.status, "done");
  assert.equal(m.resultCalls, 1);
  assert.equal(m.run.artifacts[0].content, "draft");
  assert.equal(m.timers.size, 0);
});

test("terminal event during an outstanding poll reconciles errors and partial artifacts once", async () => {
  let resolveStatus;
  const m = monitor(() => new Promise((resolve) => { resolveStatus = resolve; }));
  await m.tick();
  m.event("error", " ");
  assert.match(m.run.errorMessage, /failed/i);
  assert.equal(m.timers.size, 0);
  resolveStatus({ status: "failed", errors: ["Reviewer unavailable"] });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(m.statusCalls, 1);
  assert.equal(m.resultCalls, 1);
  assert.equal(m.run.status, "error");
  assert.equal(m.run.errorMessage, "Reviewer unavailable");
  assert.equal(m.run.artifacts[0].content, "draft");
  assert.equal(m.timers.size, 0);
});

test("monitor backs off and pauses after persistent API failures", async () => {
  const m = monitor(() => Promise.reject(new Error("offline")));
  for (let i = 0; i < 5; i++) await m.tick();
  assert.match(m.paused, /offline/);
  assert.equal(m.run.status, "running");
  assert.equal(m.resultCalls, 0);
  assert.equal(m.timers.size, 0);
});

test("unmount ignores an in-flight status response", async () => {
  let resolveStatus;
  const m = monitor(() => new Promise((resolve) => { resolveStatus = resolve; }));
  await m.tick();
  m.stop();
  resolveStatus({ status: "completed" });
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(m.resultCalls, 0);
  assert.equal(m.timers.size, 0);
});

test("storage bounds preserve an older active task and recent partial answer", () => {
  let raw;
  const api = runtime({ localStorage: { getItem: () => raw, setItem: (_, value) => { raw = value; } } });
  const message = { role: "assistant", content: "", timestamp: new Date() };
  const messages = [
    { ...message, agentRun: { taskId: "old", events: [], status: "running" } },
    ...Array.from({ length: 120 }, (_, i) => ({ ...message, content: String(i) })),
    { ...message, streaming: true, content: "a".repeat(1_000_000) },
  ];
  api.saveConversation({ mode: "answer", approvalMode: "never", input: "draft", messages });
  assert.ok(raw.length <= 1_000_000);
  const restored = api.readConversation();
  assert.equal(restored.messages.length, 100);
  assert.equal(restored.messages[0].agentRun.taskId, "old");
  assert.ok(restored.messages.at(-1).content.length > 0);
  assert.match(restored.messages.at(-1).notice, /interrupted/i);
});

test("raw task_complete recovers result even if status is temporarily unavailable", async () => {
  const m = monitor(() => Promise.reject(new Error("offline")));
  await m.tick();
  m.event("task_complete", "Finished review");
  await m.tick();
  assert.equal(m.run.status, "done");
  assert.equal(m.resultCalls, 1);
  assert.equal(m.run.resultPending, false);
  assert.equal(m.timers.size, 0);
});

test("artifacts arrive during the run and polling recovers a missed review event", async () => {
  let status = "in_progress";
  let artifacts = [];
  let reviews = [];
  const m = monitor(() => ({ status, artifact_count: artifacts.length, review_count: reviews.length }),
    () => ({ status, artifacts, reviews, errors: status === "completed" ? ["Unresolved integration question"] : [] }));
  await m.tick();
  artifacts = [{ id: "research", content: "Found source" }];
  m.event("artifact_produced", "Produced research");
  await m.tick();
  assert.equal(m.run.status, "running");
  assert.equal(m.run.artifacts[0].content, "Found source");
  m.event("artifact_produced", "Produced research");
  await m.tick();
  assert.equal(m.resultCalls, 1);
  m.close();
  artifacts = [...artifacts, { id: "draft", content: "Draft" }];
  reviews = [{ artifact_id: "draft", approved: false, issues: ["Integration missing"], reasoning: "" }];
  await m.tick();
  assert.equal(m.run.reviews[0].approved, false);
  status = "completed";
  await m.tick();
  assert.equal(m.run.errorMessage, "Unresolved integration question");
  assert.equal(m.run.status, "done");
  assert.equal(m.timers.size, 0);
});

test("a terminal event during a partial result fetch reloads the final artifacts", async () => {
  let finish;
  let completed = false;
  const m = monitor(() => ({ status: completed ? "completed" : "in_progress", artifact_count: 1 }),
    () => completed ? { status: "completed", artifacts: [{ content: "Final draft" }] }
      : new Promise((resolve) => { finish = resolve; }));
  await m.tick();
  completed = true;
  m.event("task_complete", "Finished");
  finish({ status: "in_progress", artifacts: [] });
  await new Promise((resolve) => setImmediate(resolve));
  await m.tick();
  assert.equal(m.run.artifacts[0].content, "Final draft");
  assert.equal(m.run.resultPending, false);
  assert.equal(m.timers.size, 0);
});

test("group revisions by subtask, support historical query metadata and isolate unknown drafts", () => {
  const { groupArtifacts } = runtime();
  const doc = { artifact_type: "document", source_agent: "writer" };
  const groups = groupArtifacts([
    { ...doc, id: "1", metadata: { subtask_id: "write", query: "Draft" } },
    { ...doc, id: "2", metadata: { subtask_id: "write", query: "Revise" } },
    { ...doc, id: "3", metadata: { query: "POC" } }, { ...doc, id: "4", metadata: { query: "POC" } },
    { ...doc, id: "5" }, { ...doc, id: "6" },
  ]);
  assert.deepEqual(Array.from(groups, (versions) => Array.from(versions, (a) => a.id)),
    [["1", "2"], ["3", "4"], ["5"], ["6"]]);
});
