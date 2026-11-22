# Offline workflow demonstration

[Watch the captioned video](assets/lattice-offline-demo.webm) · [Animated preview](assets/lattice-offline-preview.gif) · [Static poster](assets/lattice-demo-poster.png) · [Captured output](assets/mock-workflow.txt)

The 34-second video replays an actual run of `examples/mock_workflow.py`. It uses authored pacing to make a short execution readable. Its panels are a visualization of the captured task, drafts, reviews, and final artifact; they are not a dashboard recording.

## Reproduce the run

From the installed repository:

```bash
python examples/mock_workflow.py
```

The example runs the real `OrchestratorAgent` through a plan, a research artifact, an initial document, a rejected review, a revision, and approval. Six fixed `MockProvider` responses make the path deterministic. The script checks for completed status, two document versions, the rejected/approved review sequence, and the expected number of calls.

It makes no model or network requests and performs no external write-back. This demonstrates orchestration behavior; it does not measure the quality of a live language model or the integration of the dashboard.

See the [development guide](development.md#model-backed-demo) to run a separate model-backed example.
