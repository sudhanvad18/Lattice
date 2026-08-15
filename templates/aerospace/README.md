# Aerospace Quality Template

Pre-configured Lattice setup for aerospace engineering teams managing engine maintenance, quality documentation, and failure analysis.

## Entity Types

- **Component**: Engine parts, assemblies, sub-systems
- **FailureMode**: Known failure types (erosion, fatigue, corrosion)
- **Procedure**: Inspection, maintenance, repair workflows
- **Specification**: Engineering tolerances, material specs
- **RootCause**: Underlying causes of failures
- **Mitigation**: Actions that reduce failure risk

## Relationship Types

- `HAS_COMPONENT` — Assembly → Part
- `HAS_FAILURE_MODE` — Component → FailureMode
- `CAUSED_BY` — FailureMode → RootCause
- `MITIGATED_BY` — FailureMode → Mitigation
- `REQUIRES_PROCEDURE` — Component → Procedure
- `REFERENCES_SPEC` — Procedure → Specification

## Sample Documents

Place your documents in `./docs/` — the ingestion pipeline handles:
- PDF maintenance manuals
- Markdown engineering notes
- HTML quality reports

## Agent Configuration

```yaml
agents:
  researcher:
    system_prompt_suffix: |
      You are an aerospace engineering research assistant.
      Focus on: failure modes, inspection intervals, regulatory compliance.
      Always cite specific document sections and specification numbers.

  writer:
    system_prompt_suffix: |
      Generate documentation following AS9100 quality management standards.
      Include traceability matrices and reference applicable FAR/CFR.

  reviewer:
    criteria:
      - Technical accuracy against cited sources
      - Compliance with AS9100/ISO 9001 requirements
      - Proper unit usage (SI preferred, imperial where industry standard)
      - Complete traceability to source specifications
```

## Quick Start

```bash
lattice init --template aerospace
lattice ingest ./docs/
lattice run "Generate inspection procedure for first-stage compressor blades"
```
