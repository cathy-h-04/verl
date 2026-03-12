# Plotting Conventions

1. No monolithic plotting suite. Each plot is one Python file under `plots/tier0/`, `plots/tier1/`, or `plots/tier2/`.
2. Every plot file must declare its run selection explicitly in that file (for example, a `RUN_IDS = [...]` constant).
3. Every plot must apply the shared default analysis predicate via `plots.plotting.filters.apply_analysis_ok(...)`.
4. No plot may invent custom exclusion logic unless the file header documents the exception and rationale.
5. Use `plots.plotting.filters.explain_filtering(df_before, df_after)` for auditable filtering logs.
6. Selectors should return `(df, manifest)` (or `(df, manifest, debug_info)`), so plots do not need schema-specific manifest logic.
7. Every plot must write a JSON manifest sidecar next to the figure output via `plots.data.manifest.save_manifest(...)`.
8. Manifest payloads must stay minimal: run IDs plus data provenance (which views/files/roots produced the data), with timestamp and plot name.
9. Every plot must save a PNG under `plots/out/figures/<tier>/`.
10. Shared default filtering excludes the first 5 iterations for full-epoch, non-checkpointed runs on row-level views (`phase_fact_view`, `step_fact_view`, `device_timeseries_view`).

Smoke-check command:

```bash
python -m plots.tier0.example_smoke
```
