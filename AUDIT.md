# Packaging and content audit

Authoritative input: final conversation attachment `QWIM-project.zip` (the requested `/mnt/data` path does not exist on this Mac). SHA-256: `1ab3cb7d154d7b1e3f3609a992440f3447d73afe6bb2a2cf31d93da6e3902c57`.

Retained: `tab_multigoal.py` as `planner.py`, plus a new minimal `app.py`. AST import review shows only standard-library imports, NumPy, pandas, SciPy, Plotly and Shiny after removing unused Polars/pickle imports. `data_utils` and `reactives_shiny` arguments are unused; empty `data_inputs` deliberately invokes the existing synthetic generator. No external asset paths or network fetches are required by this module. Plotly HTML and its bundled JavaScript are generated in the UI (external CDN dependency removed).

Excluded: original `main_App.py` (other tabs, shared data loading and branding), `mgwm_model.py`, `multigoal_planner1.0.py`, `revise.py` (not imported by the active planner), other modules, docs, logs, outputs, research PDFs, macOS metadata and all five original CSVs. ETF history and portfolio allocations/values have uncertain redistribution/proprietary status. The anonymous time series has uncertain provenance. No assumption that these files are public was made.

Changes: per-visitor module namespace isolates all original global lists/counters/results and template helpers; disconnect removes the module registration. Financial diagnostic prints removed; unused disk save/load helpers removed; session save becomes a browser JSON download. Synthetic generation uses a local seeded random generator and business dates instead of calendar dates. Existing numerical model and UI otherwise retained.

Retained-source review found no obvious hard-coded credentials, email addresses, private endpoints, institution branding, or client records. This is a static audit, not a legal clearance or proof of absence. Paper references and template dollar values remain as academic/example content. No license grant was found for the retained code.

Verification results are appended below.

- Python 3.12: syntax parsing, app import and UI construction passed.
- Clean virtual environment dependency installation passed; requirements pins the tested direct package versions.
- Two separate visitor module namespaces: template goal/state isolation passed.
- Synthetic ETF fixture: 1,305 business-day rows × 7 assets; frontier generation passed.
- Small direct numerical smoke test: 3-year horizon, 12-point wealth grid, one hypothetical goal; Bellman solve and finite probability in [0,1] passed. This does not validate model correctness.
- Shiny server startup and HTTP 200 demo page check passed; process stopped after verification.
- Full browser/WebSocket interaction, downloads, all templates and large optimizations have not been tested.
- Original diagnostic `algorithm_used` can report full_dp for a heuristic run because computation_time is always defined; interpret diagnostics cautiously. Numerical behavior has not been broadly rewritten.
- No original source/input files overwritten.
