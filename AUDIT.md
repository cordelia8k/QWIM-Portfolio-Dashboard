# Packaging, Data, and Verification Record

Last updated: October 9, 2026.

This document records the scope of the standalone Multi-Goal Portfolio Planner package, its data handling, and the checks performed. It does not certify financial model accuracy or establish redistribution rights.

## Source provenance

The authoritative source archive is the final local `QWIM-project.zip`, rather than the initial GitHub archive `QWIM-project-main.zip`.

Source archive SHA-256:

```text
1ab3cb7d154d7b1e3f3609a992440f3447d73afe6bb2a2cf31d93da6e3902c57
```

The active `tab_multigoal.py` module was retained as `planner.py`, with a new standalone Shiny entrypoint in `app.py`. Original source and input files were not overwritten.

## Dependency and asset review

Static import review identified standard-library modules, NumPy, pandas, SciPy, Plotly, and Shiny in the retained application after removing unused Polars and pickle imports. Tested direct dependency versions are recorded in `requirements.txt`.

The planner does not use its `data_utils` or `reactives_shiny` arguments. Empty `data_inputs` deliberately selects the existing synthetic-data generator. The retained planner requires no original data files, image assets, or external data fetches. Plotly HTML includes bundled JavaScript, replacing the original external CDN dependency.

## Excluded content

The deployment package excludes:

- The original `main_App.py`, including unrelated tabs, shared data loading, and branding.
- `mgwm_model.py`, `multigoal_planner1.0.py`, and `revise.py`, which are not imported by the active planner.
- Other unused project modules, original documentation, logs, generated outputs, research PDFs, and macOS metadata.
- All five original CSV files: ETF prices, portfolio weights, anonymous time series, benchmark portfolio values, and portfolio values.

The source datasets have uncertain provenance or redistribution status. They were excluded rather than treated as public data.

## Synthetic demo data

The application generates seven ETF-labelled synthetic return series in memory, using demonstration parameters, regime switching, correlations, a local random generator seeded with 42, and business-day dates. The generated fixture has 1,305 rows and seven asset columns.

ETF symbols are example labels; the simulated series do not represent historical ETF prices or verified forecasts. Goal templates and their amounts are hypothetical examples. Displayed success rates are model outputs under these assumptions, not validated real-world probabilities.

## Session and packaging changes

- Each visitor receives a separate planner module namespace for goals, cash contributions, counters, results, and template helpers. Module registration is removed when the session ends.
- Financial diagnostic print statements were removed from the retained planner.
- Unused server-side save/load helpers were removed. The save control downloads goals and cash contributions as JSON to the visitor's browser.
- Synthetic generation uses a local seeded random generator and business dates rather than the original global seed and calendar dates.
- The existing numerical model and interface were otherwise retained, apart from packaging, data labeling, and the changes described above.

## Content review

Static review of the retained source found no obvious hard-coded credentials, email addresses, private endpoints, institution branding, or client records. Academic references and hypothetical goal amounts remain.

This review is not proof that all sensitive content is absent, and it does not establish ownership of inherited code. No license grant was found for the retained code. Excluding datasets does not establish redistribution rights for the remaining implementation.

## Local verification

The following checks passed during packaging:

- Python 3.12 syntax parsing, application import, and UI construction.
- Dependency installation in a clean virtual environment.
- Goal and result namespace isolation between two separately loaded visitor modules.
- Synthetic return generation and portfolio frontier construction.
- A small direct dynamic-programming smoke test with a three-year horizon, a 12-point wealth grid, and one hypothetical goal: Bellman solution and a finite goal probability within [0, 1].
- Shiny server startup and an HTTP 200 response containing the demo interface. The local test server was stopped afterward.
- Deployment ZIP integrity.

These checks establish basic execution for the tested cases; they do not validate the model's mathematical or financial accuracy.

## Deployment verification

On October 8, 2026, the application owner deployed the package to Posit Connect Cloud. A supplied screenshot showed the planner interface, two goals, and a displayed success-rate result. The owner also confirmed successful access in a private/incognito browser window without signing in.

[Deployed dashboard](https://connect.posit.cloud/cordelia8k/content/01a11c43-8e53-83a8-a4fa-e1e49820ad75)

This records the deployment evidence and access confirmation from that date. It is not a continuous availability check or an exhaustive browser test.

## Remaining limitations

- All browser controls, downloads, templates, and WebSocket reconnection behavior have not been exhaustively tested.
- Large optimization configurations, concurrent visitor workloads, and cross-session isolation under load have not been comprehensively tested.
- Large horizons and wealth grids can consume substantial resources and block the server.
- The original `algorithm_used` diagnostic can report `full_dp` for a heuristic run because `computation_time` is always defined. That label should be interpreted cautiously.
- Claims of complete compliance with the referenced research framework have not been independently verified. Numerical behavior has not been broadly rewritten or validated.
