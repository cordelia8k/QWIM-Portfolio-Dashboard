# Multi-Goal Portfolio Planner — Synthetic Demo

Standalone Shiny for Python app extracted from the user's final local `QWIM-project.zip`. The original project is untouched. The planner UI and optimization algorithms come from `tab_multigoal.py`; the original team dashboard, unrelated tabs, research PDFs and input datasets are excluded.

## Run locally

Use Python 3.12. From this folder:

```sh
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
shiny run app.py --reload
```

Add a goal or select a template, choose settings, then optimize. Download saves goals and cash infusions as JSON to your computer; it does not persist results on the server. Refreshing/reconnecting resets your session.

## Publish

Unzip and upload the files INSIDE this folder to the new GitHub repository, including `.gitignore`. Do not upload the ZIP itself as the application. Select `app.py` as the Shiny entrypoint in a Python/Shiny hosting service, and install `requirements.txt`. For a container or server use:

```sh
shiny run app.py --host 0.0.0.0 --port 8000
```

GitHub stores the source; this server app requires a Python host with WebSocket support. See [official Shiny deployment documentation](https://shiny.posit.co/py/docs/deploy.html) and [cloud hosting](https://shiny.posit.co/py/get-started/deploy-cloud.html).

## Data and limitations

All seven ETF-labelled return series are generated in memory from hard-coded demonstration assumptions using seed 42, business dates, regime switching and correlations. ETF symbols identify example asset categories; the numbers are NOT actual ETF history or verified forecasts. No original price, allocation, portfolio-value or client data is bundled. Template goals are hypothetical examples.

This is an educational prototype, not a validated planning or investment service. Original claims of full paper/framework compliance have not been independently verified. The original algorithm diagnostic can mislabel heuristic runs as full DP; see AUDIT.md. Full dynamic programming and large horizons/grids can block the server and consume considerable memory; start with a small horizon and grid and avoid unrestricted high-traffic deployment until workload controls are implemented. Browser interaction and full optimization are separate from basic startup verification; see AUDIT.md for actual checks.

Source provenance is the user's local project. No new open-source license has been granted: confirm rights to publish inherited code before making the repository public. Removing datasets cannot establish ownership of the remaining algorithms or code.
