# Multi-Goal Portfolio Planner

An interactive financial planning prototype for exploring how multiple goals, available wealth, future cash contributions, and portfolio strategies interact over time. Built with Python and Shiny, the dashboard uses synthetic ETF return scenarios for educational demonstrations.

**[Open Live Demo](https://connect.posit.cloud/cordelia8k/content/01a11c43-8e53-83a8-a4fa-e1e49820ad75)** · **[View Source](https://github.com/cordelia8k/QWIM-Portfolio-Dashboard)**

## Features

- Create multiple financial goals with target amounts, timelines, utility values, and priority weights.
- Explore partial goal achievement and predefined retirement, housing, education, and other hypothetical goals.
- Add future cash contributions and adjust initial wealth, planning horizon, and inflation assumptions.
- Compare ETF portfolio strategies and inspect model-estimated goal achievement probabilities.
- Explore results through interactive charts, portfolio analysis, and algorithm diagnostics.
- Download current goals and cash contributions as JSON.

## Try the dashboard

1. Open the live demo and add a goal or choose a template.
2. Adjust the planning settings and optionally add future cash contributions.
3. Run optimization, then explore the analysis and portfolio tabs.

Each visitor has separate session state. Refreshing or reconnecting resets the session. The JSON download contains goals and cash contributions, rather than a complete saved optimization result.

## Methods and technology

The prototype combines goal-based dynamic programming inspired by Das et al. (2022), mean–variance portfolio optimization, and heuristic calculations for larger problems. Synthetic return scenarios include regime switching and correlations; portfolio parameters use exponentially weighted estimates.

| Component | Technology |
| --- | --- |
| Web application | Shiny for Python |
| Numerical computation | NumPy and SciPy |
| Data handling | pandas |
| Interactive charts | Plotly |
| Hosting | Posit Connect Cloud |

The automatic mode selects between dynamic programming and a heuristic approach according to problem size. The implementation has not been independently verified as a complete reproduction of the referenced research framework.

## Run locally

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
# On Windows, use: .venv\Scripts\activate
python -m pip install -r requirements.txt
shiny run app.py --reload
```

Open the local URL printed in the terminal.

## Project structure

```text
app.py            Shiny entrypoint and visitor session isolation
planner.py        Planner interface, portfolio models, and optimization logic
requirements.txt  Tested direct dependency versions
README.md         Project overview and usage
AUDIT.md          Source provenance, packaging changes, and verification record
.gitignore        Local environments, caches, and generated files to exclude
```

## Data and interpretation

All seven ETF-labelled return series are generated in memory from synthetic assumptions using seed 42. ETF symbols serve as example asset labels; the series are not historical ETF prices or verified market forecasts. Goal templates are hypothetical examples. No original project input datasets, client records, or portfolio allocations are included.

Displayed success rates are outputs of the demonstration model under its assumptions. They are not validated real-world probabilities or investment guarantees. This application is an educational prototype and is not intended for investment decisions.

## Known limitations

- Large horizons and wealth grids can require substantial computation and block the server. Start with a small problem when exploring dynamic programming.
- The original algorithm diagnostic can label a heuristic run as full dynamic programming; interpret that label cautiously.
- Local checks passed for syntax, app startup, visitor namespace isolation, synthetic data generation, and a small direct dynamic-programming example. These checks do not establish mathematical or financial validity.
- The deployed app has been opened successfully without signing in. All browser interactions, downloads, and large optimization configurations have not been exhaustively tested.

See [AUDIT.md](AUDIT.md) for the detailed verification record.

## Provenance and deployment

This standalone version was adapted from the Multi-Goal Planner module in a local QWIM project. Unrelated dashboard modules, original input datasets, and research PDFs were excluded. Packaging changes added a standalone entrypoint, visitor state isolation, browser downloads, and explicit synthetic-data labeling.

No new open-source license is granted by this repository. Redistribution rights for inherited code should be confirmed before reuse.

For hosting, use `app.py` as the Shiny entrypoint and install `requirements.txt`. See the [official Shiny deployment documentation](https://shiny.posit.co/py/docs/deploy.html).
