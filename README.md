# Reinforcement Learning Workshop

Static workshop website for GitHub Pages. The three main areas are **Basics**, **Applied to finance**, and an initial **Coding** resource page. The grid-world simulation uses `app.js`; the finance page uses real Python in the browser through Pyodide 0.29.1 from jsDelivr. No Python server runs on GitHub Pages.

## Pages

- `index.html`: theory and interactive grid world, with training and diagnostics below the four live Q-learning boards and a separate frozen-policy evaluation page.
- `finance.html`: Apple, Microsoft, and SPY/TLT/cash scenarios. Each uses Introduction, Environment, Q-learning, and Evaluation. Finance Q-learning shows one state-action panel with the current policy and all action values, followed by a playback readout, progressive guided updates, and expandable training diagnostics.
- `finance-boards.js`: renders the finance state-action panel from live Python snapshots and provides one-week and ten-week playback controls.
- `coding.html`: first version of the coding area, with a guided Apple notebook and a completed counterpart, source module and data.

## Teaching flow and interface

The main navigation leads to Basics, Applied to finance, or Coding. Basics and each finance scenario use the same underlined page labels: Introduction, Environment, Q-learning, Evaluation. The boxed Apple, Microsoft, and Multi-asset selector changes the scenario, not the lesson stage.

| Stage | Basics | Applied to finance |
| --- | --- | --- |
| Introduction | Academic framing of the MDP | Allocation problem and chronological periods |
| Environment | Grid world and its rules | Portfolio state, actions, and reward |
| Q-learning | Four synchronized boards; guided step and playback; expandable training diagnostics below | One panel with 12 Apple or Microsoft states (24 multi-asset states), each displaying signals, previous allocation, current preferred action and all Q-values; guided step, fast step, ten weekly steps, Start and Pause; expandable diagnostics below |
| Evaluation | Frozen policy on fresh grid episodes | Frozen policy on later historical periods |

Keep detailed data, implementation, and notebooks in Coding. Training parameters and reward curves sit in an expandable section below the Q-learning panel. Guided step in both areas reveals one calculation at a time; the step bar at the top names every step and keeps each completed result in place. Finance holds the main board's Q-values steady during the walkthrough, then shows the update and repeats the selected state's final Q-values as a small result. The state panel contains every current Q-value, so the redundant Apple Q-table and Microsoft policy table were removed. Microsoft and multi-asset Q-value trajectories remain in expandable detail sections. The Bellman planning versus sampled-learning comparison is in Basics. Fast step advances one historical training week; 10 steps advances ten weeks, including the boundary between replay passes. One finance replay pass uses the same historical training weeks; it is not a new independent episode. Evaluation uses the current policy without Q updates, and finance validation and final holdout remain distinct from training. Methodological qualifications stay next to the claims they qualify.

The former top-level Training page was folded into Q-learning in both areas; no Python learning or evaluation functionality was removed. The finance introductions use the same numbered academic sections as Basics. The Apple and Microsoft 12-state panels remain immediately visible; the 24-state multi-asset extension uses the same components.

Serve the files locally with `python3 -m http.server 8000` from the repository root, then open `http://localhost:8000/`. The finance learning controls require internet access to load Pyodide. On restricted corporate networks, allow `cdn.jsdelivr.net/pyodide/v0.29.1/full/` or arrange a reviewed local runtime mirror before the workshop.

## Reproduction

The HTML embeds the exact content of `asset_agent.py`. If the Python engine changes, update the embedded copy and rerun the parity check before publishing. The notebooks import `asset_agent.py` and use the CSVs in `data/`; run them from the repository root or `notebooks/` with Python, NumPy, pandas and matplotlib installed. The website itself does not need those local packages.

The Apple normalized weekly data are from Plotly's bundled 2018–19 stock example. The SPY/TLT closes are derived from [m-khut/etf-analaysis](https://github.com/m-khut/etf-analaysis). The Microsoft CSV is a separate public historical collection; its adjustment quality has not been independently audited. ETF closes have not been independently verified as dividend-adjusted total returns. Cash yield, transaction costs, and portfolio accounting are simplified. Replaying historical weeks does not create new data, and the 2023–24 final period was viewed during development; it is an illustrative check, not untouched evidence of performance.

## Contributors

Kai designed and implemented the finance case study, Python agent, datasets, and notebooks. Jakob's workshop team integrated the scenarios with the site's navigation and teaching flow. A GitHub profile can be linked here when Kai provides the preferred handle.
