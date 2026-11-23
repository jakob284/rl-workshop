# Reinforcement Learning Workshop

Static workshop website for GitHub Pages. The three main areas are **Basics**, **Applied to finance**, and an initial **Coding** resource page. The grid-world simulation uses `app.js`; the finance page uses real Python in the browser through Pyodide 0.29.1 from jsDelivr. No Python server runs on GitHub Pages.

## Pages

- `index.html`: theory and interactive grid world, with a separate frozen-policy evaluation page and a short explanation of training, validation and final testing.
- `finance.html`: Apple, Microsoft, and SPY/TLT/cash scenarios. Each uses Overview, Portfolio MDP, Control room, Training, and Evaluation. The control rooms expose one real Python step and learned Q-values; training controls and reward curves follow on their own page.
- `coding.html`: first version of the coding area, with a guided Apple notebook and a completed counterpart, source module and data.

## Teaching flow and interface

The main navigation leads to Basics, Applied to finance, or Coding. Basics and each finance scenario use the same underlined page navigation. The boxed Apple, Microsoft, and Multi-asset selector changes the scenario, not the lesson stage.

| Stage | Basics | Applied to finance |
| --- | --- | --- |
| Setup | Introduction and Environment | Overview and Portfolio MDP |
| One decision | Q-learning Control room with guided step | Control room with a real weekly Q update and learned values |
| Repeated learning | Training diagnostics | Training controls and reward curves |
| Frozen policy | Evaluation on fresh grid episodes | Evaluation on later historical periods |

Keep detailed data, implementation, and notebooks in Coding. Keep the finance Q-value boards inside Control room and the training curves inside Training. Evaluation uses the current policy without further Q updates; finance validation and final holdout remain distinct from the training period. Explanatory caveats should appear next to the claim they qualify, without a separate modeling-assumption block at the bottom of a lesson page.

Serve the files locally with `python3 -m http.server 8000` from the repository root, then open `http://localhost:8000/`. The finance learning controls require internet access to load Pyodide. On restricted corporate networks, allow `cdn.jsdelivr.net/pyodide/v0.29.1/full/` or arrange a reviewed local runtime mirror before the workshop.

## Reproduction

The HTML embeds the exact content of `asset_agent.py`. If the Python engine changes, update the embedded copy and rerun the parity check before publishing. The notebooks import `asset_agent.py` and use the CSVs in `data/`; run them from the repository root or `notebooks/` with Python, NumPy, pandas and matplotlib installed. The website itself does not need those local packages.

The Apple normalized weekly data are from Plotly's bundled 2018–19 stock example. The SPY/TLT closes are derived from [m-khut/etf-analaysis](https://github.com/m-khut/etf-analaysis). The Microsoft CSV is a separate public historical collection; its adjustment quality has not been independently audited. ETF closes have not been independently verified as dividend-adjusted total returns. Cash yield, transaction costs, and portfolio accounting are simplified. Replaying historical weeks does not create new data, and the 2023–24 final period was viewed during development; it is an illustrative check, not untouched evidence of performance.

## Contributors

Kai designed and implemented the finance case study, Python agent, datasets, and notebooks. Jakob's workshop team integrated the scenarios with the site's navigation and teaching flow. A GitHub profile can be linked here when Kai provides the preferred handle.
