# Reinforcement Learning Workshop

Static workshop website for GitHub Pages. The three main areas are **Basics**, **Applied to finance**, and an initial **Coding** resource page. The grid-world simulation uses `app.js`; the finance page uses real Python in the browser through Pyodide 0.29.1 from jsDelivr. No Python server runs on GitHub Pages.

## Pages

- `index.html`: theory and interactive grid world, including a short explanation of training, validation and final testing.
- `finance.html`: Apple, Microsoft, and SPY/TLT/cash scenarios. Each uses the same Overview, Portfolio MDP, Training, and Evaluation navigation. Apple has a switchable training reward and Q-value memory display.
- `coding.html`: first version of the coding area, with a guided Apple notebook and a completed counterpart, source module and data.

Serve the files locally with `python3 -m http.server 8000` from the repository root, then open `http://localhost:8000/`. The finance learning controls require internet access to load Pyodide. On restricted corporate networks, allow `cdn.jsdelivr.net/pyodide/v0.29.1/full/` or arrange a reviewed local runtime mirror before the workshop.

## Reproduction

The HTML embeds the exact content of `asset_agent.py`. If the Python engine changes, update the embedded copy and rerun the parity check before publishing. The notebooks import `asset_agent.py` and use the CSVs in `data/`; run them from the repository root or `notebooks/` with Python, NumPy, pandas and matplotlib installed. The website itself does not need those local packages.

The Apple normalized weekly data are from Plotly's bundled 2018–19 stock example. The SPY/TLT closes are derived from [m-khut/etf-analaysis](https://github.com/m-khut/etf-analaysis). The Microsoft CSV is a separate public historical collection; its adjustment quality has not been independently audited. ETF closes have not been independently verified as dividend-adjusted total returns. Cash yield, transaction costs, and portfolio accounting are simplified. Replaying historical weeks does not create new data, and the 2023–24 final period was viewed during development; it is an illustrative check, not untouched evidence of performance.

## Contributors

Kai designed and implemented the finance case study, Python agent, datasets, and notebooks. Jakob's workshop team integrated the scenarios with the site's navigation and teaching flow. A GitHub profile can be linked here when Kai provides the preferred handle.
