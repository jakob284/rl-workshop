"use strict";

// The finance engines live in asset_agent.py. These boards only display its snapshots.
(() => {
  const snapshots = {};
  const selected = { apple: null, msft: null, multi: null };
  const $ = id => document.getElementById(id);
  const labels = {
    apple: ["0% Apple", "50% Apple", "100% Apple"],
    msft: ["0% MSFT", "50% MSFT", "100% MSFT"],
    multi: ["Cash", "SPY", "TLT", "SPY/TLT", "SPY/cash", "TLT/cash"]
  };
  const digits = value => Number(value).toFixed(3);
  const best = values => values.indexOf(Math.max(...values));
  function stateName(kind, state) {
    const width = kind === "multi" ? 6 : 3;
    const signals = Math.floor(state / width);
    const first = signals & 2 ? "↑" : "↓";
    const second = signals & 1 ? "↑" : "↓";
    return kind === "multi"
      ? `SPY ${first}, TLT ${second} · previous ${labels[kind][state % width]}`
      : `4w ${first}, 12w ${second} · previous ${["0%", "50%", "100%"][state % width]}`;
  }
  function draw(kind) {
    const snapshot = snapshots[kind];
    if (!snapshot?.q) return;
    const { q, last, enabled } = snapshot;
    const prefix = kind;
    const active = selected[kind] ?? last?.s ?? last?.state ?? 0;
    const current = last?.s ?? last?.state;
    const policy = $(prefix + "-policy-board");
    policy.innerHTML = q.map((row, state) =>
      `<button type="button" class="finance-state-tile ${state === active ? "selected" : ""} ${state === current ? "observed" : ""}" data-state="${state}" aria-pressed="${state === active}" title="${stateName(kind, state)}">
        <small>s${state} · ${stateName(kind, state)}</small><strong>${labels[kind][best(row)]}</strong>
      </button>`).join("");
    $(prefix + "-value-board").innerHTML = q.map((row, state) =>
      `<div class="finance-value-tile ${state === active ? "selected" : ""}" title="${stateName(kind, state)}"><small>s${state}</small><strong>${digits(Math.max(...row))}</strong></div>`).join("");
    $(prefix + "-selected-state").textContent = `s${active} · ${stateName(kind, active)}`;
    $(prefix + "-q-board").innerHTML = q[active].map((value, action) =>
      `<div class="finance-action-value ${action === best(q[active]) ? "best" : ""}"><small>${labels[kind][action]}</small><strong>${digits(value)}</strong></div>`).join("");
    const description = !last ? "No transition yet. Use Guided step, Fast step, or run a training pass."
      : kind === "apple"
        ? `State s${last.s} (${stateName(kind, last.s)}); chose ${labels[kind][last.a]}. Reward ${digits(last.reward)}; Q-value ${digits(last.oldQ)} → ${digits(last.updated)}.`
        : $(kind === "msft" ? "ch-last-decision" : "ma-last-decision").textContent;
    $(prefix + "-agent-board").textContent = description.replace(/\s+/g, " ").trim();
    policy.querySelectorAll("[data-state]").forEach(tile => tile.addEventListener("click", () => {
      selected[kind] = Number(tile.dataset.state);
      const select = $(kind === "msft" ? "ch-state" : kind === "multi" ? "ma-state" : "not-present");
      if (select) { select.value = selected[kind]; select.dispatchEvent(new Event("change")); }
      draw(kind);
    }));
    document.querySelectorAll(`[data-finance-action][data-for="${kind === "msft" ? "microsoft" : kind}"]`).forEach(button => {
      button.disabled = !enabled;
    });
  }
  window.renderFinanceBoards = (kind, q, last, enabled) => {
    snapshots[kind] = { q, last, enabled };
    if (last && selected[kind] === null) selected[kind] = last.s ?? last.state ?? 0;
    draw(kind);
  };
  for (const [id, kind] of [["ch-state", "msft"], ["ma-state", "multi"]]) {
    $(id).addEventListener("change", event => {
      selected[kind] = Number(event.target.value);
      draw(kind);
    });
  }
  document.querySelectorAll("[data-finance-action]").forEach(button => button.addEventListener("click", () => {
    const kind = button.dataset.for;
    if (button.dataset.financeAction === "reset") {
      selected[kind === "microsoft" ? "msft" : kind] = null;
      $(button.dataset.reset).click();
      return;
    }
    if (button.disabled) return;
    const advance = window.financeAdvance?.[kind];
    if (!advance) return;
    advance(button.dataset.financeAction === "fast" ? 0 : button.dataset.financeAction === "ten" ? 10 : 1);
  }));
})();
