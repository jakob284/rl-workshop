"use strict";

// Display the Q-tables returned by asset_agent.py; learning remains in the Python engine.
(() => {
  const $ = id => document.getElementById(id);
  const snapshots = {};
  const held = {};
  const selected = { apple: null, msft: null, multi: null };
  const choices = {
    apple: ["0%", "50%", "100%"],
    msft: ["0%", "50%", "100%"],
    multi: ["Cash", "SPY", "TLT", "SPY/TLT", "SPY/cash", "TLT/cash"]
  };
  const best = row => row.indexOf(Math.max(...row));
  const value = number => Number(number).toFixed(4);
  const guideId = { apple: "finance-guide", msft: "ch-guide", multi: "ma-guide" };
  const guideButton = { apple: "finance-step", msft: "ch-guided-step", multi: "ma-guided-step" };

  function stateLabel(kind, state) {
    const width = kind === "multi" ? 6 : 3;
    const signal = Math.floor(state / width);
    const short = signal & 2 ? "↑" : "↓";
    const long = signal & 1 ? "↑" : "↓";
    return kind === "multi"
      ? `SPY ${short} · TLT ${long} · previous ${choices[kind][state % width]}`
      : `4w ${short} · 12w ${long} · previous ${choices[kind][state % width]}`;
  }

  function draw(kind) {
    const snapshot = snapshots[kind];
    if (!snapshot?.q) return;
    const { last, enabled, meta } = snapshot;
    const q = held[kind] || snapshot.q;
    const active = selected[kind] ?? last?.s ?? last?.state ?? 0;
    const observed = last?.s ?? last?.state;
    const board = $(kind + "-state-board");
    board.innerHTML = q.map((row, state) => {
      const preferred = best(row);
      const estimates = row.map((estimate, action) =>
        `<span class="finance-state-q ${action === preferred ? "preferred" : ""}"><small>Q(${choices[kind][action]})</small><strong>${value(estimate)}</strong></span>`).join("");
      return `<button type="button" class="finance-state-card ${state === active ? "selected" : ""} ${state === observed ? "observed" : ""}" data-state="${state}" aria-pressed="${state === active}" aria-label="State s${state}: ${stateLabel(kind,state)}. Preferred allocation ${choices[kind][preferred]}. ${row.map((estimate,action) => `Q ${choices[kind][action]} ${value(estimate)}`).join(', ')}">
        <span class="finance-state-head"><strong>s${state}</strong><b>${choices[kind][preferred]}</b></span>
        <span class="finance-state-signals">${stateLabel(kind,state)}</span>
        <span class="finance-state-qs">${estimates}</span>
      </button>`;
    }).join("");
    const state = last?.s ?? last?.state;
    const action = last?.a ?? last?.action;
    const readout = { date: meta?.date ?? "—", state: last ? `s${state}` : "—",
      action: last ? choices[kind][action] : "—",
      reward: last ? `${(last.reward * 100).toFixed(2)}%` : "—",
      pass: `${((meta?.currentReward ?? 0) * 100).toFixed(2)}%` };
    for (const [field, text] of Object.entries(readout)) $(kind + "-read-" + field).textContent = text;
    board.querySelectorAll("[data-state]").forEach(card => card.addEventListener("click", () => {
      selected[kind] = Number(card.dataset.state);
      const selector = $(kind === "msft" ? "ch-state" : kind === "multi" ? "ma-state" : "no-select");
      if (selector) { selector.value = selected[kind]; selector.dispatchEvent(new Event("change")); }
      draw(kind);
    }));
    const actionKind = kind === "msft" ? "microsoft" : kind;
    document.querySelectorAll(`[data-finance-action][data-for="${actionKind}"]`).forEach(button => { button.disabled = !enabled; });
    $(guideButton[kind]).disabled = !enabled;
  }

  window.renderFinanceBoards = (kind, q, last, enabled, meta) => {
    snapshots[kind] = { q, last, enabled, meta };
    if (last && selected[kind] === null) selected[kind] = last.s ?? last.state ?? 0;
    draw(kind);
  };
  window.finishFinanceGuided = kind => { delete held[kind]; draw(kind); };
  for (const [id, kind] of [["ch-state", "msft"], ["ma-state", "multi"]]) {
    $(id).addEventListener("change", event => { selected[kind] = Number(event.target.value); draw(kind); });
  }
  for (const kind of Object.keys(guideButton)) {
    $(guideButton[kind]).disabled = true;
    $(guideButton[kind]).addEventListener("click", () => {
      if (snapshots[kind]?.q) held[kind] = snapshots[kind].q.map(row => row.slice());
    }, true);
    const start = $(kind === "apple" ? "finance-train" : kind === "msft" ? "ch-start" : "ma-start");
    start.addEventListener("click", () => { window.finishFinanceGuided(kind); $(guideId[kind]).hidden = true; }, true);
    $(kind === "apple" ? "finance-reset" : kind === "msft" ? "ch-reset" : "ma-reset")
      .addEventListener("click", () => window.finishFinanceGuided(kind), true);
  }
  document.querySelectorAll("[data-finance-action]").forEach(button => {
    button.disabled = true;
    button.addEventListener("click", () => {
      if (button.disabled) return;
      const kind = button.dataset.for;
      const boardKind = kind === "microsoft" ? "msft" : kind;
      window.finishFinanceGuided(boardKind);
      $(guideId[boardKind]).hidden = true;
      window.financeAdvance?.[kind]?.(button.dataset.financeAction === "ten" ? 10 : 1);
    });
  });
  document.querySelectorAll("[data-scenario], [data-finance-view]").forEach(button => button.addEventListener("click", () => {
    for (const kind of Object.keys(guideId)) { if (held[kind]) window.finishFinanceGuided(kind); $(guideId[kind]).hidden = true; }
  }));
})();
