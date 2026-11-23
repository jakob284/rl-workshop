"use strict";

const SIZE = 5;
const N = SIZE * SIZE;
const START = 20;
const GOAL = 4;
const LOSS = 14;
const WALLS = new Set([7, 12, 17]);
const TERMINALS = new Set([GOAL, LOSS]);
const ACTIONS = [
  { key: "u", label: "Up", arrow: "↑", dr: -1, dc: 0 },
  { key: "r", label: "Right", arrow: "→", dr: 0, dc: 1 },
  { key: "d", label: "Down", arrow: "↓", dr: 1, dc: 0 },
  { key: "l", label: "Left", arrow: "←", dr: 0, dc: -1 }
];

const settings = { alpha: .30, gamma: .92, epsilon: .20, slip: 0 };
const qLearned = Array.from({ length: N }, () => Array(4).fill(0));
const visits = Array(N).fill(0);
const pairVisits = Array.from({ length: N }, () => Array(4).fill(0));
let valuesPolicy = Array(N).fill(0);
let qPolicy = Array.from({ length: N }, () => Array(4).fill(0));
let qExact = Array.from({ length: N }, () => Array(4).fill(0));
let agentState = START;
let episodeReturn = 0;
let episodeSteps = 0;
let episodeCount = 0;
let returnHistory = [];
let path = [START];
let selectedState = START;
let guided = null;
let randomState = 934775;
let evaluationCurrent = false;

function clearEvaluation() {
  if (!evaluationCurrent) return;
  evaluationCurrent = false;
  for (const id of ["eval-greedy-return", "eval-random-return", "eval-goal-rate"]) {
    document.getElementById(id).textContent = "—";
  }
  document.getElementById("evaluation-status").textContent = "The policy changed. Run evaluation again to see the current result.";
}

function rng() {
  randomState |= 0; randomState = randomState + 0x6D2B79F5 | 0;
  let t = Math.imul(randomState ^ randomState >>> 15, 1 | randomState);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
  return ((t ^ t >>> 14) >>> 0) / 4294967296;
}

function initializeQLearned() {
  for (let s = 0; s < N; s++) {
    for (let a = 0; a < 4; a++) {
      qLearned[s][a] = WALLS.has(s) || TERMINALS.has(s) ? 0 : (rng() - .5) * .02;
    }
  }
}

const rc = s => [Math.floor(s / SIZE), s % SIZE];
const stateName = s => `s${s}`;
const isBlocked = s => s < 0 || s >= N || WALLS.has(s);
const rewardAt = s => s === GOAL ? 1 : s === LOSS ? -1 : -.04;

function move(s, actionIndex) {
  if (TERMINALS.has(s)) return s;
  const [r, c] = rc(s);
  const a = ACTIONS[actionIndex];
  const nr = r + a.dr, nc = c + a.dc;
  if (nr < 0 || nr >= SIZE || nc < 0 || nc >= SIZE) return s;
  const ns = nr * SIZE + nc;
  return WALLS.has(ns) ? s : ns;
}

function transitionDistribution(s, actionIndex) {
  if (TERMINALS.has(s)) return [{ p: 1, next: s, reward: 0 }];
  const side = settings.slip / 2;
  const candidates = [
    { p: 1 - settings.slip, a: actionIndex },
    { p: side, a: (actionIndex + 3) % 4 },
    { p: side, a: (actionIndex + 1) % 4 }
  ].filter(x => x.p > 0);
  const merged = new Map();
  candidates.forEach(x => {
    const next = move(s, x.a);
    const key = `${next}:${rewardAt(next)}`;
    const old = merged.get(key) || { p: 0, next, reward: rewardAt(next) };
    old.p += x.p; merged.set(key, old);
  });
  return [...merged.values()];
}

function sampleTransition(s, actionIndex) {
  const branches = transitionDistribution(s, actionIndex);
  let draw = rng();
  for (const branch of branches) {
    draw -= branch.p;
    if (draw <= 0) return branch;
  }
  return branches[branches.length - 1];
}

function evaluateRandomPolicy() {
  let v = Array(N).fill(0);
  for (let iter = 0; iter < 1000; iter++) {
    const nextV = v.slice();
    let delta = 0;
    for (let s = 0; s < N; s++) {
      if (WALLS.has(s) || TERMINALS.has(s)) continue;
      let value = 0;
      for (let a = 0; a < 4; a++) {
        let av = 0;
        transitionDistribution(s, a).forEach(b => av += b.p * (b.reward + (TERMINALS.has(b.next) ? 0 : settings.gamma * v[b.next])));
        value += .25 * av;
      }
      nextV[s] = value;
      delta = Math.max(delta, Math.abs(nextV[s] - v[s]));
    }
    v = nextV;
    if (delta < 1e-8) break;
  }
  valuesPolicy = v;
  qPolicy = Array.from({ length: N }, (_, s) => ACTIONS.map((_, a) => {
    if (WALLS.has(s) || TERMINALS.has(s)) return 0;
    return transitionDistribution(s, a).reduce((sum, b) => sum + b.p * (b.reward + (TERMINALS.has(b.next) ? 0 : settings.gamma * v[b.next])), 0);
  }));
}

function computeExactQ() {
  let q = Array.from({ length: N }, () => Array(4).fill(0));
  for (let iter = 0; iter < 1500; iter++) {
    const nq = q.map(row => row.slice());
    let delta = 0;
    for (let s = 0; s < N; s++) {
      if (WALLS.has(s) || TERMINALS.has(s)) continue;
      for (let a = 0; a < 4; a++) {
        nq[s][a] = transitionDistribution(s, a).reduce((sum, b) => {
          const continuation = TERMINALS.has(b.next) ? 0 : settings.gamma * Math.max(...q[b.next]);
          return sum + b.p * (b.reward + continuation);
        }, 0);
        delta = Math.max(delta, Math.abs(nq[s][a] - q[s][a]));
      }
    }
    q = nq;
    if (delta < 1e-9) break;
  }
  qExact = q;
}

function greedyActions(qRow) {
  const max = Math.max(...qRow);
  return qRow.map((v, i) => Math.abs(v - max) < 1e-10 ? i : -1).filter(i => i >= 0);
}

function chooseAction(s, forceRandom = false) {
  if (forceRandom || rng() < settings.epsilon) return Math.floor(rng() * 4);
  const best = greedyActions(qLearned[s]);
  return best[Math.floor(rng() * best.length)];
}

function qUpdate(s, a, transition) {
  clearEvaluation();
  const old = qLearned[s][a];
  const nextBest = TERMINALS.has(transition.next) ? 0 : Math.max(...qLearned[transition.next]);
  const target = transition.reward + settings.gamma * nextBest;
  const td = target - old;
  qLearned[s][a] += settings.alpha * td;
  pairVisits[s][a]++;
  return { old, nextBest, target, td, updated: qLearned[s][a] };
}

function completeTransition(s, a, transition, update = true) {
  const details = update ? qUpdate(s, a, transition) : null;
  agentState = transition.next;
  episodeReturn += transition.reward;
  episodeSteps++;
  visits[agentState]++;
  path.push(agentState);
  updateReadout(s, a, transition);
  if (TERMINALS.has(agentState) || episodeSteps >= 80) finishEpisode();
  renderAll();
  return details;
}

function finishEpisode() {
  returnHistory.push(episodeReturn);
  episodeCount++;
  agentState = START; episodeReturn = 0; episodeSteps = 0; path = [START];
}

function fastStep() {
  if (guided) { guided = null; document.getElementById("guided-card").hidden = true; }
  if (TERMINALS.has(agentState)) finishEpisode();
  const s = agentState;
  const a = chooseAction(s);
  const t = sampleTransition(s, a);
  completeTransition(s, a, t);
}

function runEpisode(render = true) {
  const startingEpisode = episodeCount;
  let guard = 0;
  while (episodeCount === startingEpisode && guard++ < 100) fastStep();
  if (render) renderAll();
}

function runEpisodes(count) {
  if (guided) { guided = null; document.getElementById("guided-card").hidden = true; }
  const oldRender = renderAll;
  for (let i = 0; i < count; i++) {
    const startingEpisode = episodeCount;
    let guard = 0;
    while (episodeCount === startingEpisode && guard++ < 100) {
      const s = agentState, a = chooseAction(s), t = sampleTransition(s, a);
      const details = qUpdate(s, a, t);
      void details;
      agentState = t.next; episodeReturn += t.reward; episodeSteps++; visits[agentState]++; path.push(agentState);
      if (TERMINALS.has(agentState) || episodeSteps >= 80) finishEpisode();
    }
  }
  renderAll();
}

function evaluateGridPolicy(randomPolicy, episodes = 100) {
  let totalReturn = 0, goals = 0;
  for (let episode = 0; episode < episodes; episode++) {
    let state = START, episodeReturn = 0;
    for (let step = 0; step < 80 && !TERMINALS.has(state); step++) {
      const action = randomPolicy ? Math.floor(rng() * ACTIONS.length) : greedyActions(qLearned[state])[0];
      const result = sampleTransition(state, action);
      episodeReturn += result.reward;
      state = result.next;
    }
    totalReturn += episodeReturn;
    if (state === GOAL) goals++;
  }
  return { meanReturn: totalReturn / episodes, goalRate: goals / episodes };
}

function cellBase(s) {
  const classes = ["cell"];
  if (WALLS.has(s)) classes.push("wall");
  if (s === GOAL) classes.push("goal");
  if (s === LOSS) classes.push("loss");
  if (s === START) classes.push("start");
  if (s === selectedState) classes.push("selected");
  return classes;
}

function makeCell(s, extra = []) {
  const el = document.createElement("button");
  el.type = "button";
  el.className = [...cellBase(s), ...extra].join(" ");
  el.setAttribute("aria-label", WALLS.has(s) ? `Obstacle at cell ${s}` : `Select state ${stateName(s)}`);
  el.disabled = WALLS.has(s);
  el.addEventListener("click", () => { selectedState = s; renderAll(); });
  return el;
}

function renderEnvironment() {
  const grid = document.getElementById("environment-grid"); grid.innerHTML = "";
  for (let s = 0; s < N; s++) {
    const cell = makeCell(s);
    if (!WALLS.has(s)) cell.innerHTML = `<span class="cell-label">${s === GOAL ? "GOAL" : s === LOSS ? "LOSS" : stateName(s)}</span>`;
    grid.appendChild(cell);
  }
}

function currentValues() {
  return qLearned.map(row => Math.max(...row));
}

function currentQ() { return qLearned; }

function valueColor(v) {
  const clamped = Math.max(-1, Math.min(1, v));
  if (clamped >= 0) return `rgba(100,230,189,${.10 + clamped * .42})`;
  return `rgba(241,111,97,${.10 + Math.abs(clamped) * .38})`;
}

function renderBoards() {
  const policyGrid = document.getElementById("policy-grid");
  const valueGrid = document.getElementById("value-grid");
  const qGrid = document.getElementById("q-grid");
  const agentGrid = document.getElementById("agent-grid");
  [policyGrid, valueGrid, qGrid, agentGrid].forEach(g => g.innerHTML = "");
  const vals = currentValues(), qs = currentQ();
  for (let s = 0; s < N; s++) {
    const pc = makeCell(s), vc = makeCell(s), qc = makeCell(s);
    const ac = makeCell(s, path.includes(s) ? ["path"] : []);
    if (!WALLS.has(s) && !TERMINALS.has(s)) {
      const best = greedyActions(qLearned[s])[0];
      pc.innerHTML = `<span class="policy-arrow single">${ACTIONS[best].arrow}</span>`;
      vc.innerHTML = `<span class="value-number">${vals[s].toFixed(2)}</span>`;
      vc.style.background = valueColor(vals[s]);
      const qbest = greedyActions(qs[s]);
      qc.innerHTML = ACTIONS.map((a,i) => `<span class="q-tri ${a.key} ${qbest.includes(i) ? "best" : ""}">${qs[s][i].toFixed(2)}</span>`).join("");
    } else if (TERMINALS.has(s)) {
      const label = s === GOAL ? "+1" : "−1";
      [pc,vc,qc].forEach(c => c.innerHTML = `<strong>${label}</strong>`);
    }
    if (s === agentState) { ac.classList.add("current"); ac.innerHTML = `<span class="agent" aria-label="Agent"></span>`; }
    if (s === GOAL && s !== agentState) ac.innerHTML = "+1";
    if (s === LOSS && s !== agentState) ac.innerHTML = "−1";
    policyGrid.appendChild(pc); valueGrid.appendChild(vc); qGrid.appendChild(qc); agentGrid.appendChild(ac);
  }
}

function renderVisits() {
  const grid = document.getElementById("visit-grid"); grid.innerHTML = "";
  const max = Math.max(1, ...visits);
  for (let s = 0; s < N; s++) {
    const cell = makeCell(s);
    if (!WALLS.has(s)) {
      const intensity = visits[s] / max;
      cell.style.background = `rgba(18,103,79,${.06 + intensity * .75})`;
      cell.textContent = visits[s];
    }
    grid.appendChild(cell);
  }
}

function renderChart() {
  const svg = document.getElementById("return-chart");
  const grid = svg.querySelector(".chart-grid");
  grid.innerHTML = [40,85,130,175,220].map(y => `<line x1="35" x2="705" y1="${y}" y2="${y}"></line>`).join("");
  const empty = svg.querySelector(".empty-chart");
  if (!returnHistory.length) { empty.style.display = "block"; svg.querySelector(".chart-line").setAttribute("points", ""); return; }
  empty.style.display = "none";
  const data = returnHistory.slice(-150), min = Math.min(-1.5, ...data), max = Math.max(1, ...data);
  const points = data.map((v,i) => {
    const x = 35 + (data.length === 1 ? 0 : i / (data.length - 1) * 670);
    const y = 220 - ((v - min) / (max - min || 1)) * 180;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  svg.querySelector(".chart-line").setAttribute("points", points);
}

function rmse() {
  let sum = 0, count = 0;
  for (let s = 0; s < N; s++) if (!WALLS.has(s) && !TERMINALS.has(s)) for (let a = 0; a < 4; a++) { sum += (qLearned[s][a] - qExact[s][a]) ** 2; count++; }
  return Math.sqrt(sum / count);
}

function renderMetrics() {
  document.getElementById("metric-episodes").textContent = episodeCount;
  document.getElementById("evaluate-grid").disabled = episodeCount === 0;
  const last = returnHistory.slice(-10);
  document.getElementById("metric-return").textContent = last.length ? (last.reduce((a,b)=>a+b,0)/last.length).toFixed(2) : "—";
  document.getElementById("metric-rmse").textContent = episodeCount ? rmse().toFixed(3) : "—";
  const covered = pairVisits.flat().filter(x => x > 0).length;
  const validPairs = (N - WALLS.size - TERMINALS.size) * 4;
  document.getElementById("metric-coverage").textContent = `${Math.round(covered / validPairs * 100)}%`;
}

function updateReadout(s, a, transition) {
  document.getElementById("read-state").textContent = stateName(s);
  document.getElementById("read-action").textContent = `${ACTIONS[a].arrow} ${ACTIONS[a].label}`;
  document.getElementById("read-reward").textContent = transition.reward.toFixed(2);
  document.getElementById("read-next").textContent = stateName(transition.next);
  document.getElementById("read-return").textContent = episodeReturn.toFixed(2);
}

function renderAll() { evaluateRandomPolicy(); computeExactQ(); renderEnvironment(); renderBoards(); renderVisits(); renderChart(); renderMetrics(); }

const formulas = {
  mdp: {
    title: "Markov decision process (MDP)", formula: "M = (S, A, p, γ)",
    definition: "An MDP is a mathematical model of sequential decisions. The Markov assumption says that the declared state contains the information needed to model the next outcome.",
    symbols: [["S","set of states"],["A","set of actions"],["p","transition-and-reward model"],["γ","discount factor"]],
    example: "The current grid cell is the state. A direction is the action. The environment supplies the next cell and reward.",
    finance: "The state must contain financially relevant information available at decision time—for example market features, current holdings, time, and constraints."
  },
  transition: {
    title: "Transition-and-reward model", formula: "p(s′, r | s, a) = P(Sₜ₊₁=s′, Rₜ₊₁=r | Sₜ=s, Aₜ=a)",
    definition: "This object describes the environment. It assigns a probability to each possible next state and immediate reward after a particular action in a particular state.",
    symbols: [["s","current state"],["a","chosen action"],["s′","possible next state"],["r","possible immediate reward"],["p","conditional probability supplied by the model"]],
    example: "With deterministic movement, moving right may reach one next cell with probability 1. With 20% movement uncertainty, it reaches the intended cell with probability 0.8 and may slip sideways with probability 0.1 each.",
    finance: "A market model would describe how market conditions and holdings may evolve after an allocation decision. Model-free RL does not require this probability model to be supplied explicitly."
  },
  policy: {
    title: "Policy", formula: "π(a | s) = P(Aₜ=a | Sₜ=s)",
    definition: "A policy is the agent's decision rule. It assigns an action probability to every state.",
    symbols: [["π","policy"],["s","observed state"],["a","possible action"],["π(a | s)","probability of choosing a in s"]],
    example: "The initial random policy assigns probability 0.25 to each direction. In learning mode, the highlighted arrow is currently greedy while ε preserves some exploration.",
    finance: "A policy maps the observed market and portfolio state to an allocation or trade. It may be deterministic or randomized."
  },
  value: {
    title: "State-value function", formula: "Vᵖⁱ(s) = Eᵖⁱ[Gₜ | Sₜ=s]",
    definition: "The state value is the expected discounted return from a state when the agent continues following policy π.",
    symbols: [["Vᵖⁱ(s)","value of state s under π"],["Gₜ","discounted future return"],["Eᵖⁱ","expectation generated by policy π"],["Sₜ=s","evaluation begins in state s"]],
    example: "A cell near the positive terminal usually has a higher value, but its exact value also depends on the policy and the risk of reaching the loss state.",
    finance: "The value can represent expected long-run utility or cumulative net reward from the current market and portfolio state under a fixed investment policy."
  },
  qvalue: {
    title: "Action-value function", formula: "Qᵖⁱ(s,a) = Eᵖⁱ[Gₜ | Sₜ=s, Aₜ=a]",
    definition: "The action value evaluates one specified first action in a state, followed by policy π afterwards.",
    symbols: [["Qᵖⁱ(s,a)","value of taking a in s"],["a","specified first action"],["π","policy followed afterwards"],["Gₜ","discounted future return"]],
    example: "The four directional numbers in a cell answer four counterfactual questions: what if the first move is up, right, down, or left?",
    finance: "Q(s,a) compares the long-run value of possible allocation decisions from the same current market and holdings state."
  },
  return: {
    title: "Discounted return", formula: "Gₜ = Rₜ₊₁ + γRₜ₊₂ + γ²Rₜ₊₃ + ⋯",
    definition: "Return is the cumulative future reward used to evaluate a trajectory. Discounting controls the relative influence of later rewards.",
    symbols: [["Gₜ","return from time t"],["Rₜ₊₁","next reward"],["γ","discount factor between 0 and 1"],["γᵏ","weight on a reward k steps ahead"]],
    example: "With γ=0.9, rewards one and two steps ahead receive weights 0.9 and 0.81. The agent still values them, but less than immediate feedback.",
    finance: "Rewards could be net period returns or utility after transaction costs. The objective must be chosen carefully; raw return alone may ignore risk and constraints."
  },
  bellman: {
    title: "Bellman optimality backup", formula: "Qₖ₊₁(s,a) = Σₛ′,ᵣ p(s′,r | s,a) [r + γ maxₐ′ Qₖ(s′,a′)]",
    definition: "When the transition-and-reward model is known, the backup evaluates every possible next outcome. It combines immediate reward with the best continuation value and averages using model probabilities.",
    symbols: [["Qₖ(s,a)","current action-value estimate"],["p(s′,r | s,a)","transition-and-reward model"],["Σₛ′,ᵣ","sum over possible next outcomes"],["maxₐ′ Qₖ(s′,a′)","best continuation value"],["γ","discount factor"]],
    example: "Select a grid state, introduce movement uncertainty, and the planning backup branches into the intended destination and the possible sideways slips.",
    finance: "This is dynamic programming: it is natural when a trustworthy state-transition model is available and the state and action spaces remain computationally manageable."
  },
  qlearning: {
    title: "Q-learning update", formula: "Q(s,a) ← Q(s,a) + α [r + γ maxₐ′Q(s′,a′) − Q(s,a)]",
    definition: "When the environment model is not supplied, Q-learning uses one experienced transition. The bracketed difference is the temporal-difference error.",
    symbols: [["α","learning rate"],["r + γ max Q(s′,·)","TD target"],["Q(s,a)","current prediction"],["δ","TD error: target minus prediction"]],
    example: "If the old Q-value is 0.20 and the TD target is 0.50, the error is 0.30. With α=0.3, the new estimate becomes 0.29.",
    finance: "The agent can learn from historical or simulated transitions without being given an explicit market-transition probability model. The quality of its experience and evaluation remains critical."
  }
};

function openFormula(key) {
  const f = formulas[key]; if (!f) return;
  document.getElementById("dialog-title").textContent = f.title;
  document.getElementById("dialog-formula").textContent = f.formula;
  document.getElementById("dialog-definition").textContent = f.definition;
  document.getElementById("dialog-symbols").innerHTML = f.symbols.map(([s,d]) => `<div><strong>${s}</strong> — ${d}</div>`).join("");
  document.getElementById("dialog-example").textContent = f.example;
  document.getElementById("dialog-finance").textContent = f.finance;
  miniMoved = false;
  renderFormulaMini();
  document.getElementById("formula-dialog").showModal();
}

function renderFormulaMini() {
  const grid = document.getElementById("formula-mini-grid");
  grid.innerHTML = "";
  for (let i = 0; i < 9; i++) {
    const cell = document.createElement("div");
    cell.className = `mini-cell ${i === 4 ? "origin" : ""} ${i === 5 ? "destination" : ""}`;
    if (i === (miniMoved ? 5 : 4)) cell.innerHTML = '<span class="mini-agent"></span>';
    grid.appendChild(cell);
  }
  document.getElementById("mini-caption").textContent = miniMoved ? "Observed next state s′ and reward r" : "Current state s; choose action a = right";
  document.getElementById("example-step").textContent = miniMoved ? "Reset example" : "Move one step";
}

function setMethod(method) {
  document.querySelectorAll("[data-method]").forEach(b => b.classList.toggle("active", b.dataset.method === method));
  const planning = method === "planning";
  document.getElementById("method-eyebrow").textContent = planning ? "Planning from a known model" : "Learning from sampled experience";
  document.getElementById("method-title").textContent = planning ? "Average over every possible next outcome." : "Update from the one transition that actually occurred.";
  document.getElementById("method-copy").innerHTML = planning
    ? "At this point, the transition-and-reward model <strong>p(s′, r | s, a)</strong> is available. A Bellman backup can therefore combine all possible next states and rewards, weighted by their probabilities."
    : "At this point, the transition-and-reward model is <strong>not supplied to the agent</strong>. Q-learning replaces the probability-weighted expectation with a sampled transition <strong>(s, a, r, s′)</strong>.";
  document.getElementById("method-formula").innerHTML = planning
    ? "Q<sub>k+1</sub>(s,a) = Σ<sub>s′,r</sub> p(s′,r | s,a) [r + γ max<sub>a′</sub> Q<sub>k</sub>(s′,a′)]"
    : "Q(s,a) ← Q(s,a) + α [r + γ max<sub>a′</sub>Q(s′,a′) − Q(s,a)]";
  document.getElementById("method-caption").textContent = planning ? "A model-based Bellman optimality backup." : "A model-free temporal-difference update from one experience.";
}

function beginGuidedStep() {
  const s = agentState, a = chooseAction(s), transition = sampleTransition(s, a);
  const old = qLearned[s][a], nextBest = TERMINALS.has(transition.next) ? 0 : Math.max(...qLearned[transition.next]);
  const target = transition.reward + settings.gamma * nextBest, td = target - old, updated = old + settings.alpha * td;
  guided = { index: 0, s, a, transition, old, nextBest, target, td, updated };
  document.getElementById("guided-card").hidden = false;
  renderGuided();
  document.getElementById("guided-card").scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderGuided() {
  const g = guided; if (!g) return;
  const steps = [
    ["1 · Observe", `The agent is in ${stateName(g.s)}.`, "The current state is the information used for the next decision.", `Sₜ = ${stateName(g.s)}`],
    ["2 · Select", `The behaviour policy chooses ${ACTIONS[g.a].label.toLowerCase()}.`, "With ε-greedy behaviour, the agent usually chooses a currently best action and sometimes explores.", `Aₜ = ${ACTIONS[g.a].label}`],
    ["3 · Experience", `The environment moves to ${stateName(g.transition.next)}.`, "This is one sampled outcome—not an average over every possible outcome.", `(${stateName(g.s)}, ${ACTIONS[g.a].label}, ${g.transition.reward.toFixed(2)}, ${stateName(g.transition.next)})`],
    ["4 · Continue", `Find the best next-state estimate.`, "Q-learning uses the largest current action value in the observed next state.", `maxₐ Q(${stateName(g.transition.next)},a) = ${g.nextBest.toFixed(3)}`],
    ["5 · Target", `Construct the temporal-difference target.`, "The target combines the realized immediate reward with the discounted estimate of the best continuation.", `${g.transition.reward.toFixed(2)} + ${settings.gamma.toFixed(2)} × ${g.nextBest.toFixed(3)} = ${g.target.toFixed(3)}`],
    ["6 · Error", `Calculate the temporal-difference error.`, "The TD error measures the discrepancy between the new target and the old prediction.", `δ = ${g.target.toFixed(3)} − ${g.old.toFixed(3)} = ${g.td.toFixed(3)}`],
    ["7 · Update", `Move the old estimate toward the target.`, "The learning rate determines how much of the error is incorporated in this update.", `${g.old.toFixed(3)} + ${settings.alpha.toFixed(2)} × ${g.td.toFixed(3)} = ${g.updated.toFixed(3)}`],
    ["8 · Refresh", `The action value and policy can now change.`, "The synchronized boards refresh after the update. Repeated experience gradually shapes the learned value surface.", `Q(${stateName(g.s)},${ACTIONS[g.a].arrow}) ← ${g.updated.toFixed(3)}`]
  ];
  const [label,title,copy,calc] = steps[g.index];
  document.getElementById("guided-index").textContent = `${g.applied ? steps.length : g.index + 1} / ${steps.length}`;
  document.getElementById("guided-bar").style.width = `${(g.applied ? 1 : (g.index + 1) / steps.length) * 100}%`;
  document.getElementById("guided-label").textContent = label;
  document.getElementById("guided-title").textContent = g.applied ? "Update applied to the Q-table." : title;
  document.getElementById("guided-copy").textContent = g.applied ? "The boards above now reflect the completed update." : copy;
  document.getElementById("guided-calculation").textContent = calc;
  const history = document.getElementById("guided-history");
  const list = document.getElementById("guided-history-list");
  list.replaceChildren();
  steps.slice(0, g.applied ? steps.length : g.index).forEach(([stepLabel,, , result]) => {
    const item = document.createElement("li");
    const name = document.createElement("span"); name.textContent = stepLabel;
    const value = document.createElement("strong"); value.textContent = result;
    item.append(name, value); list.append(item);
  });
  history.hidden = !list.childElementCount;
  document.getElementById("next-guided").hidden = !!g.applied;
  document.getElementById("next-guided").textContent = g.index === steps.length - 1 ? "Apply update" : "Next calculation step";
}

function advanceGuided() {
  if (!guided || guided.applied) return;
  if (guided.index < 7) { guided.index++; renderGuided(); return; }
  completeTransition(guided.s, guided.a, guided.transition, true);
  guided.applied = true; renderGuided();
}

function resetLearning() {
  guided = null; document.getElementById("guided-card").hidden = true;
  clearEvaluation();
  visits.fill(0); pairVisits.forEach(row => row.fill(0));
  agentState = START; episodeReturn = 0; episodeSteps = 0; episodeCount = 0; returnHistory = []; path = [START]; randomState = 934775;
  initializeQLearned();
  updateReadout(START, 0, { reward: 0, next: START }); renderAll();
}

let activeView = 0;
function setView(index) {
  activeView = Math.max(0, Math.min(3, Number(index)));
  document.querySelectorAll("[data-view-panel]").forEach(panel => panel.classList.toggle("active", Number(panel.dataset.viewPanel) === activeView));
  document.querySelectorAll(".basics-nav [data-view]").forEach(button => button.classList.toggle("active", Number(button.dataset.view) === activeView));
  document.getElementById("view-position").textContent = `${activeView + 1} / 4`;
  document.getElementById("previous-view").disabled = activeView === 0;
  document.getElementById("next-view").disabled = activeView === 3;
  window.scrollTo({ top: 0, behavior: "instant" });
}

document.addEventListener("click", event => {
  const viewButton = event.target.closest("[data-view]"); if (viewButton) setView(viewButton.dataset.view);
  const methodButton = event.target.closest("[data-method]"); if (methodButton) setMethod(methodButton.dataset.method);
});
document.getElementById("fast-step").addEventListener("click", fastStep);
document.getElementById("guided-step").addEventListener("click", beginGuidedStep);
document.getElementById("next-guided").addEventListener("click", advanceGuided);
document.getElementById("close-guided").addEventListener("click", () => { guided = null; document.getElementById("guided-card").hidden = true; });
document.getElementById("run-episode").addEventListener("click", () => runEpisode());
document.getElementById("reset-agent").addEventListener("click", resetLearning);
document.getElementById("train-10").addEventListener("click", () => runEpisodes(10));
document.getElementById("train-100").addEventListener("click", () => runEpisodes(100));
document.getElementById("evaluate-grid").addEventListener("click", () => {
  const trainingRandomState = randomState;
  const learned = evaluateGridPolicy(false);
  const random = evaluateGridPolicy(true);
  randomState = trainingRandomState;
  document.getElementById("eval-greedy-return").textContent = learned.meanReturn.toFixed(2);
  document.getElementById("eval-random-return").textContent = random.meanReturn.toFixed(2);
  document.getElementById("eval-goal-rate").textContent = `${Math.round(learned.goalRate * 100)}%`;
  document.getElementById("evaluation-status").textContent = `Evaluated the frozen Q-table after ${episodeCount} training episodes. No Q-values were updated.`;
  evaluationCurrent = true;
});
document.getElementById("previous-view").addEventListener("click", () => setView(activeView - 1));
document.getElementById("next-view").addEventListener("click", () => setView(activeView + 1));
document.getElementById("defaults").addEventListener("click", () => {
  [["alpha",.3],["gamma",.92],["epsilon",.2],["slip",0]].forEach(([id,v]) => { document.getElementById(id).value = v; document.getElementById(id).dispatchEvent(new Event("input")); }); resetLearning();
});
[["alpha","alpha-output",v => Number(v).toFixed(2)], ["gamma","gamma-output",v => Number(v).toFixed(2)], ["epsilon","epsilon-output",v => Number(v).toFixed(2)], ["slip","slip-output",v => `${Math.round(Number(v)*100)}%`]].forEach(([id,out,format]) => {
  document.getElementById(id).addEventListener("input", e => { clearEvaluation(); settings[id] = Number(e.target.value); document.getElementById(out).textContent = format(e.target.value); evaluateRandomPolicy(); computeExactQ(); renderAll(); });
});

initializeQLearned(); evaluateRandomPolicy(); computeExactQ(); renderAll(); updateReadout(START, 0, { reward: 0, next: START }); setView(0);
