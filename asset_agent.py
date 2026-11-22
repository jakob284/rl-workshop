"""Single source of truth for the RL Workshop portfolio case study.

Runs in CPython (notebook / local server) and Pyodide (static GitHub Pages).
No third-party packages; historical price observations must be passed explicitly.

This is a didactic tabular Q-learning implementation, not an investment model.
A training pass replays the SAME historical weeks, not independent experiences.
"""

from __future__ import annotations
import json
from math import sqrt

WEIGHTS = (0.0, 0.5, 1.0)
STRATEGY_NAMES = ("Q-learning", "100% Apple", "Weekly 50% target", "Cash", "Two-horizon rule")
DEFAULTS = {
    "alpha": 0.15,
    "gamma": 0.90,
    "epsilon": 0.25,
    "cost_bps": 10.0,
    "risk": 0.5,
    "budget": 500,
    "seed": 17,
}


class Mulberry32:
    """Deterministic seeded generator (matches notebook and original JS)."""
    def __init__(self, seed):
        self.state = int(seed) & 0xffffffff

    def random(self):
        self.state = (self.state + 0x6D2B79F5) & 0xffffffff
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xffffffff
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xffffffff)) & 0xffffffff)) & 0xffffffff
        t = (t ^ (t >> 14)) & 0xffffffff
        return t / 4294967296


def greedy(row, rng=None):
    best = max(row)
    ties = [idx for idx, value in enumerate(row) if abs(value - best) < 1e-12]
    if rng is None:
        return ties[0]
    return ties[int(rng.random() * len(ties))]


class PortfolioAgent:
    def __init__(self, prices, dates=None, **params):
        self.prices = [round(float(x), 8) for x in prices]
        if len(self.prices) < 20 or any(x <= 0 for x in self.prices):
            raise ValueError("Expected at least 20 strictly positive weekly price observations")
        self.dates = list(dates) if dates is not None else [str(i) for i in range(len(prices))]
        if len(self.dates) != len(self.prices):
            raise ValueError("Dates must align with price observations")
        self.decisions = list(range(12, len(self.prices) - 1))
        split = int(len(self.decisions) * .70)
        self.train_t = self.decisions[:split]
        self.test_t = self.decisions[split:]
        self.stock_returns = [self.prices[i+1] / self.prices[i] - 1 for i in range(len(self.prices)-1)]
        self.up4 = [i >= 4 and self.prices[i] / self.prices[i-4] > 1 for i in range(len(self.prices))]
        self.up12 = [i >= 12 and self.prices[i] / self.prices[i-12] > 1 for i in range(len(self.prices))]
        self.configure(**params)

    def configure(self, **params):
        p = dict(DEFAULTS)
        p.update(params)
        self.alpha = float(p["alpha"])
        self.gamma = float(p["gamma"])
        self.initial_epsilon = float(p["epsilon"])
        self.cost_bps = float(p["cost_bps"])
        self.risk = float(p["risk"])
        self.budget = int(p["budget"])
        self.seed = int(p["seed"])
        if not (0 < self.alpha <= 1 and 0 <= self.gamma < 1 and 0 <= self.initial_epsilon <= 1
                and self.cost_bps >= 0 and self.risk >= 0 and 1 <= self.budget <= 5000):
            raise ValueError("Invalid training parameters")
        self.reset()

    def reset(self):
        self.q = [[0.0, 0.0, 0.0] for _ in range(12)]
        self.visits = [0 for _ in range(12)]
        self.rng = Mulberry32(self.seed)
        self.passes = 0
        self.position = 0
        self.prev = 0
        self.current_reward = 0.0
        self.history = []
        self.greedy_history = []
        self.last = None

    def epsilon(self):
        fraction = min(self.budget - 1, self.passes) / max(1, self.budget - 1)
        return self.initial_epsilon + (min(0.03, self.initial_epsilon) - self.initial_epsilon) * fraction

    def state_id(self, t, previous):
        return (2 * int(self.up4[t]) + int(self.up12[t])) * 3 + previous

    def reward_for(self, action, previous, t):
        gross = WEIGHTS[action] * self.stock_returns[t]
        turnover = abs(WEIGHTS[action] - WEIGHTS[previous])
        trade_cost = (self.cost_bps / 10000) * turnover
        net = gross - trade_cost
        reward = net - self.risk * max(-gross, 0.0)
        return dict(gross=gross, turnover=turnover, tradeCost=trade_cost, net=net, reward=reward)

    def score_greedy_training(self):
        prev, total = 0, 0.0
        for t in self.train_t:
            action = greedy(self.q[self.state_id(t, prev)])
            total += self.reward_for(action, prev, t)["reward"]
            prev = action
        return total

    def step(self):
        if self.passes >= self.budget:
            return False
        t = self.train_t[self.position]
        previous = self.prev
        state = self.state_id(t, previous)
        eps = self.epsilon()
        if self.rng.random() < eps:
            action = int(self.rng.random() * 3)
        else:
            action = greedy(self.q[state], self.rng)
        measures = self.reward_for(action, previous, t)
        terminal = self.position == len(self.train_t) - 1
        next_state = None if terminal else self.state_id(t+1, action)
        old_q = self.q[state][action]
        next_best = 0.0 if terminal else max(self.q[next_state])
        target = measures["reward"] + (0.0 if terminal else self.gamma * next_best)
        td = target - old_q
        self.q[state][action] += self.alpha * td
        self.visits[state] += 1
        self.current_reward += measures["reward"]
        self.last = dict(t=t, s=state, a=action, oldPrev=previous, oldQ=old_q, nextBest=next_best,
                         target=target, td=td, updated=self.q[state][action], eps=eps,
                         sNext=next_state, last=terminal, **measures)
        self.prev = action
        self.position += 1
        if terminal:
            self.history.append(self.current_reward)
            self.greedy_history.append(self.score_greedy_training())
            self.passes += 1
            self.position = 0
            self.prev = 0
            self.current_reward = 0.0
        return True

    def step_many(self, count):
        for _ in range(max(0, min(100000, int(count)))):
            if not self.step():
                break
        return self.snapshot()

    def snapshot(self):
        return dict(q=self.q, visits=self.visits, passes=self.passes, position=self.position,
                    prev=self.prev, currentReward=self.current_reward,
                    history=self.history, greedyHistory=self.greedy_history, last=self.last,
                    budget=self.budget, trainWeeks=len(self.train_t), testWeeks=len(self.test_t))

    def evaluate_strategy(self, name, policy):
        prev = 0
        wealth = 1.0
        high = 1.0
        max_dd = 0.0
        turnover = 0.0
        changes = 0
        wealth_path = [wealth]
        outcomes = []
        weeks = []
        for t in self.test_t:
            action = int(policy(t, prev))
            weight = WEIGHTS[action]
            change = abs(weight - WEIGHTS[prev])
            trade_cost = self.cost_bps / 10000 * change
            gross = weight * self.stock_returns[t]
            net = gross - trade_cost
            turnover += change
            if change > 1e-10:
                changes += 1
            wealth *= 1 + net
            high = max(high, wealth)
            max_dd = min(max_dd, wealth/high - 1)
            wealth_path.append(wealth)
            weeks.append(self.dates[t+1])
            outcomes.append(dict(t=t, prev=prev, a=action, weight=weight, gross=gross, net=net,
                                 tradeCost=trade_cost, change=change, nextReturn=self.stock_returns[t]))
            prev = action
        return dict(name=name, wealth=wealth_path, totalReturn=wealth - 1,
                    maxDrawdown=max_dd, turnover=turnover, changes=changes,
                    weeks=weeks, outcome=outcomes)

    def evaluate(self):
        if not self.passes:
            raise ValueError("Complete at least one training pass before evaluating")
        return [
            self.evaluate_strategy("Q-learning", lambda t, prev: greedy(self.q[self.state_id(t, prev)])),
            self.evaluate_strategy("100% Apple", lambda t, prev: 2),
            self.evaluate_strategy("Weekly 50% target", lambda t, prev: 1),
            self.evaluate_strategy("Cash", lambda t, prev: 0),
            self.evaluate_strategy("Two-horizon rule", lambda t, prev: 2 if self.up4[t] and self.up12[t] else 0),
        ]


# API for Pyodide. Data and state live in Python, not a duplicate JavaScript engine.
_ENGINE = None

def run_command(command_json):
    global _ENGINE
    cmd = json.loads(command_json)
    action = cmd["action"]
    if action == "init":
        _ENGINE = PortfolioAgent(cmd["prices"], cmd.get("dates"), **cmd["params"])
        return json.dumps(_ENGINE.snapshot(), allow_nan=False)
    if _ENGINE is None:
        raise RuntimeError("Initialize the Python agent first")
    if action == "reset":
        _ENGINE.configure(**cmd["params"])
        return json.dumps(_ENGINE.snapshot(), allow_nan=False)
    if action == "step":
        return json.dumps(_ENGINE.step_many(cmd.get("count", 1)), allow_nan=False)
    if action == "evaluate":
        return json.dumps(_ENGINE.evaluate(), allow_nan=False)
    raise ValueError("Unknown command")

class MicrosoftChallengeAgent(PortfolioAgent):
    """Longer-horizon MSFT workshop task with a train / validation / test firewall.

    Reuses exactly the original Bellman / Q-learning implementation.
    Dates / asset data are separate from the Apple teaching example.
    Participants may tune only alpha, gamma, epsilon, and training budget.
    Transaction costs, reward-risk preference and seed are fixed by the task.
    """
    CHALLENGE_COST_BPS = 10.0
    CHALLENGE_RISK = 0.5
    CHALLENGE_SEED = 17

    def __init__(self, prices, dates=None, **params):
        permitted = {'alpha', 'gamma', 'epsilon', 'budget'}
        if unknown := set(params) - permitted:
            raise ValueError(f"Challenge parameters are fixed except {permitted}; got {unknown}")
        super().__init__(prices, dates,
                         cost_bps=self.CHALLENGE_COST_BPS, risk=self.CHALLENGE_RISK,
                         seed=self.CHALLENGE_SEED, **params)
        self.train_t = [t for t in self.decisions if self.dates[t+1][:4] <= '2019']
        self.validation_t = [t for t in self.decisions if '2020' <= self.dates[t+1][:4] <= '2022']
        self.test_t = [t for t in self.decisions if self.dates[t+1][:4] >= '2023']
        if not min(map(len, (self.train_t, self.validation_t, self.test_t))) > 30:
            raise ValueError('Microsoft challenge needs substantial nonempty train, validation and test samples')
        self.reset()

    def reset(self):
        super().reset()
        # One Q-table per completed training pass. This is the ACTUAL learned table,
        # not a mock visualization or an interpolated trajectory.
        self.q_history = []
        self.policy_history = []

    def step(self):
        completed_before = self.passes
        moved = super().step()
        if moved and self.passes > completed_before:
            self.q_history.append([row[:] for row in self.q])
            self.policy_history.append([greedy(row) for row in self.q])
        return moved

    def validation_snapshot(self):
        return {'trainWeeks':len(self.train_t), 'validationWeeks':len(self.validation_t),
                'finalWeeks':len(self.test_t), 'firstTrain':self.dates[self.train_t[0]+1],
                'lastTrain':self.dates[self.train_t[-1]+1],
                'firstValidation':self.dates[self.validation_t[0]+1],
                'lastValidation':self.dates[self.validation_t[-1]+1],
                'firstTest':self.dates[self.test_t[0]+1],
                'lastTest':self.dates[self.test_t[-1]+1]}

    def evaluate_segment(self, segment):
        if not self.passes:
            raise ValueError('Complete at least one full training pass before evaluation')
        if segment not in ('validation', 'final'):
            raise ValueError('Allowed evaluation segments: validation and final')
        indices = self.validation_t if segment == 'validation' else self.test_t
        policies = (
            ('RL · learned policy', lambda t, prev: greedy(self.q[self.state_id(t, prev)])),
            ('100% Microsoft', lambda t, prev: 2),
            ('Weekly 50% MSFT', lambda t, prev: 1),
            ('Cash', lambda t, prev: 0),
            ('Two-momentum rule', lambda t, prev: 2 if self.up4[t] and self.up12[t] else 0),
        )
        output = []
        for name, choose in policies:
            wealth, high, dd = 1.0, 1.0, 0.0
            turnover, changes, path = 0.0, 0, [1.0]
            returns, weights, prev = [], [], 0
            for t in indices:
                action = int(choose(t, prev))
                w = WEIGHTS[action]
                turn = abs(w-WEIGHTS[prev])
                net = w*self.stock_returns[t] - self.cost_bps/10000*turn
                wealth *= 1 + net
                high=max(high,wealth)
                dd = min(dd,wealth/high-1)
                turnover += turn
                changes += int(turn>1e-10)
                returns.append(net)
                weights.append(w)
                path.append(wealth)
                prev=action
            total=wealth-1.0
            report=risk_report([self.dates[t+1] for t in indices],returns,weights,changes)
            output.append(dict(name=name, totalReturn=total, maxDrawdown=dd,
                               turnover=turnover, changes=changes,
                               challengeScore=total+0.5*dd,
                               wealth=path, weeks=[self.dates[t+1] for t in indices],
                               weights=weights, weeklyNet=returns, **report))
        return output


def risk_report(weeks, returns, weights, changes):
    """Return and downside diagnostics from realized portfolio net returns.

    Volatility and Sharpe use 52 weeks/year and ZERO risk-free rate for the
    didactic comparison. None indicates zero volatility rather than infinity.
    Yearly drawdowns reset at the beginning of each calendar year.
    """
    def metrics(rr, ww):
        n=len(rr)
        if not n:
            return dict(totalReturn=0.0,maxDrawdown=0.0,annualVol=0.0,
                        sharpe=None,meanExposure=0.0,positiveWeeks=0)
        v=1.0; high=1.0; dd=0.0
        for r in rr:
            v*=1+r
            high=max(high,v); dd=min(dd,v/high-1.0)
        mu=sum(rr)/n
        std=sqrt(sum((r-mu)**2 for r in rr)/n)
        return dict(totalReturn=v-1,maxDrawdown=dd,annualVol=std*sqrt(52),
                    sharpe=(sqrt(52)*mu/std if std>1e-12 else None),
                    meanExposure=sum(ww)/n if ww else 0.0,
                    positiveWeeks=sum(r>0 for r in rr))
    years={}
    for d,r,w in zip(weeks,returns,weights):
        bucket=years.setdefault(d[:4],{'returns':[],'weights':[]})
        bucket['returns'].append(r);bucket['weights'].append(w)
    annual={year:metrics(g['returns'],g['weights']) for year,g in years.items()}
    whole=metrics(returns,weights)
    return dict(annualVol=whole['annualVol'],sharpe=whole['sharpe'],
                yearMetrics=annual,positiveYears=sum(v['totalReturn']>0 for v in annual.values()),
                worstYear=min((v['totalReturn'] for v in annual.values()),default=0.0))


# Experiment Lab — all learning still delegates to PortfolioAgent.step().
# Learning inputs can be changed freely. Market data, fees and reward preference
# remain fixed, so a change in performance is attributable to the agent policy.
_LAB = None
_LAB_RUNS = []
_LAB_FINAL_REVEALED = False
LAB_REFERENCE = {'alpha':0.15,'gamma':0.90,'epsilon':0.25,'budget':100}


def _lab_year_returns(strategy):
    grouped = {}
    for date, net in zip(strategy['weeks'], strategy['weeklyNet']):
        year = date[:4]
        grouped[year] = grouped.get(year, 1.0) * (1 + net)
    return {year: value - 1 for year, value in grouped.items()}


def _lab_summary(results):
    return [
        {k:r[k] for k in ('name','totalReturn','maxDrawdown','turnover','changes','challengeScore','annualVol','sharpe','yearMetrics','positiveYears','worstYear')}
        | {'years':_lab_year_returns(r),
           'meanEquity':sum(r['weights'])/len(r['weights']) if r['weights'] else 0.0}
        for r in results
    ]


def _lab_parameters(agent):
    return dict(alpha=agent.alpha, gamma=agent.gamma,
                epsilon=agent.initial_epsilon, budget=agent.budget)


def _lab_trained_copy(agent, seed):
    new_agent = MicrosoftChallengeAgent(agent.prices, agent.dates, **_lab_parameters(agent))
    new_agent.seed = int(seed)
    new_agent.reset()
    new_agent.step_many(len(new_agent.train_t) * new_agent.budget)
    return new_agent


def _lab_policy(q):
    return [greedy(row) for row in q]


def _lab_trace(agent, state):
    if not 0 <= state < 12:
        raise ValueError('State must be 0–11')
    return {'state':state,'q': [table[state][:] for table in agent.q_history],
            'policy':[item[state] for item in agent.policy_history],
            'visits':agent.visits[state]}


def lab_command(command_json):
    global _LAB, _LAB_RUNS, _LAB_FINAL_REVEALED
    cmd = json.loads(command_json)
    action = cmd['action']
    if action == 'init':
        _LAB_RUNS = []
        _LAB_FINAL_REVEALED = False
        _LAB = MicrosoftChallengeAgent(cmd['prices'], cmd['dates'], **cmd['params'])
        return json.dumps({'snapshot':_LAB.snapshot(), 'segments':_LAB.validation_snapshot()}, allow_nan=False)
    if _LAB is None:
        raise RuntimeError('Initialize the lab first')
    if action == 'reset':
        _LAB = MicrosoftChallengeAgent(_LAB.prices, _LAB.dates, **cmd['params'])
        return json.dumps({'snapshot':_LAB.snapshot(), 'segments':_LAB.validation_snapshot()}, allow_nan=False)
    if action == 'step':
        return json.dumps(_LAB.step_many(cmd.get('count',1)), allow_nan=False)
    if action == 'trace':
        return json.dumps(_lab_trace(_LAB, int(cmd.get('state',0))), allow_nan=False)
    if action == 'validation':
        if _LAB.passes < 1:
            raise ValueError('Complete one training pass before validation')
        results = _LAB.evaluate_segment('validation')
        record = {'id':len(_LAB_RUNS)+1, 'params':_lab_parameters(_LAB),
                  'seed':_LAB.seed, 'q':[row[:] for row in _LAB.q_history[-1]],
                  'qHistory':[[row[:] for row in table] for table in _LAB.q_history],
                  'history':list(_LAB.history), 'greedyHistory':list(_LAB.greedy_history),
                  'trainingPasses':_LAB.passes,
                  'results':_lab_summary(results)}
        _LAB_RUNS.append(record)
        return json.dumps({'record':{k:v for k,v in record.items() if k not in ('qHistory','q','history','greedyHistory')},
                           'policy':_lab_policy(record['q'])}, allow_nan=False)
    if action == 'runs':
        return json.dumps([{'id':r['id'], 'params':r['params'], 'results':r['results'],
                            'seed':r['seed'], 'trainingPasses':r['trainingPasses']}
                           for r in _LAB_RUNS], allow_nan=False)
    if action == 'compare':
        first,second=int(cmd['a']),int(cmd['b'])
        if not(1<=first<=len(_LAB_RUNS) and 1<=second<=len(_LAB_RUNS)):
            raise ValueError('Choose two saved experiments')
        a,b=_LAB_RUNS[first-1],_LAB_RUNS[second-1]
        pa,pb=_lab_policy(a['q']),_lab_policy(b['q'])
        state=int(cmd.get('state',0))
        if not 0<=state<12: raise ValueError('Unknown state')
        return json.dumps({'a':{k:v for k,v in a.items() if k not in ('qHistory','q','history','greedyHistory')},
                           'b':{k:v for k,v in b.items() if k not in ('qHistory','q','history','greedyHistory')},
                           'policyA':pa,'policyB':pb,'qA':a['q'],'qB':b['q'],
                           'differentStates':sum(x!=y for x,y in zip(pa,pb)),
                           'traceA':[x[state] for x in a['qHistory']],
                           'traceB':[x[state] for x in b['qHistory']]}, allow_nan=False)
    if action == 'seeds':
        index=int(cmd['runId'])
        if not 1<=index<=len(_LAB_RUNS): raise ValueError('Unknown saved experiment')
        rec=_LAB_RUNS[index-1]
        output=[]
        for seed in (17,29,43):
            agent=MicrosoftChallengeAgent(_LAB.prices,_LAB.dates,**rec['params'])
            agent.seed=seed
            agent.reset()
            agent.step_many(rec['trainingPasses']*len(agent.train_t))
            metrics=_lab_summary(agent.evaluate_segment('validation'))[0]
            output.append({'seed':seed,**metrics})
        return json.dumps({'runId':index,'checks':output},allow_nan=False)
    if action == 'final':
        if _LAB_FINAL_REVEALED:
            raise ValueError('Final test already revealed in this page session')
        index=int(cmd['runId'])
        if not 1<=index<=len(_LAB_RUNS): raise ValueError('Choose a saved experiment first')
        rec=_LAB_RUNS[index-1]
        agent=MicrosoftChallengeAgent(_LAB.prices,_LAB.dates,**rec['params'])
        agent.q=[row[:] for row in rec['q']]
        agent.passes=1
        out=_lab_summary(agent.evaluate_segment('final'))
        _LAB_FINAL_REVEALED=True
        return json.dumps({'runId':index,'results':out},allow_nan=False)
    if action == 'segments':
        return json.dumps(_LAB.validation_snapshot())
    raise ValueError(f'Unknown lab action: {action}')


# ── Optional extension: two risky assets (SPY / TLT) and cash ────────────
# Distinct real-data teaching environment. SPY/TLT source provides historical
# closing prices WITHOUT verified dividend adjustments; metrics are price-only.
MULTI_ACTIONS=((0.0,0.0),(1.0,0.0),(0.0,1.0),(.5,.5),(.5,0.0),(0.0,.5))
MULTI_ACTION_NAMES=('100% cash','100% SPY','100% TLT','50/50 SPY–TLT',
                    '50% SPY / cash','50% TLT / cash')

class MultiAssetAgent:
    def __init__(self, dates, spy, tlt, **params):
        self.dates=list(dates)
        self.spy=list(map(float,spy));self.tlt=list(map(float,tlt))
        if len(self.dates)!=len(self.spy) or len(self.spy)!=len(self.tlt) or len(self.spy)<100:
            raise ValueError('Two aligned price series and at least 100 weeks required')
        if min(self.spy)<=0 or min(self.tlt)<=0: raise ValueError('Invalid prices')
        self.spyret=[self.spy[i+1]/self.spy[i]-1 for i in range(len(self.spy)-1)]
        self.tltret=[self.tlt[i+1]/self.tlt[i]-1 for i in range(len(self.tlt)-1)]
        self.s_up=[i>=4 and self.spy[i]>self.spy[i-4] for i in range(len(self.spy))]
        self.t_up=[i>=4 and self.tlt[i]>self.tlt[i-4] for i in range(len(self.tlt))]
        self.decisions=list(range(4,len(self.dates)-1))
        self.train_t=[t for t in self.decisions if self.dates[t+1][:4]<='2019']
        self.val_t=[t for t in self.decisions if '2020'<=self.dates[t+1][:4]<='2022']
        self.final_t=[t for t in self.decisions if '2023'<=self.dates[t+1][:4]<='2024']
        if min(len(self.train_t),len(self.val_t),len(self.final_t))<30:
            raise ValueError('Insufficient train/validation/test observations')
        self.configure(**params)

    def configure(self, **params):
        p=dict(alpha=.15,gamma=.90,epsilon=.25,budget=100,seed=17)
        p.update(params)
        self.alpha=float(p['alpha']); self.gamma=float(p['gamma']);self.initial_epsilon=float(p['epsilon'])
        self.budget=int(p['budget']);self.seed=int(p['seed'])
        self.cost_bps=10.0;self.risk=0.5
        if not(0<self.alpha<=1 and 0<=self.gamma<1 and 0<=self.initial_epsilon<=1 and 1<=self.budget<=2000):
            raise ValueError('Invalid Q-learning parameters')
        self.reset()

    def reset(self):
        self.q=[[0.0]*6 for _ in range(24)]
        self.rng=Mulberry32(self.seed)
        self.passes=0;self.position=0;self.prev=0;self.current_reward=0.0
        self.history=[];self.greedy_history=[];self.q_trace=[];self.last=None

    def state_id(self,t,prev):
        return (int(self.s_up[t])*2 + int(self.t_up[t]))*6+prev

    def eps(self):
        f=min(self.budget-1,self.passes)/max(1,self.budget-1)
        return self.initial_epsilon+(min(.03,self.initial_epsilon)-self.initial_epsilon)*f

    def payoff(self,action,prev,t):
        w=MULTI_ACTIONS[action];p=MULTI_ACTIONS[prev]
        turnover=.5*(abs(w[0]-p[0])+abs(w[1]-p[1])+abs((1-sum(w))-(1-sum(p))))
        gross=w[0]*self.spyret[t]+w[1]*self.tltret[t]
        net=gross-.001*turnover
        # The extra challenge asks the agent to manage *active* weekly performance
        # relative to a fixed 50/50 reference, with an explicit downside penalty.
        # The benchmark term depends only on realized exogenous prices, not action.
        benchmark=.5*self.spyret[t]+.5*self.tltret[t]
        reward=net-benchmark-.25*max(-gross,0.0)
        return dict(reward=reward,net=net,gross=gross,benchmark=benchmark,turnover=turnover)

    def greedy_training(self):
        previous=0;result=0.0
        for t in self.train_t:
            action=greedy(self.q[self.state_id(t,previous)])
            result+=self.payoff(action,previous,t)['reward'];previous=action
        return result

    def step(self):
        if self.passes>=self.budget:return False
        t=self.train_t[self.position];s=self.state_id(t,self.prev)
        if self.rng.random()<self.eps():a=int(self.rng.random()*6)
        else:a=greedy(self.q[s],self.rng)
        measures=self.payoff(a,self.prev,t)
        terminal=self.position==len(self.train_t)-1
        next_s=None if terminal else self.state_id(self.train_t[self.position+1],a)
        old=self.q[s][a];next_best=0.0 if terminal else max(self.q[next_s])
        td=measures['reward'] + self.gamma*next_best - old
        self.q[s][a]+=self.alpha*td
        self.current_reward+=measures['reward']
        self.last={'date':self.dates[t+1],'state':s,'action':a,'reward':measures['reward'],
                   'old':old,'new':self.q[s][a],'td':td}
        self.prev=a;self.position+=1
        if terminal:
            self.history.append(self.current_reward)
            self.greedy_history.append(self.greedy_training())
            self.q_trace.append([row[:] for row in self.q])
            self.passes+=1;self.prev=0;self.position=0;self.current_reward=0.0
        return True

    def step_many(self,count):
        for _ in range(min(max(0,int(count)),200000)):
            if not self.step():break
        return self.snapshot()

    def snapshot(self):
        return {'q':self.q,'passes':self.passes,'position':self.position,
                'budget':self.budget,'trainWeeks':len(self.train_t),
                'valWeeks':len(self.val_t),'testWeeks':len(self.final_t),
                'history':self.history,'greedyHistory':self.greedy_history,
                'last':self.last,'currentReward':self.current_reward}

    def evaluate(self,segment):
        if self.passes<1:raise ValueError('Complete a training pass first')
        if segment not in ('validation','final'):raise ValueError('Invalid segment')
        indices=self.val_t if segment=='validation' else self.final_t
        policies=(('Q-learning',None),('100% SPY',(1.,0.)),('100% TLT',(0.,1.)),
                  ('50/50 SPY–TLT',(.5,.5)),('100% cash',(0.,0.)))
        results=[]
        for name,static in policies:
            prev=(0.0,0.0); prev_action=0; wealth=1.0;high=1.0;dd=0.0;changes=0;turnover=0.0
            rr=[]; weights=[];weeks=[];alloc=[]
            for t in indices:
                action=greedy(self.q[self.state_id(t,prev_action)]) if static is None else None
                w=MULTI_ACTIONS[action] if static is None else static
                trade=.5*(abs(w[0]-prev[0])+abs(w[1]-prev[1])+
                          abs((1-sum(w))-(1-sum(prev))))
                net=w[0]*self.spyret[t]+w[1]*self.tltret[t]-.001*trade
                wealth*=1+net;high=max(high,wealth);dd=min(dd,wealth/high-1)
                turnover+=trade;changes+=int(trade>1e-10)
                rr.append(net);weights.append(sum(w));weeks.append(self.dates[t+1]);alloc.append(w)
                prev=w
                if action is not None:prev_action=action
            risk=risk_report(weeks,rr,weights,changes)
            results.append(dict(name=name,totalReturn=wealth-1,maxDrawdown=dd,
                                turnover=turnover,changes=changes,
                                meanSPY=sum(w[0] for w in alloc)/len(alloc),
                                meanTLT=sum(w[1] for w in alloc)/len(alloc),
                                **risk))
        return results

    def trace(self,state):
        s=int(state)
        if not 0<=s<24:raise ValueError('Unknown state')
        return {'state':s,'values':[q[s] for q in self.q_trace]}

_MULTI=None
_MULTI_FINAL_REVEALED=False

def multi_command(command_json):
    global _MULTI,_MULTI_FINAL_REVEALED
    cmd=json.loads(command_json);action=cmd['action']
    if action=='init':
        _MULTI=MultiAssetAgent(cmd['dates'],cmd['spy'],cmd['tlt'],**cmd['params'])
        _MULTI_FINAL_REVEALED=False
        return json.dumps(_MULTI.snapshot(),allow_nan=False)
    if _MULTI is None:raise RuntimeError('Initialize multi-asset lab first')
    if action=='reset':
        _MULTI.configure(**cmd['params'])
        return json.dumps(_MULTI.snapshot(),allow_nan=False)
    if action=='step':return json.dumps(_MULTI.step_many(cmd.get('count',1)),allow_nan=False)
    if action=='trace':return json.dumps(_MULTI.trace(cmd.get('state',0)),allow_nan=False)
    if action=='validation':return json.dumps(_MULTI.evaluate('validation'),allow_nan=False)
    if action=='final':
        if _MULTI_FINAL_REVEALED:raise ValueError('Final test already revealed in this session')
        _MULTI_FINAL_REVEALED=True
        return json.dumps(_MULTI.evaluate('final'),allow_nan=False)
    raise ValueError('Unknown multi-asset command')
