# CARBON&trade; NLD &mdash; Autonomous Agency Core

A complete, implementation-ready architecture for building **MD-based (Markov Decision Process) Autonomous AI Agents** with **Enhanced Memory Protocols (EMP)**, **Environment Variable Modeling**, and **EMV/Utility/Risk evaluation**.

Built as a systems engineering specification — every file is required for robustness, auditability, and scale.

---

## Architecture Overview

```
autonomous_agency_core/
│
├── core_decision_framework/          I.   How the agent decides
├── enhanced_memory_protocol/         II.  How the agent remembers
├── environment_modeling/             III. How the agent perceives the world
├── emv_utility_risk/                 IV.  How the agent evaluates value & risk
├── learning_adaptation/              V.   How the agent learns from outcomes
├── governance_audit_safety/          VI.  How the agent stays safe & auditable
└── elite_modules/                    VII. Multi-agent, simulation, red teaming
```

---

## Modules

### I. Core Decision Framework

The constitutional layer — defines how the agent decides.

| File | Purpose |
|------|---------|
| `mdp_spec.yaml` | Formal MDP definition: state space, action space, transitions, reward structure, discount factor, terminal states |
| `policy_definition.py` | Policy classes (Deterministic, Epsilon-Greedy, Boltzmann) with constraints, versioning, and exploration control |
| `reward_model.py` | Converts EMV into machine-usable rewards with shaping, component weighting, and constraint penalties |

### II. Enhanced Memory Protocol (EMP)

The intelligence layer — defines how the agent remembers, forgets, and reasons over time.

| File | Purpose |
|------|---------|
| `memory_schema.json` | JSON Schema for all memories: type, confidence, salience, provenance, decay profile, associations |
| `memory_encoder.py` | Transforms raw events into structured memories with classification, confidence estimation, and salience scoring |
| `memory_salience_engine.py` | Computes memory priority via EMV impact, risk amplification, recency decay, and frequency reinforcement |
| `memory_retrieval_policy.py` | Controls when/how memory is accessed — triggers on uncertainty, risk breaches, novelty; prevents over-retrieval |
| `memory_decay_and_compression.py` | Intentional forgetting: temporal decay, utility-based pruning, memory compression, supersession logic |
| `immutable_memory_registry.yaml` | Non-erasable memories: hard constraints, safety rules, governance mandates, ethical boundaries |

### III. Environment Variable Modeling

The perception layer — defines the world the agent operates in.

| File | Purpose |
|------|---------|
| `environment_state_model.py` | Canonical state representation with observable, latent, and partially observable variables |
| `environment_variable_registry.yaml` | Authoritative variable catalog: type, range, volatility, source, update cadence, reliability score |
| `environment_observer.py` | Signal ingestion with normalization, missing data handling, outlier detection, and confidence tagging |
| `belief_state_updater.py` | Bayesian/Kalman belief updates: state estimation, noise filtering, confidence propagation |

### IV. EMV / Utility & Risk

The valuation layer — defines how value and risk are evaluated.

| File | Purpose |
|------|---------|
| `emv_calculator.py` | Expected Monetary/Economic Value computation with outcome enumeration, sensitivity analysis, and discounting |
| `utility_function.py` | Non-linear utility: Prospect Theory, risk aversion, loss aversion, probability weighting, certainty equivalents |
| `risk_model.py` | Downside modeling: VaR, CVaR, worst-case loss, Sharpe/Sortino ratios, stress testing, constraint violation probability |

### V. Learning & Adaptation

The growth layer — defines how experience compounds.

| File | Purpose |
|------|---------|
| `learning_update_engine.py` | Updates policy, reward model, salience weights, and environment model from experience replay |
| `failure_analysis_module.py` | Post-decision diagnostics: expected vs realized EMV, root cause analysis, recurring pattern detection |

### VI. Governance, Audit & Safety

The compliance layer — ensures safety, auditability, and alignment.

| File | Purpose |
|------|---------|
| `decision_trace_log.jsonl` | Append-only audit trail: state, memories, EMV calculations, action, outcome per decision |
| `constraint_enforcement_engine.py` | Hard stop mechanism with deterministic enforcement — constraints are not suggestions |
| `agent_constitution.md` | Human-readable charter: mission, boundaries, value hierarchy, alignment principles |

### VII. Elite Modules (Optional)

Advanced capabilities for strategic-grade agents.

| File | Purpose |
|------|---------|
| `multi_agent_interface.py` | Agent discovery, shared memory, task delegation, negotiation, conflict resolution |
| `scenario_simulator.py` | Monte Carlo simulation, counterfactual analysis, decision tree exploration, strategy comparison |
| `red_team_memory_partition.py` | Adversarial testing in isolated memory partitions: injection, corruption, tampering detection |

---

## Build Tiers

| Level | Modules | Use Case |
|-------|---------|----------|
| **Minimal** | Files 1-5, 10, 14 | Proof-of-concept agent |
| **Competent** | + Files 6-9, 11-16 | Production-capable agent |
| **Autonomous Strategic** | + All files | Full autonomous decision-maker |
| **Regulated / Financial** | + Governance & Audit | Compliance-ready agent |

---

## Quick Start

```python
from autonomous_agency_core.core_decision_framework import create_policy, RewardModel
from autonomous_agency_core.enhanced_memory_protocol import MemoryEncoder, MemorySalienceEngine
from autonomous_agency_core.environment_modeling import EnvironmentStateModel, BeliefStateUpdater
from autonomous_agency_core.emv_utility_risk import EMVCalculator, UtilityFunction, RiskModel
from autonomous_agency_core.learning_adaptation import LearningUpdateEngine, FailureAnalysisModule
from autonomous_agency_core.governance_audit_safety import ConstraintEnforcementEngine

# 1. Define the environment
env = EnvironmentStateModel()

# 2. Set up memory
encoder = MemoryEncoder()
salience = MemorySalienceEngine()

# 3. Create a policy
policy = create_policy(
    policy_type="epsilon_greedy",
    action_space=["gather_information", "execute_task", "delegate", "wait", "escalate"],
    epsilon=0.1,
)

# 4. Initialize decision support
emv = EMVCalculator()
utility = UtilityFunction()
risk = RiskModel()

# 5. Enforce constraints
constraints = ConstraintEnforcementEngine()

# 6. Wire up learning
learner = LearningUpdateEngine()
failure_analyzer = FailureAnalysisModule()

# 7. Run the agent loop
state = env.get_observable_state()
check = constraints.check(action="execute_task", state=state)
if check.allowed:
    action = policy.select_action(state)
    # ... execute action, observe outcome, learn
```

---

## Requirements

- Python 3.10+
- No external dependencies (standard library only)

---

## Design Principles

- **Separation of concerns** — Each file has one job. Decision logic, memory, environment, valuation, learning, and governance are fully decoupled.
- **Auditability** — Every decision is traceable through the decision trace log, constraint checks, and memory provenance.
- **Safety by default** — Immutable constraints, hard enforcement, and a value hierarchy where safety always outranks efficiency.
- **Memory intelligence** — Not flat storage. Memories have salience, decay, compression, and retrieval policies that mirror strategic reasoning.
- **Risk-aware** — Agents optimize utility, not raw EMV. Tail risks are explicitly modeled. Survivability is separate from value.

---

## License

Proprietary — CARBON&trade; NLD

---

*Built with the CARBON&trade; NLD Autonomous Agency Specification.*
