"""
SCENARIO SIMULATOR
Counterfactual testing and foresight for autonomous agents
Version: 1.0.0
Classification: CARBON[6] AUTONOMOUS AGENCY (ELITE)

Features:
- Counterfactual analysis
- What-if scenario testing
- Monte Carlo simulation
- Foresight planning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from copy import deepcopy
import random


class ScenarioType(Enum):
    """Types of scenarios."""
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    STRESS_TEST = "stress_test"
    COUNTERFACTUAL = "counterfactual"
    MONTE_CARLO = "monte_carlo"


class VariableDistribution(Enum):
    """Distributions for variable sampling."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    BETA = "beta"
    DISCRETE = "discrete"


@dataclass
class ScenarioVariable:
    """Variable that can change across scenarios."""
    name: str
    base_value: Any
    distribution: VariableDistribution
    parameters: Dict[str, Any]  # Distribution parameters
    correlation_with: Optional[List[str]] = None


@dataclass
class Scenario:
    """A simulation scenario."""
    scenario_id: str
    scenario_type: ScenarioType
    name: str
    description: str
    variable_values: Dict[str, Any]
    probability: float = 1.0  # For weighted scenarios
    created: datetime = field(default_factory=datetime.now)


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    scenario: Scenario
    action: str
    outcomes: List[Dict[str, Any]]
    expected_value: float
    variance: float
    best_outcome: Dict[str, Any]
    worst_outcome: Dict[str, Any]
    percentiles: Dict[int, float]  # 5th, 25th, 50th, 75th, 95th
    execution_time_ms: float


@dataclass
class CounterfactualAnalysis:
    """Result of counterfactual analysis."""
    original_action: str
    original_outcome: Dict[str, Any]
    counterfactual_action: str
    counterfactual_outcomes: List[Dict[str, Any]]
    expected_difference: float
    regret: float
    confidence: float


@dataclass
class SimulatorConfig:
    """Configuration for scenario simulator."""
    # Monte Carlo settings
    default_num_samples: int = 1000
    max_num_samples: int = 10000

    # Scenario generation
    num_stress_scenarios: int = 10
    stress_severity_range: Tuple[float, float] = (1.5, 3.0)

    # Correlation handling
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None

    # Convergence criteria
    convergence_threshold: float = 0.01
    min_samples_for_convergence: int = 100

    # Time limits
    max_simulation_time_seconds: int = 30


class ScenarioGenerator:
    """Generates scenarios for simulation."""

    def __init__(self, config: SimulatorConfig):
        self.config = config

    def generate_baseline(
        self,
        variables: List[ScenarioVariable]
    ) -> Scenario:
        """Generate baseline scenario with base values."""
        values = {v.name: v.base_value for v in variables}

        return Scenario(
            scenario_id="baseline",
            scenario_type=ScenarioType.BASELINE,
            name="Baseline Scenario",
            description="All variables at expected values",
            variable_values=values
        )

    def generate_optimistic(
        self,
        variables: List[ScenarioVariable]
    ) -> Scenario:
        """Generate optimistic scenario."""
        values = {}
        for v in variables:
            values[v.name] = self._get_optimistic_value(v)

        return Scenario(
            scenario_id="optimistic",
            scenario_type=ScenarioType.OPTIMISTIC,
            name="Optimistic Scenario",
            description="All variables at favorable values",
            variable_values=values,
            probability=0.1  # Typically 10% probability
        )

    def generate_pessimistic(
        self,
        variables: List[ScenarioVariable]
    ) -> Scenario:
        """Generate pessimistic scenario."""
        values = {}
        for v in variables:
            values[v.name] = self._get_pessimistic_value(v)

        return Scenario(
            scenario_id="pessimistic",
            scenario_type=ScenarioType.PESSIMISTIC,
            name="Pessimistic Scenario",
            description="All variables at unfavorable values",
            variable_values=values,
            probability=0.1
        )

    def generate_stress_scenarios(
        self,
        variables: List[ScenarioVariable]
    ) -> List[Scenario]:
        """Generate multiple stress test scenarios."""
        scenarios = []
        min_sev, max_sev = self.config.stress_severity_range

        for i in range(self.config.num_stress_scenarios):
            severity = min_sev + (max_sev - min_sev) * i / self.config.num_stress_scenarios

            values = {}
            for v in variables:
                values[v.name] = self._apply_stress(v, severity)

            scenarios.append(Scenario(
                scenario_id=f"stress_{i}",
                scenario_type=ScenarioType.STRESS_TEST,
                name=f"Stress Scenario {i+1}",
                description=f"Stress test with severity {severity:.1f}",
                variable_values=values,
                probability=0.01  # Rare events
            ))

        return scenarios

    def generate_monte_carlo_sample(
        self,
        variables: List[ScenarioVariable]
    ) -> Scenario:
        """Generate a single Monte Carlo sample."""
        values = {}
        for v in variables:
            values[v.name] = self._sample_variable(v)

        return Scenario(
            scenario_id=f"mc_{random.randint(0, 1000000)}",
            scenario_type=ScenarioType.MONTE_CARLO,
            name="Monte Carlo Sample",
            description="Random sample from distributions",
            variable_values=values
        )

    def _get_optimistic_value(self, variable: ScenarioVariable) -> Any:
        """Get optimistic value for a variable."""
        params = variable.parameters

        if variable.distribution == VariableDistribution.NORMAL:
            # 90th percentile
            return params["mean"] + 1.28 * params.get("std", 1)

        elif variable.distribution == VariableDistribution.UNIFORM:
            # 90% of range toward high end
            low, high = params["low"], params["high"]
            return low + 0.9 * (high - low)

        elif variable.distribution == VariableDistribution.TRIANGULAR:
            return params["high"]

        elif variable.distribution == VariableDistribution.DISCRETE:
            # Highest value
            return max(params["values"])

        return variable.base_value

    def _get_pessimistic_value(self, variable: ScenarioVariable) -> Any:
        """Get pessimistic value for a variable."""
        params = variable.parameters

        if variable.distribution == VariableDistribution.NORMAL:
            # 10th percentile
            return params["mean"] - 1.28 * params.get("std", 1)

        elif variable.distribution == VariableDistribution.UNIFORM:
            low, high = params["low"], params["high"]
            return low + 0.1 * (high - low)

        elif variable.distribution == VariableDistribution.TRIANGULAR:
            return params["low"]

        elif variable.distribution == VariableDistribution.DISCRETE:
            return min(params["values"])

        return variable.base_value

    def _apply_stress(
        self,
        variable: ScenarioVariable,
        severity: float
    ) -> Any:
        """Apply stress to a variable value."""
        base = variable.base_value

        if isinstance(base, (int, float)):
            # Stress moves value in adverse direction
            if severity > 0:
                return base * (1 - (severity - 1) * 0.2)  # Decrease
            return base

        return base

    def _sample_variable(self, variable: ScenarioVariable) -> Any:
        """Sample a value from variable distribution."""
        params = variable.parameters

        if variable.distribution == VariableDistribution.NORMAL:
            return np.random.normal(
                params["mean"],
                params.get("std", 1)
            )

        elif variable.distribution == VariableDistribution.UNIFORM:
            return np.random.uniform(
                params["low"],
                params["high"]
            )

        elif variable.distribution == VariableDistribution.TRIANGULAR:
            return np.random.triangular(
                params["low"],
                params["mode"],
                params["high"]
            )

        elif variable.distribution == VariableDistribution.BETA:
            return np.random.beta(
                params["alpha"],
                params["beta"]
            )

        elif variable.distribution == VariableDistribution.DISCRETE:
            probs = params.get("probabilities")
            if probs:
                return np.random.choice(params["values"], p=probs)
            return np.random.choice(params["values"])

        return variable.base_value


class ScenarioSimulator:
    """
    Simulator for counterfactual testing and foresight.
    """

    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()
        self.generator = ScenarioGenerator(self.config)
        self._outcome_function: Optional[Callable] = None

    def set_outcome_function(
        self,
        fn: Callable[[Dict[str, Any], str], Dict[str, Any]]
    ):
        """
        Set the outcome function that computes results.
        fn(scenario_variables, action) -> outcome
        """
        self._outcome_function = fn

    def run_monte_carlo(
        self,
        variables: List[ScenarioVariable],
        action: str,
        num_samples: Optional[int] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Returns distribution of outcomes across sampled scenarios.
        """
        start_time = datetime.now()
        num_samples = num_samples or self.config.default_num_samples

        outcomes = []
        values = []

        for i in range(num_samples):
            scenario = self.generator.generate_monte_carlo_sample(variables)
            outcome = self._simulate_outcome(scenario, action)
            outcomes.append(outcome)

            # Extract numeric value for statistics
            value = outcome.get("value", outcome.get("emv", 0))
            values.append(value)

            # Check convergence
            if i > self.config.min_samples_for_convergence:
                if self._check_convergence(values):
                    break

            # Check time limit
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > self.config.max_simulation_time_seconds:
                break

        # Compute statistics
        values_array = np.array(values)

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return SimulationResult(
            scenario=Scenario(
                scenario_id="monte_carlo",
                scenario_type=ScenarioType.MONTE_CARLO,
                name=f"Monte Carlo ({len(outcomes)} samples)",
                description=f"Monte Carlo simulation with {len(outcomes)} samples",
                variable_values={}
            ),
            action=action,
            outcomes=outcomes,
            expected_value=float(np.mean(values_array)),
            variance=float(np.var(values_array)),
            best_outcome=outcomes[np.argmax(values_array)],
            worst_outcome=outcomes[np.argmin(values_array)],
            percentiles={
                5: float(np.percentile(values_array, 5)),
                25: float(np.percentile(values_array, 25)),
                50: float(np.percentile(values_array, 50)),
                75: float(np.percentile(values_array, 75)),
                95: float(np.percentile(values_array, 95))
            },
            execution_time_ms=execution_time
        )

    def run_scenario_analysis(
        self,
        variables: List[ScenarioVariable],
        action: str
    ) -> Dict[str, SimulationResult]:
        """
        Run comprehensive scenario analysis.

        Returns results for baseline, optimistic, pessimistic,
        and stress scenarios.
        """
        results = {}

        # Baseline
        baseline = self.generator.generate_baseline(variables)
        results["baseline"] = self._run_single_scenario(baseline, action)

        # Optimistic
        optimistic = self.generator.generate_optimistic(variables)
        results["optimistic"] = self._run_single_scenario(optimistic, action)

        # Pessimistic
        pessimistic = self.generator.generate_pessimistic(variables)
        results["pessimistic"] = self._run_single_scenario(pessimistic, action)

        # Stress tests
        stress_scenarios = self.generator.generate_stress_scenarios(variables)
        for i, scenario in enumerate(stress_scenarios):
            results[f"stress_{i}"] = self._run_single_scenario(scenario, action)

        return results

    def run_counterfactual(
        self,
        original_action: str,
        original_outcome: Dict[str, Any],
        alternative_actions: List[str],
        scenario: Scenario
    ) -> List[CounterfactualAnalysis]:
        """
        Run counterfactual analysis.

        Answers: "What would have happened if we had done X instead?"
        """
        analyses = []
        original_value = original_outcome.get("value", original_outcome.get("emv", 0))

        for alt_action in alternative_actions:
            # Simulate alternative action
            alt_outcomes = []
            for _ in range(100):  # Small sample for counterfactual
                outcome = self._simulate_outcome(scenario, alt_action)
                alt_outcomes.append(outcome)

            alt_values = [o.get("value", o.get("emv", 0)) for o in alt_outcomes]
            expected_alt = np.mean(alt_values)

            # Calculate regret
            difference = expected_alt - original_value
            regret = max(0, difference)  # Regret is non-negative

            # Confidence based on variance
            variance = np.var(alt_values)
            confidence = 1 / (1 + variance / 100)

            analyses.append(CounterfactualAnalysis(
                original_action=original_action,
                original_outcome=original_outcome,
                counterfactual_action=alt_action,
                counterfactual_outcomes=alt_outcomes,
                expected_difference=float(difference),
                regret=float(regret),
                confidence=float(confidence)
            ))

        return analyses

    def run_foresight(
        self,
        variables: List[ScenarioVariable],
        actions: List[str],
        horizon_steps: int = 5
    ) -> Dict[str, List[SimulationResult]]:
        """
        Run foresight analysis over multiple time steps.

        Returns projected outcomes for each action over the horizon.
        """
        foresight = {action: [] for action in actions}

        current_variables = deepcopy(variables)

        for step in range(horizon_steps):
            for action in actions:
                result = self.run_monte_carlo(
                    current_variables,
                    action,
                    num_samples=100  # Smaller for foresight
                )
                foresight[action].append(result)

            # Evolve variables for next step (simplified)
            current_variables = self._evolve_variables(
                current_variables, step
            )

        return foresight

    def _run_single_scenario(
        self,
        scenario: Scenario,
        action: str
    ) -> SimulationResult:
        """Run simulation for a single scenario."""
        start_time = datetime.now()

        outcome = self._simulate_outcome(scenario, action)
        value = outcome.get("value", outcome.get("emv", 0))

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return SimulationResult(
            scenario=scenario,
            action=action,
            outcomes=[outcome],
            expected_value=float(value),
            variance=0.0,
            best_outcome=outcome,
            worst_outcome=outcome,
            percentiles={p: float(value) for p in [5, 25, 50, 75, 95]},
            execution_time_ms=execution_time
        )

    def _simulate_outcome(
        self,
        scenario: Scenario,
        action: str
    ) -> Dict[str, Any]:
        """Simulate outcome for a scenario and action."""
        if self._outcome_function:
            return self._outcome_function(scenario.variable_values, action)

        # Default outcome function (placeholder)
        base_value = 100
        for var_name, var_value in scenario.variable_values.items():
            if isinstance(var_value, (int, float)):
                base_value *= var_value

        return {
            "value": base_value,
            "success": base_value > 50,
            "scenario_id": scenario.scenario_id
        }

    def _check_convergence(self, values: List[float]) -> bool:
        """Check if Monte Carlo has converged."""
        if len(values) < self.config.min_samples_for_convergence:
            return False

        recent = values[-100:]
        older = values[-200:-100] if len(values) >= 200 else values[:-100]

        if not older:
            return False

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)

        if older_mean == 0:
            return False

        change = abs(recent_mean - older_mean) / abs(older_mean)
        return change < self.config.convergence_threshold

    def _evolve_variables(
        self,
        variables: List[ScenarioVariable],
        step: int
    ) -> List[ScenarioVariable]:
        """Evolve variables for foresight (simple drift model)."""
        evolved = []
        for v in variables:
            new_v = deepcopy(v)
            if isinstance(v.base_value, (int, float)):
                # Simple random walk
                drift = np.random.normal(0, 0.05) * v.base_value
                new_v.base_value = v.base_value + drift
            evolved.append(new_v)
        return evolved


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_scenario_simulator(
    config: Optional[SimulatorConfig] = None
) -> ScenarioSimulator:
    """Create a scenario simulator."""
    return ScenarioSimulator(config)
