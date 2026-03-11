"""
FILE 23: scenario_simulator.py
PURPOSE: Counterfactual testing and foresight
ROLE: What-if analysis and decision rehearsal

Supports:
- Monte Carlo simulation of outcomes
- Counterfactual reasoning ("what if we had done X?")
- Scenario planning and stress testing
- Decision tree exploration
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A scenario to simulate."""
    name: str
    description: str
    initial_state: dict[str, Any]
    actions: list[str]
    state_transitions: dict[str, Callable] | None = None
    probability: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    scenario_name: str
    run_id: int
    trajectory: list[dict[str, Any]]
    final_state: dict[str, Any]
    total_reward: float
    steps: int
    terminal_reason: str = ""


@dataclass
class SimulationSummary:
    """Summary statistics across multiple simulation runs."""
    scenario_name: str
    num_runs: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float
    success_rate: float
    percentile_5: float
    percentile_95: float
    results: list[SimulationResult] = field(default_factory=list)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual analysis."""
    original_action: str
    counterfactual_action: str
    original_outcome: float
    counterfactual_outcome: float
    regret: float
    analysis: str


class ScenarioSimulator:
    """
    Counterfactual testing and foresight engine.
    Enables the agent to simulate outcomes before committing to actions.
    """

    def __init__(
        self,
        transition_fn: Callable | None = None,
        reward_fn: Callable | None = None,
        terminal_fn: Callable | None = None,
    ):
        self.transition_fn = transition_fn or self._default_transition
        self.reward_fn = reward_fn or self._default_reward
        self.terminal_fn = terminal_fn or self._default_terminal
        self.simulation_history: list[SimulationSummary] = []

    def simulate(
        self,
        scenario: Scenario,
        num_runs: int = 100,
        max_steps: int = 100,
        policy_fn: Callable | None = None,
    ) -> SimulationSummary:
        """
        Run Monte Carlo simulation of a scenario.

        Args:
            scenario: The scenario definition
            num_runs: Number of simulation runs
            max_steps: Maximum steps per run
            policy_fn: Action selection function (state -> action)

        Returns:
            SimulationSummary with statistics across all runs
        """
        results: list[SimulationResult] = []

        for run_id in range(num_runs):
            result = self._run_single(scenario, run_id, max_steps, policy_fn)
            results.append(result)

        summary = self._compute_summary(scenario.name, results)
        self.simulation_history.append(summary)

        logger.info(
            f"Simulation '{scenario.name}': {num_runs} runs, "
            f"mean_reward={summary.mean_reward:.2f}, "
            f"success_rate={summary.success_rate:.1%}"
        )
        return summary

    def counterfactual(
        self,
        original_state: dict[str, Any],
        original_action: str,
        original_outcome: float,
        alternative_actions: list[str],
        num_runs: int = 50,
    ) -> list[CounterfactualResult]:
        """
        Counterfactual analysis: "What if we had chosen differently?"

        Args:
            original_state: The state at the decision point
            original_action: The action that was actually taken
            original_outcome: The realized outcome
            alternative_actions: Actions to test counterfactually
            num_runs: Monte Carlo runs per alternative

        Returns:
            List of CounterfactualResults
        """
        results = []

        for alt_action in alternative_actions:
            alt_outcomes = []
            for _ in range(num_runs):
                next_state = self.transition_fn(dict(original_state), alt_action)
                reward = self.reward_fn(original_state, alt_action, next_state)
                alt_outcomes.append(reward)

            avg_alt = sum(alt_outcomes) / len(alt_outcomes)
            regret = avg_alt - original_outcome

            analysis = ""
            if regret > 0:
                analysis = (
                    f"Alternative '{alt_action}' would have been better "
                    f"by {regret:.2f} on average"
                )
            elif regret < 0:
                analysis = (
                    f"Original '{original_action}' was better than "
                    f"'{alt_action}' by {abs(regret):.2f}"
                )
            else:
                analysis = "No meaningful difference detected"

            results.append(CounterfactualResult(
                original_action=original_action,
                counterfactual_action=alt_action,
                original_outcome=original_outcome,
                counterfactual_outcome=avg_alt,
                regret=regret,
                analysis=analysis,
            ))

        return results

    def explore_decision_tree(
        self,
        state: dict[str, Any],
        available_actions: list[str],
        depth: int = 3,
        branching_factor: int = 3,
    ) -> dict[str, Any]:
        """
        Explore a decision tree from the current state.

        Returns:
            Tree structure with action values at each node
        """
        return self._build_tree(state, available_actions, depth, branching_factor)

    def compare_strategies(
        self,
        scenario: Scenario,
        strategies: dict[str, Callable],
        num_runs: int = 100,
        max_steps: int = 100,
    ) -> dict[str, SimulationSummary]:
        """
        Compare multiple strategies on the same scenario.

        Args:
            scenario: The scenario to test
            strategies: {strategy_name: policy_fn}
            num_runs: Runs per strategy
            max_steps: Max steps per run

        Returns:
            {strategy_name: SimulationSummary}
        """
        results = {}
        for name, policy_fn in strategies.items():
            summary = self.simulate(scenario, num_runs, max_steps, policy_fn)
            results[name] = summary

        # Log comparison
        best = max(results, key=lambda k: results[k].mean_reward)
        logger.info(f"Best strategy: '{best}' (mean_reward={results[best].mean_reward:.2f})")

        return results

    # -------------------------------------------------------------------------
    # INTERNAL SIMULATION
    # -------------------------------------------------------------------------

    def _run_single(
        self,
        scenario: Scenario,
        run_id: int,
        max_steps: int,
        policy_fn: Callable | None,
    ) -> SimulationResult:
        """Execute a single simulation run."""
        state = dict(scenario.initial_state)
        trajectory = []
        total_reward = 0.0
        terminal_reason = "max_steps"

        for step in range(max_steps):
            # Select action
            if policy_fn:
                action = policy_fn(state)
            elif scenario.actions:
                action = random.choice(scenario.actions)
            else:
                break

            # Transition
            next_state = self.transition_fn(dict(state), action)

            # Reward
            reward = self.reward_fn(state, action, next_state)
            total_reward += reward

            trajectory.append({
                "step": step,
                "state": dict(state),
                "action": action,
                "reward": reward,
                "next_state": dict(next_state),
            })

            state = next_state

            # Terminal check
            is_terminal, reason = self.terminal_fn(state, step)
            if is_terminal:
                terminal_reason = reason
                break

        return SimulationResult(
            scenario_name=scenario.name,
            run_id=run_id,
            trajectory=trajectory,
            final_state=state,
            total_reward=total_reward,
            steps=len(trajectory),
            terminal_reason=terminal_reason,
        )

    def _build_tree(
        self,
        state: dict[str, Any],
        actions: list[str],
        depth: int,
        branching: int,
    ) -> dict[str, Any]:
        """Recursively build a decision tree."""
        if depth <= 0:
            return {"state": state, "value": 0.0}

        children = {}
        for action in actions[:branching]:
            next_state = self.transition_fn(dict(state), action)
            reward = self.reward_fn(state, action, next_state)
            subtree = self._build_tree(next_state, actions, depth - 1, branching)
            children[action] = {
                "reward": reward,
                "total_value": reward + subtree.get("value", 0.0),
                "subtree": subtree,
            }

        best_action = max(children, key=lambda a: children[a]["total_value"])
        return {
            "state": state,
            "children": children,
            "best_action": best_action,
            "value": children[best_action]["total_value"],
        }

    def _compute_summary(
        self, name: str, results: list[SimulationResult]
    ) -> SimulationSummary:
        """Compute summary statistics from simulation results."""
        rewards = sorted([r.total_reward for r in results])
        n = len(rewards)

        mean = sum(rewards) / n
        variance = sum((r - mean) ** 2 for r in rewards) / n
        std = variance ** 0.5

        median = rewards[n // 2]
        p5 = rewards[max(0, int(n * 0.05))]
        p95 = rewards[min(n - 1, int(n * 0.95))]

        success_count = sum(1 for r in results if r.total_reward > 0)

        return SimulationSummary(
            scenario_name=name,
            num_runs=n,
            mean_reward=mean,
            std_reward=std,
            min_reward=min(rewards),
            max_reward=max(rewards),
            median_reward=median,
            success_rate=success_count / n,
            percentile_5=p5,
            percentile_95=p95,
            results=results,
        )

    # -------------------------------------------------------------------------
    # DEFAULT FUNCTIONS (override with real implementations)
    # -------------------------------------------------------------------------

    @staticmethod
    def _default_transition(state: dict[str, Any], action: str) -> dict[str, Any]:
        """Default transition: add noise to continuous values."""
        next_state = dict(state)
        for key, value in next_state.items():
            if isinstance(value, (int, float)):
                noise = random.gauss(0, 0.1)
                next_state[key] = value + noise
        next_state["last_action"] = action
        return next_state

    @staticmethod
    def _default_reward(
        state: dict[str, Any], action: str, next_state: dict[str, Any]
    ) -> float:
        """Default reward: based on task progress."""
        progress = next_state.get("task_progress", 0) - state.get("task_progress", 0)
        return progress * 10.0

    @staticmethod
    def _default_terminal(state: dict[str, Any], step: int) -> tuple[bool, str]:
        """Default terminal check."""
        if state.get("task_progress", 0) >= 1.0:
            return True, "task_complete"
        return False, ""
