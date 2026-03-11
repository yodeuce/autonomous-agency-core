"""Elite Modules - Multi-agent, simulation, and adversarial testing."""

from .multi_agent_interface import (
    MultiAgentInterface,
    AgentRole,
    AgentMessage,
    CoordinationProtocol,
    SharedMemoryStore,
)
from .scenario_simulator import (
    ScenarioSimulator,
    Scenario,
    SimulationSummary,
    SCENARIO_PRESETS,
    get_scenario_preset,
)
from .red_team_memory_partition import (
    RedTeamMemoryPartition,
    RedTeamTest,
    MemoryPartition,
    IsolationLevel,
)

__all__ = [
    "MultiAgentInterface",
    "AgentRole",
    "AgentMessage",
    "CoordinationProtocol",
    "SharedMemoryStore",
    "ScenarioSimulator",
    "Scenario",
    "SimulationSummary",
    "SCENARIO_PRESETS",
    "get_scenario_preset",
    "RedTeamMemoryPartition",
    "RedTeamTest",
    "MemoryPartition",
    "IsolationLevel",
]
