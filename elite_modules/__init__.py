"""Elite Modules - Multi-agent, simulation, and adversarial testing."""

from .multi_agent_interface import MultiAgentInterface, AgentRole, AgentMessage
from .scenario_simulator import ScenarioSimulator, Scenario, SimulationSummary
from .red_team_memory_partition import RedTeamMemoryPartition, RedTeamTest, MemoryPartition

__all__ = [
    "MultiAgentInterface",
    "AgentRole",
    "AgentMessage",
    "ScenarioSimulator",
    "Scenario",
    "SimulationSummary",
    "RedTeamMemoryPartition",
    "RedTeamTest",
    "MemoryPartition",
]
