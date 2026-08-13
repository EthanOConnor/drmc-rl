"""Human-behavior models and the Professor Pills backend contract."""

from .conditioning import HumanSkillCondition
from .coach import analyze_choice

__all__ = ["HumanSkillCondition", "analyze_choice"]
