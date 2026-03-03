from abc import ABC, abstractmethod
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all MediFlow agents.
    Every agent must implement the async run() method.
    """
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logger.bind(agent=self.name)

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """
        Execute the agent's core logic.
        """
        pass
