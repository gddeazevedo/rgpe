from abc import abstractmethod


class BaseDemo:
    """Base class for demos."""
    @abstractmethod
    def run(self) -> None:
        pass
