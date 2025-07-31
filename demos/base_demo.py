from abc import abstractmethod


class BaseDemo:
    """Base class for demos."""
    @abstractmethod
    def exec(self):
        pass
