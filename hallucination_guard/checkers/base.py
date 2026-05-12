from __future__ import annotations

from abc import ABC, abstractmethod

from hallucination_guard.models import CheckerResult, Document


class BaseChecker(ABC):
    name: str = "base"

    @abstractmethod
    def check(
        self,
        query: str,
        response: str,
        context: list[Document],
    ) -> CheckerResult:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
