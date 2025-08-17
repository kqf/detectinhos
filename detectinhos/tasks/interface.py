from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class HasBoxesAndClasses(Protocol, Generic[T]):
    bbox: T
    label: T
    score: T

    @classmethod
    def is_dataclass(cls) -> bool: ...
