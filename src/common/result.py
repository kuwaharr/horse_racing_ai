from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass
class Result(Generic[T]):
    success: bool
    value: Optional[T] = None
    error: Optional[str] = None