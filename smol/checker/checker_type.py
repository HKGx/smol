from dataclasses import dataclass, field
from typing import Optional

from smol.parser.utils import Expression, Statement


@dataclass(eq=True, frozen=True)
class CheckerType:
    name: str
    meta: dict[str, bool] = field(
        init=False, default_factory=dict, compare=False)


@dataclass(eq=True, frozen=True)
class InvalidType(CheckerType):
    name = "invalid"


@dataclass(eq=True, frozen=True)
class ListType(CheckerType):
    type: CheckerType
    known_length: Optional[int]


@dataclass(eq=True, frozen=True)
class UnionType(CheckerType):
    types: tuple[CheckerType, ...]


@dataclass(eq=True, frozen=True)
class FunctionArgumentType(CheckerType):
    name: str
    type: CheckerType
    named: bool = False


@dataclass(eq=True, frozen=True)
class FunctionType(CheckerType):
    arg_types: tuple[FunctionArgumentType, ...]
    to_type: CheckerType

    @property
    def named_arg_types(self) -> tuple[FunctionArgumentType, ...]:
        return tuple(arg_type for arg_type in self.arg_types if arg_type.named)

    @property
    def positional_arg_types(self) -> tuple[FunctionArgumentType, ...]:
        return tuple(arg_type for arg_type in self.arg_types if not arg_type.named)

    def is_named(self, name: str) -> bool:
        return any(arg_type.name == name for arg_type in self.named_arg_types)


@dataclass(eq=True, frozen=True)
class StructFieldType(CheckerType):
    type: CheckerType


@dataclass(eq=True, frozen=True)
class StructMethodType(CheckerType):
    type: FunctionType


@dataclass(eq=True, frozen=True)
class StructType(CheckerType):
    fields: tuple[StructFieldType, ...]
    methods: tuple[StructMethodType, ...]

    def get(self, name: str) -> StructFieldType | StructMethodType | None:
        for member in [*self.fields, *self.methods]:
            if member.name == name:
                return member
        return None


@dataclass(eq=True, frozen=True)
class ModuleType(CheckerType):
    name: str
    types: dict[str, CheckerType]


@dataclass(eq=True, frozen=True)
class TypedExpression:
    type: CheckerType
    value: Expression


@dataclass
class TypedStatement:
    type: CheckerType
    value: Statement
