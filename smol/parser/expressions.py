from dataclasses import dataclass
from typing import Literal
from smol.parser.utils import Expression, Statement


@dataclass
class IntegerExpression(Expression):
    value: int


@dataclass
class IdentifierExpression(Expression):
    name: str


@dataclass
class BooleanExpression(Expression):
    value: bool


@dataclass
class StringExpression(Expression):
    value: str


@dataclass
class ArrayExpression(Expression):
    elements: list[Expression]


@dataclass
class RangeExpression(Expression):
    left: Expression
    right: Expression
    step: Expression


@dataclass
class EqualityExpression(Expression):
    left: Expression
    sign: Literal['=='] | Literal['!=']
    right: Expression


@dataclass
class ComparisonExpression(Expression):
    left: Expression
    sign: Literal['<'] | Literal[">"] | Literal[">="] | Literal["<="]
    right: Expression


@dataclass
class AdditionExpression(Expression):
    left: Expression
    sign: Literal["+"] | Literal["-"]
    right: Expression


@dataclass
class MultiplicationExpression(Expression):
    left: Expression
    sign: Literal["*"] | Literal["/"]
    right: Expression


@dataclass
class ExponentiationExpression(Expression):
    left: Expression
    sign: Literal["^"]
    right: Expression


@dataclass
class NegationExpression(Expression):
    value: Expression


@dataclass
class PropertyAccessExpression(Expression):
    object: Expression
    property: str


@dataclass
class FunctionCallArgument:
    name: str | None
    value: Expression


@dataclass
class FunctionCallExpression(Expression):
    object: Expression  # object which is being called
    args: list[FunctionCallArgument]


@dataclass
class IfExpression(Expression):
    condition: Expression
    body: Expression
    else_ifs: list[tuple[Expression, Expression]]
    else_body: Expression | None


@dataclass
class BlockExpression(Expression):
    body: list[Statement]


@dataclass
class BreakExpression(Expression):
    pass


@dataclass
class ContinueExpression(Expression):
    pass


@dataclass
class TypeExpression(Expression):
    pass


@dataclass
class TypeDeduceExpression(TypeExpression):
    pass


@dataclass
class TypeBuiltInExpression(TypeExpression):
    name: Literal["int"] | Literal["string"] | Literal["bool"] | Literal["none"]


@dataclass
class TypeIdentifierExpression(TypeExpression):
    name: str


@dataclass
class TypeArrayExpression(TypeExpression):
    # TODO: implement this
    element: TypeExpression


@dataclass
class TypeUnionExpression(TypeExpression):
    elements: list[TypeExpression]
