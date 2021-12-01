from dataclasses import dataclass, field
from smol.parser.expressions import IdentifierExpression, TypeExpression
from smol.parser.utils import Expression, Statement


@dataclass
class ExpressionStatement(Statement):
    value: Expression


@dataclass
class AssignmentStatement(Statement):
    name: IdentifierExpression
    value: Expression
    type: TypeExpression
    mutable: bool


@dataclass
class WhileStatement(Statement):
    condition: Expression
    body: Expression


@dataclass
class ForStatement(Statement):
    ident: IdentifierExpression
    value: Expression
    body: Expression


@dataclass
class FunctionArgument(IdentifierExpression):
    type: TypeExpression
    mutable: bool
    default: Expression | None


@dataclass
class FunctionDefinitionStatement(Statement):
    name: str
    args: list[FunctionArgument]
    body: Expression
    return_type: TypeExpression


@dataclass
class StructField(IdentifierExpression):
    type: TypeExpression
    mutable: bool


@dataclass
class StructMethod(FunctionDefinitionStatement):
    @classmethod
    def from_function(cls, function: FunctionDefinitionStatement):
        return cls(
            name=function.name,
            args=function.args,
            body=function.body,
            return_type=function.return_type,
        )


@dataclass
class StructDefinitionStatement(Statement):
    name: str
    fields: list[StructField]
    methods: list[StructMethod]


@dataclass
class ImportStatement(Statement):
    name: str
    add_to_scope: bool = field(default=False, kw_only=True)
