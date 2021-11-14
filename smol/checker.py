from dataclasses import dataclass
from enum import Enum
from os import name
from smol.utils import Scope
from smol.parser import AdditionExpression, ArrayExpression, AssignmentStatement, ExponentiationExpression, Expression, ExpressionStatement, ForStatement, IdentifierExpression, IntegerExpression, MultiplicationExpression, Program, Statement, StringExpression


@dataclass
class CheckerType:
    name: str


@dataclass
class ListType(CheckerType):
    type: CheckerType


@dataclass
class Mapping(CheckerType):
    from_type: CheckerType
    to_type: CheckerType


@dataclass
class TypedExpression:
    type: CheckerType
    value: Expression


class UnknownIdentifier(Exception):
    pass


class InvalidOperation(Exception):
    pass


ScopeValue = TypeVar("ScopeValue")


class Scope(dict[str, ScopeValue]):
    parent: Optional["Scope"] = None

    @classmethod
    def from_dict(cls, d: dict[str, ScopeValue]):
        new = cls()
        for k, v in d.items():
            new[k] = v
        return new

    def __init__(self, parent: "Scope" = None):
        super().__init__()
        self.parent = parent

    def rec_contains(self, o: str) -> bool:
        if o in self:
            return True
        if self.parent is None:
            return False
        return self.parent.rec_contains(o)

    def rec_get(self, key: str) -> ScopeValue:
        if key in self:
            return self[key]
        if self.parent is None:
            raise KeyError(key)
        return self.parent.rec_get(key)

    def rec_set(self, key: str, value: ScopeValue) -> bool:
        if self.parent is None:
            self[key] = value
            return True
        if self.parent is not None:
            if self.parent.rec_contains(key):
                return self.parent.rec_set(key, value)
            self[key] = value
            return True
        return False

    def spawn_child(self):
        return Scope(parent=self)


class BuiltInTypes(Enum):
    INT = CheckerType("int")
    STRING = CheckerType("string")
    BOOL = CheckerType("bool")
    NONE = CheckerType("none")


class Checker:
    program: Program
    errors: list[str] = []
    scope: Scope = Scope.from_dict({
        "print": Mapping("function", BuiltInTypes.STRING.value, BuiltInTypes.NONE.value),
        "str": Mapping("function", BuiltInTypes.INT.value, BuiltInTypes.STRING.value),
    })

    def __init__(self, program: Program):
        self.program = program
        self.errors = []

    def check(self):
        self.check_program(self.program)
        return self.errors

    def check_program(self, program: Program):
        for statement in program.statements:
            self.check_statement(statement)

    def evaluate_type_expression(self, expression: Expression, scope: Scope = None) -> TypedExpression:
        if scope is None:
            scope = self.scope
        match expression:
            case IntegerExpression() as e:
                return TypedExpression(BuiltInTypes.INT.value, e)
            case StringExpression() as e:
                return TypedExpression(BuiltInTypes.STRING.value, e)
            case IdentifierExpression(name) as e:
                return TypedExpression(scope.rec_get(name), e)
            case AdditionExpression(left, sign,  right) as e:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                match sign:
                    case "+":
                        match (left_type.type, right_type.type):
                            case (BuiltInTypes.INT.value, BuiltInTypes.INT.value):
                                return TypedExpression(BuiltInTypes.INT.value, e)
                            case (BuiltInTypes.STRING.value, BuiltInTypes.STRING.value):
                                return TypedExpression(BuiltInTypes.STRING.value, e)
                            case _:
                                self.errors.append(
                                    f"Invalid operation: {left_type.type.name} + {right_type.type.name}")
                    case "-":
                        match (left_type.type, right_type.type):
                            case (BuiltInTypes.INT.value, BuiltInTypes.INT.value):
                                return TypedExpression(BuiltInTypes.INT.value, e)
                            case _:
                                self.errors.append(
                                    f"Invalid operation: {left_type.type.name} - {right_type.type.name}")
            case MultiplicationExpression(left, sign, right) as e:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                match (left_type.type, right_type.type):
                    case (BuiltInTypes.INT.value, BuiltInTypes.INT.value):
                        return TypedExpression(BuiltInTypes.INT.value, e)
                    case _:
                        self.errors.append(
                            f"Invalid operation: {left_type.type.name} {sign} {right_type.type.name}")
            case ExponentiationExpression(left, sign, right) as e:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                match (left_type.type, right_type.type):
                    case (BuiltInTypes.INT.value, BuiltInTypes.INT.value):
                        return TypedExpression(BuiltInTypes.INT.value, e)
                    case _:
                        self.errors.append(
                            f"Invalid operation: {left_type.type} {sign} {right_type.type}"
                        )
            case ArrayExpression(elements) as e:
                element_types = [self.evaluate_type_expression(
                    element, scope) for element in elements]
                if len(element_types) == 0:
                    return TypedExpression(BuiltInTypes.NONE.value, e)
                if element_types.count(element_types[0]) == len(element_types):
                    return TypedExpression(ListType("list", element_types[0].type), e)
                self.errors.append(
                    f"Invalid operation: {element_types[0].type}[]"
                )
            case _:
                self.errors.append(f"Unknown expression: {expression}")

    def check_statement(self, statement: Statement, scope: Scope = None):
        if scope is None:
            scope = self.scope
        match statement:
            case ExpressionStatement(expression):
                self.evaluate_type_expression(expression, scope)
            case AssignmentStatement(identifier, expression):
                if scope.rec_contains(identifier.name):
                    self.errors.append(
                        f"Variable {identifier.name} already defined")
                else:
                    t = self.evaluate_type_expression(expression, scope)
                    scope.rec_set(identifier.name, t.type)
            case ForStatement(identifier, value, body):
                if scope.rec_contains(identifier.name):
                    self.errors.append(
                        f"Variable {identifier.name} already defined")
                else:
                    value_type = self.evaluate_type_expression(value, scope)
                    if not isinstance(value_type, ListType):
                        self.errors.append(
                            f"Invalid operation: {value_type.type}[]")
                    else:
                        inner_scope = scope.spawn_child()
                        inner_scope.rec_set(identifier.name, value_type.type)
                        self.evaluate_type_expression(body, inner_scope)
