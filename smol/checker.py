from dataclasses import dataclass
from typing import NamedTuple, Optional

from smol.utils import Scope
from smol.parser import (AdditionExpression, ArrayExpression,
                         AssignmentStatement, BlockExpression, ComparisonExpression, EqualityExpression, ExponentiationExpression,
                         Expression, ExpressionStatement, ForStatement,
                         FunctionCallExpression, IdentifierExpression,
                         IfExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Program,
                         Statement, StringExpression)


@dataclass
class CheckerType:
    name: str


@dataclass
class InvalidType(CheckerType):
    name = "invalid"


@dataclass
class ListType(CheckerType):
    type: CheckerType
    known_length: Optional[int]


@dataclass
class MappingType(CheckerType):
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


class BuiltInType(NamedTuple):
    int = CheckerType("int")
    string = CheckerType("string")
    bool = CheckerType("bool")
    none = CheckerType("none")
    invalid = InvalidType("invalid")


class Checker:
    program: Program
    errors: list[str] = []
    scope: Scope = Scope.from_dict({
        "print": MappingType("function", BuiltInType.string, BuiltInType.none),
        "str": MappingType("function", BuiltInType.int, BuiltInType.string),
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
        # TODO: Improve upon errors as they're really confusing
        if scope is None:
            scope = self.scope
        match expression:
            case IntegerExpression() as expr:
                return TypedExpression(BuiltInType.int, expr)
            case StringExpression() as expr:
                return TypedExpression(BuiltInType.string, expr)
            case IdentifierExpression(name) as expr:
                return TypedExpression(scope.rec_get(name), expr)
            case EqualityExpression(left, sign, right) as expr:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                if left_type.type != right_type.type:
                    self.errors.append(
                        f"{expr} has different types: {left_type.type} and {right_type.type}")
                    return TypedExpression(BuiltInType.invalid, expr)
                return TypedExpression(BuiltInType.bool, expr)
            case ComparisonExpression(left, sign, right) as expr:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                if left_type.type != right_type.type:
                    self.errors.append(
                        f"{expr} has different types: {left_type.type} and {right_type.type}")
                    return TypedExpression(BuiltInType.invalid, expr)
                return TypedExpression(BuiltInType.bool, expr)

            case AdditionExpression(left, sign,  right) as expr:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                match sign:
                    case "+":
                        match (left_type.type, right_type.type):
                            case (BuiltInType.int, BuiltInType.int):
                                return TypedExpression(BuiltInType.int, expr)
                            case (BuiltInType.string, BuiltInType.string):
                                return TypedExpression(BuiltInType.string, expr)
                            case _:
                                self.errors.append(
                                    f"Invalid operation: {left_type.type.name} + {right_type.type.name}")
                    case "-":
                        match (left_type.type, right_type.type):
                            case (BuiltInType.int, BuiltInType.int):
                                return TypedExpression(BuiltInType.int, expr)
                            case _:
                                self.errors.append(
                                    f"Invalid operation: {left_type.type.name} - {right_type.type.name}")
                                return TypedExpression(BuiltInType.invalid, expr)
            case MultiplicationExpression(left, sign, right) as expr:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                match (left_type.type, right_type.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expr)
                    case _:
                        self.errors.append(
                            f"Invalid operation: {left_type.type.name} {sign} {right_type.type.name}")
                        return TypedExpression(BuiltInType.invalid, expr)
            case ExponentiationExpression(left, sign, right) as expr:
                left_type = self.evaluate_type_expression(left, scope)
                right_type = self.evaluate_type_expression(right, scope)
                match (left_type.type, right_type.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expr)
                    case _:
                        self.errors.append(
                            f"Invalid operation: {left_type.type} {sign} {right_type.type}"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
            case NegationExpression(expression) as expr:
                body_typ = self.evaluate_type_expression(expression, scope)
                match body_typ.type:
                    case BuiltInType.int:
                        return TypedExpression(BuiltInType.int, expr)
                    case _:
                        self.errors.append(
                            f"Invalid operation: -{body_typ.type}"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
            case IfExpression(condition, body, else_ifs, else_body) as expr:
                condition_type = self.evaluate_type_expression(
                    condition, scope)
                if condition_type.type != BuiltInType.bool:
                    self.errors.append(
                        f"Invalid condition: {condition_type.type}")
                    return TypedExpression(BuiltInType.invalid, expr)
                body_typ = self.evaluate_type_expression(
                    body, scope.spawn_child())
                if body_typ is BuiltInType.invalid:
                    self.errors.append(
                        f"Invalid if statement: {body} is not a valid expression"
                    )
                    return TypedExpression(BuiltInType.invalid, expr)
                body_types: list[CheckerType] = [body_typ.type]
                for else_if in else_ifs:
                    else_if_cond_type = self.evaluate_type_expression(
                        else_if[0], scope)
                    if else_if_cond_type is BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else if statement: {else_if[0]} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
                    else_if_body_type = self.evaluate_type_expression(
                        else_if[1], scope.spawn_child())
                    if else_if_body_type is BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else if statement: {else_if[1]} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
                    body_types.append(else_if_body_type.type)
                if else_body is not None:
                    else_body_type = self.evaluate_type_expression(
                        else_body, scope.spawn_child())
                    if else_body_type is BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else statement: {else_body} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
                    body_types.append(else_body_type.type)

                # TODO: Need to discuss whether we allow multiple types in an if statement to be returned

                if body_types.count(body_types[0]) != len(body_types):
                    self.errors.append(
                        f"Invalid types of if bodies")
                    return TypedExpression(BuiltInType.invalid, expr)

                return TypedExpression(body_types[0], expr)
            case ArrayExpression(elements) as expr:
                element_types = [self.evaluate_type_expression(
                    element, scope) for element in elements]
                if len(element_types) == 0:
                    return TypedExpression(ListType("list", BuiltInType.none, 0), expr)
                if element_types.count(element_types[0]) == len(element_types):
                    return TypedExpression(ListType("list", element_types[0].type, len(element_types)), expr)
                self.errors.append(
                    f"Invalid operation: {element_types[0].type}[]"
                )
                return TypedExpression(BuiltInType.invalid, expr)
            case FunctionCallExpression(name, args) as expr:
                function_type = scope.rec_get(name.name)
                match function_type:
                    case MappingType(_, from_type, to_type):
                        arg_types = [self.evaluate_type_expression(
                            arg, scope) for arg in args]
                        if not all(arg_type.type == from_type for arg_type in arg_types):
                            self.errors.append(
                                f"Invalid operation: {name.name}({arg_types[0].type})"
                            )
                            return TypedExpression(BuiltInType.invalid, expr)
                        return TypedExpression(to_type, expr)
                    case _:
                        self.errors.append(
                            f"Invalid operation: {name.name}({function_type.name})"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
            case BlockExpression(statements) as expr:
                for statement in statements[:-1]:
                    self.check_statement(statement, scope)
                stat_type = self.check_statement(statements[-1], scope)
                if stat_type is BuiltInType.invalid:
                    self.errors.append(
                        f"Invalid block: {statements[-1]} is not a valid statement"
                    )
                    return TypedExpression(BuiltInType.invalid, expr)
                if stat_type is None:
                    return TypedExpression(BuiltInType.none, expr)
                return TypedExpression(stat_type, expr)

        self.errors.append(f"Unknown expression: {expression}")
        return TypedExpression(BuiltInType.invalid, expression)

    def check_statement(self, statement: Statement, scope: Scope = None) -> Optional[CheckerType]:
        if scope is None:
            scope = self.scope
        match statement:
            case ExpressionStatement(expression):
                typ = self.evaluate_type_expression(expression, scope)
                return typ.type
            case AssignmentStatement(identifier, expression):
                if scope.rec_contains(identifier.name):
                    self.errors.append(
                        f"Variable {identifier.name} already defined")
                else:
                    typ = self.evaluate_type_expression(expression, scope)
                    scope.rec_set(identifier.name, typ.type)
                    return typ.type
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
