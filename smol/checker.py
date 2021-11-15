import dataclasses
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

from smol.parser import (AdditionExpression, ArrayExpression,
                         AssignmentStatement, BlockExpression,
                         ComparisonExpression, EqualityExpression,
                         ExponentiationExpression, Expression,
                         ExpressionStatement, ForStatement,
                         FunctionCallExpression, FunctionDefinitionStatement, IdentifierExpression,
                         IfExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Program,
                         Statement, StringExpression, WhileStatement)
from smol.utils import Scope


@dataclass(eq=True, frozen=True)
class CheckerType:
    name: str
    meta: dict[str, bool] = dataclasses.field(
        init=False, default_factory=dict, compare=False)


@dataclass(eq=True, frozen=True)
class InvalidType(CheckerType):
    name = "invalid"


@dataclass(eq=True, frozen=True)
class ListType(CheckerType):
    type: CheckerType
    known_length: Optional[int]


@dataclass(eq=True, frozen=True)
class MappingType(CheckerType):
    from_type: CheckerType
    to_type: CheckerType


@dataclass(eq=True, frozen=True)
class TypedExpression:
    type: CheckerType
    value: Expression


@dataclass
class TypedStatement:
    type: CheckerType
    value: Statement


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
        "range": MappingType("function", BuiltInType.int, ListType("list", BuiltInType.int, None)),
    })

    def __init__(self, program: Program):
        self.program = program
        self.errors = []

    def check(self):
        self.check_program(self.program)
        return self.errors

    @property
    def has_errors(self):
        return len(self.errors) > 0

    def check_program(self, program: Program):
        for statement in program.statements:
            self.check_statement(statement)

    def lr_evaluate(self, lhs: Expression, rhs: Expression, scope: Scope = None) -> Tuple[TypedExpression, TypedExpression]:
        if scope is None:
            scope = self.scope
        lhs_type = self.evaluate_type_expression(lhs, scope)
        rhs_type = self.evaluate_type_expression(rhs, scope)
        return lhs_type, rhs_type

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
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                if l_typ.type != r_typ.type:
                    self.errors.append(
                        f"{expr} has different types: {l_typ.type} and {r_typ.type}")
                    return TypedExpression(BuiltInType.invalid, expr)
                return TypedExpression(BuiltInType.bool, expr)
            case ComparisonExpression(left, sign, right) as expr:
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                if l_typ.type != r_typ.type:
                    self.errors.append(
                        f"{expr} has different types: {l_typ.type} and {r_typ.type}")
                    return TypedExpression(BuiltInType.invalid, expr)
                return TypedExpression(BuiltInType.bool, expr)

            case AdditionExpression(left, sign,  right) as expr:
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match sign:
                    case "+":
                        match (l_typ.type, r_typ.type):
                            case (BuiltInType.int, BuiltInType.int):
                                return TypedExpression(BuiltInType.int, expr)
                            case (BuiltInType.string, BuiltInType.string):
                                return TypedExpression(BuiltInType.string, expr)

                    case "-":
                        match (l_typ.type, r_typ.type):
                            case (BuiltInType.int, BuiltInType.int):
                                return TypedExpression(BuiltInType.int, expr)

                self.errors.append(
                    f"Invalid operation: {l_typ.type.name} {sign} {r_typ.type.name}")
                return TypedExpression(BuiltInType.invalid, expr)
            case MultiplicationExpression(left, sign, right) as expr:
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match (l_typ.type, r_typ.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expr)

                self.errors.append(
                    f"Invalid operation: {l_typ.type.name} {sign} {r_typ.type.name}")
                return TypedExpression(BuiltInType.invalid, expr)
            case ExponentiationExpression(left, sign, right) as expr:
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match (l_typ.type, r_typ.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expr)

                self.errors.append(
                    f"Invalid operation: {l_typ.type} {sign} {r_typ.type}"
                )
                return TypedExpression(BuiltInType.invalid, expr)
            case NegationExpression(expression) as expr:
                typ = self.evaluate_type_expression(expression, scope)
                if typ.type != BuiltInType.int:
                    self.errors.append(
                        f"Invalid operation: -{typ.type}"
                    )
                    return TypedExpression(BuiltInType.invalid, expr)
                return TypedExpression(BuiltInType.int, expr)
            case IfExpression(condition, body, else_ifs, else_body) as expr:
                condition_type = self.evaluate_type_expression(
                    condition, scope)
                if condition_type.type != BuiltInType.bool:
                    self.errors.append(
                        f"Invalid condition: {condition_type.type}")
                    return TypedExpression(BuiltInType.invalid, expr)
                body_typ = self.evaluate_type_expression(
                    body, scope.spawn_child())
                if body_typ.type == BuiltInType.invalid:
                    self.errors.append(
                        f"Invalid if statement: {body} is not a valid expression"
                    )
                    return TypedExpression(BuiltInType.invalid, expr)
                body_types: list[CheckerType] = [body_typ.type]
                for else_if in else_ifs:
                    else_if_cond_type = self.evaluate_type_expression(
                        else_if[0], scope)
                    if else_if_cond_type.type == BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else if statement: {else_if[0]} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
                    else_if_body_type = self.evaluate_type_expression(
                        else_if[1], scope.spawn_child())
                    if else_if_body_type.type == BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else if statement: {else_if[1]} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
                    body_types.append(else_if_body_type.type)
                if else_body is not None:
                    else_body_type = self.evaluate_type_expression(
                        else_body, scope.spawn_child())
                    if else_body_type.type == BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else statement: {else_body} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expr)
                    body_types.append(else_body_type.type)

                # TODO: Need to discuss whether we allow multiple types in an if statement to be returned

                if body_types.count(body_types[0]) != len(body_types):
                    self.errors.append("Invalid types of if bodies")
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
                if not scope.rec_get(name.name):
                    self.errors.append(
                        f"Invalid function call: {name} is not a valid function")
                    return TypedExpression(BuiltInType.invalid, expr)

                function_type = scope.rec_get(name.name)
                if not isinstance(function_type, MappingType):
                    self.errors.append(
                        f"Invalid function call: {name} is not a valid function")
                    return TypedExpression(BuiltInType.invalid, expr)
                from_type = function_type.from_type
                to_type = function_type.to_type
                arg_types = [self.evaluate_type_expression(
                    arg, scope) for arg in args]
                if not all(arg_type.type == from_type for arg_type in arg_types):
                    self.errors.append(
                        f"Invalid operation: {name.name}({arg_types[0].type})"
                    )
                    return TypedExpression(BuiltInType.invalid, expr)
                return TypedExpression(to_type, expr)
            case BlockExpression(statements) as expr:
                for statement in statements[:-1]:
                    self.check_statement(statement, scope)
                stat = self.check_statement(statements[-1], scope)
                stat_type = stat.type
                if stat.type == BuiltInType.invalid:
                    return TypedExpression(BuiltInType.invalid, expr)
                if stat_type == BuiltInType.none:
                    return TypedExpression(BuiltInType.none, expr)
                return TypedExpression(stat_type, expr)

        self.errors.append(f"Unknown expression: {expression}")
        return TypedExpression(BuiltInType.invalid, expression)

    def check_expr_statement(self, statement: ExpressionStatement, scope: Scope) -> TypedStatement:
        typ = self.evaluate_type_expression(statement.value, scope)
        if typ == BuiltInType.invalid:
            self.errors.append(
                f"Invalid expression statement: {statement.value} is not a valid expression"
            )
            return TypedStatement(BuiltInType.invalid, statement)
        return TypedStatement(typ.type, statement)

    def check_assignment_statement(self, statement: AssignmentStatement, scope: Scope[CheckerType]) -> TypedStatement:
        ident_name = statement.name.name
        expr = statement.value
        if not scope.rec_contains(ident_name):
            typ = self.evaluate_type_expression(expr, scope)
            if statement.mutable:
                new_typ = dataclasses.replace(typ.type)  # copy type
                new_typ.meta["mut"] = True
                scope.rec_set(ident_name, new_typ)
            else:
                scope.rec_set(ident_name, typ.type)
            return TypedStatement(typ.type, statement)
        if statement.mutable:
            self.errors.append(
                f"Invalid assignment statement: {ident_name} is already defined."
                "`mut` modifier can only be used when defining variables.")

            return TypedStatement(BuiltInType.invalid, statement)
        ident_type = scope.rec_get(ident_name)
        expr_type = self.evaluate_type_expression(expr, scope)
        if ident_type == expr_type.type and ident_type.meta.get("mut"):
            scope.rec_set(ident_name, ident_type)
            return TypedStatement(ident_type, statement)
        if ident_type != expr_type.type and ident_type.meta.get("mut"):
            self.errors.append(
                f"Type mismatch: Tried assigning {expr_type.type} to {ident_name} of type {ident_type}")
            return TypedStatement(BuiltInType.invalid, statement)
        self.errors.append(
            f"Invalid assignment statement: {ident_name} was assigned before and is not mutable"
        )
        return TypedStatement(BuiltInType.invalid, statement)

    def check_while_statement(self, statement: WhileStatement, scope: Scope) -> TypedStatement:
        condition = statement.condition
        body = statement.body
        typ = self.evaluate_type_expression(condition, scope)
        if typ.type != BuiltInType.bool:
            self.errors.append(
                f"Invalid while condition: {typ} is not a valid boolean expression")
            return TypedStatement(BuiltInType.invalid, statement)
        self.evaluate_type_expression(body, scope)
        return TypedStatement(BuiltInType.none, statement)

    def check_for_statement(self, statement: ForStatement, scope: Scope) -> TypedStatement:
        ident_name = statement.ident.name
        value = statement.value
        body = statement.body
        if scope.rec_contains(ident_name):
            self.errors.append(
                f"Variable {ident_name} already defined")
            return TypedStatement(BuiltInType.invalid, statement)
        iterable_type = self.evaluate_type_expression(
            value, scope).type

        if not isinstance(iterable_type, ListType):
            self.errors.append(
                f"Type {iterable_type} is not iterable")
            return TypedStatement(BuiltInType.invalid, statement)
        inner_scope = scope.spawn_child()

        # pylint: disable=no-member
        inner_scope.rec_set(ident_name, iterable_type.type)
        self.evaluate_type_expression(body, inner_scope)
        return TypedStatement(BuiltInType.none, statement)

    def check_statement(self, statement: Statement, scope: Scope = None) -> TypedStatement:
        if scope is None:
            scope = self.scope
        match statement:
            case ExpressionStatement():
                return self.check_expr_statement(statement, scope)
            case AssignmentStatement():
                return self.check_assignment_statement(statement, scope)
            case ForStatement():
                return self.check_for_statement(statement, scope)
            case WhileStatement():
                return self.check_while_statement(statement, scope)
            case FunctionDefinitionStatement(name, args, body):
                # TODO: blocked by upstream, need to implement typings
                if scope.rec_contains(name.name):
                    self.errors.append(
                        f"Invalid function definition: {name} is already defined")
                    return TypedStatement(BuiltInType.invalid, statement)
                scope.rec_set(name.name, MappingType(
                    "function", BuiltInType.int, BuiltInType.int))
                return TypedStatement(BuiltInType.none, statement)

        raise NotImplementedError(f"Unknown statement: {statement}")
