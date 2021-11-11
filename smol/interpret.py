from typing import Any, Optional
from collections.abc import Iterable
from smol.parser import (AdditionExpression, ArrayExpression, AssignmentStatement, BlockExpression, BreakExpression,
                         ComparisonExpression, ContinueExpression, EqualityExpression,
                         ExponentiationExpression, Expression,
                         ExpressionStatement, ForStatement, FunctionCallExpression,
                         IdentifierExpression, IfExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Program,
                         Statement, StringExpression)


RETURN_TYPE = int | float | None | str | list["RETURN_TYPE"]


class Scope(dict[str, Any]):
    parent: Optional["Scope"] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        new = cls()
        for k, v in d.items():
            new[k] = v
        return new

    def __init__(self, parent: "Scope" = None):
        self.parent = parent

    def rec_contains(self, o: object) -> bool:
        if o in self:
            return True
        if self.parent is None:
            return False
        return self.parent.rec_contains(o)

    def rec_get(self, key: str):
        if key in self:
            return self[key]
        if self.parent is None:
            raise KeyError(key)
        return self.parent.rec_get(key)

    def rec_set(self, key: str, value: Any) -> bool:
        if self.parent is None:
            self[key] = value
            return True
        if self.parent is not None:
            return self.parent.rec_set(key, value)
        return False

    def spawn_child(self):
        return Scope(parent=self)


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


class Interpreter:
    program: Program
    scope: Scope = Scope.from_dict({
        'print': print,
        'range': range,
        'str': str
    })

    def __init__(self, program: Program):
        self.program = program

    def evaluate(self, expression: Expression, scope: Scope = None) -> RETURN_TYPE:
        if scope is None:
            scope = self.scope
        match expression:
            case IntegerExpression(value):
                return value
            case ExponentiationExpression(left, sign, right):
                lhs = self.evaluate(left, scope)
                rhs = self.evaluate(right, scope)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                return lhs ** rhs
            case MultiplicationExpression(left, sign, right):
                lhs = self.evaluate(left, scope)
                rhs = self.evaluate(right, scope)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                if sign == '*':
                    return lhs * rhs
                return lhs / rhs
            case AdditionExpression(left, sign, right):
                lhs = self.evaluate(left, scope)
                rhs = self.evaluate(right, scope)
                if isinstance(lhs, str) and isinstance(rhs, str):
                    return lhs + rhs
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                if sign == '+':
                    return lhs + rhs
                return lhs - rhs
            case ComparisonExpression(left, sign, right):
                lhs = self.evaluate(left, scope)
                rhs = self.evaluate(right, scope)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                comparison_map = {
                    ">": "__gt__",
                    ">=": "__ge__",
                    "<": "__lt__",
                    "<=": "__le__"
                }
                fun = getattr(lhs, comparison_map[sign])
                return fun(rhs)

            case EqualityExpression(left, sign, right):
                lhs = self.evaluate(left, scope)
                rhs = self.evaluate(right, scope)

                if sign == "=":
                    return lhs == rhs
                return lhs != rhs

            case NegationExpression(expression):
                value = self.evaluate(expression, scope)
                assert isinstance(value, (int, float)
                                  ), f"{value} is not a number"
                return -1 * value
            case FunctionCallExpression(ident, args):
                assert scope.rec_contains(
                    ident.name), f"Function {ident.name} not found"
                return scope.rec_get(ident.name)(*[self.evaluate(arg, scope) for arg in args])
            case IdentifierExpression(name):
                assert scope.rec_contains(
                    name), f"Undefined identifier: {name}"
                return scope.rec_get(name)
            case IfExpression(condition, then_block, else_ifs, else_block):
                if self.evaluate(condition):
                    return self.evaluate(then_block, scope)
                for else_if in else_ifs:
                    if self.evaluate(else_if[0], scope):
                        return self.evaluate(else_if[1])
                if else_block:
                    return self.evaluate(else_block)
            case BlockExpression(statements):
                inner_scope = scope.spawn_child()
                last: RETURN_TYPE | None = None
                for statement in statements:
                    last = self.execute(statement, inner_scope)
                return last
            case ArrayExpression(values):
                return [self.evaluate(value, scope) for value in values]
            case BreakExpression():
                raise BreakException()
            case ContinueExpression():
                raise ContinueException()
            case StringExpression(value):
                return value
            case _:
                raise NotImplementedError(
                    f"Unsupported expression: {expression}")

    def execute(self, statement: Statement, scope: Scope) -> RETURN_TYPE:
        match statement:
            case AssignmentStatement(ident, expression):
                value = self.evaluate(expression, scope)
                scope.rec_set(ident.name, value)
                return value
            case ExpressionStatement(expression):
                return self.evaluate(expression, scope)
            case ForStatement(ident, value, body):
                values = self.evaluate(value, scope)
                if not isinstance(values, Iterable):
                    raise TypeError(f"{values} is not iterable")
                for val in values.__iter__():
                    scope.rec_set(ident.name, val)
                    try:
                        self.evaluate(body, scope)
                    except BreakException:
                        break
                    except ContinueException:
                        continue
            case _:
                raise NotImplementedError(
                    f"Unsupported statement: {statement}")

    def run(self) -> RETURN_TYPE | None:
        # Execute all statements and return last
        last: RETURN_TYPE | None = None
        for statement in self.program.statements:
            last = self.execute(statement, self.scope)
        return last
