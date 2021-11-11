from typing import Any

from smol.parser import (AdditionExpression, AssignmentStatement,
                         ComparisonExpression, EqualityExpression,
                         ExponentatiotnExpression, Expression,
                         ExpressionStatement, FunctionCallExpression,
                         IdentifierExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Program,
                         Statement)


RETURN_TYPE = int | float


class Interpreter:
    program: Program
    state: dict[str, Any] = {
        'print': print,
    }

    def __init__(self, program: Program):
        self.program = program

    def evaluate(self, expression: Expression) -> RETURN_TYPE:
        # TODO: Implement blocks and ifs
        match expression:
            case IntegerExpression(value):
                return value
            case ExponentatiotnExpression(left, sign, right):
                return self.evaluate(left) ** self.evaluate(right)
            case MultiplicationExpression(left, sign, right):
                if sign == '*':
                    return self.evaluate(left) * self.evaluate(right)
                else:
                    return self.evaluate(left) / self.evaluate(right)
            case AdditionExpression(left, sign, right):
                if sign == '+':
                    return self.evaluate(left) + self.evaluate(right)
                else:
                    return self.evaluate(left) - self.evaluate(right)
            case ComparisonExpression(left, sign, right):
                lhs = self.evaluate(left)
                rhs = self.evaluate(right)
                comparison_map = {
                    ">": "__gt__",
                    ">=": "__ge__",
                    "<": "__lt__",
                    "<=": "__le__"
                }
                fun = getattr(lhs, comparison_map[sign])
                return fun(rhs)

            case EqualityExpression(left, sign, right):
                if sign == "=":
                    return self.evaluate(left) == self.evaluate(right)
                return self.evaluate(left) != self.evaluate(right)

            case NegationExpression(expression):
                return -1 * self.evaluate(expression)
            case FunctionCallExpression(ident, args):
                assert ident.name in self.state, f"Function {ident.name} not found"
                return self.state[ident.name](*[self.evaluate(arg) for arg in args])
            case IdentifierExpression(name):
                assert name in self.state, f"Undefined identifier: {name}"
                return self.state[name]
            case _:
                raise NotImplementedError(
                    f"Unsupported expression: {expression}")

    def execute(self, statement: Statement, state: dict[str, Any]) -> RETURN_TYPE:
        match statement:
            case AssignmentStatement(ident, expression):
                state[ident.name] = self.evaluate(expression)
                return state[ident.name]
            case ExpressionStatement(expression):
                return self.evaluate(expression)
            case _:
                raise NotImplementedError(
                    f"Unsupported statement: {statement}")

    def run(self) -> RETURN_TYPE:
        # Execute all statements and return last
        last: Any = None
        for statement in self.program.statements:
            last = self.execute(statement, self.state)
        return last
