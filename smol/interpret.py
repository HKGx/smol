from typing import Any

from smol.parser import (AdditionExpression, AssignmentStatement,
                         ComparisonExpression, EqualityExpression,
                         ExponentatiotnExpression, Expression,
                         ExpressionStatement, FunctionCallExpression,
                         IdentifierExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Program,
                         Statement)


class Interpreter:
    program: Program
    state: dict[str, Any] = {
        'print': print,
    }

    def __init__(self, program: Program):
        self.program = program

    def evaluate(self, expression: Expression) -> Any:
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
                value = self.evaluate(expression)
                assert value is int
                return value
            case FunctionCallExpression(ident, args):
                return self.state[ident.name](*[self.evaluate(arg) for arg in args])
            case IdentifierExpression(name):
                return self.state[name]

    def execute(self, statement: Statement, state: dict[str, Any]) -> Any:
        match statement:
            case AssignmentStatement(ident, expression):
                state[ident.name] = self.evaluate(expression)
                return state[ident.name]
            case ExpressionStatement(expression):
                return self.evaluate(expression)

    def run(self) -> Any:
        # Execute all statements and return last
        last: Any = None
        for statement in self.program.statements:
            last = self.execute(statement, self.state)
        return last
