from typing import Any

from smol.parser import (AdditionExpression, AssignmentStatement, BlockExpression,
                         ComparisonExpression, EqualityExpression,
                         ExponentatiotnExpression, Expression,
                         ExpressionStatement, FunctionCallExpression,
                         IdentifierExpression, IfExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Program,
                         Statement)


RETURN_TYPE = int | float | None


class Interpreter:
    program: Program
    state: dict[str, Any] = {
        'print': print,
    }

    def __init__(self, program: Program):
        self.program = program

    def evaluate(self, expression: Expression, state: dict[str, Any] = None) -> RETURN_TYPE:
        if state is None:
            state = self.state
        match expression:
            case IntegerExpression(value):
                return value
            case ExponentatiotnExpression(left, sign, right):
                lhs = self.evaluate(left, state)
                rhs = self.evaluate(right, state)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                return lhs ** rhs
            case MultiplicationExpression(left, sign, right):
                lhs = self.evaluate(left, state)
                rhs = self.evaluate(right, state)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                if sign == '*':
                    return lhs * rhs
                return lhs / rhs
            case AdditionExpression(left, sign, right):
                lhs = self.evaluate(left, state)
                rhs = self.evaluate(right, state)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                if sign == '+':
                    return lhs + rhs
                return lhs - rhs
            case ComparisonExpression(left, sign, right):
                lhs = self.evaluate(left, state)
                rhs = self.evaluate(right, state)
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
                lhs = self.evaluate(left, state)
                rhs = self.evaluate(right, state)

                if sign == "=":
                    return lhs == rhs
                return lhs != rhs

            case NegationExpression(expression):
                value = self.evaluate(expression, state)
                assert isinstance(value, (int, float)
                                  ), f"{value} is not a number"
                return -1 * value
            case FunctionCallExpression(ident, args):
                assert ident.name in state, f"Function {ident.name} not found"
                return state[ident.name](*[self.evaluate(arg, state) for arg in args])
            case IdentifierExpression(name):
                assert name in state, f"Undefined identifier: {name}"
                return state[name]
            case IfExpression(condition, then_block, else_ifs, else_block):
                if self.evaluate(condition):
                    return self.evaluate(then_block, state)
                for else_if in else_ifs:
                    if self.evaluate(else_if[0], state):
                        return self.evaluate(else_if[1], state)
                if else_block:
                    return self.evaluate(else_block, state)
            case BlockExpression(statements):
                state = state.copy()
                last: RETURN_TYPE | None = None
                for statement in statements:
                    last = self.execute(statement, state)
                return last
            case _:
                raise NotImplementedError(
                    f"Unsupported expression: {expression}")

    def execute(self, statement: Statement, state: dict[str, Any]) -> RETURN_TYPE:
        match statement:
            case AssignmentStatement(ident, expression):
                state[ident.name] = self.evaluate(expression, state)
                return state[ident.name]
            case ExpressionStatement(expression):
                return self.evaluate(expression, state)
            # TODO: implement for loops
            case _:
                raise NotImplementedError(
                    f"Unsupported statement: {statement}")

    def run(self) -> RETURN_TYPE | None:
        # Execute all statements and return last
        last: RETURN_TYPE | None = None
        for statement in self.program.statements:
            last = self.execute(statement, self.state)
        return last
