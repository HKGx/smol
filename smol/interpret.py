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

    def evaluate(self, expression: Expression) -> RETURN_TYPE:
        match expression:
            case IntegerExpression(value):
                return value
            case ExponentatiotnExpression(left, sign, right):
                lhs = self.evaluate(left)
                rhs = self.evaluate(right)
                assert lhs is not None and rhs is not None, f"Exponentiation expression evaluated to None: {expression}"
                return lhs ** rhs
            case MultiplicationExpression(left, sign, right):
                lhs = self.evaluate(left)
                rhs = self.evaluate(right)
                assert lhs is not None and rhs is not None, f"Multiplication expression evaluated to None: {expression}"
                if sign == '*':
                    return lhs * rhs
                return lhs / rhs
            case AdditionExpression(left, sign, right):
                lhs = self.evaluate(left)
                rhs = self.evaluate(right)
                assert lhs is not None and rhs is not None, f"Addition expression evaluated to None: {expression}"
                if sign == '+':
                    return lhs + rhs
                return lhs - rhs
            case ComparisonExpression(left, sign, right):
                lhs = self.evaluate(left)
                rhs = self.evaluate(right)
                assert lhs is not None and rhs is not None, f"Comparison expression evaluated to None: {expression}"
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
                assert value is not None, f"Negation expression evaluated to None: {expression}"
                return -1 * value
            case FunctionCallExpression(ident, args):
                assert ident.name in self.state, f"Function {ident.name} not found"
                return self.state[ident.name](*[self.evaluate(arg) for arg in args])
            case IdentifierExpression(name):
                assert name in self.state, f"Undefined identifier: {name}"
                return self.state[name]
            case IfExpression(condition, then_block, else_ifs, else_block):
                if self.evaluate(condition):
                    return self.evaluate(then_block)
                for else_if in else_ifs:
                    if self.evaluate(else_if[0]):
                        return self.evaluate(else_if[1])
                if else_block:
                    return self.evaluate(else_block)
            case BlockExpression(statements):
                state = self.state.copy()
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
                state[ident.name] = self.evaluate(expression)
                return state[ident.name]
            case ExpressionStatement(expression):
                return self.evaluate(expression)
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
