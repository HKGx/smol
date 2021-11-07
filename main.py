from smol.tokenizer import Tokenizer
from smol.parser import AdditionExpression, ExponentatiotnExpression, Expression, FunctionCallExpression, IdentifierExpression, IntegerExpression, MultiplicationExpression, NegationExpression, Parser
from pprint import pprint


def stringify(expression: Expression) -> str:
    match expression:
        case IdentifierExpression(value):
            return value
        case IntegerExpression(value):
            return str(value)
        case AdditionExpression(lhs, sign, rhs):
            return f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case MultiplicationExpression(lhs, sign, rhs):
            return f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case NegationExpression(value):
            return f"(-{stringify(value)})"
        case ExponentatiotnExpression(lhs, sign, rhs):
            return f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case FunctionCallExpression(name, args):
            return f"{stringify(name)}({', '.join(stringify(arg) for arg in args)})"
    raise Exception(f"Unexpected expression: {expression}")

if __name__ == "__main__":
    tokens = Tokenizer("3^2^2^2 * (2 + 2 + 2 + 2)").tokenize()
    pprint(tokens)
    parser = Parser(tokens)
    expr = parser.expression()
    pprint(expr)
    print(stringify(expr))