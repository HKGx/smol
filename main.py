from smol.tokenizer import Tokenizer
from smol.parser import AdditionExpression, Expression, IdentifierExpression, IntegerExpression, MultiplicationExpression, NegationExpression, Parser
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
    raise Exception(f"Unexpected expression: {expression}")

if __name__ == "__main__":
    tokens = Tokenizer("2 + 3 + 3123 * -Πa4 - 3").tokenize()
    pprint(tokens)
    parser = Parser(tokens)
    expr = parser.expression()
    print(stringify(expr))
