from smol.interpret import Interpreter
from smol.tokenizer import TokenType, Tokenizer
from smol.parser import AdditionExpression, AssignmentStatement, ExponentatiotnExpression, Expression, ExpressionStatement, FunctionCallExpression, IdentifierExpression, IntegerExpression, MultiplicationExpression, NegationExpression, Parser, Program, Statement
from pprint import pprint

def program(program: Program) -> str:
    return '\n'.join(statement(stmt) for stmt in program.statements)

def statement(stmt: Statement) -> str:
    match stmt:
        case AssignmentStatement(name, value):
            return f"{stringify(name)} = {stringify(value)}"
        case ExpressionStatement(expression):
            return stringify(expression)
    raise Exception(f"Unexpected statement: {stmt}")


# TODO: Implement comparisons
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
    print("/ᐠ｡ꞈ｡ᐟ\\")
    print("Tokenizing")
    tokens = Tokenizer("""
    print(x != 0)""").tokenize()
    # pprint(tokens)
    print("Parsing")
    parser = Parser(tokens)
    expr = parser.program()
    #pprint(expr)
    #print(program(expr))
    print("Interpreting")
    interpreter = Interpreter(expr)
    interpreter.run()