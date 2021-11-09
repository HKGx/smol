from io import TextIOWrapper
from pprint import pprint
from smol.parser import (AdditionExpression, AssignmentStatement,
                         ExponentatiotnExpression, Expression,
                         ExpressionStatement, FunctionCallExpression,
                         IdentifierExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Parser,
                         Program, Statement)
from smol.tokenizer import Tokenizer


def program(prog: Program) -> str:
    return '\n'.join(statement(stmt) for stmt in prog.statements)


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


def compile_file(file: TextIOWrapper):
    pass


def interpret_file(file: TextIOWrapper):
    tokens = Tokenizer(file.read()).tokenize()
    prog = Parser(tokens).program()
    print(program(prog))
    print(prog.execute())


def repl():
    while True:
        try:
            line = input('> ')
            if line == 'exit':
                break
            tokens = Tokenizer(line).tokenize()
            prog = Parser(tokens).program()
            print(program(prog))
            print(prog.execute())
        except Exception as exception:
            print(exception)


if __name__ == "__main__":
    # If none arguments are passed then invoke repl()
    # Otherwise based on the argument passed start interpreting or compilation of passed file
    import argparse
    parser = argparse.ArgumentParser(
        description='Interpreter for the smol language')
    parser.add_argument("run_type", nargs="?", choices=[
                        'repl', "r", 'interpret', "i", 'compile', "c`"], help="Run type")
    parser.add_argument('file', nargs='?', help='File to interpret', type=open)
    prased_args = parser.parse_args()
    # If no args or run_type == "repl"
    if not prased_args.run_type or prased_args.run_type == "repl":
        repl()
    # If file not provided raise error
    if not prased_args.file:
        raise Exception("File not provided")
    if prased_args.run_type == "interpret":
        interpret_file(prased_args.file)
    elif prased_args.run_type == "compile":
        compile_file(prased_args.file)
