from io import TextIOWrapper
from pprint import pprint

from smol.interpret import Interpreter
from smol.parser import (AdditionExpression, AssignmentStatement,
                         BlockExpression, ComparisonExpression,
                         EqualityExpression, ExponentatiotnExpression,
                         Expression, ExpressionStatement,
                         FunctionCallExpression, IdentifierExpression,
                         IfExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Parser,
                         Program, Statement)
from smol.tokenizer import Tokenizer


def program(prog: Program) -> str:
    return '\n'.join(statement(stmt) for stmt in prog.statements)


def statement(stmt: Statement, indent: int = 0) -> str:
    match stmt:
        case AssignmentStatement(name, value):
            return f"{stringify(name, indent)} = {stringify(value)}"
        case ExpressionStatement(expression):
            return stringify(expression, indent)
    raise Exception(f"Unexpected statement: {stmt}")


def stringify(expression: Expression, indent: int = 0) -> str:
    # TODO: improve indentation
    match expression:
        case IdentifierExpression(value):
            return "\t" * indent + value
        case IntegerExpression(value):
            return "\t" * indent + str(value)
        case AdditionExpression(lhs, sign, rhs):
            return "\t" * indent + f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case MultiplicationExpression(lhs, sign, rhs):
            return "\t" * indent + f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case NegationExpression(value):
            return "\t" * indent + f"(-{stringify(value)})"
        case ExponentatiotnExpression(lhs, sign, rhs):
            return "\t" * indent + f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case FunctionCallExpression(name, args):
            return "\t" * indent + f"{stringify(name)}({', '.join(stringify(arg) for arg in args)})"
        case ComparisonExpression(left, sign, right):
            return "\t" * indent + f"{stringify(left)} {sign} {stringify(right)}"
        case EqualityExpression(left, sign, right):
            return "\t" * indent + f"{stringify(left)} {sign} {stringify(right)}"
        case IfExpression(condition, body, elifs, else_body):
            main_branch = "\t" * indent + \
                f"if {stringify(condition)}:\n{stringify(body, indent + 1)}"
            elif_branches = ("\t" * indent + '\n').join(
                f"elif {stringify(elif_[0])}:\n{stringify(elif_[1], indent + 1)}" for elif_ in elifs)
            else_branch = f"else:\n{stringify(else_body, indent + 1)}" if else_body else ''
            return f"{main_branch}\n{elif_branches}\n{else_branch}"
        case BlockExpression(statements):
            stmts = ('\n').join(statement(stmt, indent + 1)
                                for stmt in statements)
            return "\t" * indent + f"do\n{stmts}\n" + "\t" * indent + "end"

    raise Exception(f"Unexpected expression: {expression}")


def compile_file(file: TextIOWrapper):
    raise NotImplementedError("Compiling files is not implemented yet")


def interpret_file(file: TextIOWrapper):
    tokens = Tokenizer(file.read()).tokenize()
    prog = Parser(tokens).program()
    print(program(prog))
    interpreter = Interpreter(prog)
    print(interpreter.run())


def repl():
    while True:
        line = input('> ')
        if line == 'exit':
            break
        tokens = Tokenizer(line).tokenize()
        prog = Parser(tokens).program()
        print(program(prog))
        interpreter = Interpreter(prog)
        print(interpreter.run())


if __name__ == "__main__":
    # If none arguments are passed then invoke repl()
    # Otherwise based on the argument passed start interpreting or compilation of passed file
    import argparse
    parser = argparse.ArgumentParser(
        description='Interpreter for the smol language')
    parser.add_argument("run_type", nargs="?", choices=[
                        'repl', "r", 'interpret', "i", 'compile', "c`"], help="Run type")
    parser.add_argument('file', nargs='?',
                        help='File to interpret', type=open)
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
