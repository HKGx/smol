from io import TextIOWrapper
from pprint import pprint
from smol.checker import Checker

from smol.interpret import Interpreter
from smol.parser import (AdditionExpression, ArrayExpression, AssignmentStatement,
                         BlockExpression, BreakExpression, ComparisonExpression, ContinueExpression,
                         EqualityExpression, ExponentiationExpression,
                         Expression, ExpressionStatement, ForStatement,
                         FunctionCallExpression, IdentifierExpression,
                         IfExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Parser,
                         Program, Statement, StringExpression)
from smol.tokenizer import Tokenizer


def program(prog: Program) -> str:
    return '\n'.join(statement(stmt) for stmt in prog.statements)


def statement(stmt: Statement, indent: int = 0) -> str:
    match stmt:
        case AssignmentStatement(name, value):
            return f"{stringify(name, indent)} = {stringify(value)}"
        case ExpressionStatement(expression):
            return stringify(expression, indent)
        case ForStatement(ident, expr, body):
            return f"for {stringify(ident)} in {stringify(expr)} do \n{stringify(body, indent + 1)}\nend"

    raise Exception(f"Unexpected statement: {stmt}")


def stringify(expression: Expression, indent: int = 0) -> str:
    # TODO: improve indentation
    match expression:
        case IdentifierExpression(value):
            return "\t" * indent + value
        case IntegerExpression(value):
            return "\t" * indent + str(value)
        case StringExpression(value):
            return "\t" * indent + f'"{value}"'
        case AdditionExpression(lhs, sign, rhs):
            return "\t" * indent + f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case MultiplicationExpression(lhs, sign, rhs):
            return "\t" * indent + f"({stringify(lhs)} {sign} {stringify(rhs)})"
        case NegationExpression(value):
            return "\t" * indent + f"(-{stringify(value)})"
        case ExponentiationExpression(lhs, sign, rhs):
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
        case ArrayExpression(values):
            return f"[{', '.join(stringify(value) for value in values)}]"
        case BreakExpression():
            return "\t" * indent + "break"
        case ContinueExpression():
            return "\t" * indent + "continue"
        case BlockExpression(statements):
            stmts = ('\n').join(statement(stmt, indent + 1)
                                for stmt in statements)
            return "\t" * indent + f"do\n{stmts}\n" + "\t" * indent + "end"

    raise Exception(f"Unexpected expression: {expression}")


def compile_file(file: TextIOWrapper, debug: bool = False):
    raise NotImplementedError("Compiling files is not implemented yet")


def check_file(file: TextIOWrapper, debug: bool = False):
    tokens = Tokenizer(file.read()).tokenize()
    if debug:
        pprint(tokens)
    prog = Parser(tokens).program()
    if debug:
        pprint(prog)
    checker = Checker(prog)
    pprint(checker.check())


def interpret_file(file: TextIOWrapper, debug: bool = False):
    tokens = Tokenizer(file.read()).tokenize()
    if debug:
        pprint(tokens)
    prog = Parser(tokens).program()
    if debug:
        pprint(prog)
    pprint(program(prog))
    interpreter = Interpreter(prog)
    pprint(interpreter.run())


def repl(debug: bool = False):
    while True:
        line = input('> ')
        if line == 'exit':
            break
        tokens = Tokenizer(line).tokenize()
        if debug:
            pprint(tokens)
        prog = Parser(tokens).program()
        if debug:
            pprint(prog)
        pprint(program(prog))
        checker = Checker(prog)
        pprint(checker.check())
        interpreter = Interpreter(prog)
        pprint(interpreter.run())


if __name__ == "__main__":
    # If none arguments are passed then invoke repl()
    # Otherwise based on the argument passed start interpreting or compilation of passed file
    import argparse
    parser = argparse.ArgumentParser(
        description='Interpreter for the smol language')
    parser.add_argument("run_type", nargs="?", choices=[
                        'repl', "r", 'interpret', "i", 'compile', "c", "check"], help="Run type")
    parser.add_argument('file', nargs='?',
                        help='File to interpret', type=open)
    parser.add_argument("--debug", "-d", action="store_true")
    parsed_args = parser.parse_args()
    match (parsed_args.file, parsed_args.run_type):
        case (None, 'repl' | "r"):
            repl(parsed_args.debug)
        case (None, _):
            print("No file specified")
        case (_, "interpret" | "i"):
            interpret_file(parsed_args.file, parsed_args.debug)
        case (_, "compile" | "c"):
            compile_file(parsed_args.file, parsed_args.debug)
        case (_, "check"):
            check_file(parsed_args.file, parsed_args.debug)
        case (_, _):
            raise Exception("Invalid run_type")
