from io import TextIOWrapper
from pprint import pprint

from smol.checker import Checker
from smol.interpret import Interpreter
from smol.parser import Parser
from smol.tokenizer import Tokenizer


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
    for error in checker.check():
        print(error)
    if not checker.has_errors:
        print("No errors found")


def interpret_file(file: TextIOWrapper, debug: bool = False, no_checker: bool = False):
    tokens = Tokenizer(file.read()).tokenize()
    if debug:
        pprint(tokens)
    prog = Parser(tokens).program()
    if debug:
        pprint(prog)
    if no_checker:
        interpreter = Interpreter(prog)
        pprint(interpreter.run())
        return
    checker = Checker(prog)
    for error in checker.check():
        print(error)
    if not checker.has_errors:
        interpreter = Interpreter(prog)
        pprint(interpreter.run())


def repl(debug: bool = False):
    content: list[str] = []
    while True:
        line = input('> ' if len(content) == 0 else '. ')
        # if two last lines are empty and user presses enter, we execute the program
        if line != '' or not (len(content) > 0 and content[-1] == ''):
            content.append(line)
            continue
        joined = '\n'.join(content)
        tokens = Tokenizer(joined).tokenize()
        if debug:
            pprint(tokens)
        prog = Parser(tokens).program()
        if debug:
            pprint(prog)
        checker = Checker(prog)
        for error in checker.check():
            print(error)
        if not checker.has_errors:
            print("No errors found during typecheking")
            interpreter = Interpreter(prog)
            pprint(interpreter.run())
        content = []


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
    parser.add_argument("--no-checker", action="store_true")
    parsed_args = parser.parse_args()
    match (parsed_args.file, parsed_args.run_type):
        case (None, 'repl' | "r"):
            repl(parsed_args.debug)
        case (None, _):
            print("No file specified")
        case (_, "interpret" | "i"):
            interpret_file(parsed_args.file, parsed_args.debug,
                           parsed_args.no_checker)
        case (_, "compile" | "c"):
            compile_file(parsed_args.file, parsed_args.debug)
        case (_, "check"):
            check_file(parsed_args.file, parsed_args.debug)
        case (_, _):
            raise Exception("Invalid run_type")
