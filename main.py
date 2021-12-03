from dataclasses import dataclass
from importlib.resources import path
from io import TextIOWrapper
from pathlib import Path
from pprint import pprint

from smol.checker import Checker, CheckerContext
from smol.interpreter import Interpreter, InterpreterContext
from smol.parser import Parser
from smol.lexer import Lexer
from smol.utils import StageContext


@dataclass(frozen=True)
class SmolContext:
    file: TextIOWrapper | None
    path: Path | None
    debug: bool = False
    no_checker: bool = False


def compile_file(ctx: SmolContext):
    raise NotImplementedError("Compiling files is not implemented yet")


def check_file(ctx: SmolContext):
    assert ctx.file is not None, "File is not specified"
    assert ctx.path is not None, "Path is not specified"
    kw_context = {
        'current_directory': ctx.path.parent,
        'current_file': ctx.path.name,
    }
    context = StageContext(**kw_context)
    lexer = Lexer.from_file(ctx.file, context)
    parser = Parser.from_lexer(lexer)
    parsed = parser.parse()
    context = CheckerContext(**kw_context)
    checker = Checker(parsed, context)
    for error in checker.check():
        print(error)
    if not checker.has_errors:
        print("No errors found")


def interpret_file(ctx: SmolContext):
    assert ctx.file is not None, "File is not specified"
    assert ctx.path is not None, "Path is not specified"
    kw_context = {
        'current_directory': ctx.path.parent,
        'current_file': ctx.path.name,
    }
    context = StageContext(**kw_context)
    lexer = Lexer.from_file(ctx.file, context)
    parser = Parser.from_lexer(lexer)
    program = parser.parse()
    icontext = InterpreterContext(**kw_context)
    ccontext = CheckerContext(**kw_context)
    if ctx.no_checker:
        interpreter = Interpreter(program, icontext)
        pprint(interpreter.run())
        return
    checker = Checker(program, ccontext)
    for error in checker.check():
        print(error)
    if not checker.has_errors:
        interpreter = Interpreter(program, icontext)
        pprint(interpreter.run())


def repl(ctx: SmolContext):
    content: list[str] = []
    while True:
        line = input('> ' if len(content) == 0 else '. ')
        # if two last lines are empty and user presses enter, we execute the program
        if line != '' or not (len(content) > 0 and content[-1] == ''):
            content.append(line)
            continue
        joined = '\n'.join(content)
        context = StageContext(current_directory=Path.cwd())
        lexer = Lexer(joined, context)
        parser = Parser.from_lexer(lexer)
        parsed = parser.parse()
        icontext = InterpreterContext(current_directory=Path.cwd())
        ccontext = CheckerContext(current_directory=Path.cwd())
        checker = Checker(parsed, ccontext)
        for error in checker.check():
            print(error)
        if not checker.has_errors:
            print("No errors found during typecheking")
            interpreter = Interpreter(parsed, icontext)
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
    file_path: Path | None = Path(
        parsed_args.file.name).resolve() if parsed_args.file else None
    smol_context = SmolContext(
        parsed_args.file, file_path, parsed_args.debug, parsed_args.no_checker)
    match (parsed_args.file, parsed_args.run_type):
        case (None, 'repl' | "r" | None):
            repl(smol_context)
        case (None, _):
            print("No file specified")
        case (_, "interpret" | "i"):
            interpret_file(smol_context)
        case (_, "compile" | "c"):
            compile_file(smol_context)
        case (_, "check"):
            check_file(smol_context)
        case (_, _):
            raise Exception("Invalid run_type")
