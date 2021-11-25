import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

from smol.parser import (AdditionExpression, ArrayExpression,
                         AssignmentStatement, BlockExpression,
                         BooleanExpression, BreakExpression,
                         ComparisonExpression, ContinueExpression,
                         EqualityExpression, ExponentiationExpression,
                         Expression, ExpressionStatement, ForStatement,
                         FunctionCallExpression, FunctionDefinitionStatement,
                         IdentifierExpression, IfExpression, ImportStatement,
                         IntegerExpression, MultiplicationExpression,
                         NegationExpression, Parser, Program,
                         PropertyAccessExpression, RangeExpression, Statement,
                         StringExpression, StructDefinitionStatement,
                         WhileStatement)
from smol.tokenizer import Tokenizer
from smol.utils import Scope, StageContext, resolve_module_path

RETURN_TYPE = int | float | None | str | list["RETURN_TYPE"] | dict[str, "RETURN_TYPE"]


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass

# TODO: Add support for types inherited from checker


def iprint(value: str) -> None:
    print(value)


def istr(value: int) -> str:
    return str(value)


@dataclass()
class InterpreterContext(StageContext):
    module_cache: dict[str, dict[str, RETURN_TYPE]
                       ] = dataclasses.field(default_factory=dict)

    def copy(self):
        return InterpreterContext(
            current_directory=self.current_directory,
            current_file=self.current_file,
            import_stack=self.import_stack.copy(),
            module_cache=self.module_cache.copy()
        )


def iopen_file(path: str) -> dict[str, Any]:
    f = open(path, "r")
    return {
        "path": path,
        "read": f.read,
        "close": f.close,
        "seek": f.seek,
    }


class Interpreter:
    program: Program
    context: InterpreterContext
    scope: Scope[Any] = Scope.from_dict({
        'print': iprint,
        'str': istr
    })

    def __init__(self, program: Program, context: InterpreterContext):
        self.program = program
        self.context = context
        self.scope = Scope.from_dict({
            'print': iprint,
            'str': istr
        })

    def struct(self) -> Callable[..., dict[str, "RETURN_TYPE"]]:
        def struct_fn(**kwargs):
            return kwargs
        struct_fn.__name__ = "__struct__"
        return struct_fn

    def import_(self, name: str) -> dict[str, RETURN_TYPE]:
        # TODOOOOO: Get rid of this hack
        if name in ("std.file"):
            return {
                "File": self.struct(),
                "close_file": lambda file: file["close"](),
                "open_file": iopen_file,
                "read_file": lambda file: file["read"](),
                "seek_file": lambda file, offset: file["seek"](offset),
            }  # type: ignore
        if name in self.context.import_stack:
            raise ImportError(f"Recursive import: {name}")
        if name in self.context.module_cache:
            return self.context.module_cache[name]
        module_path = resolve_module_path(self.context.current_directory, name)
        # Tokenize module
        tokens = Tokenizer.from_file(module_path)
        # Parse module
        module = Parser.from_tokenizer(tokens)
        # Copy context
        new_context = self.context.copy()
        new_context.import_stack.append(name)
        # Create new interpreter
        interpreter = Interpreter(module.program(), new_context)
        # Run module
        interpreter.run()
        # Return module scope
        self.context.module_cache[name] = interpreter.scope
        return interpreter.scope

    def lr_evaluate(self,
                    lhs: Expression,
                    rhs: Expression,
                    scope: Scope = None) -> tuple[RETURN_TYPE, RETURN_TYPE]:
        if scope is None:
            scope = self.scope
        lhs_val = self.evaluate(lhs, scope)
        rhs_val = self.evaluate(rhs, scope)
        return lhs_val, rhs_val

    def evaluate(self, expression: Expression, scope: Scope = None) -> RETURN_TYPE:
        # TODO: assist runtime type checking with compile-time type checking
        if scope is None:
            scope = self.scope
        match expression:
            case BooleanExpression(value) | IntegerExpression(value) | StringExpression(value):
                return value
            case ExponentiationExpression(left, sign, right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                return lhs ** rhs
            case MultiplicationExpression(left, sign, right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                if sign == '*':
                    return lhs * rhs
                return lhs // rhs
            case AdditionExpression(left, sign, right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                if isinstance(lhs, str) and isinstance(rhs, str):
                    return lhs + rhs
                assert isinstance(lhs, (int, float)), f"{lhs} is not a number"
                assert isinstance(rhs, (int, float)), f"{rhs} is not a number"
                if sign == '+':
                    return lhs + rhs
                return lhs - rhs
            case ComparisonExpression(left, sign, right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
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
                lhs, rhs = self.lr_evaluate(left, right, scope)

                if sign == "=":
                    return lhs == rhs
                return lhs != rhs

            case NegationExpression(expression):
                value = self.evaluate(expression, scope)
                assert isinstance(value, (int, float)
                                  ), f"{value} is not a number"
                return -1 * value
            case PropertyAccessExpression(expression, property):
                value = self.evaluate(expression, scope)
                assert isinstance(value, dict), f"{value} is not a struct"
                return value[property]
            case FunctionCallExpression(object, args):
                object_val = self.evaluate(object, scope)
                assert isinstance(
                    object_val, Callable), f"{object_val} is not callable"
                pos, kwd = [], {}
                for arg in args:
                    if arg.name is None:
                        assert object_val.__name__ != "__struct__", f"{object_val} cannot be a struct"
                        pos.append(self.evaluate(arg.value, scope))
                    else:
                        kwd[arg.name] = self.evaluate(arg.value, scope)
                return object_val(*pos, **kwd)
            case IdentifierExpression(name):
                assert scope.rec_contains(
                    name), f"Undefined identifier: {name}"
                return scope.rec_get(name)
            case IfExpression(condition, then_block, else_ifs, else_block):
                if self.evaluate(condition, scope):
                    return self.evaluate(then_block, scope)
                for else_if in else_ifs:
                    if self.evaluate(else_if[0], scope):
                        return self.evaluate(else_if[1], scope)
                if else_block:
                    return self.evaluate(else_block, scope)
            case BlockExpression(statements):
                inner_scope = scope.spawn_child()
                last: RETURN_TYPE | None = None
                for statement in statements:
                    last = self.execute(statement, inner_scope)
                return last
            case ArrayExpression(values):
                return [self.evaluate(value, scope) for value in values]
            case RangeExpression(start, end, step):
                start_value = self.evaluate(start, scope)
                end_value = self.evaluate(end, scope)
                step_value = self.evaluate(step, scope)
                assert isinstance(start_value, (int)
                                  ), f"{start_value} is not a number"
                assert isinstance(
                    end_value, (int)), f"{end_value} is not a number"
                assert isinstance(
                    step_value, (int)), f"{step_value} is not a number"
                return list(range(start_value, end_value, step_value))
            case BreakExpression():
                raise BreakException()
            case ContinueExpression():
                raise ContinueException()
            case _:
                raise NotImplementedError(
                    f"Unsupported expression: {expression}")

    def execute(self, statement: Statement, scope: Scope) -> RETURN_TYPE:
        match statement:
            case AssignmentStatement(ident, expression):
                value = self.evaluate(expression, scope)
                scope.rec_set(ident.name, value)
                return value
            case ExpressionStatement(expression):
                return self.evaluate(expression, scope)
            case ForStatement(ident, value, body):
                values = self.evaluate(value, scope)
                if not isinstance(values, Iterable):
                    raise TypeError(f"{values} is not iterable")
                for val in values.__iter__():
                    scope.rec_set(ident.name, val)
                    try:
                        self.evaluate(body, scope)
                    except BreakException:
                        break
                    except ContinueException:
                        continue
            case WhileStatement(condition, body):
                while self.evaluate(condition, scope):
                    try:
                        self.evaluate(body, scope)
                    except BreakException:
                        break
                    except ContinueException:
                        continue
            case FunctionDefinitionStatement(name, fn_args, body):
                def fn(*args):
                    inner_scope = scope.spawn_child()
                    for arg, val in zip(fn_args, args):
                        inner_scope.rec_set(arg.name, val)
                    return self.evaluate(body, inner_scope)
                scope.rec_set(name, fn)
            case StructDefinitionStatement(name):
                # struct is technically typechecked in checker phase
                scope.rec_set(name, self.struct())
            case ImportStatement(name):
                assert scope.parent is None, f"Cannot import in inner scope"
                paths = name.split(".")
                mname = paths[-1]
                scope.rec_set(mname, self.import_(name))
            case _:
                raise NotImplementedError(
                    f"Unsupported statement: {statement}")

    def run(self) -> RETURN_TYPE | None:
        # Execute all statements and return last
        for statement in self.program.statements[:-1]:
            self.execute(statement, self.scope)
        if self.program.statements:
            return self.execute(self.program.statements[-1], self.scope)
