import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
import os
from typing import Any, Callable
from smol.interpreter.utils import RETURN_TYPE, BreakException, ContinueException

from smol.parser.expressions import *
from smol.parser.parser import Parser, Program
from smol.parser.statements import *
from smol.lexer import Lexer
from smol.utils import Scope, StageContext, resolve_module_path


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


def overwrite_module(interpreter: "Interpreter", name: str) -> None:
    # FIXME: it's a mess, but it works for now
    # We need a way to directly use native code with auto-conversion
    string_type = interpreter.scope.rec_get("string")
    if name == "std.file":

        file_struct = interpreter.scope.rec_get("File")

        def iopen(path: RETURN_TYPE):
            file = file_struct(path=path)
            file["__file__"] = open(path["value"], "r")  # type: ignore
            file["read"] = lambda: string_type(
                value=file["__file__"].read())  # type: ignore
            file["seek"] = lambda i: file["__file__"].seek(
                i["value"])  # type: ignore
            file["close"] = lambda: file["__file__"].close()  # type: ignore
            return file
        interpreter.scope.rec_set("open_file", iopen)
    if name == "std.os":
        def ishell(value: RETURN_TYPE):
            os.system(value["value"])  # type: ignore
        interpreter.scope.rec_set("shell", ishell)
    if name == "std.std":
        def istr(value: RETURN_TYPE):
            return string_type(value=str(value["value"]))  # type: ignore

        def iprint(value: RETURN_TYPE):
            print(value["value"])  # type: ignore
        interpreter.scope.rec_set("str", istr)
        interpreter.scope.rec_set("print", iprint)


class Interpreter:
    program: Program
    context: InterpreterContext
    scope: Scope[Any]

    def __init__(self, program: Program, context: InterpreterContext):
        self.program = program
        self.context = context
        self.scope = Scope()

    def import_(self, name: str) -> dict[str, RETURN_TYPE]:
        if name in self.context.import_stack:
            raise ImportError(f"Recursive import: {name}")
        if name in self.context.module_cache:
            return self.context.module_cache[name]
        module_path = resolve_module_path(self.context.current_directory, name)
        # Lex module
        tokens = Lexer.from_file(module_path)
        # Parse module
        module = Parser.from_lexer(tokens)
        # Copy context
        new_context = self.context.copy()
        new_context.current_file = module_path.name
        new_context.import_stack.append(name)
        # Create new interpreter
        interpreter = Interpreter(module.program(), new_context)
        # Run module
        interpreter.run()
        overwrite_module(interpreter, name)  # Overwrite std.std and std.file.
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

    def typeof(self, value: RETURN_TYPE) -> str:
        if isinstance(value, list):
            print(value)
            if len(value) == 0:
                return "list.none"
            return "list." + self.typeof(value[0])
        return value["__constructor__"].__name__  # type: ignore

    def evaluate(self, expression: Expression, scope: Scope = None) -> RETURN_TYPE:
        # TODO: assist runtime type checking with compile-time type checking
        if scope is None:
            scope = self.scope
        bool_c = scope.rec_get("bool")
        int_c = scope.rec_get("int")
        float_c = scope.rec_get("float")
        string_c = scope.rec_get("string")
        none_c = scope.rec_get("none")
        match expression:
            case IntegerExpression():
                return int_c(value=expression.value)
            case StringExpression():
                return string_c(value=expression.value)
            case BooleanExpression():
                return bool_c(value=expression.value)
            case ExponentiationExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs["value"], rhs["value"]  # type: ignore
                assert self.typeof(
                    lhs) == "int", f"{self.typeof(lhs)} is not int"
                assert self.typeof(
                    rhs) == "int", f"{self.typeof(rhs)} is not int"
                return int_c(value=l_val ** r_val)  # type: ignore
            case MultiplicationExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs["value"], rhs["value"]  # type: ignore
                assert self.typeof(
                    lhs) == "int", f"{self.typeof(lhs)} is not int"
                assert self.typeof(
                    rhs) == "int", f"{self.typeof(rhs)} is not int"

                if sign == "*":
                    return int_c(value=l_val * r_val)  # type: ignore
                return int_c(value=l_val // r_val)  # type: ignore

            case AdditionExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs["value"], rhs["value"]  # type: ignore
                if sign == "-":
                    assert self.typeof(
                        lhs) == "int", f"{self.typeof(lhs)} is not int"
                    assert self.typeof(
                        rhs) == "int", f"{self.typeof(rhs)} is not int"
                    return int_c(value=l_val - r_val)   # type: ignore
                assert self.typeof(lhs) in (
                    "int", "string"), f"{self.typeof(lhs)} is not int or string, {expression.edges}"
                assert self.typeof(rhs) in (
                    "int", "string"), f"{self.typeof(rhs)} is not int or string"
                if self.typeof(lhs) == "int":
                    return int_c(value=l_val + r_val)  # type: ignore
                # type: ignore
                return string_c(value=l_val + r_val)  # type: ignore
            case ComparisonExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs["value"], rhs["value"]  # type: ignore
                assert self.typeof(
                    lhs) == "int", f"{self.typeof(lhs)} is not int"
                assert self.typeof(
                    rhs) == "int", f"{self.typeof(rhs)} is not int"
                comparison_map = {
                    ">": "__gt__",
                    ">=": "__ge__",
                    "<": "__lt__",
                    "<=": "__le__"
                }

                fun = getattr(l_val, comparison_map[sign])  # type: ignore
                return bool_c(value=fun(r_val))  # type: ignore

            case EqualityExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs["value"], rhs["value"]  # type: ignore
                assert self.typeof(
                    lhs) == self.typeof(rhs), f"{self.typeof(lhs)} != {self.typeof(rhs)}"
                comparison_map = {
                    "==": "__eq__",
                    "!=": "__ne__"
                }
                fun = getattr(l_val, comparison_map[sign])  # type: ignore
                return bool_c(value=fun(r_val))  # type: ignore

            case NegationExpression():
                evaluated = self.evaluate(expression.value, scope)
                assert self.typeof(
                    evaluated) == "int", f"{self.typeof(evaluated)} is not int"
                return int_c(value=-evaluated["value"])  # type: ignore

            case PropertyAccessExpression(object=obj, property=property):
                value = self.evaluate(obj, scope)
                assert isinstance(value, dict), f"{value} is not a struct"
                return value[property]
            case FunctionCallExpression(object=object, args=args):
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
            case IdentifierExpression(name=name):
                assert scope.rec_contains(
                    name), f"Undefined identifier: {name}"
                val = scope.rec_get(name)
                return val
            case IfExpression(condition=condition, body=then_block, else_ifs=else_ifs, else_body=else_block):
                condition_val = self.evaluate(condition, scope)
                if condition_val["value"]:  # type: ignore
                    return self.evaluate(then_block, scope)
                for else_if in else_ifs:
                    condition_val = self.evaluate(else_if[0], scope)
                    if condition_val["value"]:  # type: ignore
                        return self.evaluate(else_if[1], scope)
                if else_block:
                    return self.evaluate(else_block, scope)
            case BlockExpression(body=statements):
                inner_scope = scope.spawn_child()
                last: RETURN_TYPE = none_c()
                for statement in statements:
                    last = self.execute(statement, inner_scope) or none_c()
                return last
            case ArrayExpression(elements=values):
                return [self.evaluate(value, scope) for value in values]
            case RangeExpression(left=start, right=end, step=step):
                start_value = self.evaluate(start, scope)
                end_value = self.evaluate(end, scope)
                step_value = self.evaluate(step, scope)
                assert isinstance(start_value, (int)
                                  ), f"{start_value} is not a number"
                assert isinstance(
                    end_value, (int)), f"{end_value} is not a number"
                assert isinstance(
                    step_value, (int)), f"{step_value} is not a number"
                return [int_c(value=i) for i in range(start_value, end_value, step_value)]
            case BreakExpression():
                raise BreakException()
            case ContinueExpression():
                raise ContinueException()
        raise NotImplementedError(
            f"Unsupported expression: {expression}")

    def execute_function_definition_statement(self, statement: FunctionDefinitionStatement, scope: Scope):
        def fn(*args):
            inner_scope = scope.spawn_child()
            for arg, val in zip(statement.args, args):
                inner_scope.rec_set(arg.name, val)
            return self.evaluate(statement.body, inner_scope)
        scope.rec_set(statement.name, fn)

    def execute_struct_definition_statement(self, statement: StructDefinitionStatement, scope: Scope):
        def constructor(**kwargs):
            struct = {}
            struct["__constructor__"] = constructor
            for method in statement.methods:
                def fn(*args, **kwargs):
                    inner_scope = scope.spawn_child()
                    inner_scope["self"] = struct
                    for arg, val in zip(method.args, args):
                        inner_scope[arg.name] = val
                    return self.evaluate(method.body, inner_scope)
                struct[method.name] = fn
            for field in statement.fields:
                struct[field.name] = kwargs[field.name]
            return struct
        constructor.__name__ = statement.name
        scope.rec_set(statement.name, constructor)

    def execute_import_statement(self, statement: ImportStatement, scope: Scope):
        assert scope.parent is None, f"Cannot import in inner scope"
        name = statement.name
        if name == "std.std" and self.context.current_file == "std.smol":
            return  # Don't import std.std if we're in std.smol
        module_name = name.split(".")[-1]
        module = self.import_(name)
        if statement.add_to_scope:
            for k, v in module.items():
                assert not scope.rec_contains(k), f"Duplicate identifier: {k}"
                scope.rec_set(k, v)
        scope.rec_set(module_name, module)

    def execute(self, statement: Statement, scope: Scope) -> RETURN_TYPE | None:
        match statement:
            case AssignmentStatement(ident, expression):
                value = self.evaluate(expression, scope)
                scope.rec_set(ident.name, value)
                return value
            case ExpressionStatement(expression):
                return self.evaluate(expression, scope)
            case ForStatement(ident, value, body):
                values = self.evaluate(value, scope)  # type: ignore
                if not isinstance(values, Iterable):
                    raise TypeError(f"{values} is not iterable")
                for val in values:
                    scope.rec_set(ident.name, val)
                    try:
                        self.evaluate(body, scope)
                    except BreakException:
                        break
                    except ContinueException:
                        continue
            case WhileStatement(condition, body):
                while self.evaluate(condition, scope)["value"]:  # type: ignore
                    try:
                        self.evaluate(body, scope)
                    except BreakException:
                        break
                    except ContinueException:
                        continue
            case FunctionDefinitionStatement():
                self.execute_function_definition_statement(statement, scope)
            case StructDefinitionStatement():
                self.execute_struct_definition_statement(statement, scope)
            case ImportStatement():
                self.execute_import_statement(statement, scope)
            case _:
                raise NotImplementedError(
                    f"Unsupported statement: {statement}")

    def run(self) -> RETURN_TYPE | None:
        for module in self.program.imports:
            self.execute_import_statement(module, self.scope)
        for struct in self.program.structs:
            self.execute_struct_definition_statement(struct, self.scope)
        for fn in self.program.functions:
            self.execute_function_definition_statement(fn, self.scope)
        for statement in self.program.statements:
            if isinstance(statement, (ImportStatement, StructDefinitionStatement, FunctionDefinitionStatement)):
                continue
            self.execute(statement, self.scope)
