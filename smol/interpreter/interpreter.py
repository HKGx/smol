import dataclasses
from dataclasses import dataclass
from typing import Any, Callable
from smol.interpreter.overwrites import OVERWRITE_TABLE
from smol.interpreter.utils import RETURN_TYPE, BreakException, ContinueException

from smol.parser.expressions import *
from smol.parser.parser import Parser, Program
from smol.parser.statements import *
from smol.lexer.lexer import Lexer
from smol.utils import Scope, StageContext, resolve_module_path


@dataclass()
class InterpreterContext(StageContext):
    module_cache: dict[str, dict[str, RETURN_TYPE]] = dataclasses.field(
        default_factory=dict
    )

    def copy(self):
        return InterpreterContext(
            current_directory=self.current_directory,
            current_file=self.current_file,
            import_stack=self.import_stack.copy(),
            module_cache=self.module_cache.copy(),
        )


def overwrite_module(interpreter: "Interpreter", name: str) -> None:
    # FIXME: We need a way to directly use native code with auto-conversion
    if name in OVERWRITE_TABLE:
        OVERWRITE_TABLE[name](interpreter)


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
        # Copy context
        new_context = self.context.copy()
        new_context.current_file = module_path.name
        new_context.import_stack.append(name)
        # Lex module
        tokens = Lexer.from_file(module_path, new_context)
        # Parse module
        module = Parser.from_lexer(tokens)
        # Create new interpreter
        interpreter = Interpreter(module.program(), new_context)
        # Run module
        interpreter.run()
        overwrite_module(interpreter, name)
        # Return module scope
        self.context.module_cache[name] = interpreter.scope
        return interpreter.scope

    def lr_evaluate(
        self, lhs: Expression, rhs: Expression, scope: Scope = None
    ) -> tuple[RETURN_TYPE, RETURN_TYPE]:
        if scope is None:
            scope = self.scope
        lhs_val = self.evaluate(lhs, scope)
        rhs_val = self.evaluate(rhs, scope)
        assert isinstance(
            lhs_val, dict
        ), f"Left hand side of assignment must be a dict, got {lhs_val}"
        assert isinstance(
            rhs_val, dict
        ), f"Right hand side of assignment must be a dict, got {rhs_val}"
        return lhs_val, rhs_val

    def typeof(self, value: RETURN_TYPE) -> str:
        if isinstance(value, list):
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
        v_prop = "__value__"
        match expression:
            case IntegerExpression():
                i = int_c()
                i["__value__"] = expression.value  # type: ignore
                return i
            case StringExpression():
                i_len = int_c()
                i_len["__value__"] = len(expression.value)  # type: ignore
                s = string_c(length=i_len)
                s["__value__"] = expression.value  # type: ignore
                return s
            case BooleanExpression():
                b = bool_c()
                b["__value__"] = expression.value  # type: ignore
                return b
            case ExponentiationExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs[v_prop], rhs[v_prop]  # type: ignore
                assert self.typeof(lhs) == "int", f"{self.typeof(lhs)} is not int"
                assert self.typeof(rhs) == "int", f"{self.typeof(rhs)} is not int"
                i = int_c()
                i["__value__"] = l_val**r_val  # type: ignore
                return i
            case MultiplicationExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs[v_prop], rhs[v_prop]  # type: ignore
                assert self.typeof(lhs) == "int", f"{self.typeof(lhs)} is not int"
                assert self.typeof(rhs) == "int", f"{self.typeof(rhs)} is not int"
                i = int_c()
                if sign == "*":
                    i["__value__"] = l_val * r_val  # type: ignore
                else:
                    i["__value__"] = l_val // r_val  # type: ignore
                return i

            case AdditionExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs[v_prop], rhs[v_prop]  # type: ignore
                if sign == "-":
                    assert self.typeof(lhs) == "int", f"{self.typeof(lhs)} is not int"
                    assert self.typeof(rhs) == "int", f"{self.typeof(rhs)} is not int"
                    i = int_c()
                    i["__value__"] = l_val - r_val  # type: ignore
                    return i
                assert self.typeof(lhs) in (
                    "int",
                    "string",
                ), f"{self.typeof(lhs)} is not int or string, {expression.edges}"
                assert self.typeof(rhs) in (
                    "int",
                    "string",
                ), f"{self.typeof(rhs)} is not int or string"
                if self.typeof(lhs) == "int":
                    i = int_c()
                    i["__value__"] = l_val + r_val  # type: ignore
                    return i
                # type: ignore
                i_len = int_c()
                i_len["__value__"] = len(l_val) + len(r_val)
                s = string_c(length=i_len)
                s["__value__"] = l_val + r_val  # type: ignore
                return s
            case ComparisonExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs[v_prop], rhs[v_prop]  # type: ignore
                assert self.typeof(lhs) == "int", f"{self.typeof(lhs)} is not int"
                assert self.typeof(rhs) == "int", f"{self.typeof(rhs)} is not int"
                comparison_map = {
                    ">": "__gt__",
                    ">=": "__ge__",
                    "<": "__lt__",
                    "<=": "__le__",
                }

                fun = getattr(l_val, comparison_map[sign])  # type: ignore
                b = bool_c()
                b["__value__"] = fun(r_val)  # type: ignore
                return b

            case EqualityExpression(left=left, sign=sign, right=right):
                lhs, rhs = self.lr_evaluate(left, right, scope)
                l_val, r_val = lhs[v_prop], rhs[v_prop]  # type: ignore
                assert self.typeof(lhs) == self.typeof(
                    rhs
                ), f"{self.typeof(lhs)} != {self.typeof(rhs)}"
                comparison_map = {"==": "__eq__", "!=": "__ne__"}
                fun = getattr(l_val, comparison_map[sign])  # type: ignore
                b = bool_c()
                b["__value__"] = fun(r_val)  # type: ignore
                return b

            case NegationExpression():
                evaluated = self.evaluate(expression.value, scope)
                assert (
                    self.typeof(evaluated) == "int"
                ), f"{self.typeof(evaluated)} is not int"
                i = int_c()
                i["__value__"] = -evaluated[v_prop]  # type: ignore
                return i

            case PropertyAccessExpression(object=obj, property=property):
                value = self.evaluate(obj, scope)
                if isinstance(value, list):
                    # TODO: don't use interpreter magic in future
                    if property == "length":
                        i = int_c()
                        i["__value__"] = len(value)  # type: ignore
                        return i
                    if property == "push":

                        def ipush(to_push):
                            value.append(to_push)

                        return ipush  # type: ignore
                    if property == "set":

                        def iset(index, to_set):
                            assert (
                                self.typeof(index) == "int"
                            ), f"{self.typeof(index)} is not int"
                            value[index[v_prop]] = to_set

                        return iset  # type: ignore

                assert isinstance(value, dict), f"{value} is not a struct"
                return value[property]
            case ArrayAccessExpression(array=obj, index=index):
                value = self.evaluate(obj, scope)
                assert self.typeof(value) == "string" or isinstance(
                    value, list
                ), f"{value} is not a string or array"
                if isinstance(index, RangeExpression):
                    start, end = self.lr_evaluate(index.start, index.end, scope)
                    assert (
                        self.typeof(start) == "int"
                    ), f"{self.typeof(start)} is not int"
                    assert self.typeof(end) == "int", f"{self.typeof(end)} is not int"
                    start_val: int = start[v_prop]  # type: ignore
                    end_val: int = end[v_prop]  # type: ignore
                    if isinstance(value, list):
                        return value[start_val:end_val]
                    if self.typeof(value) == "string":
                        str_val: str = value[v_prop]  # type: ignore
                        slice_ = str_val[start_val:end_val]
                        i_len = int_c()
                        i_len["__value__"] = len(slice_)
                        val = string_c(length=i_len)
                        val["__value__"] = slice_  # type: ignore
                        return val  # type: ignore
                index_value = self.evaluate(index, scope)
                assert (
                    self.typeof(index_value) == "int"
                ), f"{self.typeof(index_value)} is not int"
                if isinstance(value, list):
                    return value[index_value[v_prop]]  # type: ignore
                if self.typeof(value) == "string":
                    str_val: str = value[v_prop]  # type: ignore
                    i_len = int_c()
                    i_len["__value__"] = len(str_val)
                    val = string_c(length=i_len)
                    val["__value__"] = str_val[index_value[v_prop]]  # type: ignore
                    return val  # type: ignore

            case FunctionCallExpression(object=object, args=args):
                object_val = self.evaluate(object, scope)
                assert isinstance(object_val, Callable), f"{object_val} is not callable"
                pos, kwd = [], {}
                for arg in args:
                    if arg.name is None:
                        assert (
                            object_val.__name__ != "__struct__"
                        ), f"{object_val} cannot be a struct"
                        pos.append(self.evaluate(arg.value, scope))
                    else:
                        kwd[arg.name] = self.evaluate(arg.value, scope)
                return object_val(*pos, **kwd)
            case IdentifierExpression(name=name):
                assert scope.rec_contains(
                    name
                ), f"Undefined identifier: {name} at {expression.edges}"
                val = scope.rec_get(name)
                if isinstance(val, (Callable, list)):
                    return val  # type: ignore
                if v_prop in val:
                    assert isinstance(
                        val[v_prop], (str, bool, int)
                    ), f"{val[v_prop]} is not a string, bool or int"
                return val
            case IfExpression(
                condition=condition, body=body, else_ifs=else_ifs, else_body=else_body
            ):
                condition_val = self.evaluate(condition, scope)
                assert (
                    self.typeof(condition_val) == "bool"
                ), f"{self.typeof(condition_val)} is not bool"
                if condition_val[v_prop]:  # type: ignore
                    return self.evaluate(body, scope)
                for else_if in else_ifs:
                    condition_val = self.evaluate(else_if[0], scope)
                    if condition_val[v_prop]:  # type: ignore
                        return self.evaluate(else_if[1], scope)
                if else_body:
                    return self.evaluate(else_body, scope)
                return none_c()
            case BlockExpression(body=statements):
                inner_scope = scope.spawn_child()
                last: RETURN_TYPE = none_c()
                for statement in statements:
                    last = self.execute(statement, inner_scope) or none_c()
                return last
            case ArrayExpression(elements=values):
                return [self.evaluate(value, scope) for value in values]
            case RangeExpression(start=start, end=end, step=step):
                start_value = self.evaluate(start, scope)[v_prop]  # type: ignore
                end_value = self.evaluate(end, scope)[v_prop]  # type: ignore
                step_value = (
                    self.evaluate(step, scope)[v_prop] if step else 1
                )  # type: ignore
                assert isinstance(start_value, int), f"{start_value} is not int"
                assert isinstance(end_value, int), f"{end_value} is not int"
                assert isinstance(step_value, int), f"{step_value} is not int"
                output = []
                for i in range(start_value, end_value, step_value):
                    ic = int_c()
                    ic["__value__"] = i
                    output.append(ic)
                return output
            case BreakExpression():
                raise BreakException()
            case ContinueExpression():
                raise ContinueException()
        raise NotImplementedError(f"Unsupported expression: {expression}")

    def execute_function_definition_statement(
        self, statement: FunctionDefinitionStatement, scope: Scope
    ):
        def fn(*args, **kwargs):
            inner_scope = scope.spawn_child()
            for arg, val in zip(statement.args, args):
                inner_scope.rec_set(arg.name, val)
            for arg, val in kwargs.items():
                inner_scope.rec_set(arg, val)
            return self.evaluate(statement.body, inner_scope)

        scope.rec_set(statement.name, fn)

    def execute_struct_definition_statement(
        self, statement: StructDefinitionStatement, scope: Scope
    ):
        def constructor(**kwargs):
            struct = {}
            struct["__constructor__"] = constructor
            for method in statement.methods:

                def make_fn(m):
                    def fn(*pos, **kwd):
                        inner_scope = scope.spawn_child()
                        inner_scope["self"] = struct
                        for arg, val in zip(m.args, pos):
                            inner_scope[arg.name] = val
                        for arg, val in kwd.items():
                            inner_scope[arg] = val
                        return self.evaluate(m.body, inner_scope)

                    return fn

                struct[method.name] = make_fn(method)
            for field in statement.fields:
                struct[field.name] = kwargs[field.name]
            return struct

        constructor.__name__ = statement.name
        scope.rec_set(statement.name, constructor)

    def execute_import_statement(self, statement: ImportStatement, scope: Scope):
        assert scope.parent is None, "Cannot import in inner scope"
        name = statement.name
        if name == "std/std" and self.context.current_file == "std.smol":
            return  # Don't import std/std if we're in std.smol
        module_name = name.split(".")[-1]
        module = self.import_(name)
        if statement.add_to_scope:
            for k, v in module.items():
                assert not scope.rec_contains(k), f"Duplicate identifier: {k}"
                scope.rec_set(k, v)
        scope.rec_set(module_name, module)

    def execute(self, statement: Statement, scope: Scope) -> RETURN_TYPE | None:
        v_prop = "__value__"
        match statement:
            case AssignmentStatement(ident, expression):
                value = self.evaluate(expression, scope)
                if v_prop in value:
                    val = value[v_prop]  # type: ignore
                    assert isinstance(
                        val, (str, bool, int)
                    ), f"{val} is not a string, bool or int"
                scope.rec_set(ident.name, value)
                return value
            case ExpressionStatement(expression):
                return self.evaluate(expression, scope)
            case ForStatement(ident, value, body):
                values = self.evaluate(value, scope)  # type: ignore
                assert isinstance(values, list), f"{values} is not a list"

                for val in values:
                    scope.rec_set(ident.name, val)
                    try:
                        self.evaluate(body, scope)
                    except BreakException:
                        break
                    except ContinueException:
                        continue
            case WhileStatement(condition, body):
                while self.evaluate(condition, scope)[v_prop]:  # type: ignore
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
                raise NotImplementedError(f"Unsupported statement: {statement}")

    def run(self) -> RETURN_TYPE | None:
        for module in self.program.imports:
            self.execute_import_statement(module, self.scope)
        for struct in self.program.structs:
            self.execute_struct_definition_statement(struct, self.scope)
        for fn in self.program.functions:
            self.execute_function_definition_statement(fn, self.scope)
        for statement in self.program.statements:
            if isinstance(
                statement,
                (
                    ImportStatement,
                    StructDefinitionStatement,
                    FunctionDefinitionStatement,
                ),
            ):
                continue
            self.execute(statement, self.scope)
