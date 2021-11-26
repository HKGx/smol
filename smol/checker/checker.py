from dataclasses import dataclass, field, replace as dataclass_replace
from typing import NamedTuple
from smol.checker.checker_type import *
from smol.parser.expressions import *
from smol.parser.statements import *
from smol.parser.parser import Parser, Program
from smol.lexer import Lexer

from smol.utils import Scope, SourcePositionable, StageContext, resolve_module_path


class BuiltInType(NamedTuple):
    int = CheckerType("int")
    string = CheckerType("string")
    bool = CheckerType("bool")
    none = CheckerType("none")
    invalid = InvalidType("invalid")


@dataclass
class CheckerContext(StageContext):
    module_cache: dict[str, ModuleType] = field(default_factory=dict)

    def copy(self):
        return CheckerContext(
            current_directory=self.current_directory,
            current_file=self.current_file,
            import_stack=self.import_stack[:],
            module_cache=self.module_cache.copy()
        )


class Checker:
    program: Program
    context: CheckerContext
    _errors: list[str]
    scope: Scope[CheckerType]

    def __init__(self, program: Program, context: CheckerContext):
        self.program = program
        self.context = context
        self._errors = []
        self.scope = Scope.from_dict({  # type: ignore
            "print": FunctionType("print", (FunctionArgumentType("to_print", BuiltInType.string),), BuiltInType.none),
            "str": FunctionType("str", (FunctionArgumentType("from", BuiltInType.int),), BuiltInType.string)
        })

    def error(self, message: str, expr: SourcePositionable = None) -> None:
        if expr is not None:
            message += f" at {expr.source_position()}"
        self._errors.append(message)

    def check(self):
        self.check_program(self.program)
        return self._errors

    @property
    def has_errors(self):
        return len(self._errors) > 0

    def check_program(self, program: Program):
        """
        Check a program in two cycles. First add all the types to the scope, then check the program.
        """
        # Check imports first and add them to the scope
        for import_statement in program.imports:
            self.check_import_statement(
                import_statement, self.scope, is_def=True)
        assert not self.has_errors, "Import errors"
        # Add struct types to the scope
        for struct_definition in program.structs:
            self.check_struct_definition_statement(
                struct_definition, self.scope, is_def=True)
            assert not self.has_errors, "Struct errors"
        # Add function types to the scope
        for function_definition in program.functions:
            self.check_function_definition_statement(
                function_definition, self.scope, is_def=True)
            assert not self.has_errors, "Function errors"
        for statement in program.statements:
            self.check_statement(statement)

    def lr_evaluate(self, lhs: Expression, rhs: Expression, scope: Scope = None) -> tuple[TypedExpression, TypedExpression]:
        if scope is None:
            scope = self.scope
        lhs_type = self.evaluate_type_expression(lhs, scope)
        rhs_type = self.evaluate_type_expression(rhs, scope)
        return lhs_type, rhs_type

    def evaluate_type(self, t_expression: TypeExpression, scope: Scope) -> Optional[CheckerType]:
        match t_expression:
            case TypeBuiltInExpression("int"): return BuiltInType.int
            case TypeBuiltInExpression("string"): return BuiltInType.string
            case TypeBuiltInExpression("bool"): return BuiltInType.bool
            case TypeBuiltInExpression("none"): return BuiltInType.none
            case TypeIdentifierExpression(name):
                if not scope.rec_contains(name):
                    self.error(f"Unknown type: {name}", t_expression)
                    return BuiltInType.invalid
                typ = scope.rec_get(name)
                if not isinstance(typ, StructType):
                    self.error(f"Type {name} is not a struct", t_expression)
                    return BuiltInType.invalid
                return typ
            case TypeDeduceExpression(): return None
        raise NotImplementedError(
            f"Unsupported type expression: {t_expression}")

    def evaluate_type_expression(self, expression: Expression, scope: Scope[CheckerType] = None) -> TypedExpression:
        # TODO: Improve upon errors as they're really confusing
        if scope is None:
            scope = self.scope
        match expression:
            case IntegerExpression():
                return TypedExpression(BuiltInType.int, expression)
            case BooleanExpression():
                return TypedExpression(BuiltInType.bool, expression)
            case StringExpression():
                return TypedExpression(BuiltInType.string, expression)
            case IdentifierExpression(name):
                if not scope.rec_contains(name):
                    self.error(
                        f"Identifier {name} is not defined in scope", expression)
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(scope.rec_get(name), expression)
            case EqualityExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                if l_typ.type != r_typ.type:
                    self.error(
                        f"{expression} has different types: {l_typ.type} and {r_typ.type}", expression)
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(BuiltInType.bool, expression)
            case ComparisonExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                if l_typ.type != r_typ.type:
                    self.error(
                        f"{expression} has different types: {l_typ.type} and {r_typ.type}", expression)
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(BuiltInType.bool, expression)

            case AdditionExpression(left, sign,  right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match sign:
                    case "+":
                        match (l_typ.type, r_typ.type):
                            case (BuiltInType.int, BuiltInType.int):
                                return TypedExpression(BuiltInType.int, expression)
                            case (BuiltInType.string, BuiltInType.string):
                                return TypedExpression(BuiltInType.string, expression)

                    case "-":
                        match (l_typ.type, r_typ.type):
                            case (BuiltInType.int, BuiltInType.int):
                                return TypedExpression(BuiltInType.int, expression)

                self.error(
                    f"Invalid operation: {l_typ.type.name} {sign} {r_typ.type.name}", expression)
                return TypedExpression(BuiltInType.invalid, expression)
            case MultiplicationExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match (l_typ.type, r_typ.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expression)

                self.error(
                    f"Invalid operation: {l_typ.type.name} {sign} {r_typ.type.name}", expression)
                return TypedExpression(BuiltInType.invalid, expression)
            case ExponentiationExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match (l_typ.type, r_typ.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expression)

                self.error(
                    f"Invalid operation: {l_typ.type} {sign} {r_typ.type}", expression)
                return TypedExpression(BuiltInType.invalid, expression)
            case NegationExpression(inner_expr):
                typ = self.evaluate_type_expression(inner_expr, scope)
                if typ.type != BuiltInType.int:
                    self.error(
                        f"Invalid operation: -{typ.type}", expression
                    )
                    return TypedExpression(BuiltInType.invalid, inner_expr)
                return TypedExpression(BuiltInType.int, expression)
            case IfExpression(condition, body, else_ifs, else_body):
                condition_type = self.evaluate_type_expression(
                    condition, scope)
                if condition_type.type != BuiltInType.bool:
                    self.error(
                        f"Invalid condition: {condition_type.type}", condition)
                    return TypedExpression(BuiltInType.invalid, expression)
                body_typ = self.evaluate_type_expression(
                    body, scope.spawn_child())
                if body_typ.type == BuiltInType.invalid:
                    self.error(
                        f"Invalid if statement: {body} is not a valid expression", body
                    )
                    return TypedExpression(BuiltInType.invalid, expression)
                body_types: list[CheckerType] = [body_typ.type]
                for else_if in else_ifs:
                    else_if_cond_type = self.evaluate_type_expression(
                        else_if[0], scope)
                    if else_if_cond_type.type == BuiltInType.invalid:
                        self.error(
                            f"Invalid else if condition: {else_if[0]} is not a valid expression", else_if[0]
                        )
                        return TypedExpression(BuiltInType.invalid, expression)
                    else_if_body_type = self.evaluate_type_expression(
                        else_if[1], scope.spawn_child())
                    if else_if_body_type.type == BuiltInType.invalid:
                        self.error(
                            f"Invalid else if body: {else_if[1]} is not a valid expression", else_if[1]
                        )
                        return TypedExpression(BuiltInType.invalid, expression)
                    body_types.append(else_if_body_type.type)
                if else_body is not None:
                    else_body_type = self.evaluate_type_expression(
                        else_body, scope.spawn_child())
                    if else_body_type.type == BuiltInType.invalid:
                        self.error(
                            f"Invalid else statement: {else_body} is not a valid expression", else_body
                        )
                        return TypedExpression(BuiltInType.invalid, expression)
                    body_types.append(else_body_type.type)

                # TODO: Need to discuss whether we allow multiple types in an if statement to be returned

                if body_types.count(body_types[0]) != len(body_types):
                    self.error(
                        "Invalid types of if bodies in if statement", expression)
                    return TypedExpression(BuiltInType.invalid, expression)

                return TypedExpression(body_types[0], expression)
            case ArrayExpression(elements):
                elements = [self.evaluate_type_expression(
                    element, scope) for element in elements]
                element_types = [element.type for element in elements]
                if len(element_types) == 0:
                    return TypedExpression(ListType("list", BuiltInType.none, 0), expression)
                if element_types.count(element_types[0]) == len(element_types):
                    return TypedExpression(ListType("list", element_types[0], len(element_types)), expression)
                self.error("Invalid types of array elements", expression)
                return TypedExpression(BuiltInType.invalid, expression)
            case PropertyAccessExpression(obj, property):
                typ = self.evaluate_type_expression(obj, scope)
                if isinstance(typ.type, ModuleType):
                    if property in typ.type.types:
                        return TypedExpression(typ.type.types[property], expression)
                    self.error(
                        f"Invalid property access: {property} is not a valid property of {typ.type.name}", expression
                    )
                    return TypedExpression(BuiltInType.invalid, expression)
                if not isinstance(typ.type, StructType):
                    self.error(
                        f"Invalid operation: {typ.type} has no property {property}", expression)
                    return TypedExpression(BuiltInType.invalid, obj)
                member_typ = typ.type.get(property)
                if member_typ is None:
                    self.error(
                        f"Invalid operation: {typ.type} has no property {property}", expression)
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(member_typ.type, expression)
            case FunctionCallExpression(object):
                typ = self.evaluate_type_expression(object, scope)
                function = typ.type
                match function:
                    case FunctionType():
                        return self.function_call_from_function(function, expression, scope)
                    case StructType():
                        return self.struct_constructor_call(function, expression, scope)
                    case _:
                        self.error(
                            f"Invalid function call: {function.name} is not a valid function", expression)
                        return TypedExpression(BuiltInType.invalid, expression)

            case BlockExpression(statements):
                inner_scope = scope.spawn_child()
                for statement in statements[:-1]:
                    self.check_statement(statement, inner_scope)
                if len(statements) == 0:
                    return TypedExpression(BuiltInType.none, expression)
                stat = self.check_statement(statements[-1], inner_scope)
                stat_type = stat.type
                if stat.type == BuiltInType.invalid:
                    return TypedExpression(BuiltInType.invalid, expression)
                if stat_type == BuiltInType.none:
                    return TypedExpression(BuiltInType.none, expression)
                return TypedExpression(stat_type, expression)
            case RangeExpression(start, end, step):
                start_type = self.evaluate_type_expression(start, scope)
                end_type = self.evaluate_type_expression(end, scope)
                step_type = self.evaluate_type_expression(step, scope)
                if start_type.type != BuiltInType.int or end_type.type != BuiltInType.int or step_type.type != BuiltInType.int:
                    self.error(
                        f"Invalid range: {start_type.type} to {end_type.type} by {step_type.type}", expression)
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(ListType("list", BuiltInType.int, None), expression)

        raise NotImplementedError(f"{expression}")

    def struct_constructor_call(self, struct: StructType, expression: FunctionCallExpression, scope: Scope[CheckerType]) -> TypedExpression:
        if len(expression.args) != len(struct.fields):
            self.error(
                f"Invalid struct constructor call: invalid count of members", expression)
            return TypedExpression(BuiltInType.invalid, expression)
        for arg in expression.args:
            if arg.name is None:
                self.error(
                    f"Invalid struct constructor call: invalid member name", expression)
                return TypedExpression(BuiltInType.invalid, expression)
            defined_arg = struct.get(arg.name)
            if defined_arg is None:
                self.error(
                    f"Invalid struct constructor call: struct doesn't have member named {arg.name}", expression)
                return TypedExpression(BuiltInType.invalid, expression)
            arg_type = self.evaluate_type_expression(arg.value, scope)
            if arg_type.type != defined_arg.type:
                self.error(
                    f"Invalid struct constructor call: invalid type of member {arg.name}", expression)
                return TypedExpression(BuiltInType.invalid, expression)
        return TypedExpression(struct, expression)

    def function_call_from_function(self, function: FunctionType, expression: FunctionCallExpression, scope: Scope[CheckerType]) -> TypedExpression:
        args = expression.args
        for defined_arg, passed_arg in zip(function.positional_arg_types, args):
            if passed_arg.name is not None and passed_arg.name != defined_arg.name:
                self.error(
                    f"Invalid argument name: {passed_arg.name} is not a valid argument name for {function.name}")
                return TypedExpression(BuiltInType.invalid, expression)
            arg_type = self.evaluate_type_expression(
                passed_arg.value, scope)
            if arg_type.type != defined_arg.type:
                self.error(
                    f"Invalid function call {function.name}: {passed_arg.name or args.index(passed_arg)}'s type: `{arg_type.type}` doesn't match {defined_arg.type} ")
                return TypedExpression(BuiltInType.invalid, expression)
        named_offset = len(function.positional_arg_types)
        for defined_arg, passed_arg in zip(function.named_arg_types, args[named_offset:]):
            if passed_arg.name is None:
                self.error(
                    f"Invalid argument: expected argument name")
                return TypedExpression(BuiltInType.invalid, expression)
            if not function.is_named(passed_arg.name):
                self.error(
                    f"Invalid argument name: {passed_arg.name} is not a valid argument name for {function.name}")
                return TypedExpression(BuiltInType.invalid, expression)
            arg_type = self.evaluate_type_expression(
                passed_arg.value, scope)
            if arg_type.type != defined_arg.type:
                self.error(
                    f"Invalid function call: {function.name} is not a valid function")
                return TypedExpression(BuiltInType.invalid, expression)
        return TypedExpression(function.to_type, expression)

    def check_expr_statement(self, statement: ExpressionStatement, scope: Scope) -> TypedStatement:
        typ = self.evaluate_type_expression(statement.value, scope)
        if typ == BuiltInType.invalid:
            self.error(
                f"Invalid expression statement: {statement.value} is not a valid expression"
            )
            return TypedStatement(BuiltInType.invalid, statement)
        return TypedStatement(typ.type, statement)

    def check_assignment_statement(self, statement: AssignmentStatement, scope: Scope[CheckerType]) -> TypedStatement:
        ident_name = statement.name.name
        expr = statement.value
        if not scope.rec_contains(ident_name):
            expr_type = self.evaluate_type_expression(expr, scope)
            defined_type = self.evaluate_type(
                statement.type, scope) or expr_type.type
            if expr_type.type != defined_type:
                self.error(
                    f"Invalid assignment: Defined type {defined_type} is not equal to {expr_type.type}!")
                return TypedStatement(BuiltInType.invalid, statement)
            if statement.mutable:
                new_typ = dataclass_replace(expr_type.type)  # copy type
                new_typ.meta["mut"] = True
                scope.rec_set(ident_name, new_typ)
            else:
                scope.rec_set(ident_name, expr_type.type)
            return TypedStatement(expr_type.type, statement)
        if statement.mutable:
            self.error(
                f"Invalid assignment statement: {ident_name} is already defined."
                "`mut` modifier can only be used when defining variables.")

            return TypedStatement(BuiltInType.invalid, statement)
        ident_type = scope.rec_get(ident_name)
        expr_type = self.evaluate_type_expression(expr, scope)
        defined_type = self.evaluate_type(
            statement.type, scope) or expr_type.type
        if expr_type.type != defined_type:
            self.error(
                f"Invalid assignment: Defined type {defined_type} is not equal to {expr_type.type}!")
            return TypedStatement(BuiltInType.invalid, statement)
        if ident_type == expr_type.type and ident_type.meta.get("mut"):
            scope.rec_set(ident_name, ident_type)
            return TypedStatement(ident_type, statement)
        if ident_type != expr_type.type and ident_type.meta.get("mut"):
            self.error(
                f"Type mismatch: Tried assigning {expr_type.type} to {ident_name} of type {ident_type}")
            return TypedStatement(BuiltInType.invalid, statement)
        self.error(
            f"Invalid assignment statement: {ident_name} was assigned before and is not mutable"
        )
        return TypedStatement(BuiltInType.invalid, statement)

    def check_while_statement(self, statement: WhileStatement, scope: Scope) -> TypedStatement:
        condition = statement.condition
        body = statement.body
        typ = self.evaluate_type_expression(condition, scope)
        if typ.type != BuiltInType.bool:
            self.error(
                f"Invalid while condition: {typ} is not a valid boolean expression")
            return TypedStatement(BuiltInType.invalid, statement)
        self.evaluate_type_expression(body, scope)
        return TypedStatement(BuiltInType.none, statement)

    def check_for_statement(self, statement: ForStatement, scope: Scope) -> TypedStatement:
        ident_name = statement.ident.name
        value = statement.value
        body = statement.body
        if scope.rec_contains(ident_name):
            self.error(
                f"Variable {ident_name} already defined")
            return TypedStatement(BuiltInType.invalid, statement)
        iterable_type = self.evaluate_type_expression(
            value, scope).type

        if not isinstance(iterable_type, ListType):
            self.error(
                f"Type {iterable_type} is not iterable")
            return TypedStatement(BuiltInType.invalid, statement)
        inner_scope = scope.spawn_child()

        # pylint: disable=no-member
        inner_scope.rec_set(ident_name, iterable_type.type)
        self.evaluate_type_expression(body, inner_scope)
        return TypedStatement(BuiltInType.none, statement)

    def check_function_arguments(self, args: list[FunctionArgument], scope: Scope) -> tuple[FunctionArgumentType, ...] | None:
        defined_args: list[FunctionArgumentType] = []
        for arg in args:
            if any(arg.name == a.name for a in defined_args):
                self.error(
                    f"Argument {arg.name} already defined")
                return None
            arg_type = self.evaluate_type(arg.type, scope)
            is_named = False
            if arg.default is not None:
                is_named = True
            if arg_type is None:
                if arg.default is None:
                    self.error(
                        f"Argument {arg.name} has no type and no default value")
                    return None
                is_named = True
                arg_type = self.evaluate_type_expression(
                    arg.default, scope).type
            if len(defined_args) > 0 and defined_args[-1].named and not is_named:
                self.error(
                    f"Argument {arg.name} is not named but previous argument is named")
                return None
            defined_args.append(FunctionArgumentType(
                arg.name, arg_type, is_named))

        for i, arg in enumerate(defined_args):
            if arg.named:
                # Check whether next argument is also named
                if i + 1 < len(defined_args) and not defined_args[i + 1].named:
                    self.error(
                        f"Argument {arg.name} is named but next argument is not")
                    return None
        return tuple(defined_args)

    def check_function_definition_statement(self, statement: FunctionDefinitionStatement, scope: Scope, is_def: bool = False) -> TypedStatement:
        name = statement.name
        args = statement.args
        body = statement.body
        if scope.rec_contains(name):
            fun = scope.rec_get(name)
            if not isinstance(fun, FunctionType):
                self.error(
                    f"Function {name} already defined")
                return TypedStatement(BuiltInType.invalid, statement)
            if fun.meta["is_def"] and is_def:
                self.error(
                    f"Function {name} already defined")
                return TypedStatement(BuiltInType.invalid, statement)
            if not fun.meta["is_def"] and not is_def:
                self.error(
                    f"Function {name} already defined")
                return TypedStatement(BuiltInType.invalid, statement)
        return_type = self.evaluate_type(statement.return_type, scope)
        if return_type is None:
            return_type = BuiltInType.none
        defined_args = self.check_function_arguments(args, scope)
        if defined_args is None:
            return TypedStatement(BuiltInType.invalid, statement)
        fun = FunctionType(name, defined_args, return_type)
        fun.meta["is_def"] = is_def
        if is_def:
            scope.rec_set(name, fun)
            return TypedStatement(fun, statement)
        inner_scope = scope.spawn_child()
        for arg in defined_args:
            inner_scope.rec_set(arg.name, arg.type)
        inner_scope.rec_set(name, fun)
        body_return_type = self.evaluate_type_expression(body, inner_scope)
        if body_return_type.type != return_type:
            self.error(
                f"Invalid return type: {body_return_type} is not equal to {return_type}")
            return TypedStatement(BuiltInType.invalid, statement)
        scope.rec_set(name, fun)
        return TypedStatement(fun, statement)

    def check_struct_field(self, field: StructField, scope: Scope) -> CheckerType:
        field_type = self.evaluate_type(field.type, scope)
        if field_type is None:
            self.error(
                f"Invalid field type: {field.type}")
            return BuiltInType.invalid
        return field_type

    def check_struct_definition_statement(self, statement: StructDefinitionStatement, scope: Scope, is_def: bool = False) -> TypedStatement:
        if scope.rec_contains(statement.name):
            fun = scope.rec_get(statement.name)
            if not isinstance(fun, StructType):
                self.error(
                    f"Invalid struct definition: Name {statement.name} is already defined!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            if fun.meta["is_def"] and is_def:
                self.error(
                    f"Invalid struct definition: Name {statement.name} is already defined!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            if not fun.meta["is_def"] and not is_def:
                self.error(
                    f"Invalid struct definition: Name {statement.name} is already defined!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
        defined_names: list[str] = []
        defined_fields: list[StructFieldType] = []
        for field in statement.fields:
            if field.name in defined_names:
                self.error(
                    f"Invalid struct definition: Field {field.name} is already defined!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            field_type = self.check_struct_field(field, scope)
            if field_type == BuiltInType.invalid:
                self.error(
                    f"Invalid struct definition: Field {field.name} has invalid type!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            defined_names.append(field.name)
            defined_fields.append(StructFieldType(field.name, field_type))

        defined_methods: list[StructMethodType] = []
        for method in statement.methods:
            if method.name in defined_names:
                self.error(
                    f"Invalid struct definition: Method {method.name} is already defined!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            defined_names.append(method.name)
            args = self.check_function_arguments(method.args, scope)
            if args is None:
                self.error(
                    f"Invalid struct definition: Method {method.name} has invalid arguments!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            return_type = self.evaluate_type(method.return_type, scope)
            if return_type is None:
                return_type = BuiltInType.none
            function_type = FunctionType(method.name, args, return_type)
            defined_methods.append(
                StructMethodType(method.name, function_type))
        struct_type = StructType(statement.name, tuple(
            defined_fields), tuple(defined_methods))
        struct_type.meta["is_def"] = is_def
        if is_def:
            scope.rec_set(statement.name, struct_type)
            return TypedStatement(struct_type, statement)
        for defined_method, method in zip(defined_methods, statement.methods):
            # Check method bodies
            inner_scope = scope.spawn_child()
            inner_scope.rec_set("self", struct_type)
            for arg in defined_method.type.arg_types:
                inner_scope.rec_set(arg.name, arg.type)
            typ = self.evaluate_type_expression(method.body, inner_scope)
            return_typ = self.evaluate_type(method.return_type, scope)
            if return_typ is None:
                return_typ = BuiltInType.none
            if typ.type != return_typ:
                self.error(
                    f"Invalid struct definition: Method {method.name} has invalid return type!")
                return TypedStatement(BuiltInType.invalid, statement)
        scope.rec_set(statement.name, struct_type)
        return TypedStatement(struct_type, statement)

    def import_(self, name: str) -> ModuleType:
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
        # Create new checker
        program = module.program()
        checker = Checker(program, new_context)
        # Run module
        checker.check()
        # Check for errors
        assert not checker.has_errors, f"Errors in module {name}: {checker._errors}"
        # Return module scope
        types: dict[str, CheckerType] = {}
        for n, typ in checker.scope.items():
            types[n] = typ
        module = ModuleType(name, types)
        self.context.module_cache[name] = module
        return module

    def check_import_statement(self, statement: ImportStatement, scope: Scope, is_def: bool = False) -> TypedStatement:
        if scope.parent is not None:
            self.error(
                f"Import statement is not allowed in inner scope")
            return TypedStatement(BuiltInType.invalid, statement)
        paths = statement.name.split(".")
        name = paths[-1]
        if scope.rec_contains(name):
            mod = scope.rec_get(name)
            conditions = [
                mod.meta["is_def"] and is_def,
                not mod.meta["is_def"] and not is_def
            ]
            if any(conditions):
                self.error(
                    f"Name {name} is already defined. Cannot import {statement.name}")
                return TypedStatement(BuiltInType.invalid, statement)
        module = self.import_(statement.name)
        module.meta["is_def"] = is_def
        scope.rec_set(name, module)
        stat = TypedStatement(module, statement)
        return stat

    def check_statement(self, statement: Statement, scope: Scope = None) -> TypedStatement:
        if scope is None:
            scope = self.scope
        match statement:
            case ExpressionStatement():
                return self.check_expr_statement(statement, scope)
            case AssignmentStatement():
                return self.check_assignment_statement(statement, scope)
            case ForStatement():
                return self.check_for_statement(statement, scope)
            case WhileStatement():
                return self.check_while_statement(statement, scope)
            case FunctionDefinitionStatement():
                return self.check_function_definition_statement(statement, scope)
            case StructDefinitionStatement():
                return self.check_struct_definition_statement(statement, scope)
            case ImportStatement():
                return self.check_import_statement(statement, scope)

        raise NotImplementedError(f"Unknown statement: {statement}")
