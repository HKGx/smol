import dataclasses
from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple
from smol.tokenizer import Tokenizer

from smol.parser import (AdditionExpression, ArrayExpression,
                         AssignmentStatement, BlockExpression, BooleanExpression, FunctionArgument, ImportStatement, Parser, PropertyAccessExpression, StructDefinitionStatement, StructField, StructMethod, TypeBuiltInExpression,
                         ComparisonExpression, EqualityExpression,
                         ExponentiationExpression, Expression,
                         ExpressionStatement, ForStatement,
                         FunctionCallExpression, FunctionDefinitionStatement, IdentifierExpression,
                         IfExpression, IntegerExpression,
                         MultiplicationExpression, NegationExpression, Program, RangeExpression,
                         Statement, StringExpression, TypeDeduceExpression, TypeExpression, TypeIdentifierExpression, WhileStatement)
from smol.utils import Scope, StageContext, resolve_module_path


@dataclass(eq=True, frozen=True)
class CheckerType:
    name: str
    meta: dict[str, bool] = dataclasses.field(
        init=False, default_factory=dict, compare=False)


@dataclass(eq=True, frozen=True)
class InvalidType(CheckerType):
    name = "invalid"


@dataclass(eq=True, frozen=True)
class ListType(CheckerType):
    type: CheckerType
    known_length: Optional[int]


@dataclass(eq=True, frozen=True)
class FunctionArgumentType(CheckerType):
    name: str
    type: CheckerType
    named: bool = False


@dataclass(eq=True, frozen=True)
class FunctionType(CheckerType):
    arg_types: Tuple[FunctionArgumentType, ...]
    to_type: CheckerType

    @property
    def named_arg_types(self) -> Tuple[FunctionArgumentType, ...]:
        return tuple(arg_type for arg_type in self.arg_types if arg_type.named)

    @property
    def positional_arg_types(self) -> Tuple[FunctionArgumentType, ...]:
        return tuple(arg_type for arg_type in self.arg_types if not arg_type.named)

    def is_named(self, name: str) -> bool:
        return any(arg_type.name == name for arg_type in self.named_arg_types)


@dataclass(eq=True, frozen=True)
class StructFieldType(CheckerType):
    type: CheckerType


@dataclass(eq=True, frozen=True)
class StructMethodType(CheckerType):
    type: FunctionType


@dataclass(eq=True, frozen=True)
class StructType(CheckerType):
    fields: Tuple[StructFieldType, ...]
    methods: Tuple[StructMethodType, ...]

    def get(self, name: str) -> Optional[StructFieldType | StructMethodType]:
        for member in [*self.fields, *self.methods]:
            if member.name == name:
                return member
        return None


@dataclass(eq=True, frozen=True)
class ModuleType(CheckerType):
    name: str
    types: dict[str, CheckerType]


@dataclass(eq=True, frozen=True)
class TypedExpression:
    type: CheckerType
    value: Expression


@dataclass
class TypedStatement:
    type: CheckerType
    value: Statement


class BuiltInType(NamedTuple):
    int = CheckerType("int")
    string = CheckerType("string")
    bool = CheckerType("bool")
    none = CheckerType("none")
    invalid = InvalidType("invalid")


@dataclass
class CheckerContext(StageContext):
    module_cache: dict[str, ModuleType] = dataclasses.field(
        default_factory=dict)

    def copy(self):
        return CheckerContext(
            current_directory=self.current_directory,
            current_file=self.current_file,
            import_stack=self.import_stack.copy(),
            module_cache=self.module_cache.copy()
        )


class Checker:
    program: Program
    context: CheckerContext
    errors: list[str] = []
    scope: Scope[CheckerType] = Scope.from_dict({
        "print": FunctionType("print", (FunctionArgumentType("to_print", BuiltInType.string),), BuiltInType.none),
        "str": FunctionType("str", (FunctionArgumentType("from", BuiltInType.int),), BuiltInType.string)
    })  # type: ignore

    def __init__(self, program: Program, context: CheckerContext):
        self.program = program
        self.context = context

    def load_stdlib(self):
        if self.context.current_file != "std.smol":
            self.import_("std.std")

    def check(self):
        # Before checking load stdlib
        print(self.context)
        self.load_stdlib()
        self.check_program(self.program)
        return self.errors

    @property
    def has_errors(self):
        return len(self.errors) > 0

    def check_program(self, program: Program):
        for statement in program.statements:
            self.check_statement(statement)

    def lr_evaluate(self, lhs: Expression, rhs: Expression, scope: Scope = None) -> Tuple[TypedExpression, TypedExpression]:
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
                    self.errors.append(f"Unknown type: {name}")
                    return BuiltInType.invalid
                typ = scope.rec_get(name)
                if not isinstance(typ, StructType):
                    self.errors.append(f"Type {name} is not a struct")
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
                    self.errors.append(
                        f"Identifier {name} is not defined in scope")
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(scope.rec_get(name), expression)
            case EqualityExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                if l_typ.type != r_typ.type:
                    self.errors.append(
                        f"{expression} has different types: {l_typ.type} and {r_typ.type}")
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(BuiltInType.bool, expression)
            case ComparisonExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                if l_typ.type != r_typ.type:
                    self.errors.append(
                        f"{expression} has different types: {l_typ.type} and {r_typ.type}")
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

                self.errors.append(
                    f"Invalid operation: {l_typ.type.name} {sign} {r_typ.type.name}")
                return TypedExpression(BuiltInType.invalid, expression)
            case MultiplicationExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match (l_typ.type, r_typ.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expression)

                self.errors.append(
                    f"Invalid operation: {l_typ.type.name} {sign} {r_typ.type.name}")
                return TypedExpression(BuiltInType.invalid, expression)
            case ExponentiationExpression(left, sign, right):
                l_typ, r_typ = self.lr_evaluate(left, right, scope)
                match (l_typ.type, r_typ.type):
                    case (BuiltInType.int, BuiltInType.int):
                        return TypedExpression(BuiltInType.int, expression)

                self.errors.append(
                    f"Invalid operation: {l_typ.type} {sign} {r_typ.type}"
                )
                return TypedExpression(BuiltInType.invalid, expression)
            case NegationExpression(expression):
                typ = self.evaluate_type_expression(expression, scope)
                if typ.type != BuiltInType.int:
                    self.errors.append(
                        f"Invalid operation: -{typ.type}"
                    )
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(BuiltInType.int, expression)
            case IfExpression(condition, body, else_ifs, else_body):
                condition_type = self.evaluate_type_expression(
                    condition, scope)
                if condition_type.type != BuiltInType.bool:
                    self.errors.append(
                        f"Invalid condition: {condition_type.type}")
                    return TypedExpression(BuiltInType.invalid, expression)
                body_typ = self.evaluate_type_expression(
                    body, scope.spawn_child())
                if body_typ.type == BuiltInType.invalid:
                    self.errors.append(
                        f"Invalid if statement: {body} is not a valid expression"
                    )
                    return TypedExpression(BuiltInType.invalid, expression)
                body_types: list[CheckerType] = [body_typ.type]
                for else_if in else_ifs:
                    else_if_cond_type = self.evaluate_type_expression(
                        else_if[0], scope)
                    if else_if_cond_type.type == BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else if statement: {else_if[0]} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expression)
                    else_if_body_type = self.evaluate_type_expression(
                        else_if[1], scope.spawn_child())
                    if else_if_body_type.type == BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else if statement: {else_if[1]} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expression)
                    body_types.append(else_if_body_type.type)
                if else_body is not None:
                    else_body_type = self.evaluate_type_expression(
                        else_body, scope.spawn_child())
                    if else_body_type.type == BuiltInType.invalid:
                        self.errors.append(
                            f"Invalid else statement: {else_body} is not a valid expression"
                        )
                        return TypedExpression(BuiltInType.invalid, expression)
                    body_types.append(else_body_type.type)

                # TODO: Need to discuss whether we allow multiple types in an if statement to be returned

                if body_types.count(body_types[0]) != len(body_types):
                    self.errors.append("Invalid types of if bodies")
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
                self.errors.append("Invalid types of array elements")
                return TypedExpression(BuiltInType.invalid, expression)
            case PropertyAccessExpression(obj, property):
                typ = self.evaluate_type_expression(obj, scope)
                if isinstance(typ.type, ModuleType):
                    if property in typ.type.types:
                        return TypedExpression(typ.type.types[property], expression)
                    self.errors.append(
                        f"Invalid property access: {property} is not a valid property of {typ.type.name}"
                    )
                    return TypedExpression(BuiltInType.invalid, expression)
                if not isinstance(typ.type, StructType):
                    self.errors.append(
                        f"Invalid operation: {typ.type} has no property {property}")
                    return TypedExpression(BuiltInType.invalid, obj)
                member_typ = typ.type.get(property)
                if member_typ is None:
                    self.errors.append(
                        f"Invalid operation: {typ.type} has no property {property}")
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
                        self.errors.append(
                            f"Invalid function call: {function.name} is not a valid function")
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
                    self.errors.append(
                        f"Invalid range: {start_type.type} to {end_type.type} by {step_type.type}")
                    return TypedExpression(BuiltInType.invalid, expression)
                return TypedExpression(ListType("list", BuiltInType.int, None), expression)

        raise NotImplementedError(f"{expression}")

    def struct_constructor_call(self, struct: StructType, expression: FunctionCallExpression, scope: Scope[CheckerType]) -> TypedExpression:
        if len(expression.args) != len(struct.fields):
            self.errors.append(
                f"Invalid struct constructor call: invalid count of members")
            return TypedExpression(BuiltInType.invalid, expression)
        for arg in expression.args:
            if arg.name is None:
                self.errors.append(
                    f"Invalid struct constructor call: invalid member name")
                return TypedExpression(BuiltInType.invalid, expression)
            defined_arg = struct.get(arg.name)
            if defined_arg is None:
                self.errors.append(
                    f"Invalid struct constructor call: struct doesn't have member named {arg.name}")
                return TypedExpression(BuiltInType.invalid, expression)
            arg_type = self.evaluate_type_expression(arg.value, scope)
            if arg_type.type != defined_arg.type:
                self.errors.append(
                    f"Invalid struct constructor call: invalid type of member {arg.name}")
                return TypedExpression(BuiltInType.invalid, expression)
        return TypedExpression(struct, expression)

    def function_call_from_function(self, function: FunctionType, expression: FunctionCallExpression, scope: Scope[CheckerType]) -> TypedExpression:
        args = expression.args
        for defined_arg, passed_arg in zip(function.positional_arg_types, args):
            if passed_arg.name is not None and passed_arg.name != defined_arg.name:
                self.errors.append(
                    f"Invalid argument name: {passed_arg.name} is not a valid argument name for {function.name}")
                return TypedExpression(BuiltInType.invalid, expression)
            arg_type = self.evaluate_type_expression(
                passed_arg.value, scope)
            if arg_type.type != defined_arg.type:
                self.errors.append(
                    f"Invalid function call {function.name}: {passed_arg.name or args.index(passed_arg)}'s type: `{arg_type.type}` doesn't match {defined_arg.type} ")
                return TypedExpression(BuiltInType.invalid, expression)
        named_offset = len(function.positional_arg_types)
        for defined_arg, passed_arg in zip(function.named_arg_types, args[named_offset:]):
            if passed_arg.name is None:
                self.errors.append(
                    f"Invalid argument: expected argument name")
                return TypedExpression(BuiltInType.invalid, expression)
            if not function.is_named(passed_arg.name):
                self.errors.append(
                    f"Invalid argument name: {passed_arg.name} is not a valid argument name for {function.name}")
                return TypedExpression(BuiltInType.invalid, expression)
            arg_type = self.evaluate_type_expression(
                passed_arg.value, scope)
            if arg_type.type != defined_arg.type:
                self.errors.append(
                    f"Invalid function call: {function.name} is not a valid function")
                return TypedExpression(BuiltInType.invalid, expression)
        return TypedExpression(function.to_type, expression)

    def check_expr_statement(self, statement: ExpressionStatement, scope: Scope) -> TypedStatement:
        typ = self.evaluate_type_expression(statement.value, scope)
        if typ == BuiltInType.invalid:
            self.errors.append(
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
                self.errors.append(
                    f"Invalid assignment: Defined type {defined_type} is not equal to {expr_type.type}!")
                return TypedStatement(BuiltInType.invalid, statement)
            if statement.mutable:
                new_typ = dataclasses.replace(expr_type.type)  # copy type
                new_typ.meta["mut"] = True
                scope.rec_set(ident_name, new_typ)
            else:
                scope.rec_set(ident_name, expr_type.type)
            return TypedStatement(expr_type.type, statement)
        if statement.mutable:
            self.errors.append(
                f"Invalid assignment statement: {ident_name} is already defined."
                "`mut` modifier can only be used when defining variables.")

            return TypedStatement(BuiltInType.invalid, statement)
        ident_type = scope.rec_get(ident_name)
        expr_type = self.evaluate_type_expression(expr, scope)
        defined_type = self.evaluate_type(
            statement.type, scope) or expr_type.type
        if expr_type.type != defined_type:
            self.errors.append(
                f"Invalid assignment: Defined type {defined_type} is not equal to {expr_type.type}!")
            return TypedStatement(BuiltInType.invalid, statement)
        if ident_type == expr_type.type and ident_type.meta.get("mut"):
            scope.rec_set(ident_name, ident_type)
            return TypedStatement(ident_type, statement)
        if ident_type != expr_type.type and ident_type.meta.get("mut"):
            self.errors.append(
                f"Type mismatch: Tried assigning {expr_type.type} to {ident_name} of type {ident_type}")
            return TypedStatement(BuiltInType.invalid, statement)
        self.errors.append(
            f"Invalid assignment statement: {ident_name} was assigned before and is not mutable"
        )
        return TypedStatement(BuiltInType.invalid, statement)

    def check_while_statement(self, statement: WhileStatement, scope: Scope) -> TypedStatement:
        condition = statement.condition
        body = statement.body
        typ = self.evaluate_type_expression(condition, scope)
        if typ.type != BuiltInType.bool:
            self.errors.append(
                f"Invalid while condition: {typ} is not a valid boolean expression")
            return TypedStatement(BuiltInType.invalid, statement)
        self.evaluate_type_expression(body, scope)
        return TypedStatement(BuiltInType.none, statement)

    def check_for_statement(self, statement: ForStatement, scope: Scope) -> TypedStatement:
        ident_name = statement.ident.name
        value = statement.value
        body = statement.body
        if scope.rec_contains(ident_name):
            self.errors.append(
                f"Variable {ident_name} already defined")
            return TypedStatement(BuiltInType.invalid, statement)
        iterable_type = self.evaluate_type_expression(
            value, scope).type

        if not isinstance(iterable_type, ListType):
            self.errors.append(
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
                self.errors.append(
                    f"Argument {arg.name} already defined")
                return None
            arg_type = self.evaluate_type(arg.type, scope)
            is_named = False
            if arg.default is not None:
                is_named = True
            if arg_type is None:
                if arg.default is None:
                    self.errors.append(
                        f"Argument {arg.name} has no type and no default value")
                    return None
                is_named = True
                arg_type = self.evaluate_type_expression(
                    arg.default, scope).type
            if len(defined_args) > 0 and defined_args[-1].named and not is_named:
                self.errors.append(
                    f"Argument {arg.name} is not named but previous argument is named")
                return None
            defined_args.append(FunctionArgumentType(
                arg.name, arg_type, is_named))

        for i, arg in enumerate(defined_args):
            if arg.named:
                # Check whether next argument is also named
                if i + 1 < len(defined_args) and not defined_args[i + 1].named:
                    self.errors.append(
                        f"Argument {arg.name} is named but next argument is not")
                    return None
        return tuple(defined_args)

    def check_function_definition_statement(self, statement: FunctionDefinitionStatement, scope: Scope) -> TypedStatement:
        name = statement.name
        args = statement.args
        body = statement.body
        if scope.rec_contains(name):
            self.errors.append(
                f"Function {name} already defined")
            return TypedStatement(BuiltInType.invalid, statement)
        return_type = self.evaluate_type(statement.return_type, scope)
        if return_type is None:
            return_type = BuiltInType.none
        defined_args = self.check_function_arguments(args, scope)
        if defined_args is None:
            return TypedStatement(BuiltInType.invalid, statement)
        fun = FunctionType(name, defined_args, return_type)
        inner_scope = scope.spawn_child()
        for arg in defined_args:
            inner_scope.rec_set(arg.name, arg.type)
        inner_scope.rec_set(name, fun)
        body_return_type = self.evaluate_type_expression(body, inner_scope)
        if body_return_type.type != return_type:
            self.errors.append(
                f"Invalid return type: {body_return_type} is not equal to {return_type}")
            return TypedStatement(BuiltInType.invalid, statement)
        scope.rec_set(name, fun)
        return TypedStatement(fun, statement)

    def check_struct_field(self, field: StructField, scope: Scope) -> CheckerType:
        field_type = self.evaluate_type(field.type, scope)
        if field_type is None:
            self.errors.append(
                f"Invalid field type: {field.type}")
            return BuiltInType.invalid
        return field_type

    def check_struct_definition_statement(self, statement: StructDefinitionStatement, scope: Scope) -> TypedStatement:
        if scope.rec_contains(statement.name):
            self.errors.append(
                f"Invalid struct definition: Name {statement.name} is already defined!"
            )
            return TypedStatement(BuiltInType.invalid, statement)
        defined_names: list[str] = []
        defined_fields: list[StructFieldType] = []
        for field in statement.fields:
            if field.name in defined_names:
                self.errors.append(
                    f"Invalid struct definition: Field {field.name} is already defined!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            field_type = self.check_struct_field(field, scope)
            if field_type == BuiltInType.invalid:
                self.errors.append(
                    f"Invalid struct definition: Field {field.name} has invalid type!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            defined_names.append(field.name)
            defined_fields.append(StructFieldType(field.name, field_type))

        defined_methods: list[StructMethodType] = []
        for method in statement.methods:
            if method.name in defined_names:
                self.errors.append(
                    f"Invalid struct definition: Method {method.name} is already defined!"
                )
                return TypedStatement(BuiltInType.invalid, statement)
            defined_names.append(method.name)
            args = self.check_function_arguments(method.args, scope)
            if args is None:
                self.errors.append(
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

        inner_scope = scope.spawn_child()
        inner_scope.rec_set("self", struct_type)
        for method in statement.methods:
            # Check method bodies
            typ = self.evaluate_type_expression(method.body, inner_scope)
            return_typ = self.evaluate_type(method.return_type, scope)
            if typ.type != return_typ:
                self.errors.append(
                    f"Invalid struct definition: Method {method.name} has invalid return type!"
                )
                return TypedStatement(BuiltInType.invalid, statement)

        scope.rec_set(statement.name, struct_type)
        return TypedStatement(struct_type, statement)

    def import_(self, name: str) -> ModuleType:
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
        new_context.current_file = module_path.name
        new_context.import_stack.append(name)
        # Create new checker
        checker = Checker(module.program(), new_context)
        # Run module
        checker.check()
        # Check for errors
        assert len(
            checker.errors) == 0, f"Errors in module {name}: {checker.errors}"
        # Return module scope
        types: dict[str, CheckerType] = {}
        for n, typ in checker.scope.items():
            types[n] = typ
        module = ModuleType(name, types)
        self.context.module_cache[name] = module
        return module

    def check_import_statement(self, statement: ImportStatement, scope: Scope) -> TypedStatement:
        if scope.parent is not None:
            self.errors.append(
                f"Import statement is not allowed in inner scope")
            return TypedStatement(BuiltInType.invalid, statement)
        paths = statement.name.split(".")
        name = paths[-1]
        if scope.rec_contains(name):
            self.errors.append(
                f"Name {name} is already defined. Cannot import {statement.name}")
            return TypedStatement(BuiltInType.invalid, statement)
        module = self.import_(statement.name)
        scope.rec_set(name, module)
        return TypedStatement(module, statement)

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
