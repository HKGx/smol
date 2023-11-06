from dataclasses import dataclass
from textwrap import dedent
from typing import Literal

import smol.parser.expressions as E
import smol.parser.statements as S
from smol.lexer.lexer import Lexer
from smol.lexer.token import Token, TokenType
from smol.utils import StageContext


@dataclass
class Program:
    statements: tuple[S.Statement, ...]
    structs: tuple[S.StructDefinitionStatement, ...]
    functions: tuple[S.FunctionDefinitionStatement, ...]
    imports: tuple[S.ImportStatement, ...]


class Parser:
    """
    Parses tokens into an AST.
    This can be then fed into Checker or Interpreter.
    """

    tokens: list[Token]
    structs: list[S.StructDefinitionStatement]
    functions: list[S.FunctionDefinitionStatement]
    imports: list[S.ImportStatement]
    current: int = 0
    context: StageContext

    def __init__(self, tokens: list[Token], context: StageContext):
        self.tokens = tokens
        self.structs = []
        self.functions = []
        self.imports = [S.ImportStatement("std/std", add_to_scope=True)]
        self.current = 0
        self.context = context

    @classmethod
    def from_lexer(cls, lexer: Lexer) -> "Parser":
        """
        Creates a parser from lexer.
        If lexer has ended it'll return it's tokens.
        If lexer hasn't ended it'll run .tokenize() on it.
        """
        if lexer.ended:
            return cls(lexer._tokens, lexer.context)
        else:
            return cls(lexer.lex(), lexer.context)

    @classmethod
    def from_file(cls, filename: str, context: StageContext) -> "Parser":
        return cls.from_lexer(Lexer.from_file(filename, context))

    @property
    def current_token(self) -> Token:
        return self.tokens[self.current]

    @property
    def peek_next(self) -> Token | None:
        return (
            self.tokens[self.current + 1]
            if self.current + 1 < len(self.tokens)
            else None
        )

    @property
    def ended(self):
        return self.current >= len(self.tokens)

    def next(self, increment: int = 1):
        self.current += increment

    def edges(self, start: Token) -> tuple[Token, Token]:
        return start, self.current_token

    def type_expression(self) -> E.TypeExpression:
        return self.type_union_expression()

    def type_union_expression(self) -> E.TypeExpression:
        start = self.current_token
        expr = self.type_array_expression()
        assert not self.ended, "Expected expression after 'or'"
        exprs: list[E.TypeExpression] = [expr]
        while not self.ended:
            match self.current_token:
                case Token(TokenType.KEYWORD, "or"):
                    self.next()
                    exprs.append(self.type_array_expression())
                case _:
                    break
        if len(exprs) == 1:
            return expr
        return E.TypeUnionExpression(exprs, edges=self.edges(start))

    def type_array_expression(self) -> E.TypeExpression:
        start = self.current_token
        expr = self.type_atomic()
        while not self.ended and self.current_token.match(TokenType.LEFT_BRACKET):
            self.next()
            length = None
            assert not self.ended, "Expected `]` or length but got EOF"
            if self.current_token.match(TokenType.INTEGER_LITERAL):
                length = int(self.current_token.image)
                self.next()
            assert not self.ended, "Expected `]` but got EOF"
            assert self.current_token.match(
                TokenType.RIGHT_BRACKET
            ), f"Expected `]` but got {self.current_token.image}"
            self.next()
            expr = E.TypeArrayExpression(expr, length, edges=self.edges(start))
        return expr

    def type_atomic(self) -> E.TypeExpression:
        assert not self.ended, "Expected type but got EOF"
        expr: E.TypeExpression
        edges = self.edges(self.current_token)
        match self.current_token:
            case Token(TokenType.IDENTIFIER_LITERAL, name):
                expr = E.TypeIdentifierExpression(name, edges=edges)
            case _:
                assert False, f"Expected type but got {self.current_token.image}"
        self.next()
        return expr

    def expression(self) -> E.Expression:
        return self.equality()

    def equality(self) -> E.Expression:
        start = self.current_token
        lhs = self.comparison()
        while not self.ended and self.current_token.match(
            TokenType.EQUALS, TokenType.NOT_EQUALS
        ):
            assert self.current_token.image == "==" or self.current_token.image == "!="
            sign: Literal["==", "!="] = self.current_token.image
            self.next()
            edges = self.edges(start)
            lhs = E.EqualityExpression(lhs, sign, self.comparison(), edges=edges)
        return lhs

    def comparison(self) -> E.Expression:
        start = self.current_token
        lhs = self.range_expression()
        while not self.ended and self.current_token.match(
            TokenType.SMALLER_THAN,
            TokenType.GREATER_THAN,
            TokenType.SMALLER_OR_EQUAL_THAN,
            TokenType.GREATER_OR_EQUAL_THAN,
        ):
            assert (
                self.current_token.image == "<"
                or self.current_token.image == ">"
                or self.current_token.image == "<="
                or self.current_token.image == ">="
            )
            sign: Literal["<", ">", "<=", ">="] = self.current_token.image
            self.next()
            edges = self.edges(start)
            lhs = E.ComparisonExpression(
                lhs, sign, self.range_expression(), edges=edges
            )
        return lhs

    def range_expression(self) -> E.Expression:
        start = self.current_token
        lhs = self.addition()
        if not self.ended and self.current_token.match(TokenType.RANGE):
            assert self.current_token.image == ".."
            self.next()
            assert not self.ended, "Expected expression after '..'"
            rhs = self.addition()
            edges = self.edges(start)
            step = None
            if not self.ended and self.current_token.match(TokenType.RANGE):
                assert self.current_token.image == ".."
                self.next()
                assert not self.ended, "Expected expression after '..'"
                step = self.addition()
                edges = self.edges(start)

            return E.RangeExpression(lhs, rhs, step, edges=edges)
        return lhs

    def addition(self) -> E.Expression:
        start = self.current_token
        lhs = self.multiplication()
        while not self.ended and self.current_token.match(
            TokenType.PLUS, TokenType.MINUS
        ):
            assert self.current_token.image == "+" or self.current_token.image == "-"

            sign: Literal["+"] | Literal["-"] = self.current_token.image
            self.next()
            edges = self.edges(start)
            lhs = E.AdditionExpression(lhs, sign, self.multiplication(), edges=edges)

        return lhs

    def multiplication(self) -> E.Expression:
        lhs = self.exponentiation()
        while not self.ended and self.current_token.match(
            TokenType.STAR, TokenType.SLASH
        ):
            assert self.current_token.image == "*" or self.current_token.image == "/"
            sign: Literal["*"] | Literal["/"] = self.current_token.image
            self.next()
            lhs = E.MultiplicationExpression(lhs, sign, self.exponentiation())
        return lhs

    def exponentiation(self) -> E.Expression:
        lhs = self.negation()
        while not self.ended and self.current_token.typ == TokenType.CARET:
            self.next()
            lhs = E.ExponentiationExpression(
                lhs, "^", self.exponentiation()
            )  # it just works?
        return lhs

    def negation(self) -> E.Expression:
        if self.current_token.typ == TokenType.MINUS:
            self.next()
            return E.NegationExpression(self.array_access())
        return self.array_access()

    def array_access(self) -> E.Expression:
        start = self.current_token
        expr = self.function_call()
        if not self.ended and self.current_token.match(TokenType.LEFT_BRACKET):
            self.next()
            assert not self.ended, "Expected expression after '['"
            index = self.expression()
            assert not self.ended, "Expected ']'"
            assert self.current_token.match(TokenType.RIGHT_BRACKET), "Expected ']'"
            self.next()
            return E.ArrayAccessExpression(expr, index, edges=self.edges(start))
        return expr

    def function_call(self) -> E.Expression:
        start = self.current_token
        object = self.property_access()
        if self.ended or self.current_token.typ != TokenType.LEFT_PAREN:
            return object
        assert not self.ended, "Expected `(` but found `EOF`"
        assert self.current_token.typ == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        # parse arguments
        args: list[E.FunctionCallArgument] = []
        while not self.ended:
            match self.current_token:
                case Token(TokenType.RIGHT_PAREN):
                    break
                case Token(TokenType.COMMA):
                    self.next()
                    continue
                case _:
                    args.append(self.function_call_argument())
        assert not self.ended, "Expected `)` but found `EOF`"
        assert self.current_token.typ == TokenType.RIGHT_PAREN, "Expected ')'"
        edges = self.edges(start)
        self.next()
        return E.FunctionCallExpression(object, args, edges=edges)

    def property_access(self) -> E.Expression:
        start = self.current_token
        lhs = self.atomic()
        while not self.ended and self.current_token.typ == TokenType.DOT:
            self.next()
            assert not self.ended, "Expected identifier after '.' but found `EOF`"
            assert (
                self.current_token.typ == TokenType.IDENTIFIER_LITERAL
            ), f"Expected identifier after '.' but found {self.current_token.image}"
            edges = self.edges(start)
            lhs = E.PropertyAccessExpression(lhs, self.current_token.image, edges=edges)
            self.next()
        return lhs

    def array_literal(self) -> E.Expression:
        start = self.current_token
        assert self.current_token.typ == TokenType.LEFT_BRACKET
        self.next()
        elements = []
        while not self.ended:
            match self.current_token:
                case Token(TokenType.RIGHT_BRACKET):
                    return E.ArrayExpression(elements, edges=self.edges(start))
                case Token(TokenType.COMMA):
                    self.next()
                    assert not self.ended, "Expected expression after comma"
                case _:
                    elements.append(self.expression())
                    assert self.current_token.typ in (
                        TokenType.COMMA,
                        TokenType.RIGHT_BRACKET,
                    ), f"Expected comma or right bracket, but got {self.current_token.image}"
        assert False, "Unexpected end of file, expected closing `]`"

    def atomic(self) -> E.Expression:
        start = self.current_token
        expr: E.Expression
        match self.current_token:
            case Token(TokenType.KEYWORD, "break"):
                expr = E.BreakExpression()
            case Token(TokenType.KEYWORD, "continue"):
                expr = E.ContinueExpression()
            case Token(TokenType.KEYWORD, "do"):
                expr = self.do_block_expression()
            case Token(TokenType.KEYWORD, "if"):
                expr = self.if_expression()
            case Token(TokenType.INTEGER_LITERAL):
                expr = E.IntegerExpression(int(self.current_token.image))
            case Token(TokenType.BOOLEAN_LITERAL, image):
                expr = E.BooleanExpression(image == "true")
            case Token(TokenType.LEFT_BRACKET):
                expr = self.array_literal()
            case Token(TokenType.LEFT_PAREN):
                expr = self.parenthesized_expression()
            case Token(TokenType.STRING_LITERAL):
                expr = E.StringExpression(self.current_token.image)
            case Token(TokenType.IDENTIFIER_LITERAL):
                expr = E.IdentifierExpression(self.current_token.image)
            case _:
                # TODO: improve error reporting
                start = max(0, self.current - 10)
                end = min(len(self.tokens), self.current + 10)
                last_tokens_images = [t.image for t in self.tokens[start:end]]
                assert False, dedent(
                    f"""
                Expected integer, identifier, function call or `(` but got `{self.current_token.image}` at {self.current_token.line}:{self.current_token.column}.
                {" ".join(last_tokens_images)}
                """
                )
        edges = None
        if not self.ended:
            # FIXME: sometimes self.current_token is outside the bounds of self.tokens
            # this is a bit of a hack, need to be fixed
            edges = self.edges(start)
        if not isinstance(expr, (E.BlockExpression, E.IfExpression)):
            self.next()
        expr.edges = edges
        return expr

    def do_block_expression(self) -> E.Expression:
        assert self.current_token.typ == TokenType.KEYWORD
        assert self.current_token.image == "do"
        self.next()
        statements = []

        while not self.ended:
            if (
                self.current_token.typ == TokenType.KEYWORD
                and self.current_token.image == "end"
            ):
                self.next()
                return E.BlockExpression(statements)
            statements.append(self.statement())

        assert False, "Expected `end` but got end of file"

    def enter_body(self) -> E.Expression:
        match self.current_token:
            case Token(TokenType.KEYWORD, "do"):
                return self.do_block_expression()
            case Token(TokenType.COLON):
                self.next()
                assert not self.ended, "Expected expression after colon"
                return self.expression()
            case got:
                assert (
                    False
                ), f"Expected `do` or `:` but got {got.image} at {got.line}:{got.column}"

    def if_expression(self) -> E.Expression:
        # TODO: Investigate as there might be a bug with `end` keyword
        assert self.current_token.typ == TokenType.KEYWORD
        assert self.current_token.image == "if"
        self.next()
        assert not self.ended, "Expected expression after `if` but found `EOF`"
        condition = self.expression()
        assert not self.ended, "Expected `:` or `do` but found `EOF`"
        body = self.enter_body()
        elifs: list[tuple[E.Expression, E.Expression]] = []
        while not self.ended:
            match (self.current_token, self.peek_next):
                case (Token(TokenType.KEYWORD, "else"), Token(TokenType.KEYWORD, "if")):
                    self.next(2)
                    assert (
                        not self.ended
                    ), "Expected expression after `if` but found `EOF`"
                    elif_condition = self.expression()
                    assert not self.ended, "Expected `:` or `do` but found `EOF`"
                    elif_body = self.enter_body()
                    elifs.append((elif_condition, elif_body))
                case (
                    Token(TokenType.KEYWORD, "else"),
                    Token(TokenType.KEYWORD, "do") | Token(TokenType.COLON),
                ):
                    self.next()
                    else_body = self.enter_body()
                    return E.IfExpression(condition, body, elifs, else_body)
                case _:
                    break
        return E.IfExpression(condition, body, elifs, None)

    def parenthesized_expression(self) -> E.Expression:
        assert self.current_token.typ == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        expr = self.expression()
        assert not self.ended, "Expected ')' but found 'EOF'"
        assert (
            self.current_token.typ == TokenType.RIGHT_PAREN
        ), f"Expected ')' but found {self.current_token.image}"
        return expr

    def function_call_argument(self) -> E.FunctionCallArgument:
        if self.current_token.typ == TokenType.IDENTIFIER_LITERAL:
            assert self.peek_next is not None, "Expected ?? but found EOF"
            if self.peek_next.typ == TokenType.DEFINE:
                name = self.current_token.image
                self.next(2)
                assert not self.ended, "Expected expression after `:=` but found `EOF`"
                return E.FunctionCallArgument(name, value=self.expression())
        return E.FunctionCallArgument(None, self.expression())

    def expression_statement(self) -> S.Statement:
        return S.ExpressionStatement(self.expression())

    def while_statement(self) -> S.Statement:
        assert self.current_token.typ == TokenType.KEYWORD
        assert self.current_token.image == "while"
        self.next()
        assert not self.ended, "Expected expression after `while` but found `EOF`"
        condition = self.expression()
        assert not self.ended, "Expected `:` or `do` but found `EOF`"
        body = self.enter_body()
        return S.WhileStatement(condition, body)

    def for_statement(self) -> S.Statement:
        assert self.current_token.typ == TokenType.KEYWORD
        assert self.current_token.image == "for"
        self.next()
        assert not self.ended, "Expected identifier after `for` but found `EOF`"
        assert self.current_token.typ == TokenType.IDENTIFIER_LITERAL
        var = E.IdentifierExpression(self.current_token.image)
        self.next()
        assert not self.ended, "Expected `in` but found `EOF`"
        assert (
            self.current_token.typ == TokenType.KEYWORD
            and self.current_token.image == "in"
        )
        self.next()
        assert not self.ended, "Expected expression after `in` but found `EOF`"
        collection = self.expression()
        body = self.enter_body()
        return S.ForStatement(var, collection, body)

    def assignment_statement(self) -> S.Statement:
        is_mutable = False
        if (
            self.current_token.typ == TokenType.KEYWORD
            and self.current_token.image == "mut"
        ):
            is_mutable = True
            self.next()
        assert not self.ended, "Expected identifier but found `EOF`"
        assert (
            self.current_token.typ == TokenType.IDENTIFIER_LITERAL
        ), f"Expected identifier but found `{self.current_token.image}`"
        identifier = E.IdentifierExpression(self.current_token.image)
        self.next()
        typ = E.TypeDeduceExpression()
        assert not self.ended, "Expected `:` but found `EOF`"
        assert (
            self.current_token.typ == TokenType.COLON
        ), f"Expected `:` but found `{self.current_token.image}`"
        self.next()
        assert (
            not self.ended
        ), "Expected type expression or `=` after `:` but found `EOF`"
        if self.current_token.typ == TokenType.DEFINE:
            self.next()
            assert not self.ended, "Expected expression after `=` but found `EOF`"
            value = self.expression()
            return S.AssignmentStatement(identifier, value, typ, is_mutable)
        typ = self.type_expression()
        assert not self.ended, "Expected `=` but found `EOF`"
        assert (
            self.current_token.typ == TokenType.DEFINE
        ), f"Expected `=` but found `{self.current_token.image}`"
        self.next()
        assert not self.ended, "Expected expression after `=` but found `EOF`"
        value = self.expression()
        return S.AssignmentStatement(identifier, value, typ, is_mutable)

    def function_definition_statement(self) -> S.FunctionDefinitionStatement:
        # TODO: investigate whether this can be simplified
        assert self.current_token.typ == TokenType.KEYWORD
        assert self.current_token.image == "fn"
        self.next()
        assert not self.ended, "Expected identifier after `fn` but found `EOF`"
        assert (
            self.current_token.typ == TokenType.IDENTIFIER_LITERAL
        ), "Expected identifier after `fn` but found `{self.current_token.image}`"
        name = self.current_token.image
        self.next()
        assert not self.ended, "Expected `(` but found `EOF`"
        assert self.current_token.typ == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        # parse arguments
        args: list[S.FunctionArgument] = []
        while not self.ended:
            if self.current_token.typ == TokenType.RIGHT_PAREN:
                break
            if self.current_token.typ == TokenType.COMMA:
                assert (
                    self.peek_next is not None
                ), "Expected expression after `,` but found `EOF`"
                assert (
                    self.peek_next.typ != TokenType.RIGHT_PAREN
                ), "Expected expression after `,` but found `)`"
                assert (
                    self.peek_next.typ != TokenType.COMMA
                ), "Expected expression after `,` but found `,`"
                self.next()
            else:
                is_mutable = False
                if (
                    self.current_token.typ == TokenType.KEYWORD
                    and self.current_token.image == "mut"
                ):
                    is_mutable = True
                    self.next()
                assert not self.ended, "Expected identifier but found `EOF`"
                assert (
                    self.current_token.typ == TokenType.IDENTIFIER_LITERAL
                ), f"Expected identifier but found `{self.current_token.image}`"
                image = self.current_token.image
                self.next()
                assert not self.ended, "Expected `:` but found `EOF`"
                assert self.current_token.typ == TokenType.COLON, "Expected `:`"
                self.next()
                assert not self.ended, "Expected type but found `EOF`"
                typ = self.type_expression()
                assert not self.ended, "Expected `)` but found `EOF`"
                default_value: E.Expression | None = None
                if self.current_token.typ == TokenType.DEFINE:
                    self.next()
                    assert (
                        not self.ended
                    ), "Expected expression after `:=` but found `EOF`"
                    default_value = self.expression()
                args.append(S.FunctionArgument(image, typ, is_mutable, default_value))
        assert (
            self.current_token.typ == TokenType.RIGHT_PAREN
        ), "Expected ')', unterminated function definition"
        self.next()
        assert not self.ended, "Expected type but found `EOF`"
        typ = E.TypeDeduceExpression()
        if not self.current_token.match(TokenType.COLON, TokenType.KEYWORD):
            typ = self.type_expression()
        assert not self.ended, "Expected `:` or `do` but found `EOF`"
        body = self.enter_body()
        statement = S.FunctionDefinitionStatement(name, args, body, typ)
        self.functions.append(statement)
        return statement

    def struct_member(self) -> S.StructField:
        is_mutable = False
        if (
            self.current_token.typ == TokenType.KEYWORD
            and self.current_token.image == "mut"
        ):
            self.next()
            is_mutable = True
        assert not self.ended, "Expected identifier but found `EOF`"
        assert self.current_token.typ == TokenType.IDENTIFIER_LITERAL
        name = self.current_token.image
        self.next()
        assert not self.ended, "Expected `:` but found `EOF`"
        assert self.current_token.typ == TokenType.COLON, "Expected `:`"
        self.next()
        assert not self.ended, "Expected type but found `EOF`"
        typ = self.type_expression()
        return S.StructField(name, typ, is_mutable)

    def struct_method(self) -> S.StructMethod:
        assert not self.ended, "Expected identifier but found `EOF`"
        assert (
            self.current_token.typ == TokenType.KEYWORD
        ), f"Expected `fn` but found `{self.current_token.image}`"
        assert (
            self.current_token.image == "fn"
        ), f"Expected `fn` but found `{self.current_token.image}`"
        func_def = self.function_definition_statement()
        return S.StructMethod.from_function(func_def)

    def struct_definition_statement(self) -> S.Statement:
        assert self.current_token.typ == TokenType.KEYWORD
        assert self.current_token.image == "struct"
        self.next()
        assert not self.ended, "Expected identifier after `struct` but found `EOF`"
        assert (
            self.current_token.typ == TokenType.IDENTIFIER_LITERAL
        ), f"Expected identifier after `struct` but found `{self.current_token.image}`"
        name = self.current_token.image
        self.next()
        assert not self.ended, "Expected member or end of struct but found `EOF`"
        fields: list[S.StructField] = []
        methods: list[S.StructMethod] = []
        while not self.ended:
            match self.current_token:
                case Token(TokenType.KEYWORD, "fn"):
                    methods.append(self.struct_method())
                case (
                    Token(TokenType.IDENTIFIER_LITERAL)
                    | Token(TokenType.KEYWORD, "mut")
                ):
                    fields.append(self.struct_member())
                case Token(TokenType.KEYWORD, "end"):
                    break
                case _:
                    raise Exception(
                        f"Expected member or end of struct but found `{self.current_token.image}`"
                    )
        self.next()
        statement = S.StructDefinitionStatement(name, fields, methods)
        self.structs.append(statement)
        return statement

    def import_statement(self) -> S.Statement:
        assert self.current_token.typ == TokenType.KEYWORD
        assert self.current_token.image == "import"
        self.next()
        assert not self.ended, "Expected identifier after `import` but found `EOF`"
        assert (
            self.current_token.typ == TokenType.IDENTIFIER_LITERAL
        ), f"Expected identifier after `import` but found `{self.current_token.image}`"
        path_parts = [self.current_token.image]
        self.next()
        while not self.ended and self.current_token.typ == TokenType.SLASH:
            self.next()
            assert not self.ended, "Expected identifier after '/'"
            assert (
                self.current_token.typ == TokenType.IDENTIFIER_LITERAL
            ), "Expected identifier after '/'"
            path_parts.append(self.current_token.image)
            self.next()
        statement = S.ImportStatement("/".join(path_parts))
        self.imports.append(statement)
        return statement

    def statement(self) -> S.Statement:
        match self.current_token:
            case Token(TokenType.KEYWORD, "struct"):
                return self.struct_definition_statement()
            case Token(TokenType.KEYWORD, "fn"):
                return self.function_definition_statement()
            case Token(TokenType.KEYWORD, "mut"):
                return self.assignment_statement()
            case Token(TokenType.IDENTIFIER_LITERAL):
                if self.peek_next and self.peek_next.match(
                    TokenType.COLON, TokenType.DEFINE
                ):
                    return self.assignment_statement()
            case Token(TokenType.KEYWORD, "for"):
                return self.for_statement()
            case Token(TokenType.KEYWORD, "while"):
                return self.while_statement()
            case Token(TokenType.KEYWORD, "import"):
                return self.import_statement()
        return self.expression_statement()

    def program(self) -> Program:
        statements = []
        while not self.ended:
            statements.append(self.statement())
        return Program(
            tuple(statements),
            tuple(self.structs),
            tuple(self.functions),
            tuple(self.imports),
        )

    def parse(self) -> Program:
        try:
            return self.program()
        except AssertionError as e:
            if self.context.current_file:
                file = self.context.current_directory / self.context.current_file
                print(f"{file}:{self.current_token.line}:{self.current_token.column}")

            raise e
