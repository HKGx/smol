from dataclasses import dataclass, field
from textwrap import dedent
from typing import Literal
from smol.parser.expressions import *
from smol.parser.statements import *

from smol.lexer import Token, Lexer, TokenType


@dataclass
class Program:
    statements: tuple[Statement, ...]
    structs: tuple[StructDefinitionStatement, ...]
    functions: tuple[FunctionDefinitionStatement, ...]
    imports: tuple[ImportStatement, ...]


class Parser:
    """
    Parses tokens into an AST.
    This can be then fed into Checker or Interpreter.
    """
    tokens: list[Token]
    structs: list[StructDefinitionStatement]
    functions: list[FunctionDefinitionStatement]
    imports: list[ImportStatement]
    current: int = 0

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.structs = []
        self.functions = []
        self.imports = [
            ImportStatement("std.std", add_to_scope=True)
        ]
        self.current = 0

    @classmethod
    def from_lexer(cls, lexer: Lexer) -> "Parser":
        """
        Creates a parser from lexer.
        If lexer has ended it'll return it's tokens.
        If lexer hasn't ended it'll run .tokenize() on it.
        """
        if lexer.ended:
            return cls(lexer._tokens)
        else:
            return cls(lexer.lex())

    @classmethod
    def from_file(cls, filename: str) -> "Parser":
        return cls.from_lexer(Lexer.from_file(filename))

    @property
    def current_token(self) -> Token:
        return self.tokens[self.current]

    @property
    def peek_next(self) -> Token | None:
        return self.tokens[self.current + 1] if self.current + 1 < len(self.tokens) else None

    @property
    def ended(self):
        return self.current >= len(self.tokens)

    def next(self, increment: int = 1):
        self.current += increment

    def edges(self, start: Token) -> tuple[Token, Token]:
        return start, self.current_token

    def type_expression(self) -> TypeExpression:
        return self.type_union_expression()

    def type_union_expression(self) -> TypeExpression:
        start = self.current_token
        expr = self.type_atomic()
        assert not self.ended, "Expected expression after 'or'"
        exprs: list[TypeExpression] = [expr]
        while not self.ended:
            match self.current_token:
                case Token(TokenType.KEYWORD, "or"):
                    self.next()
                    exprs.append(self.type_atomic())
                case _:
                    break
        if len(exprs) == 1:
            return expr
        return TypeUnionExpression(exprs, edges=self.edges(start))

    def type_atomic(self) -> TypeExpression:
        assert not self.ended, "Expected type but got EOF"
        expr: TypeExpression
        edges = self.edges(self.current_token)
        match (self.current_token):
            case Token(TokenType.IDENTIFIER_LITERAL, name):
                expr = TypeIdentifierExpression(name, edges=edges)
            case _:
                assert False, f"Expected type but got {self.current_token.image}"
        self.next()
        return expr

    def expression(self) -> Expression:
        return self.equality()

    def equality(self) -> Expression:
        start = self.current_token
        lhs = self.comparison()
        while (not self.ended and self.current_token.match(TokenType.EQUALS, TokenType.NOT_EQUALS)):
            assert self.current_token.image == "==" or self.current_token.image == "!="
            sign: Literal["==", "!="] = self.current_token.image
            self.next()
            edges = self.edges(start)
            lhs = EqualityExpression(lhs, sign, self.comparison(), edges=edges)
        return lhs

    def comparison(self) -> Expression:
        start = self.current_token
        lhs = self.range_expression()
        while (not self.ended and self.current_token.match(
            TokenType.SMALLER_THAN,
            TokenType.GREATER_THAN,
            TokenType.SMALLER_OR_EQUAL_THAN,
            TokenType.GREATER_OR_EQUAL_THAN)
        ):
            assert (self.current_token.image == "<"
                    or self.current_token.image == ">"
                    or self.current_token.image == "<="
                    or self.current_token.image == ">=")
            sign: Literal["<", ">", "<=", ">="] = self.current_token.image
            self.next()
            edges = self.edges(start)
            lhs = ComparisonExpression(
                lhs, sign, self.range_expression(), edges=edges)
        return lhs

    def range_expression(self) -> Expression:
        start = self.current_token
        lhs = self.addition()
        if (not self.ended and self.current_token.match(TokenType.RANGE)):
            assert self.current_token.image == ".."
            self.next()
            assert not self.ended, "Expected expression after '..'"
            rhs = self.addition()
            edges = self.edges(start)
            step = IntegerExpression(1)
            if (not self.ended and self.current_token.match(TokenType.RANGE)):
                assert self.current_token.image == ".."
                self.next()
                assert not self.ended, "Expected expression after '..'"
                step = self.addition()
                edges = self.edges(start)
            return RangeExpression(lhs, rhs, step, edges=edges)
        return lhs

    def addition(self) -> Expression:
        lhs = self.multiplication()
        while (not self.ended and self.current_token.match(TokenType.PLUS, TokenType.MINUS)):
            assert self.current_token.image == "+" or self.current_token.image == "-"
            sign: Literal["+"] | Literal["-"] = self.current_token.image
            self.next()
            lhs = AdditionExpression(lhs, sign, self.multiplication())
        return lhs

    def multiplication(self) -> Expression:
        lhs = self.exponentiation()
        while (not self.ended and self.current_token.match(TokenType.STAR, TokenType.SLASH)):
            assert self.current_token.image == "*" or self.current_token.image == "/"
            sign: Literal["*"] | Literal["/"] = self.current_token.image
            self.next()
            lhs = MultiplicationExpression(lhs, sign, self.exponentiation())
        return lhs

    def exponentiation(self) -> Expression:
        lhs = self.negation()
        while (not self.ended and self.current_token.type == TokenType.CARET):
            self.next()
            lhs = ExponentiationExpression(
                lhs, "^", self.exponentiation())  # it just works?
        return lhs

    def negation(self) -> Expression:
        if self.current_token.type == TokenType.MINUS:
            self.next()
            return NegationExpression(self.function_call())
        return self.function_call()

    def function_call(self) -> Expression:
        start = self.current_token
        object = self.property_access()
        if self.ended or self.current_token.type != TokenType.LEFT_PAREN:
            return object
        assert not self.ended, "Expected `(` but found `EOF`"
        assert self.current_token.type == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        # parse arguments
        args: list[FunctionCallArgument] = []
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
        assert self.current_token.type == TokenType.RIGHT_PAREN, "Expected ')'"
        edges = self.edges(start)
        self.next()
        return FunctionCallExpression(object, args, edges=edges)

    def property_access(self) -> Expression:
        lhs = self.atomic()
        while (not self.ended and self.current_token.type == TokenType.DOT):
            self.next()
            assert not self.ended, f"Expected identifier after '.' but found `EOF`"
            assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, f"Expected identifier after '.' but found {self.current_token.image}"
            lhs = PropertyAccessExpression(lhs, self.current_token.image)
            self.next()
        return lhs

    def array_literal(self) -> Expression:
        assert self.current_token.type == TokenType.LEFT_BRACKET
        self.next()
        elements = []
        while not self.ended:
            match self.current_token:
                case Token(TokenType.RIGHT_BRACKET):
                    return ArrayExpression(elements)
                case Token(TokenType.COMMA):
                    self.next()
                    assert not self.ended, "Expected expression after comma"
                case _:
                    elements.append(self.expression())
                    assert self.current_token.type in (
                        TokenType.COMMA, TokenType.RIGHT_BRACKET), f"Expected comma or right bracket, but got {self.current_token.image}"
        assert False, "Unexpected end of file, expected closing `]`"

    def atomic(self) -> Expression:
        start = self.current_token
        expr: Expression
        match self.current_token:
            case Token(TokenType.KEYWORD, "break"):
                expr = BreakExpression()
            case Token(TokenType.KEYWORD, "continue"):
                expr = ContinueExpression()
            case Token(TokenType.KEYWORD, "do"):
                expr = self.do_block_expression()
            case Token(TokenType.KEYWORD, "if"):
                expr = self.if_expression()
            case Token(TokenType.INTEGER_LITERAL):
                expr = IntegerExpression(int(self.current_token.image))
            case Token(TokenType.BOOLEAN_LITERAL, image):
                expr = BooleanExpression(image == "true")
            case Token(TokenType.LEFT_BRACKET):
                expr = self.array_literal()
            case Token(TokenType.LEFT_PAREN):
                expr = self.parenthesized_expression()
            case Token(TokenType.STRING_LITERAL):
                expr = StringExpression(self.current_token.image)
            case Token(TokenType.IDENTIFIER_LITERAL):
                expr = IdentifierExpression(self.current_token.image)
            case _:
                # TODO: improve error reporting
                start = max(0, self.current - 10)
                end = min(len(self.tokens), self.current + 10)
                last_tokens_images = [t.image for t in self.tokens[start:end]]
                assert False, dedent(f"""
                Expected integer, identifier, function call or `(` but got `{self.current_token.image}` at {self.current_token.line}:{self.current_token.column}.
                {" ".join(last_tokens_images)}
                """)
        if not isinstance(expr, (BlockExpression, IfExpression)):
            self.next()
        edges = self.edges(start)
        expr.edges = edges
        return expr

    def do_block_expression(self) -> Expression:
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "do"
        self.next()
        statements = []
        while not self.ended:
            if self.current_token.type == TokenType.KEYWORD and self.current_token.image == "end":
                self.next()
                return BlockExpression(statements)
            statements.append(self.statement())
        assert False, "Expected `end` but got end of file"

    def enter_body(self) -> Expression:
        match self.current_token:
            case Token(TokenType.KEYWORD, "do"):
                return self.do_block_expression()
            case Token(TokenType.COLON):
                self.next()
                assert not self.ended, "Expected expression after colon"
                return self.expression()
            case got:
                assert False, f"Expected `do` or `:` but got {got.image} at {got.line}:{got.column}"

    def if_expression(self) -> Expression:
        # TODO: Investigate as there might be a bug with `end` keyword
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "if"
        self.next()
        assert not self.ended, "Expected expression after `if` but found `EOF`"
        condition = self.expression()
        assert not self.ended, "Expected `:` or `do` but found `EOF`"
        body = self.enter_body()
        elifs: list[tuple[Expression, Expression]] = []
        while not self.ended:
            match (self.current_token, self.peek_next):
                case (Token(TokenType.KEYWORD, "else"), Token(TokenType.KEYWORD, "if")):
                    self.next(2)
                    assert not self.ended, "Expected expression after `if` but found `EOF`"
                    elif_condition = self.expression()
                    assert not self.ended, "Expected `:` or `do` but found `EOF`"
                    elif_body = self.enter_body()
                    elifs.append((elif_condition, elif_body))
                case (Token(TokenType.KEYWORD, "else"), Token(TokenType.KEYWORD, "do") | Token(TokenType.COLON)):
                    self.next()
                    else_body = self.enter_body()
                    return IfExpression(condition, body, elifs, else_body)
                case _:
                    break
        return IfExpression(condition, body, elifs, None)

    def parenthesized_expression(self) -> Expression:
        assert self.current_token.type == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        expr = self.expression()
        assert not self.ended, "Expected ')' but found 'EOF'"
        assert self.current_token.type == TokenType.RIGHT_PAREN, f"Expected ')' but found {self.current_token.image}"
        return expr

    def function_call_argument(self) -> FunctionCallArgument:
        if self.current_token.type == TokenType.IDENTIFIER_LITERAL:
            assert self.peek_next is not None, "Expected ?? but found EOF"
            if self.peek_next.type == TokenType.DEFINE:
                name = self.current_token.image
                self.next(2)
                assert not self.ended, "Expected expression after `:=` but found `EOF`"
                return FunctionCallArgument(name, value=self.expression())
        return FunctionCallArgument(None, self.expression())

    def expression_statement(self) -> Statement:
        return ExpressionStatement(self.expression())

    def while_statement(self) -> Statement:
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "while"
        self.next()
        assert not self.ended, "Expected expression after `while` but found `EOF`"
        condition = self.expression()
        assert not self.ended, "Expected `:` or `do` but found `EOF`"
        body = self.enter_body()
        return WhileStatement(condition, body)

    def for_statement(self) -> Statement:
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "for"
        self.next()
        assert not self.ended, "Expected identifier after `for` but found `EOF`"
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL
        var = IdentifierExpression(self.current_token.image)
        self.next()
        assert not self.ended, "Expected `in` but found `EOF`"
        assert self.current_token.type == TokenType.KEYWORD and self.current_token.image == "in"
        self.next()
        assert not self.ended, "Expected expression after `in` but found `EOF`"
        collection = self.expression()
        body = self.enter_body()
        return ForStatement(var, collection, body)

    def assignment_statement(self) -> Statement:
        is_mutable = False
        if self.current_token.type == TokenType.KEYWORD and self.current_token.image == "mut":
            is_mutable = True
            self.next()
        assert self.current_token.type == TokenType.KEYWORD, "Expected `let`"
        assert self.current_token.image == "let", "Expected `let`"
        self.next()
        assert not self.ended, "Expected identifier after `let` but found `EOF`"
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, "Expected identifier after `let` but found `{self.current_token.image}`"
        identifier = IdentifierExpression(self.current_token.image)
        self.next()
        typ = TypeDeduceExpression()
        if self.current_token.type == TokenType.COLON:
            self.next()
            assert not self.ended, "Expected type after `:` but found `EOF`"
            typ = self.type_expression()

        assert not self.ended, "Expected `:=` but found `EOF`"
        assert self.current_token.type == TokenType.DEFINE, f"Expected `:=` but found {self.current_token.image}"
        self.next()
        assert not self.ended, "Expected expression after `:=` but found `EOF`"
        value = self.expression()
        return AssignmentStatement(identifier, value, typ, is_mutable)

    def function_definition_statement(self) -> FunctionDefinitionStatement:
        # TODO: investigate whether this can be simplified
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "fn"
        self.next()
        assert not self.ended, "Expected identifier after `fn` but found `EOF`"
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, "Expected identifier after `fn` but found `{self.current_token.image}`"
        name = self.current_token.image
        self.next()
        assert not self.ended, "Expected `(` but found `EOF`"
        assert self.current_token.type == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        # parse arguments
        args: list[FunctionArgument] = []
        while not self.ended:
            if self.current_token.type == TokenType.RIGHT_PAREN:
                break
            if self.current_token.type == TokenType.COMMA:
                assert self.peek_next is not None, "Expected expression after `,` but found `EOF`"
                assert self.peek_next.type != TokenType.RIGHT_PAREN, "Expected expression after `,` but found `)`"
                assert self.peek_next.type != TokenType.COMMA, "Expected expression after `,` but found `,`"
                self.next()
            else:
                is_mutable = False
                if self.current_token.type == TokenType.KEYWORD and self.current_token.image == "mut":
                    is_mutable = True
                    self.next()
                assert not self.ended, "Expected identifier but found `EOF`"
                assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, f"Expected identifier but found `{self.current_token.image}`"
                image = self.current_token.image
                self.next()
                assert not self.ended, "Expected `:` but found `EOF`"
                assert self.current_token.type == TokenType.COLON, "Expected `:`"
                self.next()
                assert not self.ended, "Expected type but found `EOF`"
                typ = self.type_expression()
                assert not self.ended, "Expected `)` but found `EOF`"
                default_value: Expression | None = None
                if self.current_token.type == TokenType.DEFINE:
                    self.next()
                    assert not self.ended, "Expected expression after `:=` but found `EOF`"
                    default_value = self.expression()
                args.append(FunctionArgument(
                    image, typ, is_mutable, default_value))
        assert self.current_token.type == TokenType.RIGHT_PAREN, "Expected ')', unterminated function definition"
        self.next()
        assert not self.ended, "Expected type but found `EOF`"
        typ = TypeDeduceExpression()
        if not self.current_token.match(TokenType.COLON, TokenType.KEYWORD):
            typ = self.type_expression()
        assert not self.ended, "Expected `:` or `do` but found `EOF`"
        body = self.enter_body()
        statement = FunctionDefinitionStatement(name, args, body, typ)
        self.functions.append(statement)
        return statement

    def struct_member(self) -> StructField:
        is_mutable = False
        if self.current_token.type == TokenType.KEYWORD and self.current_token.image == "mut":
            self.next()
            is_mutable = True
        assert not self.ended, "Expected identifier but found `EOF`"
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL
        name = self.current_token.image
        self.next()
        assert not self.ended, "Expected `:` but found `EOF`"
        assert self.current_token.type == TokenType.COLON, "Expected `:`"
        self.next()
        assert not self.ended, "Expected type but found `EOF`"
        typ = self.type_expression()
        return StructField(name, typ, is_mutable)

    def struct_method(self) -> StructMethod:
        assert not self.ended, "Expected identifier but found `EOF`"
        assert self.current_token.type == TokenType.KEYWORD, f"Expected `fn` but found `{self.current_token.image}`"
        assert self.current_token.image == "fn", f"Expected `fn` but found `{self.current_token.image}`"
        func_def = self.function_definition_statement()
        return StructMethod.from_function(func_def)

    def struct_definition_statement(self) -> Statement:
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "struct"
        self.next()
        assert not self.ended, "Expected identifier after `struct` but found `EOF`"
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, f"Expected identifier after `struct` but found `{self.current_token.image}`"
        name = self.current_token.image
        self.next()
        assert not self.ended, "Expected member or end of struct but found `EOF`"
        fields: list[StructField] = []
        methods: list[StructMethod] = []
        while not self.ended:
            match self.current_token:
                case Token(TokenType.KEYWORD, "fn"):
                    methods.append(self.struct_method())
                case Token(TokenType.IDENTIFIER_LITERAL) | Token(TokenType.KEYWORD, "mut"):
                    fields.append(self.struct_member())
                case Token(TokenType.KEYWORD, "end"):
                    break
                case _:
                    raise Exception(
                        f"Expected member or end of struct but found `{self.current_token.image}`")
        self.next()
        statement = StructDefinitionStatement(name, fields, methods)
        self.structs.append(statement)
        return statement

    def import_statement(self) -> Statement:
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "import"
        self.next()
        assert not self.ended, "Expected identifier after `import` but found `EOF`"
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, f"Expected identifier after `import` but found `{self.current_token.image}`"
        name = self.current_token.image
        self.next()
        while (not self.ended and self.current_token.type == TokenType.DOT):
            self.next()
            assert not self.ended, "Expected identifier after '.'"
            assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, "Expected identifier after '.'"
            name += f".{self.current_token.image}"
            self.next()
        statement = ImportStatement(name)
        self.imports.append(statement)
        return statement

    def statement(self) -> Statement:
        match(self.current_token):
            case Token(TokenType.KEYWORD, "struct"):
                return self.struct_definition_statement()
            case Token(TokenType.KEYWORD, "fn"):
                return self.function_definition_statement()
            case Token(TokenType.KEYWORD, "let" | "mut"):
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
        return Program(tuple(statements), tuple(self.structs), tuple(self.functions), tuple(self.imports))

    def parse(self) -> Program:
        return self.program()
