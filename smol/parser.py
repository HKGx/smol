from dataclasses import dataclass
from textwrap import dedent
from typing import Callable, Literal, TypeVar

from smol.tokenizer import Token, TokenType, Tokenizer


@dataclass
class Expression:
    pass


@dataclass
class IntegerExpression(Expression):
    value: int


@dataclass
class IdentifierExpression(Expression):
    name: str


@dataclass
class BooleanExpression(Expression):
    value: bool


@dataclass
class StringExpression(Expression):
    value: str


@dataclass
class ArrayExpression(Expression):
    elements: list[Expression]


@dataclass
class RangeExpression(Expression):
    left: Expression
    right: Expression
    step: Expression


@dataclass
class EqualityExpression(Expression):
    left: Expression
    sign: Literal['=='] | Literal['!=']
    right: Expression


@dataclass
class ComparisonExpression(Expression):
    left: Expression
    sign: Literal['<'] | Literal[">"] | Literal[">="] | Literal["<="]
    right: Expression


@dataclass
class AdditionExpression(Expression):
    left: Expression
    sign: Literal["+"] | Literal["-"]
    right: Expression


@dataclass
class MultiplicationExpression(Expression):
    left: Expression
    sign: Literal["*"] | Literal["/"]
    right: Expression


@dataclass
class ExponentiationExpression(Expression):
    left: Expression
    sign: Literal["^"]
    right: Expression


@dataclass
class NegationExpression(Expression):
    value: Expression


@dataclass
class PropertyAccessExpression(Expression):
    object: Expression
    property: str


@dataclass
class FunctionCallArgument:
    name: str | None
    value: Expression


@dataclass
class FunctionCallExpression(Expression):
    name: IdentifierExpression
    args: list[FunctionCallArgument]


@dataclass
class IfExpression(Expression):
    condition: Expression
    body: Expression
    else_ifs: list[tuple[Expression, Expression]]
    else_body: Expression | None


@dataclass
class BlockExpression(Expression):
    body: list["Statement"]


@dataclass
class BreakExpression(Expression):
    pass


@dataclass
class ContinueExpression(Expression):
    pass


@dataclass
class Statement:
    pass


@dataclass
class ExpressionStatement(Statement):
    value: Expression


@dataclass
class TypeExpression(Expression):
    pass


@dataclass
class TypeDeduceExpression(TypeExpression):
    pass


@dataclass
class TypeBuiltInExpression(TypeExpression):
    name: Literal["int"] | Literal["string"] | Literal["bool"] | Literal["none"]


@dataclass
class TypeIdentifierExpression(TypeExpression):
    name: str


@dataclass
class AssignmentStatement(Statement):
    name: IdentifierExpression
    value: Expression
    type: TypeExpression
    mutable: bool


@dataclass
class WhileStatement(Statement):
    condition: Expression
    body: Expression


@dataclass
class ForStatement(Statement):
    ident: IdentifierExpression
    value: Expression
    body: Expression


@dataclass
class StructMember(IdentifierExpression):
    type: TypeExpression
    mutable: bool


@dataclass
class StructDefinitionStatement(Statement):
    name: str
    body: list[StructMember]


@dataclass
class FunctionArgument(IdentifierExpression):
    type: TypeExpression
    mutable: bool
    default: Expression | None


@dataclass
class FunctionDefinitionStatement(Statement):
    name: str
    args: list[FunctionArgument]
    body: Expression
    return_type: TypeExpression


@dataclass
class Program:
    statements: list[Statement]


RuleReturnType = TypeVar("RuleReturnType", Expression, Program, Statement)


class Parser:
    """
    Parsers tokens into an AST.
    """
    tokens: list[Token]
    current: int = 0

    @classmethod
    def from_tokenizer(cls, tokenizer: Tokenizer) -> "Parser":
        """
        Creates a parser from a tokenizer.
        If tokenizer has ended it'll return it's tokens.
        If tokenizer hasn't ended it'll run .tokenize() on it.
        """
        if tokenizer.ended:
            return cls(tokenizer._tokens)
        else:
            return cls(tokenizer.tokenize())

    @classmethod
    def from_file(cls, filename: str) -> "Parser":
        return cls.from_tokenizer(Tokenizer.from_file(filename))
    @property
    def current_token(self) -> Token:
        return self.tokens[self.current]

    @property
    def peek_next(self) -> Token | None:
        return self.tokens[self.current + 1] if self.current + 1 < len(self.tokens) else None

    @property
    def ended(self):
        return self.current >= len(self.tokens)

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens

    def next(self, increment: int = 1):
        self.current += increment

    def type_expression(self) -> TypeExpression:
        return self.type_atomic()

    def type_atomic(self) -> TypeExpression:
        expr: TypeExpression
        match self.current_token:
            case Token(TokenType.IDENTIFIER_LITERAL, "int" | "string" | "bool" | "none" as name):
                expr = TypeBuiltInExpression(name)
            case Token(TokenType.IDENTIFIER_LITERAL, name):
                expr = TypeIdentifierExpression(name)
            case _:
                assert False, "Unexpected token"
        self.next()
        return expr

    def expression(self) -> Expression:
        return self.equality()

    def equality(self) -> Expression:
        lhs = self.comparison()
        while (not self.ended and self.current_token.match(TokenType.EQUALS, TokenType.NOT_EQUALS)):
            assert self.current_token.image == "==" or self.current_token.image == "!="
            sign: Literal["==", "!="] = self.current_token.image
            self.next()
            lhs = EqualityExpression(lhs, sign, self.comparison())
        return lhs

    def comparison(self) -> Expression:
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
            lhs = ComparisonExpression(lhs, sign, self.range_expression())
        return lhs

    def range_expression(self) -> Expression:
        lhs = self.addition()
        if (not self.ended and self.current_token.match(TokenType.RANGE)):
            assert self.current_token.image == ".."
            self.next()
            assert not self.ended, "Expected expression after '..'"
            rhs = self.addition()
            if (not self.ended and self.current_token.match(TokenType.RANGE)):
                assert self.current_token.image == ".."
                self.next()
                assert not self.ended, "Expected expression after '..'"
                return RangeExpression(lhs, rhs, self.addition())
            return RangeExpression(lhs, rhs, IntegerExpression(1))
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
            return NegationExpression(self.property_access())
        return self.property_access()

    def property_access(self) -> Expression:
        lhs = self.atomic()
        while (not self.ended and self.current_token.type == TokenType.DOT):
            self.next()
            assert not self.ended, "Expected identifier after '.'"
            assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, "Expected identifier after '.'"
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
                if (self.peek_next and self.peek_next.type == TokenType.LEFT_PAREN):
                    expr = self.function_call()
                else:
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
        assert self.current_token.type == TokenType.RIGHT_PAREN, "Expected ')'"
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

    def function_call(self) -> Expression:
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL
        name = IdentifierExpression(self.current_token.image)
        self.next()
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
        assert self.current_token.type == TokenType.RIGHT_PAREN, "Expected ')'"
        return FunctionCallExpression(name, args)

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
        assert self.current_token.type == TokenType.DEFINE, "Expected `:=`"
        self.next()
        assert not self.ended, "Expected expression after `:=` but found `EOF`"
        value = self.expression()
        return AssignmentStatement(identifier, value, typ, is_mutable)

    def function_definition_statement(self) -> Statement:
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
        typ = self.type_expression()
        assert not self.ended, "Expected `:` or `do` but found `EOF`"
        body = self.enter_body()
        return FunctionDefinitionStatement(name, args, body, typ)

    def struct_member(self) -> StructMember:
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
        return StructMember(name, typ, is_mutable)

    def struct_definition_statement(self) -> Statement:
        assert self.current_token.type == TokenType.KEYWORD
        assert self.current_token.image == "struct"
        self.next()
        assert not self.ended, "Expected identifier after `struct` but found `EOF`"
        assert self.current_token.type == TokenType.IDENTIFIER_LITERAL, f"Expected identifier after `struct` but found `{self.current_token.image}`"
        name = self.current_token.image
        self.next()
        assert not self.ended, "Expected member or end of struct but found `EOF`"
        members: list[StructMember] = []
        while not self.ended:
            if self.current_token.type == TokenType.KEYWORD and self.current_token.image == "end":
                break
            members.append(self.struct_member())
        self.next()
        return StructDefinitionStatement(name, members)

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
        return self.expression_statement()

    def program(self) -> Program:
        statements = []
        while not self.ended:
            statements.append(self.statement())
        return Program(statements)

    def _try_parse(self, rule: Callable[[], RuleReturnType]) -> RuleReturnType | None:
        start_pos = self.current
        try:
            return rule()
        except AssertionError:
            self.current = start_pos
            return None
