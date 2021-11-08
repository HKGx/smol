from dataclasses import dataclass
from typing import Callable, Literal, TypeVar
from smol.tokenizer import Token, TokenType
from textwrap import dedent

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
class ExponentatiotnExpression(Expression):
    left: Expression
    sign: Literal["^"]
    right: Expression

@dataclass
class NegationExpression(Expression):
    value: Expression

@dataclass
class FunctionCallExpression(Expression):
    name: IdentifierExpression
    args: list[Expression]

@dataclass
class Statement:
    pass

@dataclass
class ExpressionStatement(Statement):
    value: Expression

@dataclass
class AssignmentStatement(Statement):
    name: IdentifierExpression
    value: Expression


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

    def next(self):
        self.current += 1

    def expression(self) -> Expression:
        return self.addition()

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
            lhs = ExponentatiotnExpression(lhs, "^", self.exponentiation()) # it just works?
        return lhs


    def negation(self) -> Expression:
        if self.current_token.type == TokenType.MINUS:
            self.next()
            return NegationExpression(self.atomic())
        return self.atomic()


    def atomic(self) -> Expression:
        expr: Expression
        match self.current_token:
            case Token(TokenType.INTEGER_LITERAL):
                expr =  IntegerExpression(int(self.current_token.image))
            case Token(TokenType.LEFT_PAREN):
                expr =  self.parenthesized_expression()
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
        self.next()
        assert expr is not None, "Technically unreachable"
        return expr

    def parenthesized_expression(self) -> Expression:
        assert self.current_token.type == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        expr = self.expression()
        assert self.current_token.type == TokenType.RIGHT_PAREN, "Expected ')'"
        self.next()
        return expr
    
    def function_call(self) -> Expression:
        name = IdentifierExpression(self.current_token.image)
        self.next()
        assert self.current_token.type == TokenType.LEFT_PAREN, "Expected '('"
        self.next()
        # parse arguments
        # TODO: implement named arguments
        args = []
        while (not self.ended and self.current_token.type != TokenType.RIGHT_PAREN):
            if (self.current_token.type == TokenType.COMMA):
                self.next()
            args.append(self.expression())
        assert self.current_token.type == TokenType.RIGHT_PAREN, "Expected ')'"
        self.next()
        return FunctionCallExpression(name, args)


    def expressionStatement(self) -> Statement:
        return ExpressionStatement(self.expression())

    def assignmentStatement(self) -> Statement:
        if self.current_token.type != TokenType.KEYWORD:
            assert False
        if self.current_token.image != "let":
            assert False
        self.next()
        if self.current_token.type != TokenType.IDENTIFIER_LITERAL:
            assert False
        identifier = IdentifierExpression(self.current_token.image)
        self.next()
        assert self.current_token.type == TokenType.EQUALS
        self.next()
        value = self.expression()
        return AssignmentStatement(identifier, value)


    def statement(self) -> Statement:
        if self.current_token.type == TokenType.KEYWORD:
            if self.current_token.image == "let":
                return self.assignmentStatement()
        return self.expressionStatement()

    def program(self) -> Program:
        statements = []
        while (not self.ended):
            statements.append(self.statement())
        return Program(statements)

    def _tryParse(self, rule: Callable[[], RuleReturnType]) -> RuleReturnType | None:
        start_pos = self.current
        try:
            return rule()
        except AssertionError:
            self.current = start_pos
            return None