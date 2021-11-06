from enum import Enum, auto
from dataclasses import dataclass
from locale import currency
from sre_parse import State
from typing import Literal
from smol.tokenizer import Token, TokenType


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
class NegationExpression(Expression):
    value: Expression

@dataclass

class FunctionCallExpression(Expression):
    name: IdentifierExpression
    args: list[Expression]
@dataclass
class Statement:
    pass

class ExpressionStatement(Statement):
    value: Expression



@dataclass
class Program:
    statements: list[Statement]


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
        lhs = self.negation()
        while (not self.ended and self.current_token.match(TokenType.STAR, TokenType.SLASH)):
            assert self.current_token.image == "*" or self.current_token.image == "/"
            sign: Literal["*"] | Literal["/"] = self.current_token.image
            self.next()
            lhs = MultiplicationExpression(lhs, sign, self.negation())
            self.next()
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
                assert False, "Expected expression token"
        self.next()
        assert expr is not None, "Expected expression"
        return expr

    def parenthesized_expression(self) -> Expression:
        if (self.current_token.type != TokenType.LEFT_PAREN): 
            assert False
        self.next()
        expr = self.expression()
        self.next()
        if (self.current_token.type != TokenType.RIGHT_PAREN):
            assert False
        self.next()
        return expr
    
    def function_call(self) -> Expression:
        name = IdentifierExpression(self.current_token.image)
        self.next()
        assert self.current_token.type == TokenType.LEFT_PAREN
        # TODO: parse arguments
        self.next()
        assert self.current_token.type == TokenType.RIGHT_PAREN
        self.next()
        return FunctionCallExpression(name, [])




    def statement(self) -> Statement:
        return Statement()

    def program(self) -> Program:
        return Program([])
