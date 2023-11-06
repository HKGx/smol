from dataclasses import dataclass
from enum import Enum, auto
from functools import cache


class TokenType(Enum):
    # Two or more character tokens and operators
    DEFINE = "="
    EQUALS = "=="
    NOT_EQUALS = "!="
    SMALLER_OR_EQUAL_THAN = "<="
    GREATER_OR_EQUAL_THAN = ">="
    RANGE = ".."
    # Single-character tokens and operators
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    LEFT_BRACE = "{"
    RIGHT_BRACE = "}"
    LEFT_BRACKET = "["
    RIGHT_BRACKET = "]"
    LEFT_POINTY_BRACKET = "<"
    RIGHT_POINTY_BRACKET = ">"
    COMMA = ","
    DOT = "."
    PLUS = "+"
    MINUS = "-"
    SLASH = "/"
    STAR = "*"
    CARET = "^"
    COLON = ":"
    SEMICOLON = ";"
    SMALLER_THAN = "<"
    GREATER_THAN = ">"

    # Literal tokens

    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    IDENTIFIER_LITERAL = auto()
    BOOLEAN_LITERAL = auto()
    KEYWORD = auto()

    @staticmethod
    @cache
    def first_characters() -> list[str]:
        return [
            str(t.value)[0]
            for t in TokenType.__members__.values()
            if not isinstance(t.value, int)
        ]


@dataclass
class Token:
    typ: TokenType
    image: str
    line: int
    column: int

    def match(self, *types: TokenType):
        return self.typ in types

    def __str__(self):
        return (
            f"Token(type=<{self.typ.name}>"
            f", image=`{self.image}`"
            f", position=[{self.line}:{self.column}])"
        )

    def __repr__(self):
        return self.__str__()

    def source_position(self) -> str:
        return f"{self.line}:{self.column}"
