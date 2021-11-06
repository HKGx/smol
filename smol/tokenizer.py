from dataclasses import dataclass
from email.mime import image
from enum import Enum, auto
from webbrowser import get


class TokenType(Enum):
    # Single-character tokens and operators
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    PLUS = auto()
    MINUS = auto()
    SLASH = auto()
    STAR = auto()
    CARET = auto()
    SEMICOLON = auto()

    SMALLER_THAN = auto()
    GREATER_THAN = auto()
    EQUALS = auto()
    # Two or more character tokens and operators
    NOT_EQUALS = auto()
    SMALLER_OR_EQUAL_THAN = auto()
    GREATER_OR_EQUAL_THAN = auto()
    # Identifier tokens
    IDENTIFIER = auto()
    # Literal tokens
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    KEYWORD = auto()
    IDENTIFIER_LITERAL = auto()


@dataclass
class Token:
    type: TokenType
    image: str
    line: int
    column: int

    def match(self, *types: TokenType):
        return self.type in types

    def __str__(self):
        return f"Token(type=<{self.type.name}>, image=`{self.image}`, position=[{self.line}:{self.column}])"

    def __repr__(self):
        return self.__str__()


SINGLE_CHAR_TOKENS = {
    "(": TokenType.LEFT_PAREN,
    ")": TokenType.RIGHT_PAREN,
    "{": TokenType.LEFT_BRACE,
    "}": TokenType.RIGHT_BRACE,
    ",": TokenType.COMMA,
    ".": TokenType.DOT,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "/": TokenType.SLASH,
    "*": TokenType.STAR,
    "^": TokenType.CARET,
    ";": TokenType.SEMICOLON,
    "=": TokenType.EQUALS,
    "<": TokenType.SMALLER_THAN,
    ">": TokenType.GREATER_THAN,
}

KEYWORDS = ["if", "else", "let"]


class Tokenizer:
    _tokens: list[Token] = []
    current_source_idx: int = 0
    current_line: int = 1
    current_column: int = 1
    source: str

    @property
    def ended(self) -> bool:
        return self.current_source_idx >= len(self.source)

    @property
    def current_character(self) -> str:
        return self.source[self.current_source_idx]

    def __init__(self, source: str):
        self.source = source

    def skip_whitespace(self):
        """
        Skips whitespaces in source incrementing current_source_index, current_column and current_line if needed
        """
        while not self.ended and self.current_character.isspace():
            if self.current_character == "\n":
                self.current_line += 1
                self.current_column = 0
            else:
                self.current_column += 1
            self.current_source_idx += 1

    def integer_literal(self) -> Token:
        """
        Parse integer literal and return Token
        """
        start = self.current_source_idx
        while not self.ended and self.current_character.isdigit():
            self.current_source_idx += 1
            self.current_column += 1
        return Token(
            type=TokenType.INTEGER_LITERAL,
            image=self.source[start : self.current_source_idx],
            line=self.current_line,
            column=self.current_column,
        )

    def identifier_literal(self) -> Token:
        """
        Parse identifier literal and return Token
        """
        start = self.current_source_idx
        while not self.ended and self.current_character.isalnum():
            self.current_source_idx += 1
            self.current_column += 1

        image = self.source[start : self.current_source_idx]

        return Token(
            type=TokenType.KEYWORD
            if image in KEYWORDS
            else TokenType.IDENTIFIER_LITERAL,
            image=image,
            line=self.current_line,
            column=self.current_column,
        )

    def tokenize(self) -> list[Token]:
        """
        Tokenizes the source string and returns list of Tokens
        """
        self._tokens = []
        self.current_source_idx = 0
        self.current_line = 1
        self.current_column = 1
        while not self.ended:
            self.skip_whitespace()
            if self.ended:
                break
            if self.current_character in SINGLE_CHAR_TOKENS:
                self._tokens.append(
                    Token(
                        SINGLE_CHAR_TOKENS[self.current_character],
                        self.current_character,
                        self.current_line,
                        self.current_column,
                    )
                )
                self.current_source_idx += 1
                self.current_column += 1
            elif self.current_character.isdigit():  # integer literal
                self._tokens.append(self.integer_literal())
            elif self.current_character.isalpha():  # identifier literal
                self._tokens.append(self.identifier_literal())
            else:
                assert False, "other tokens not yet implemented"
        return self._tokens
