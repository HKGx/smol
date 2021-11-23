from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from io import TextIOWrapper
from pathlib import Path
from typing import Union


class TokenType(Enum):
    # Two or more character tokens and operators
    DEFINE = ":="
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
    @lru_cache
    def first_characters() -> list[str]:
        return [
            str(t.value)[0] for t in TokenType.__members__.values() if not isinstance(t.value, int)
        ]


@dataclass
class Token:
    type: TokenType
    image: str
    line: int
    column: int

    def match(self, *types: TokenType):
        return self.type in types

    def __str__(self):
        return (f"Token(type=<{self.type.name}>"
                f", image=`{self.image}`"
                f", position=[{self.line}:{self.column}])")

    def __repr__(self):
        return self.__str__()


KEYWORDS = {"if", "else", "mut", "let", "do", "end",
            "while", "for", "in", "break", "continue", "fn", "struct", "import"}


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

    @property
    def peek(self) -> str | None:
        """
        Returns the next character without incrementing current_source_index
        """
        if self.current_source_idx + 1 >= len(self.source):
            return None
        return self.source[self.current_source_idx + 1]

    def increment(self):
        """
        Increments current_source_idx and current_column
        """
        self.current_source_idx += 1
        self.current_column += 1

    def __init__(self, source: str):
        self.source = source

    def skip_whitespace(self):
        """
        Skips whitespaces in source
        """
        while not self.ended and self.current_character.isspace():
            if self.current_character == "\n":
                self.current_line += 1
                self.current_column = 1
            else:
                self.current_column += 1
            self.current_source_idx += 1

    def skip_comment(self):
        """
        Skips comments in source
        """
        assert self.current_character == "#"
        self.increment()
        while not self.ended and self.current_character != "\n":
            self.increment()

    def integer_literal(self) -> Token:
        """
        Parse integer literal and return Token
        """
        start = self.current_source_idx
        while not self.ended and self.current_character.isdigit():
            self.increment()
        image = self.source[start: self.current_source_idx]
        return Token(
            type=TokenType.INTEGER_LITERAL,
            image=self.source[start: self.current_source_idx],
            line=self.current_line,
            column=self.current_column - len(image),
        )

    def string_literal(self) -> Token:
        """
        Parse string literal and return Token
        """
        # not sure about the implementation, but it seems to work
        assert self.current_character == '"'
        self.increment()
        start = self.current_source_idx
        while not self.ended:
            match self.current_character:
                case '"':
                    break
                case "\n":
                    self.current_line += 1
                    self.current_column = 0
                    self.current_source_idx += 1
                    continue
                case "\\":
                    self.increment()
                    self.increment()
                    continue
            self.increment()
        assert not self.ended, "Unterminated string literal"
        assert self.current_character == '"', "Unterminated string literal"
        image = self.source[start: self.current_source_idx]
        self.increment()
        return Token(
            type=TokenType.STRING_LITERAL,
            image=image,
            line=self.current_line,
            column=self.current_column - len(image),
        )

    def identifier_literal(self) -> Token:
        """
        Parse identifier literal and return Token
        """
        start = self.current_source_idx
        while (not self.ended
               and (self.current_character.isalnum()
                    or self.current_character == "_")):
            self.increment()
        image = self.source[start: self.current_source_idx]
        typ: TokenType = TokenType.IDENTIFIER_LITERAL
        if image in KEYWORDS:
            typ = TokenType.KEYWORD
        if image in ("true", "false"):
            typ = TokenType.BOOLEAN_LITERAL
        return Token(
            type=typ,
            image=image,
            line=self.current_line,
            column=self.current_column - len(image),
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
            if self.current_character == "#":
                self.skip_comment()
            elif self.current_character == '"':
                self._tokens.append(self.string_literal())
            elif self.current_character in TokenType.first_characters():
                match (self.current_character, self.peek):
                    case (">" | "<" | "!" | "=" | ":", "=") | (".", "."):
                        self._tokens.append(
                            Token(
                                type=TokenType(
                                    self.current_character + self.peek),
                                image=self.current_character + self.peek,
                                line=self.current_line,
                                column=self.current_column - 2,
                            )
                        )
                        self.increment()
                    case _:
                        self._tokens.append(
                            Token(
                                type=TokenType(self.current_character),
                                image=self.current_character,
                                line=self.current_line,
                                column=self.current_column - 1,
                            )
                        )
                self.increment()
            elif self.current_character.isdigit():  # integer literal
                self._tokens.append(self.integer_literal())
            elif (self.current_character.isalpha()
                  or self.current_character in ("_")):  # identifier literal
                self._tokens.append(self.identifier_literal())
            else:
                assert False, f"Unknown character {self.current_character}"
        return self._tokens

    @classmethod
    def from_file(cls, filename: Union[TextIOWrapper, str, Path]) -> "Tokenizer":
        """
        Returns a Tokenizer instance from a file
        """
        if isinstance(filename, (str, Path)):
            with open(filename, "r") as f:
                return cls(f.read())
        else:
            return cls(filename.read())
