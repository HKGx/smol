from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache


class TokenType(Enum):
    # Single-character tokens and operators
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    LEFT_BRACE = "{"
    RIGHT_BRACE = "}"
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
    EQUALS = "="
    # Two or more character tokens and operators
    NOT_EQUALS = "!="
    SMALLER_OR_EQUAL_THAN = "<="
    GREATER_OR_EQUAL_THAN = ">="
    # Identifier tokens
    IDENTIFIER = auto()
    # Literal tokens
    INTEGER_LITERAL = auto()
    FLOAT_LITERAL = auto()
    STRING_LITERAL = auto()
    KEYWORD = auto()
    IDENTIFIER_LITERAL = auto()

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


KEYWORDS = ["if", "else", "let", "do", "end"]


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
        if self.ended:
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
            self.increment()
        return Token(
            type=TokenType.INTEGER_LITERAL,
            image=self.source[start: self.current_source_idx],
            line=self.current_line,
            column=self.current_column,
        )

    def identifier_literal(self) -> Token:
        """
        Parse identifier literal and return Token
        """
        start = self.current_source_idx
        while not self.ended and (self.current_character.isalnum() or self.current_character == "_"):
            self.increment()
        image = self.source[start: self.current_source_idx]

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
            if self.current_character in TokenType.first_characters():
                match (self.current_character, self.peek):
                    case (">" | "<" | "!", "="):
                        self._tokens.append(
                            Token(
                                type=TokenType(
                                    self.current_character + self.peek),
                                image=self.current_character + self.peek,
                                line=self.current_line,
                                column=self.current_column,
                            )
                        )
                        self.increment()
                    case _:
                        self._tokens.append(
                            Token(
                                type=TokenType(self.current_character),
                                image=self.current_character,
                                line=self.current_line,
                                column=self.current_column,
                            )
                        )
                self.increment()
            elif self.current_character.isdigit():  # integer literal
                self._tokens.append(self.integer_literal())
            elif self.current_character.isalpha():  # identifier literal
                self._tokens.append(self.identifier_literal())
            else:
                print(f"Unknown character {self.current_character}")
                assert False, "other tokens not yet implemented"
        return self._tokens
