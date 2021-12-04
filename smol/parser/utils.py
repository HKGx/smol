from dataclasses import dataclass, field

from smol.lexer import Token


@dataclass
class Statement:
    pass


@dataclass
class Expression:
    edges: tuple[Token, Token] | None = field(
        default=None, kw_only=True)

    def source_position(self) -> str:
        if self.edges is None:
            return "?"
        return f"{self.edges[0].line}:{self.edges[0].column}"
