from typing import Any
import os


RETURN_TYPE = int | float | None | str | list["RETURN_TYPE"] | dict[str, "RETURN_TYPE"]


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass

# TODO: Add support for types inherited from checker


def iprint(value: str) -> None:
    print(value)


def istr(value: int) -> str:
    return str(value)


def iopen_file(path: str) -> dict[str, Any]:
    f = open(path, "r")
    return {
        "path": path,
        "read": f.read,
        "close": f.close,
        "seek": f.seek,
    }


def ishell(cmd: str) -> None:
    os.system(cmd)
