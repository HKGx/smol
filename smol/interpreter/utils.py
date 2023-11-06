type RETURN_TYPE = dict[str, RETURN_TYPE] | list[RETURN_TYPE]


class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass
