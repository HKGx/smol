import os

from .utils import RETURN_TYPE


def overwrite_std_file(interpreter):
    file_struct = interpreter.scope.rec_get("File")
    int_type = interpreter.scope.rec_get("int")
    string_type = interpreter.scope.rec_get("string")

    def open_overwrite(path: RETURN_TYPE):
        file = file_struct(path=path)

        def read_overwrite():
            read = file["__file__"].read()
            i_len = int_type()
            i_len["__value__"] = len(read)
            s = string_type(length=i_len)
            s["__value__"] = read
            return s

        file["__file__"] = open(path["__value__"], "r")  # type: ignore
        file["read"] = read_overwrite
        file["seek"] = lambda i: file["__file__"].seek(i["__value__"])  # type: ignore
        file["close"] = lambda: file["__file__"].close()  # type: ignore
        return file

    interpreter.scope.rec_set("open_file", open_overwrite)


def overwrite_std_os(interpreter):
    def shell_overwrite(value: RETURN_TYPE):
        os.system(value["__value__"])  # type: ignore

    interpreter.scope.rec_set("shell", shell_overwrite)


def overwrite_std_std(interpreter):
    string_type = interpreter.scope.rec_get("string")
    int_type = interpreter.scope.rec_get("int")
    v_prop = "__value__"

    def str_overwrite(value: RETURN_TYPE):
        stringified = str(value[v_prop])  # type: ignore
        i_len = int_type()
        i_len[v_prop] = len(stringified)
        s = string_type(length=i_len)
        s[v_prop] = stringified
        return s

    def char_at_overwrite(value: RETURN_TYPE, index: RETURN_TYPE):
        s_value: str = value[v_prop]  # type: ignore
        i_index: int = index[v_prop]  # type: ignore
        char = s_value[i_index]
        i = int_type()
        i[v_prop] = ord(char)
        return i

    def print_overwrite(value: RETURN_TYPE):
        print(value[v_prop])  # type: ignore

    interpreter.scope.rec_set("str", str_overwrite)
    interpreter.scope.rec_set("print", print_overwrite)
    interpreter.scope.rec_set("_charcode_at", char_at_overwrite)


OVERWRITE_TABLE = {
    "std/file": overwrite_std_file,
    "std/os": overwrite_std_os,
    "std/std": overwrite_std_std,
}
