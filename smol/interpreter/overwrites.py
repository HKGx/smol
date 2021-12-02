from .utils import RETURN_TYPE
import os


def overwrite_std_file(interpreter):
    file_struct = interpreter.scope.rec_get("File")
    string_type = interpreter.scope.rec_get("string")

    def iopen(path: RETURN_TYPE):
        file = file_struct(path=path)

        def iread():
            s = string_type()
            s["__value__"] = file["__file__"].read()
            return s
        file["__file__"] = open(path["__value__"], "r")  # type: ignore
        file["read"] = iread
        file["seek"] = lambda i: file["__file__"].seek(
            i["__value__"])  # type: ignore
        file["close"] = lambda: file["__file__"].close()  # type: ignore
        return file
    interpreter.scope.rec_set("open_file", iopen)


def overwrite_std_os(interpreter):
    def ishell(value: RETURN_TYPE):
        os.system(value["__value__"])  # type: ignore
    interpreter.scope.rec_set("shell", ishell)


def overwrite_std_std(interpreter):
    string_type = interpreter.scope.rec_get("string")

    def istr(value: RETURN_TYPE):
        s = string_type()
        s["__value__"] = value["__value__"]  # type: ignore
        return s

    def iprint(value: RETURN_TYPE):
        print(value["__value__"])  # type: ignore
    interpreter.scope.rec_set("str", istr)
    interpreter.scope.rec_set("print", iprint)


OVERWRITE_TABLE = {
    "std.file": overwrite_std_file,
    "std.os": overwrite_std_os,
    "std.std": overwrite_std_std,
}
