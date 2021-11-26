from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, TypeVar

ScopeValue = TypeVar("ScopeValue")


class Scope(dict[str, ScopeValue]):
    parent: Optional["Scope"] = None

    @classmethod
    def from_dict(cls, dict_: dict[str, ScopeValue]):
        new = cls()
        for key, value in dict_.items():
            new[key] = value
        return new

    def __init__(self, parent: "Scope" = None):
        super().__init__()
        self.parent = parent

    def rec_contains(self, value: str) -> bool:
        if value in self:
            return True
        if self.parent is None:
            return False
        return self.parent.rec_contains(value)

    def rec_get(self, key: str) -> ScopeValue:
        if key in self:
            return self[key]
        if self.parent is None:
            raise KeyError(key)
        return self.parent.rec_get(key)

    def rec_set(self, key: str, value: ScopeValue) -> bool:
        if self.parent is None:
            self[key] = value
            return True
        if self.parent is not None:
            if self.parent.rec_contains(key):
                return self.parent.rec_set(key, value)
            self[key] = value
            return True
        return False

    def spawn_child(self):
        return Scope(parent=self)


to = TypeVar("to")


@dataclass()
class StageContext:
    current_directory: Path
    current_file: Optional[str] = None
    import_stack: list[str] = field(default_factory=list)


def resolve_module_path(file_dir: Path, module_name: str) -> Path:
    # module path is relative to the current working directory
    # module name "foo" is resolved to "foo.smol"
    # module name "foo.bar" is resolved to "foo/bar.smol"
    if module_name.startswith("std."):
        # strip "std." prefix
        module_name = module_name[4:]
        std_path = Path(__file__).parent.parent / "std"
        return Path(std_path / f"{module_name}.smol")
    submodules = module_name.split(".")
    if len(submodules) == 1:
        return file_dir / (module_name + ".smol")
    else:
        dir = file_dir
        for submodule in submodules[:-1]:
            dir = dir / submodule
        return dir / (submodules[-1] + ".smol")


class SourcePositionable(Protocol):
    def source_position(self) -> str:
        ...
