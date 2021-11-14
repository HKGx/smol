from typing import Optional, TypeVar


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
