from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, get_args, get_origin, get_type_hints
from urllib.parse import urlparse


class HttpUrl(str):
    def unicode_string(self) -> str:
        return str(self)


def Field(*, default_factory):
    return {"__default_factory__": default_factory}


class BaseModel:
    def __init__(self, **data: Any) -> None:
        annotations = get_type_hints(self.__class__)
        for name in annotations:
            if name in data:
                value = data[name]
            elif hasattr(self.__class__, name):
                default = getattr(self.__class__, name)
                if isinstance(default, dict) and "__default_factory__" in default:
                    value = default["__default_factory__"]()
                else:
                    value = deepcopy(default)
            else:
                value = None
            setattr(self, name, self._coerce(value, annotations[name]))

    def _coerce(self, value: Any, annotation: Any) -> Any:
        if value is None:
            return None
        origin = get_origin(annotation)
        if origin is None:
            if annotation is HttpUrl:
                parsed = urlparse(str(value))
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError("Invalid URL")
                return HttpUrl(str(value))
            return value

        if origin is list:
            subtype = get_args(annotation)[0]
            return [self._coerce(item, subtype) for item in value]
        if origin is tuple:
            subtypes = get_args(annotation)
            return tuple(self._coerce(v, t) for v, t in zip(value, subtypes))

        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return self._coerce(value, args[0])
        return value

    def model_dump(self) -> dict[str, Any]:
        annotations = get_type_hints(self.__class__)
        return {k: getattr(self, k) for k in annotations.keys()}

    def model_dump_json(self, indent: int | None = None) -> str:
        return json.dumps(self.model_dump(), indent=indent)

    def model_copy(self, *, update: dict[str, Any] | None = None):
        payload = self.model_dump()
        if update:
            payload.update(update)
        return self.__class__(**payload)
