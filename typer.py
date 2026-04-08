from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass


class colors:
    RED = "red"


class Exit(Exception):
    def __init__(self, code: int = 0) -> None:
        self.code = code


@dataclass
class _Command:
    name: str
    callback: callable


class Typer:
    def __init__(self, help: str | None = None) -> None:
        self.help = help
        self._commands: dict[str, _Command] = {}

    def command(self, name: str):
        def decorator(func):
            self._commands[name] = _Command(name=name, callback=func)
            return func

        return decorator

    def __call__(self) -> None:
        argv = sys.argv[1:]
        if not argv or argv[0] not in self._commands:
            echo(self.help or "")
            raise SystemExit(1)
        cmd = self._commands[argv[0]]
        fn = cmd.callback
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        values = argv[1:]
        if len(values) < len(params):
            raise SystemExit(1)
        try:
            fn(*values[: len(params)])
        except Exit as exc:
            raise SystemExit(exc.code) from exc


def echo(msg: str) -> None:
    print(msg)


def secho(msg: str, fg: str | None = None, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    print(msg, file=stream)
