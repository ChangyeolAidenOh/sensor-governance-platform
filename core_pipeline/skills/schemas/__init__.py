"""core_pipeline/skills/schemas — JSON Schemas of Skill contract v1.0. The servers validate their OWN output envelopes
against these before returning; the client validates inputs before sending; goldens and H0 tests validate both.
A schema violation inside a server is reported as ERROR(SCHEMA_INVALID), never silently passed on."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

HERE = Path(__file__).parent
TOOLS = ("gate_check", "detect_diagnose", "spend_query", "retrieve", "work_order")


@lru_cache(maxsize=None)
def _schema(name: str) -> dict:
    return json.loads((HERE / f"{name}.schema.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _registry() -> Registry:
    reg = Registry()
    for n in ("envelope",) + TOOLS:
        reg = reg.with_resource(_schema(n)["$id"], Resource.from_contents(_schema(n)))
    return reg


def _validator(schema: dict) -> Draft202012Validator:
    return Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())


def errors(schema: dict, obj) -> list[str]:
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in _validator(schema).iter_errors(obj)]


def _sub(tool: str, sub: str) -> dict:
    # validate through a $ref into the registered file so that the file's own '#/$defs/...' refs resolve
    return {"$ref": f"{_schema(tool)['$id']}#/{sub}"}


def validate_input(tool: str, obj: dict, sub: str = "input") -> list[str]:
    return errors(_sub(tool, sub), obj)


def validate_envelope(obj: dict) -> list[str]:
    return errors(_schema("envelope"), obj)


def validate_payload(tool: str, obj: dict, sub: str = "payload") -> list[str]:
    return errors(_sub(tool, sub), obj)


def validate_result(tool: str, env: dict, sub: str = "payload") -> list[str]:
    """Envelope + (when OK) payload of that tool. Used by servers on their own output."""
    errs = validate_envelope(env)
    if env.get("status") == "OK" and env.get("payload") is not None:
        errs += validate_payload(tool, env["payload"], sub)
    return errs
