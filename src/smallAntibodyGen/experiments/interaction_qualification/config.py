"""Closed-schema qualification config and the strict-JSON helpers ``records`` shares (pure stdlib).

The config carries no paths and no dataset: source/checkpoint/device/run paths are CLI
arguments. Every key is required, unknown keys are refused, numbers are never coerced
(``True`` is not an integer or a real, ``1.0`` is not an integer) and JSON with duplicate
keys or ``NaN``/``Infinity`` is refused. Nothing here tunes a hyperparameter or switches
test evaluation on implicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

CONFIG_SCHEMA = "aa-qualification/1"
PARTITIONS: Tuple[str, ...] = ("train", "development", "test")
GEOMETRIES: Tuple[str, ...] = ("suffix", "subset")
FULL_MASK = "full"

MaskCount = Union[int, str]


class SchemaError(ValueError):
    """A config, manifest or record broke its closed schema; the whole input is rejected."""


# --- strict JSON ----------------------------------------------------------------------------


def _refuse_constant(name: str) -> Any:
    raise SchemaError(f"non-finite JSON constant {name} is not allowed")


def _refuse_non_finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise SchemaError(f"JSON number {text} is not finite")
    return value


def _refuse_duplicates(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SchemaError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def strict_json_loads(text: str) -> Any:
    """``json.loads`` that refuses duplicate keys and non-finite numbers/constants."""
    if not isinstance(text, str):
        raise SchemaError(f"JSON text must be a str, got {type(text).__name__}")
    try:
        return json.loads(
            text,
            object_pairs_hook=_refuse_duplicates,
            parse_constant=_refuse_constant,
            parse_float=_refuse_non_finite,
        )
    except SchemaError:
        raise
    except (ValueError, RecursionError) as exc:
        raise SchemaError(f"invalid JSON: {exc}") from None


def read_file_bytes(path: Union[str, Path], what: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except (OSError, ValueError) as exc:
        raise SchemaError(f"cannot read {what} {str(path)!r}: {exc}") from None


def decode_utf8(data: bytes, what: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{what} is not valid UTF-8: {exc}") from None


# --- small closed-schema validators ---------------------------------------------------------


def expect_object(value: object, where: str, required: Sequence[str], optional: Sequence[str] = ()) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{where} must be a JSON object, got {type(value).__name__}")
    missing = [key for key in required if key not in value]
    unexpected = sorted(key for key in value if key not in required and key not in optional)
    if missing or unexpected:
        raise SchemaError(f"{where} is a closed schema; missing {missing}, unexpected {unexpected}")
    return value


def expect_int(value: object, where: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{where} must be an integer, got {value!r}")
    if value < minimum:
        raise SchemaError(f"{where} must be >= {minimum}, got {value}")
    return value


def expect_real(value: object, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{where} must be a real number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise SchemaError(f"{where} is too large to be a finite real") from None
    if not math.isfinite(number):
        raise SchemaError(f"{where} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise SchemaError(f"{where} must be positive, got {value!r}")
    return number


def expect_flag(value: object, where: str) -> bool:
    if value is not True and value is not False:
        raise SchemaError(f"{where} must be the JSON literal true or false, got {value!r}")
    return value


def expect_id(value: object, where: str) -> str:
    """A non-empty identifier with no surrounding whitespace (never coerced)."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise SchemaError(f"{where} must be a non-empty string without surrounding whitespace, got {value!r}")
    return value


def expect_text(value: object, where: str) -> str:
    """Free text that is not blank."""
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{where} must be a non-blank string, got {value!r}")
    return value


# --- config ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalGapConfig:
    enabled: bool
    max_pairs: int
    kappa: float


@dataclass(frozen=True)
class DiagnosticsConfig:
    seed: int
    partitions: Tuple[str, ...]
    mask_counts: Tuple[MaskCount, ...]
    geometries: Tuple[str, ...]
    max_examples: int
    max_forwards: int
    local_gap: LocalGapConfig


@dataclass(frozen=True)
class QualificationConfig:
    qualification_id: str
    max_tokens: int
    ridge_alpha: float
    evaluate_test: bool
    diagnostics: DiagnosticsConfig

    def as_dict(self) -> Dict[str, Any]:
        diag = self.diagnostics
        return {
            "schema": CONFIG_SCHEMA,
            "qualification_id": self.qualification_id,
            "max_tokens": self.max_tokens,
            "ridge_alpha": self.ridge_alpha,
            "evaluate_test": self.evaluate_test,
            "diagnostics": {
                "seed": diag.seed,
                "partitions": list(diag.partitions),
                "mask_counts": list(diag.mask_counts),
                "geometries": list(diag.geometries),
                "max_examples": diag.max_examples,
                "max_forwards": diag.max_forwards,
                "local_gap": {
                    "enabled": diag.local_gap.enabled,
                    "max_pairs": diag.local_gap.max_pairs,
                    "kappa": diag.local_gap.kappa,
                },
            },
        }

    def content_sha256(self) -> str:
        text = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        return hashlib.sha256(text.encode("ascii")).hexdigest()


_CONFIG_KEYS = ("schema", "qualification_id", "max_tokens", "ridge_alpha", "evaluate_test", "diagnostics")
_DIAGNOSTICS_KEYS = (
    "seed",
    "partitions",
    "mask_counts",
    "geometries",
    "max_examples",
    "max_forwards",
    "local_gap",
)
_LOCAL_GAP_KEYS = ("enabled", "max_pairs", "kappa")


def _choice_list(value: object, where: str, allowed: Sequence[str]) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SchemaError(f"{where} must be a non-empty list, got {value!r}")
    for item in value:
        if not isinstance(item, str) or item not in allowed:
            raise SchemaError(f"{where} entries must be among {tuple(allowed)}, got {item!r}")
    if len(set(value)) != len(value):
        raise SchemaError(f"{where} must not repeat an entry, got {value!r}")
    return tuple(value)


def _mask_counts(value: object) -> Tuple[MaskCount, ...]:
    where = "diagnostics.mask_counts"
    if not isinstance(value, list) or not value:
        raise SchemaError(f"{where} must be a non-empty list, got {value!r}")
    counts = []
    for item in value:
        if isinstance(item, str):
            if item != FULL_MASK:
                raise SchemaError(f"{where} string entries must be {FULL_MASK!r}, got {item!r}")
            counts.append(item)
        else:
            counts.append(expect_int(item, f"{where} entry", minimum=1))
    if len(set(counts)) != len(counts):
        raise SchemaError(f"{where} must not repeat an entry, got {value!r}")
    return tuple(counts)


def _parse_local_gap(value: object) -> LocalGapConfig:
    gap = expect_object(value, "diagnostics.local_gap", _LOCAL_GAP_KEYS)
    return LocalGapConfig(
        enabled=expect_flag(gap["enabled"], "diagnostics.local_gap.enabled"),
        max_pairs=expect_int(gap["max_pairs"], "diagnostics.local_gap.max_pairs", minimum=1),
        kappa=expect_real(gap["kappa"], "diagnostics.local_gap.kappa", positive=True),
    )


def _parse_diagnostics(value: object, evaluate_test: bool) -> DiagnosticsConfig:
    diag = expect_object(value, "diagnostics", _DIAGNOSTICS_KEYS)
    partitions = _choice_list(diag["partitions"], "diagnostics.partitions", PARTITIONS)
    if "test" in partitions and not evaluate_test:
        raise SchemaError("diagnostics.partitions names 'test' but evaluate_test is false; test is never implicit")
    return DiagnosticsConfig(
        seed=expect_int(diag["seed"], "diagnostics.seed", minimum=0),
        partitions=partitions,
        mask_counts=_mask_counts(diag["mask_counts"]),
        geometries=_choice_list(diag["geometries"], "diagnostics.geometries", GEOMETRIES),
        max_examples=expect_int(diag["max_examples"], "diagnostics.max_examples", minimum=1),
        max_forwards=expect_int(diag["max_forwards"], "diagnostics.max_forwards", minimum=1),
        local_gap=_parse_local_gap(diag["local_gap"]),
    )


def parse_config(text: str) -> QualificationConfig:
    root = expect_object(strict_json_loads(text), "config", _CONFIG_KEYS)
    if root["schema"] != CONFIG_SCHEMA:
        raise SchemaError(f"config schema must be {CONFIG_SCHEMA!r}, got {root['schema']!r}")
    evaluate_test = expect_flag(root["evaluate_test"], "evaluate_test")
    return QualificationConfig(
        qualification_id=expect_id(root["qualification_id"], "qualification_id"),
        max_tokens=expect_int(root["max_tokens"], "max_tokens", minimum=1),
        ridge_alpha=expect_real(root["ridge_alpha"], "ridge_alpha", positive=True),
        evaluate_test=evaluate_test,
        diagnostics=_parse_diagnostics(root["diagnostics"], evaluate_test),
    )


def load_config(path: Union[str, Path]) -> QualificationConfig:
    return parse_config(decode_utf8(read_file_bytes(path, "config"), "config"))
