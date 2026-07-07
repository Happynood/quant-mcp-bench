"""Schema Complexity Index — SCI(schema_j) per spec §4.3.

SCI = a1*z(depth) + a2*z(prop_count) + a3*1[oneOf/anyOf present] + a4*z(desc_len)

z-normalized across the full tool corpus passed to compute_sci, default
alpha_k = 0.25. This is what operationalizes "MCP schemas are messier" for
the H2 regression (does degradation correlate with SCI independent of quant
level) — computed across the 3 real tiers so far (see docs/RUN_REAL.md),
though a real regression/correlation still needs more tiers than that to
mean anything; the metric itself has no execution dependency, which is why
it could be implemented well before enough tiers existed to use it fully.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


def _max_depth(schema: dict[str, Any], depth: int = 0) -> int:
    """Depth of the deepest nested property, traversing both object
    `properties` and array `items` schemas.

    Originally only recursed through `properties`, so an array-of-objects
    argument (e.g. the memory tier's `create_entities([{name, entityType,
    observations: [...]}, ...])`) stopped at the array wrapper itself: an
    array schema has no `properties` key, so the old code returned
    immediately instead of continuing into `items`, undercounting any
    tool whose real complexity lives inside its array arguments.
    """
    if not isinstance(schema, dict):
        return depth
    props = schema.get("properties", {})
    items = schema.get("items")
    child_depths = [_max_depth(v, depth + 1) for v in props.values() if isinstance(v, dict)]
    if isinstance(items, dict):
        child_depths.append(_max_depth(items, depth + 1))
    return max(child_depths, default=depth)


def _prop_count(schema: dict[str, Any]) -> int:
    """Count of tool-argument properties, including those reachable only by
    traversing into an array property's `items` schema (see `_max_depth`'s
    docstring for the same array-items gap). Deliberately still counts a
    nested *object* property's own sub-properties as a single property at
    the parent level, unchanged from the original design -- only the
    array-items gap was ever measured as wrong, so only that gap is fixed.
    """
    if not isinstance(schema, dict):
        return 0
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return 0
    count = len(props)
    for v in props.values():
        if isinstance(v, dict):
            items = v.get("items")
            if isinstance(items, dict):
                count += _prop_count(items)
    return count


def _has_union(schema: dict[str, Any]) -> bool:
    if "oneOf" in schema or "anyOf" in schema:
        return True
    for v in schema.get("properties", {}).values():
        if isinstance(v, dict) and _has_union(v):
            return True
    return False


@dataclass(frozen=True)
class RawSchemaFeatures:
    name: str
    depth: int
    prop_count: int
    has_union: bool
    description_len: int


def extract_features(name: str, schema: dict[str, Any], description: str = "") -> RawSchemaFeatures:
    return RawSchemaFeatures(
        name=name,
        depth=_max_depth(schema),
        prop_count=_prop_count(schema),
        has_union=_has_union(schema),
        description_len=len(description),
    )


def _zscore(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return [0.0 for _ in values]
    return [(v - mean) / stdev for v in values]


def compute_sci(
    features: list[RawSchemaFeatures],
    alphas: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> dict[str, float]:
    """Return {tool_name: SCI} for the given corpus of schema features."""
    if not features:
        return {}
    depths = _zscore([float(f.depth) for f in features])
    props = _zscore([float(f.prop_count) for f in features])
    descs = _zscore([float(f.description_len) for f in features])
    a1, a2, a3, a4 = alphas
    scores: dict[str, float] = {}
    for f, d, p, ds in zip(features, depths, props, descs, strict=True):
        scores[f.name] = a1 * d + a2 * p + a3 * (1.0 if f.has_union else 0.0) + a4 * ds
    return scores
