from __future__ import annotations

from quantmcp.schema.complexity import (
    RawSchemaFeatures,
    _has_union,
    _max_depth,
    _prop_count,
    compute_sci,
    extract_features,
)


def test_max_depth_flat_schema():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    assert _max_depth(schema) == 1


def test_max_depth_nested_schema():
    schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }
    assert _max_depth(schema) == 2


def test_prop_count():
    schema = {"properties": {"a": {}, "b": {}, "c": {}}}
    assert _prop_count(schema) == 3


def test_has_union_top_level():
    assert _has_union({"oneOf": [{"type": "string"}, {"type": "integer"}]}) is True


def test_has_union_nested():
    schema = {"properties": {"x": {"anyOf": [{"type": "string"}]}}}
    assert _has_union(schema) is True


def test_has_union_false():
    assert _has_union({"properties": {"x": {"type": "string"}}}) is False


def test_extract_features():
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    f = extract_features("get_weather", schema, description="Get the weather")
    assert f.name == "get_weather"
    assert f.depth == 1
    assert f.prop_count == 1
    assert f.has_union is False
    assert f.description_len == len("Get the weather")


def test_compute_sci_empty():
    assert compute_sci([]) == {}


def test_compute_sci_more_complex_scores_higher():
    simple = RawSchemaFeatures(
        name="simple", depth=1, prop_count=1, has_union=False, description_len=10
    )
    complex_ = RawSchemaFeatures(
        name="complex", depth=4, prop_count=10, has_union=True, description_len=200
    )
    scores = compute_sci([simple, complex_])
    assert scores["complex"] > scores["simple"]


def test_compute_sci_identical_features_score_equal():
    a = RawSchemaFeatures(name="a", depth=2, prop_count=3, has_union=False, description_len=20)
    b = RawSchemaFeatures(name="b", depth=2, prop_count=3, has_union=False, description_len=20)
    scores = compute_sci([a, b])
    assert scores["a"] == scores["b"] == 0.0


# Real shape of the memory tier's create_entities argument: an array of
# objects, one of whose own properties is itself an array of strings. This is
# the exact pattern the README flags as under-scored by SCI's depth metric.
_NESTED_ARRAY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entityType": {"type": "string"},
                    "observations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "entityType", "observations"],
            },
        }
    },
    "required": ["entities"],
}


def test_max_depth_recurses_into_array_of_objects():
    # entities(+1) -> item object(+1) -> observations property(+1) ->
    # its own items(+1) = 4, vs. 1 under the old array-stops-recursion bug.
    assert _max_depth(_NESTED_ARRAY_SCHEMA) == 4


def test_max_depth_recurses_into_plain_array_of_strings():
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    # tags(+1) -> its string item(+1) = 2, vs. 1 under the old bug.
    assert _max_depth(schema) == 2


def test_prop_count_recurses_into_array_of_objects():
    # top-level "entities" (1) + the item object's 3 properties (name,
    # entityType, observations) = 4, vs. 1 under the old top-level-only bug.
    assert _prop_count(_NESTED_ARRAY_SCHEMA) == 4


def test_prop_count_unaffected_for_plain_nested_object():
    # A nested *object* property (not behind an array) is deliberately still
    # counted as a single property at the parent level -- only the
    # array-items gap was ever measured as wrong, so this shape's count is
    # unchanged by the fix.
    schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
            }
        },
    }
    assert _prop_count(schema) == 1
