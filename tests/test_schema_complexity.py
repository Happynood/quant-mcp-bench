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
