from __future__ import annotations

from router.hashing import hash_object, hash_text


def test_hash_object_is_order_independent():
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert hash_object(a) == hash_object(b)


def test_hash_object_differs_on_value_change():
    assert hash_object({"a": 1}) != hash_object({"a": 2})


def test_hash_text_deterministic():
    assert hash_text("same") == hash_text("same")
    assert hash_text("a") != hash_text("b")
