import dataclasses

from context_layer.context_policy import DEFAULT_POLICY, ContextType


def test_default_policy_rows():
    assert DEFAULT_POLICY["system"].pinned is True
    assert DEFAULT_POLICY["system"].base_priority == 1000
    assert DEFAULT_POLICY["tool"].truncatable is True
    assert DEFAULT_POLICY["doc"].truncatable is True
    assert DEFAULT_POLICY["user"].base_priority == 900


def test_enum_and_str_keys_interoperate():
    assert ContextType.SYSTEM == "system"
    assert DEFAULT_POLICY[ContextType.SYSTEM] is DEFAULT_POLICY["system"]


def test_typepolicy_is_frozen():
    try:
        DEFAULT_POLICY["user"].base_priority = 1
        assert False, "expected frozen"
    except dataclasses.FrozenInstanceError:
        pass
