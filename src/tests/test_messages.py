from agent_sdk.messages import identity, sys, tool_msg, user


def test_message_builders():
    assert sys("s") == {"role": "system", "content": "s"}
    assert user("u") == {"role": "user", "content": "u"}
    assert tool_msg("c1", "ok") == {"role": "tool", "tool_call_id": "c1", "content": "ok"}


def test_identity_render_returns_history():
    history = [user("hi")]
    assert identity(history) is history
