from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import llm


def make_fake_response():
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        prompt_cache_hit_tokens=200,
        prompt_cache_miss_tokens=800
    )

    message = SimpleNamespace(
        content="这是模拟的模型回答"
    )

    choice = SimpleNamespace(
        message=message
    )

    return SimpleNamespace(
        usage=usage,
        choices=[choice]
    )


@pytest.fixture
def fake_client(monkeypatch):
    client = MagicMock()

    client.chat.completions.create.return_value = (
        make_fake_response()
    )

    monkeypatch.setattr(
        llm,
        "client",
        client
    )

    return client


def test_get_llm_response_returns_content(
    fake_client
):
    # Arrange
    messages = [
        {
            "role": "user",
            "content": "什么是机器学习？"
        }
    ]

    # Act
    result = llm.get_llm_response(messages)

    # Assert
    assert result["content"] == "这是模拟的模型回答"


def test_get_llm_response_calls_correct_model(
    fake_client
):
    # Arrange
    messages = [
        {
            "role": "user",
            "content": "什么是机器学习？"
        }
    ]

    # Act
    llm.get_llm_response(messages)

    # Assert
    call_arguments = (
        fake_client
        .chat
        .completions
        .create
        .call_args
        .kwargs
    )

    assert call_arguments["model"] == "deepseek-chat"
    assert call_arguments["temperature"] == 0.7
    assert call_arguments["max_tokens"] == 500


def test_get_llm_response_calculates_usage(
    fake_client
):
    # Arrange
    messages = [
        {
            "role": "user",
            "content": "什么是机器学习？"
        }
    ]

    # Act
    result = llm.get_llm_response(messages)

    # Assert
    usage = result["usage"]

    assert usage["prompt_tokens"] == 1000
    assert usage["completion_tokens"] == 500
    assert usage["total_tokens"] == 1500
    assert usage["estimated_cost"] == pytest.approx(
        0.001804
    )


def test_get_llm_response_adds_system_prompt(
    fake_client
):
    # Arrange
    messages = [
        {
            "role": "user",
            "content": "你好"
        }
    ]

    # Act
    llm.get_llm_response(messages)

    # Assert
    call_arguments = (
        fake_client
        .chat
        .completions
        .create
        .call_args
        .kwargs
    )

    sent_messages = call_arguments["messages"]

    assert sent_messages[0] == {
        "role": "system",
        "content": llm.SYSTEM_PROMPT
    }

    assert sent_messages[1] == {
        "role": "user",
        "content": "你好"
    }


def test_get_rag_response_builds_rag_messages(
    monkeypatch
):
    # Arrange
    fake_get_llm_response = MagicMock(
        return_value={
            "content": "模拟 RAG 回答",
            "usage": {}
        }
    )

    monkeypatch.setattr(
        llm,
        "get_llm_response",
        fake_get_llm_response
    )

    messages = [
        {
            "role": "user",
            "content": "你好"
        },
        {
            "role": "assistant",
            "content": "你好，有什么可以帮你？"
        },
        {
            "role": "user",
            "content": "什么是注意力机制？"
        }
    ]

    context = "注意力机制允许模型关注重要信息。"

    # Act
    result = llm.get_rag_response(
        messages,
        context
    )

    # Assert
    sent_messages = (
        fake_get_llm_response
        .call_args
        .args[0]
    )

    assert sent_messages[0] == messages[0]
    assert sent_messages[1] == messages[1]

    final_user_message = sent_messages[2]

    assert final_user_message["role"] == "user"

    assert "什么是注意力机制？" in (
        final_user_message["content"]
    )

    assert context in final_user_message["content"]
    assert result["content"] == "模拟 RAG 回答"


def test_get_rag_response_does_not_modify_messages(
    monkeypatch
):
    # Arrange
    fake_get_llm_response = MagicMock(
        return_value={
            "content": "模拟回答",
            "usage": {}
        }
    )

    monkeypatch.setattr(
        llm,
        "get_llm_response",
        fake_get_llm_response
    )

    messages = [
        {
            "role": "user",
            "content": "什么是注意力机制？"
        }
    ]

    original_messages = [
        message.copy()
        for message in messages
    ]

    # Act
    llm.get_rag_response(
        messages,
        context="参考资料"
    )

    # Assert
    assert messages == original_messages


def test_get_llm_response_without_client(
    monkeypatch
):
    # Arrange
    monkeypatch.setattr(
        llm,
        "client",
        None
    )

    messages = [
        {
            "role": "user",
            "content": "你好"
        }
    ]

    # Act 与 Assert
    with pytest.raises(
        RuntimeError,
        match="DEEPSEEK_API_KEY"
    ):
        llm.get_llm_response(messages)