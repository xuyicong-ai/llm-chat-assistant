from prompt import build_rag_prompt


def test_build_rag_prompt_contains_question():
    # Arrange
    question = "什么是注意力机制？"
    context = "注意力机制允许模型关注重要信息。"

    # Act
    result = build_rag_prompt(
        question,
        context
    )

    # Assert
    assert question in result


def test_build_rag_prompt_contains_context():
    # Arrange
    question = "什么是注意力机制？"
    context = "注意力机制允许模型关注重要信息。"

    # Act
    result = build_rag_prompt(
        question,
        context
    )

    # Assert
    assert context in result


def test_build_rag_prompt_contains_answer_rules():
    # Arrange
    question = "什么是注意力机制？"
    context = "注意力机制允许模型关注重要信息。"

    # Act
    result = build_rag_prompt(
        question,
        context
    )

    # Assert
    assert "优先依据参考资料回答" in result
    assert "不要编造" in result