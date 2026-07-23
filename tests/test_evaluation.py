from langchain_core.documents import Document

from evaluation.evaluate_retrieval import (
    evaluate_question,
    get_document_location,
    is_expected_document
)


class FakeVectorstore:
    def __init__(self, results):
        self.results = results
        self.received_query = None
        self.received_k = None

    def similarity_search_with_score(
        self,
        query,
        k
    ):
        self.received_query = query
        self.received_k = k

        return self.results[:k]


def test_get_document_location():
    # Arrange
    document = Document(
        page_content="测试内容",
        metadata={
            "source": (
                "uploads/许易璁 论文.pdf"
            ),
            "page": 23
        }
    )

    # Act
    source_name, page_number = (
        get_document_location(document)
    )

    # Assert
    assert source_name == "许易璁 论文.pdf"

    # metadata 中的 23 是从 0 开始，
    # 用户看到的实际页码应该是 24。
    assert page_number == 24


def test_expected_document_matches_source_and_page():
    # Arrange
    document = Document(
        page_content="分组标准",
        metadata={
            "source": "uploads/许易璁 论文.pdf",
            "page": 23
        }
    )

    expected_locations = [
        {
            "source": "许易璁 论文.pdf",
            "pages": [18, 24]
        }
    ]

    # Act
    result = is_expected_document(
        document,
        expected_locations
    )

    # Assert
    assert result is True


def test_wrong_page_does_not_match():
    # Arrange
    document = Document(
        page_content="目录",
        metadata={
            "source": "uploads/许易璁 论文.pdf",
            "page": 10
        }
    )

    expected_locations = [
        {
            "source": "许易璁 论文.pdf",
            "pages": [24]
        }
    ]

    # Act
    result = is_expected_document(
        document,
        expected_locations
    )

    # Assert
    assert result is False


def test_evaluate_question_calculates_rank():
    # Arrange
    wrong_document = Document(
        page_content="不相关内容",
        metadata={
            "source": "uploads/许易璁 论文.pdf",
            "page": 10
        }
    )

    correct_document = Document(
        page_content="正确内容",
        metadata={
            "source": "uploads/许易璁 论文.pdf",
            "page": 23
        }
    )

    fake_vectorstore = FakeVectorstore([
        (wrong_document, 0.3),
        (correct_document, 0.4)
    ])

    question_data = {
        "id": "q_test",
        "question": "测试问题",
        "expected_locations": [
            {
                "source": "许易璁 论文.pdf",
                "pages": [24]
            }
        ]
    }

    # Act
    result = evaluate_question(
        vectorstore=fake_vectorstore,
        question_data=question_data,
        k=2,
        max_distance=1.2
    )

    # Assert
    assert result["recall_at_k"] == 1
    assert result["first_relevant_rank"] == 2
    assert result["reciprocal_rank"] == 0.5

    assert fake_vectorstore.received_query == (
        "测试问题"
    )

    assert fake_vectorstore.received_k == 2


def test_evaluate_question_handles_miss():
    # Arrange
    wrong_document = Document(
        page_content="不相关内容",
        metadata={
            "source": "uploads/其他文件.pdf",
            "page": 0
        }
    )

    fake_vectorstore = FakeVectorstore([
        (wrong_document, 0.3)
    ])

    question_data = {
        "id": "q_test",
        "question": "测试问题",
        "expected_locations": [
            {
                "source": "许易璁 论文.pdf",
                "pages": [24]
            }
        ]
    }

    # Act
    result = evaluate_question(
        vectorstore=fake_vectorstore,
        question_data=question_data,
        k=1,
        max_distance=1.2
    )

    # Assert
    assert result["recall_at_k"] == 0
    assert result["first_relevant_rank"] is None
    assert result["reciprocal_rank"] == 0.0