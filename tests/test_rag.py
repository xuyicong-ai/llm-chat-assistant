from langchain_core.documents import Document

from rag import (
    build_context,
    retrieve_from_vectorstore
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

def test_build_context_combines_documents():
    # Arrange
    documents = [
        Document(page_content="第一段资料"),
        Document(page_content="第二段资料")
    ]

    # Act
    result = build_context(documents)

    # Assert
    assert result == "第一段资料\n\n第二段资料"


def test_build_context_with_empty_documents():
    # Arrange
    documents = []

    # Act
    result = build_context(documents)

    # Assert
    assert result == ""


def test_retrieve_filters_documents_by_distance():
    # Arrange
    close_document = Document(
        page_content="相关资料",
        metadata={"source": "attention.pdf"}
    )

    far_document = Document(
        page_content="无关资料",
        metadata={"source": "other.pdf"}
    )

    fake_vectorstore = FakeVectorstore([
        (close_document, 0.4),
        (far_document, 1.3)
    ])

    # Act
    result = retrieve_from_vectorstore(
        fake_vectorstore,
        query="什么是注意力机制？",
        k=2,
        max_distance=0.8
    )

    # Assert
    assert result == [close_document]
    assert close_document.metadata[
        "distance_score"
    ] == 0.4


def test_retrieve_passes_query_and_k_to_vectorstore():
    # Arrange
    fake_vectorstore = FakeVectorstore([])

    # Act
    retrieve_from_vectorstore(
        fake_vectorstore,
        query="测试问题",
        k=5
    )

    # Assert
    assert fake_vectorstore.received_query == "测试问题"
    assert fake_vectorstore.received_k == 5


def test_retrieve_filters_by_lecture_document_type():
    # Arrange
    lecture_document = Document(
        page_content="讲稿中的注意力机制",
        metadata={"source": "注意力机制讲稿.pdf"}
    )

    paper_document = Document(
        page_content="论文中的注意力机制",
        metadata={"source": "注意力机制论文.pdf"}
    )

    fake_vectorstore = FakeVectorstore([
        (lecture_document, 0.3),
        (paper_document, 0.2)
    ])

    # Act
    result = retrieve_from_vectorstore(
        fake_vectorstore,
        query="讲稿中如何解释注意力机制？",
        k=2,
        max_distance=0.8
    )

    # Assert
    assert result == [lecture_document]


def test_retrieve_filters_by_paper_document_type():
    # Arrange
    lecture_document = Document(
        page_content="讲稿中的注意力机制",
        metadata={"source": "注意力机制讲稿.pdf"}
    )

    paper_document = Document(
        page_content="论文中的注意力机制",
        metadata={"source": "注意力机制论文.pdf"}
    )

    fake_vectorstore = FakeVectorstore([
        (lecture_document, 0.2),
        (paper_document, 0.3)
    ])

    # Act
    result = retrieve_from_vectorstore(
        fake_vectorstore,
        query="论文如何定义注意力机制？",
        k=2,
        max_distance=0.8
    )

    # Assert
    assert result == [paper_document]


def test_retrieve_keeps_all_types_when_not_specified():
    # Arrange
    lecture_document = Document(
        page_content="讲稿内容",
        metadata={"source": "注意力机制讲稿.pdf"}
    )

    paper_document = Document(
        page_content="论文内容",
        metadata={"source": "注意力机制论文.pdf"}
    )

    fake_vectorstore = FakeVectorstore([
        (lecture_document, 0.2),
        (paper_document, 0.3)
    ])

    # Act
    result = retrieve_from_vectorstore(
        fake_vectorstore,
        query="什么是注意力机制？",
        k=2,
        max_distance=0.8
    )

    # Assert
    assert result == [
        lecture_document,
        paper_document
    ]