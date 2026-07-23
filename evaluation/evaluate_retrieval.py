import argparse
import json
from pathlib import Path

from config import (
    MAX_RETRIEVAL_DISTANCE,
    RETRIEVAL_K
)
from rag import (
    load_saved_vectorstore,
    retrieve_from_vectorstore
)


QUESTIONS_FILE = (
    Path(__file__).parent / "questions.json"
)


def load_questions():
    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def get_document_location(document):
    source_path = document.metadata.get(
        "source",
        ""
    )

    source_name = Path(source_path).name

    # LangChain 保存的 PDF 页码从 0 开始，
    # 网页和 questions.json 中的页码从 1 开始。
    page_number = document.metadata.get("page")

    if page_number is not None:
        page_number += 1

    return source_name, page_number


def is_expected_document(
    document,
    expected_locations
):
    source_name, page_number = (
        get_document_location(document)
    )

    for location in expected_locations:
        expected_source = location["source"]
        expected_pages = location["pages"]

        source_matches = (
            source_name == expected_source
        )

        page_matches = (
            page_number in expected_pages
        )

        if source_matches and page_matches:
            return True

    return False


def evaluate_question(
    vectorstore,
    question_data,
    k=RETRIEVAL_K,
    max_distance=MAX_RETRIEVAL_DISTANCE
):
    question = question_data["question"]

    expected_locations = question_data[
        "expected_locations"
    ]

    documents = retrieve_from_vectorstore(
        vectorstore=vectorstore,
        query=question,
        k=k,
        max_distance=max_distance
    )

    first_relevant_rank = None

    for rank, document in enumerate(
        documents,
        start=1
    ):
        is_relevant = is_expected_document(
            document,
            expected_locations
        )

        if is_relevant:
            first_relevant_rank = rank
            break

    recall_at_k = int(
        first_relevant_rank is not None
    )

    if first_relevant_rank is None:
        reciprocal_rank = 0.0
    else:
        reciprocal_rank = (
            1.0 / first_relevant_rank
        )

    return {
        "id": question_data["id"],
        "question": question,
        "recall_at_k": recall_at_k,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": (
            first_relevant_rank
        ),
        "retrieved_count": len(documents),
        "documents": documents
    }


def print_question_result(result):
    print("\n" + "=" * 60)

    print(
        f'{result["id"]}: '
        f'{result["question"]}'
    )

    print(
        f'Recall: {result["recall_at_k"]}'
    )

    print(
        "首个正确结果排名:",
        result["first_relevant_rank"]
    )

    print(
        f'RR: {result["reciprocal_rank"]:.3f}'
    )

    print("\n检索结果：")

    for rank, document in enumerate(
        result["documents"],
        start=1
    ):
        source_name, page_number = (
            get_document_location(document)
        )

        distance = document.metadata.get(
            "distance_score"
        )

        if distance is None:
            distance_text = "未知"
        else:
            distance_text = f"{distance:.4f}"

        print(
            f"{rank}. "
            f"{source_name} "
            f"第{page_number}页 "
            f"距离={distance_text}"
        )


def evaluate_all(
    vectorstore,
    questions,
    k=RETRIEVAL_K,
    max_distance=MAX_RETRIEVAL_DISTANCE
):
    results = []

    for question_data in questions:
        result = evaluate_question(
            vectorstore=vectorstore,
            question_data=question_data,
            k=k,
            max_distance=max_distance
        )

        results.append(result)

        print_question_result(result)

    question_count = len(results)

    if question_count == 0:
        print("标准问题集为空。")
        return results

    recall_sum = sum(
        result["recall_at_k"]
        for result in results
    )

    reciprocal_rank_sum = sum(
        result["reciprocal_rank"]
        for result in results
    )

    average_recall = (
        recall_sum / question_count
    )

    mean_reciprocal_rank = (
        reciprocal_rank_sum
        / question_count
    )

    print("\n" + "=" * 60)
    print("整体评估结果")
    print(f"问题数量：{question_count}")

    print(
        f"Recall@{k}: "
        f"{average_recall:.2%}"
    )

    print(
        "MRR:",
        f"{mean_reciprocal_rank:.3f}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="评估 RAG 检索质量"
    )

    parser.add_argument(
        "vectorstore_path",
        help="已保存的 FAISS 知识库目录"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=RETRIEVAL_K,
        help=(
            "每道题最多检索多少个片段，"
            f"默认值为 {RETRIEVAL_K}"
        )
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=MAX_RETRIEVAL_DISTANCE,
        help=(
            "允许的最大检索距离，"
            f"默认值为 "
            f"{MAX_RETRIEVAL_DISTANCE}"
        )
    )

    args = parser.parse_args()

    print("正在加载知识库……")

    vectorstore = load_saved_vectorstore(
        args.vectorstore_path
    )

    questions = load_questions()

    print(
        f"已加载 {len(questions)} 道标准问题"
    )

    evaluate_all(
        vectorstore=vectorstore,
        questions=questions,
        k=args.k,
        max_distance=args.max_distance
    )


if __name__ == "__main__":
    main()