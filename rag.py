from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# 读取文档
loader = TextLoader(
    "knowledge/ai_notes.txt",
    encoding="utf-8"
)

documents = loader.load()


# 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)

print("chunk数量:", len(chunks))


# Qwen3 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name="./models/qwen3-embedding-0.6b",
    model_kwargs={
        "device": "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# FAISS
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS数据库创建成功")


# 测试检索
query = "什么是RAG？"

results = vectorstore.similarity_search(
    query,
    k=2
)


print("\n搜索结果:")

for doc in results:
    print("----------------")
    print(doc.page_content)