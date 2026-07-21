import os

from dotenv import load_dotenv


load_dotenv()


# DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500


# DeepSeek Token 价格，单位：元/百万 Token
CACHE_HIT_PRICE = 0.02
CACHE_MISS_PRICE = 1.0
OUTPUT_TOKEN_PRICE = 2.0




# Embedding 配置
EMBEDDING_MODEL_NAME = (
    "./models/qwen3-embedding-0.6b"
)

EMBEDDING_DEVICE = "cpu"


# 文本切分配置
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# 检索配置
RETRIEVAL_K = 6
MAX_RETRIEVAL_DISTANCE = 1.2


# PDF 上传配置
MAX_PDF_COUNT = 10
MAX_TOTAL_PDF_SIZE_MB = 50