"""Cấu hình chung cho hệ multi-agent dự đoán bệnh tiểu đường."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_PATH = BASE_DIR / "data" / "diabetes_prediction_dataset.csv"
MODELS_DIR = BASE_DIR / "ml" / "models"
FIGURES_DIR = MODELS_DIR / "figures"
KB_DIR = BASE_DIR / "kb"
# Ghi đè bằng biến môi trường khi muốn chạy thử pipeline trên dữ liệu khác (vd. trang mẫu trong tests/fixtures)
RAW_HTML_DIR = Path(os.getenv("KB_RAW_DIR", KB_DIR / "raw_html"))
CACHE_DIR = KB_DIR / "cache"
PROCESSED_DIR = Path(os.getenv("KB_PROCESSED_DIR", KB_DIR / "processed"))
SEED_URLS_PATH = KB_DIR / "seed_urls.csv"
EVAL_DIR = BASE_DIR / "eval"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
KG_LABEL = os.getenv("KG_LABEL", "DKG")  # mọi node của đồ án mang nhãn này để không đụng dữ liệu cũ trong Neo4j

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_DIM = 1024
NUM_CTX = int(os.getenv("NUM_CTX", "16384"))

# Bật/tắt chế độ thinking theo vai trò. Đo trên Advisor: bật thinking sinh ~5.200 token (154s) so với ~310 token (11s)
# khi tắt, chất lượng không khác biệt rõ vì Critic đã kiểm soát phía sau → mặc định tắt hết.
THINK = {role: os.getenv("LLM_THINK", "0") == "1" for role in
         ("supervisor", "intake", "extract", "risk", "knowledge", "advisor", "critic", "single")}

RANDOM_STATE = 42
GREY_MARGIN = 0.15  # |xác suất - ngưỡng| < GREY_MARGIN thì coi là vùng xám
MAX_RISK_RETRIES = 2
MAX_KNOWLEDGE_ROUNDS = 3
MAX_CRITIC_ROUNDS = 2
