import os



CSV_PATH_MMSkinQA = "/home/william/dataset/skin/MM-SkinQA/caption.csv"
JSON_PATH_SKINgpt = "/home/william/dataset/skin/SKINgpt/20250717205150_SKINgpt_all.json"
JSON_PATH_skindisease = "/home/william/dataset/skin/skin_disease/shivvamm_skin_disease_20250723010100.json"

IMAGE_DIR_MMSkinQA = "/home/william/dataset/skin/MM-SkinQA/image_HD"
IMAGE_DIR_SKINgpt = "/home/william/dataset/skin/SKINgpt/image_HD"
IMAGE_DIR_skindisease = "/home/william/dataset/skin/skin_disease/image_HD"

DATASET_NAME ="MMSkinQA_SKINgpt_skindisease_HD"

PROJECT_ROOT = "/home/william/model/Skinalor/RAG/RAGDataSet"

DB_DIR = os.path.join(PROJECT_ROOT, DATASET_NAME)
CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
CHROMA_COLLECTION_NAME = r"skin_cases_multivector_"+DATASET_NAME

# model config
EMBEDDING_MODEL_NAME = 'openai/clip-vit-base-patch32'