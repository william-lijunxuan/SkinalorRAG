import os
DATASET_NAME ="MMSkinQA"
PROJECT_ROOT = "/home/william/model/Skinalor/RAG/RAGDataSet"
DB_DIR = os.path.join(PROJECT_ROOT, DATASET_NAME)
CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
CHROMA_COLLECTION_NAME = r"skin_cases_multivector_"+DATASET_NAME

IMAGE_DIR = "/home/william/dataset/skin/MM-SkinQA"
# model config
EMBEDDING_MODEL_NAME = 'openai/clip-vit-base-patch32'