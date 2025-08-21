import os


DATA_DIR = "/home/william/dataset/skin/SkinCAP"

CSV_PATH = "/home/william/dataset/skin/SkinCAP/split/skincap_all_20250723_200925.csv"

IMAGE_DIR = "/home/william/dataset/skin/SkinCAP/skincap"




DATASET_NAME ="SkinCAP_train_test"
PROJECT_ROOT = "/home/william/model/Skinalor/RAG/RAGDataSet"
DB_DIR = os.path.join(PROJECT_ROOT, DATASET_NAME)
CHROMA_PERSIST_PATH = os.path.join(DB_DIR, "chroma_db_skin")
CHROMA_COLLECTION_NAME = "skin_cases_multivector_"+DATASET_NAME

# model config
EMBEDDING_MODEL_NAME = 'openai/clip-vit-base-patch32'