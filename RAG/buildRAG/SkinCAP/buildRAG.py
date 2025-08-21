import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb
import os
import sys
from tqdm import tqdm  # Used to display a nice progress bar
import shutil

from config import CSV_PATH, IMAGE_DIR, CHROMA_PERSIST_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME


def get_clip_model_and_processor():
    """Loads the CLIP model and its corresponding preprocessor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {EMBEDDING_MODEL_NAME} onto device: {device}")
    model = CLIPModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.")
    return model, processor, device


def main():
    """Main function to perform data processing and index building."""
    if os.path.exists(CHROMA_PERSIST_PATH):
        try:
            shutil.rmtree(CHROMA_PERSIST_PATH)
            print("Successfully deleted the existing database directory.")
        except Exception as e:
            print(f"An error occurred while deleting the directory: {e}")
            print("Please check the file permissions, or delete the directory manually.")
    else:
        print("The database directory does not exist; there’s no need to delete it.")

    # 1. Load the model
    model, processor, device = get_clip_model_and_processor()

    # 2. Initialize the vector database client
    print(f"Initializing or connecting to the vector database: {CHROMA_PERSIST_PATH}")
    if not os.path.exists(CHROMA_PERSIST_PATH):
        os.makedirs(CHROMA_PERSIST_PATH)

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

    # 3. Create or re-create the Collection

    print(f"Creating new collection: '{CHROMA_COLLECTION_NAME}'")
    collection = client.create_collection(name=CHROMA_COLLECTION_NAME)

    # 4. Read the CSV data
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Successfully loaded {len(df)} case records from {CSV_PATH}.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at the specified path {CSV_PATH}")
        return
    except KeyError as e:
        print(f"Error: The CSV file must contain the expected column names. Missing: {e}")
        return

    # 5. Loop through data, generate the three vectors, and prepare for database insertion
    print("Starting to generate vectors and store them in the database...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing case..."):
        case_id = str(index)
        text_description = row.get('caption_zh_polish_en', '')
        image_file = row.get('skincap_file_path', '')

        # --- MODIFICATION 1: Get the disease name ---
        disease_name = row.get('disease', 'Unknown')

        if not text_description or not image_file:
            continue

        image_full_path = os.path.join(IMAGE_DIR, image_file)
        if not os.path.exists(image_full_path):
            continue

        try:
            # --- a. Generate text vector ---
            text_inputs = processor(text=text_description, return_tensors="pt", padding=True, truncation=True).to(
                device)
            with torch.no_grad():
                text_embedding = model.get_text_features(**text_inputs).cpu().numpy().flatten().tolist()

            # --- b. Generate image vector ---
            image = Image.open(image_full_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embedding = model.get_image_features(**image_inputs).cpu().numpy().flatten().tolist()

            # --- c. Generate multimodal vector (by averaging) ---
            multimodal_embedding = (
                        (torch.tensor(text_embedding) + torch.tensor(image_embedding)) / 2.0).numpy().tolist()

            # --- d. Add all three vectors and their metadata to the database ---

            # --- MODIFICATION 2: Add 'disease' to the base metadata ---
            base_metadata = {
                "case_id": case_id,
                "description": text_description,
                "image_filename": image_file,
                "disease": disease_name
            }

            # Add data, ensuring each record has a unique ID and clear metadata
            collection.add(
                embeddings=[text_embedding, image_embedding, multimodal_embedding],
                metadatas=[
                    {**base_metadata, "type": "text"},  # Metadata for the text vector
                    {**base_metadata, "type": "image"},  # Metadata for the image vector
                    {**base_metadata, "type": "multimodal"}  # Metadata for the multimodal vector
                ],
                ids=[
                    f"case_{case_id}_text",  # Unique ID for the text vector
                    f"case_{case_id}_image",  # Unique ID for the image vector
                    f"case_{case_id}_multimodal"  # Unique ID for the multimodal vector
                ]
            )

        except Exception as e:
            print(f"An error occurred while processing case {case_id} (image: {image_file}): {e}")
            continue

    print("\n" + "=" * 50)
    print("RAG index construction complete!")
    print(f"Total number of vectors in the database: {collection.count()}")
    print(f"Database has been persisted to: {CHROMA_PERSIST_PATH}")
    print("=" * 50)


if __name__ == '__main__':
    main()