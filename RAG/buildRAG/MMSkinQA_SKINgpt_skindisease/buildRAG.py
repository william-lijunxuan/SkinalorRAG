import pandas as pd
from PIL import Image
import torch

from transformers import CLIPProcessor, CLIPModel
import chromadb
import os
import sys
import json
from tqdm import tqdm
import shutil

from config import CSV_PATH_MMSkinQA,JSON_PATH_SKINgpt, IMAGE_DIR_SKINgpt,IMAGE_DIR_MMSkinQA, CHROMA_PERSIST_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME,JSON_PATH_skindisease,IMAGE_DIR_skindisease


def get_clip_model_and_processor():
    """Loads the CLIP model and its corresponding preprocessor."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {EMBEDDING_MODEL_NAME} onto device: {device}")
    model = CLIPModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
    print("Model loaded successfully.")
    return model, processor, device
def is_unclear(text: str) -> bool:
    if not text or pd.isna(text):
        return True
    s = text.strip()

    if len(s.split()) < 3:
        return True

    lower = s.lower()
    if "?" in s or "n/a" in lower:
        return True
    return False

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
        MMSkinQA_df = pd.read_csv(CSV_PATH_MMSkinQA)

        with open(JSON_PATH_SKINgpt, 'r', encoding='utf-8') as f:
            skin_json_data = json.load(f)
        SKINgpt_df = pd.DataFrame(skin_json_data)

        with open(JSON_PATH_skindisease, 'r', encoding='utf-8') as f:
            skindisease_json_data = json.load(f)
        skindisease_df = pd.DataFrame(skindisease_json_data)

        MMSkinQA_df = MMSkinQA_df.rename(columns={
            'image': 'image_file',
            'cleaned_caption': 'text_description'
        })[['image_file', 'text_description']]

        MMSkinQA_df['image_dir'] = IMAGE_DIR_MMSkinQA
        SKINgpt_df['image_dir'] = IMAGE_DIR_SKINgpt
        skindisease_df['image_dir'] = IMAGE_DIR_skindisease

        SKINgpt_df = SKINgpt_df.rename(columns={
            'answer': 'text_description',
            'image_name': 'image_file'
        }).drop(columns=['url', 'option'], errors='ignore')


        skindisease_df = skindisease_df.rename(columns={
            'image_name': 'image_file',
            'answer': 'text_description'
        })[['image_file', 'text_description','image_dir']]


        SKINgpt_df['image_file'] = SKINgpt_df['image_file'].apply(lambda x: os.path.basename(x))



        def valid_row(row):
            disease = str(row.get('disease_name', '') or '')
            description = str(row.get('text_description', '') or '')
            if not description:
                return False
            if disease and disease not in description:
                return False
            return True


        MMSkinQA_df = MMSkinQA_df.dropna(subset=['text_description'])
        MMSkinQA_df = MMSkinQA_df[MMSkinQA_df.apply(valid_row, axis=1)].reset_index(drop=True)


        df = pd.concat([SKINgpt_df, MMSkinQA_df,skindisease_df], ignore_index=True)

        # print(df.head())
        # df.to_csv("combined_output1.csv", index=False)
        print(f"Successfully loaded {len(df)} case records from {CSV_PATH_MMSkinQA} and {JSON_PATH_SKINgpt}.")
    except FileNotFoundError:
        print(f"Error: CSV file not found at the specified path {CSV_PATH_MMSkinQA} and {JSON_PATH_SKINgpt}")
        return
    except KeyError as e:
        print(f"Error: The CSV file must contain the expected column names. Missing: {e}")
        return

    # 5. Loop through data, generate the three vectors, and prepare for database insertion
    print("Starting to generate vectors and store them in the database...")

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing case..."):
        case_id = str(index)
        text_description = row.get('text_description', '')
        disease_name = row.get('disease_name', '')
        image_file = row.get('image_file', '').replace('dataset/', '', 1)


        # if not text_description or not image_file:
        #     print(f"text_description:{text_description}")
        #     continue

        # if is_unclear(text_description) or not image_file:
        #     print(f"is_unclear:{text_description}")
        #     continue

        image_full_path = os.path.join(row.get('image_dir'), image_file)
        if not os.path.exists(image_full_path):
            print(f"not os.path.exists:{image_full_path}")
            continue

        try:
            # --- a. Generate text vector ---
            text_inputs = processor(text=text_description, return_tensors="pt", padding=True, truncation=True).to(
                device)
            with torch.no_grad():
                text_embedding = model.get_text_features(**text_inputs).cpu().numpy().flatten().tolist()

            # --- b. Generate image vector ---
            image = Image.open(image_full_path).convert("RGB")

            # os.makedirs("image", exist_ok=True)
            # filename = os.path.basename(image_full_path)
            # save_path = os.path.join("image", filename)
            # image.save(save_path)
            # print(f"Image saved to: {save_path}")

            image_inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embedding = model.get_image_features(**image_inputs).cpu().numpy().flatten().tolist()

            # --- c. Generate multimodal vector (by averaging) ---
            multimodal_embedding = (
                        (torch.tensor(text_embedding) + torch.tensor(image_embedding)) / 2.0).numpy().tolist()

            # --- d. Add all three vectors and their metadata to the database ---

            base_metadata = {
                "case_id": case_id,
                "image_filename": image_file,
                # "disease": disease_name
            }
            if text_description:
                base_metadata["description"] = text_description

            if disease_name:
                base_metadata["disease"] = disease_name

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
            # if disease_name:
            #     try:
            #         disease_name = str(disease_name).strip()
            #         if not disease_name or disease_name.lower() in ['nan', 'none', 'n/a']:
            #             print(f"[Skip] Case {case_id} skipped due to invalid disease_name: '{disease_name}'")
            #             continue
            #         disease_inputs = processor(text=disease_name, return_tensors="pt", padding=True,
            #                                    truncation=True).to(device)
            #         with torch.no_grad():
            #             disease_embedding = model.get_text_features(**disease_inputs).cpu().numpy().flatten().tolist()
            #
            #         collection.add(
            #             embeddings=[disease_embedding],
            #             metadatas=[{
            #                 **base_metadata,
            #                 "type": "label_text"
            #             }],
            #             ids=[f"case_{case_id}_label"]
            #         )
            #     except Exception as e:
            #         print(f"Failed to vectorize disease name for case {case_id}: {e}")

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