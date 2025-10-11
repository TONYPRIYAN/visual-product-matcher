import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os

METADATA_FILE_PATH = "data/metadata.json"
VECTORS_OUTPUT_FILE_PATH = "data/product_vectors.json"
MODEL_NAME = "openai/clip-vit-base-patch32"

def main():
    print("Starting image pre-processing...")
    if not os.path.exists(METADATA_FILE_PATH):
        print(f"ERROR: Metadata file not found at {METADATA_FILE_PATH}")
        return

    print(f"Loading CLIP model: '{MODEL_NAME}'...")
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    with open(METADATA_FILE_PATH, 'r') as f:
        products = json.load(f)
    print(f"Found {len(products)} products in metadata file.")

    all_product_vectors = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)

    print("Generating embeddings for each product...")
    for product in products:
        image_path = product.get("image_path")
        product_id = product.get("id")

        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Image path for product ID '{product_id}' not found at '{image_path}'. Skipping.")
            continue
        
        try:
            image = Image.open(image_path)
            inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # --- THIS IS THE FINAL CORRECTED LINE ---
                image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
            
            embedding = image_features.cpu().numpy().flatten().tolist()
            
            all_product_vectors.append({
                "id": product_id,
                "vector": embedding
            })
            print(f"  - Successfully processed product ID: {product_id}")
        except Exception as e:
            print(f"ERROR: Failed to process image for product ID '{product_id}'. Error: {e}")

    if not all_product_vectors:
        print("\nCRITICAL ERROR: No product vectors were generated. Aborting save.")
        return

    print(f"\nSaving {len(all_product_vectors)} generated vectors to '{VECTORS_OUTPUT_FILE_PATH}'...")
    with open(VECTORS_OUTPUT_FILE_PATH, 'w') as f:
        json.dump(all_product_vectors, f, indent=4)
    
    print("Pre-processing complete!")

if __name__ == "__main__":
    main()

