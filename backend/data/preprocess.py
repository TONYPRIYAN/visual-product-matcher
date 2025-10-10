import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import os

# --- Configuration ---
METADATA_FILE_PATH = "data/metadata.json"
VECTORS_OUTPUT_FILE_PATH = "data/product_vectors.json"
MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Main Script ---
def main():
    """
    Analyzes all product images and saves their numerical signatures.
    """
    print("Starting image analysis...")

    if not os.path.exists(METADATA_FILE_PATH):
        print(f"ERROR: Product list file not found at {METADATA_FILE_PATH}")
        return

    print(f"Loading image analysis library: '{MODEL_NAME}'...")
    try:
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        print("Library loaded successfully.")
    except Exception as e:
        print(f"Error loading library: {e}")
        return

    print(f"Loading product list from '{METADATA_FILE_PATH}'...")
    with open(METADATA_FILE_PATH, 'r') as f:
        products = json.load(f)
    print(f"Found {len(products)} products in the list.")

    all_product_signatures = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Generating a unique signature for each product image...")
    for product in products:
        image_path = product.get("image_path")
        product_id = product.get("id")

        if not image_path or not os.path.exists(image_path):
            print(f"WARNING: Image for product ID '{product_id}' not found. Skipping.")
            continue
        
        try:
            image = Image.open(image_path)
            
            # Use the library to analyze the image and get its numerical signature
            inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = model.get_image_features(pixel_values=inputs.pixel_values)
            
            signature = image_features.cpu().numpy().flatten().tolist()
            
            all_product_signatures.append({
                "id": product_id,
                "signature": signature 
            })
            print(f"  - Successfully processed product ID: {product_id}")

        except Exception as e:
            print(f"ERROR: Failed to process image for product ID '{product_id}'. Error: {e}")

    print(f"Saving {len(all_product_signatures)} generated signatures to '{VECTORS_OUTPUT_FILE_PATH}'...")
    with open(VECTORS_OUTPUT_FILE_PATH, 'w') as f:
        # We rename 'signature' back to 'vector' in the file for consistency with the main app
        json.dump([{"id": p["id"], "vector": p["signature"]} for p in all_product_signatures], f, indent=4)
    
    print("\nAnalysis complete!")
    print(f"Product signatures are saved in {VECTORS_OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    main()