import json
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import os

MODEL_NAME = "openai/clip-vit-base-patch32"
VECTORS_FILE = "data/product_vectors.json"
METADATA_FILE = "data/metadata.json"

app = FastAPI(
    title="Visual Product Matcher API",
    description="Finds visually similar products from an uploaded image.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_resources():
    print("Server starting up...")
    global model, processor, product_vectors, metadata_map
    
    if not os.path.exists(VECTORS_FILE) or not os.path.exists(METADATA_FILE):
        raise RuntimeError("Data files not found. Please run preprocess.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: '{MODEL_NAME}' on device: {device}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    print(f"Loading product vectors from '{VECTORS_FILE}'...")
    with open(VECTORS_FILE, 'r') as f:
        product_data = json.load(f)
    product_vectors = {item['id']: item.get('vector', []) for item in product_data}
    print(f"Loaded {len(product_vectors)} product vectors.")

    print(f"Loading product metadata from '{METADATA_FILE}'...")
    with open(METADATA_FILE, 'r') as f:
        metadata_list = json.load(f)
    metadata_map = {item['id']: item for item in metadata_list}
    print(f"Loaded metadata for {len(metadata_map)} products.")
    print("Startup complete. Server is ready.")

def get_image_embedding(image: Image.Image) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
    return image_features.cpu().numpy().flatten().tolist()

@app.post("/find-similar-products/")
async def find_similar_products(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents))
        # --- THIS IS THE FIX ---
        # Ensure the image is in RGB format before processing
        query_image = query_image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    query_vector = get_image_embedding(query_image)
    db_ids = list(product_vectors.keys())
    db_vectors = list(product_vectors.values())
    
    similarities = cosine_similarity([query_vector], db_vectors)[0]

    results = []
    for i, product_id in enumerate(db_ids):
        product_metadata = metadata_map.get(product_id)
        if product_metadata:
            results.append({
                "product": product_metadata,
                "similarity": float(similarities[i])
            })

    results.sort(key=lambda x: x['similarity'], reverse=True)
    return {"results": results[:10]}

