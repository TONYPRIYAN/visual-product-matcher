import json
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware

GITHUB_USERNAME = "TONYPRIYAN"
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/product-matcher-data/main/"
VECTORS_URL = f"{BASE_URL}data/product_vectors.json"
METADATA_URL = f"{BASE_URL}data/metadata.json"

MODEL_NAME = "openai/clip-vit-base-patch32"
model_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    print(f"Loading CLIP model: '{MODEL_NAME}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_data['model'] = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model_data['processor'] = CLIPProcessor.from_pretrained(MODEL_NAME)
    model_data['device'] = device
    print(f"Model loaded successfully on device: {device}")

    print(f"Loading product vectors from '{VECTORS_URL}'...")
    response = requests.get(VECTORS_URL)
    response.raise_for_status()
    product_data = response.json()
    model_data['product_vectors'] = {item['id']: item['vector'] for item in product_data}
    print(f"Loaded {len(model_data['product_vectors'])} product vectors.")

    print(f"Loading product metadata from '{METADATA_URL}'...")
    response = requests.get(METADATA_URL)
    response.raise_for_status()
    metadata = response.json()
    model_data['product_metadata'] = {item['id']: item for item in metadata}
    print(f"Loaded metadata for {len(model_data['product_metadata'])} products.")
    
    print("Startup complete. Server is ready to accept requests.")
    yield
    print("Server shutting down...")
    model_data.clear()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/data", StaticFiles(directory="data"), name="data")

class Product(BaseModel):
    id: str
    name: str
    category: str
    image_path: str

class Result(BaseModel):
    product: Product
    similarity: float

class SearchResponse(BaseModel):
    results: List[Result]

def get_image_embedding(image: Image.Image):
    image = image.convert("RGB")
    inputs = model_data['processor'](images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(model_data['device']) for k, v in inputs.items()}
    with torch.no_grad():
        embedding = model_data['model'].get_image_features(pixel_values=inputs['pixel_values'])
    return embedding.cpu().numpy().flatten()

@app.post("/find-similar-products/", response_model=SearchResponse)
async def find_similar_products(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    query_vector = get_image_embedding(query_image)
    
    product_ids = list(model_data['product_vectors'].keys())
    db_vectors = np.array(list(model_data['product_vectors'].values()))
    
    similarities = cosine_similarity([query_vector], db_vectors)[0]
    
    top_indices = np.argsort(similarities)[-10:][::-1]
    
    results = []
    for i in top_indices:
        product_id = product_ids[i]
        product_info = model_data['product_metadata'].get(product_id)
        if product_info:
            results.append(Result(
                product=Product(**product_info),
                similarity=similarities[i]
            ))
            
    return SearchResponse(results=results)

@app.get("/")
def read_root():
    return {"message": "Visual Product Matcher API is running."}

