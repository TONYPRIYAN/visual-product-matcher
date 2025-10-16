from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import io
from contextlib import asynccontextmanager

# --- Configuration ---
VECTORS_FILE = "data/product_vectors.json"
METADATA_FILE = "data/metadata.json"
MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Application State ---
app_state = {
    "model": None,
    "processor": None,
    "product_vectors": {},
    "product_metadata": {},
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Lifespan Events (Startup) ---
@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    print(f"Loading CLIP model: '{MODEL_NAME}'...")
    app_state["model"] = CLIPModel.from_pretrained(MODEL_NAME)
    app_state["processor"] = CLIPProcessor.from_pretrained(MODEL_NAME)
    app_state["model"].to(app_state["device"])
    print(f"Model loaded successfully on device: {app_state['device']}")

    print(f"Loading product vectors from '{VECTORS_FILE}'...")
    with open(VECTORS_FILE, 'r') as f:
        product_data = json.load(f)
    app_state["product_vectors"] = {item['id']: item['vector'] for item in product_data}
    print(f"Loaded {len(app_state['product_vectors'])} product vectors.")

    print(f"Loading product metadata from '{METADATA_FILE}'...")
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    app_state["product_metadata"] = {item['id']: item for item in metadata}
    print(f"Loaded metadata for {len(app_state['product_metadata'])} products.")
    
    print("Startup complete. Server is ready to accept requests.")

# --- CORS Middleware Configuration ---
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static File Serving ---
app.mount("/data", StaticFiles(directory="data"), name="data")

# --- Helper Functions ---
def get_image_embedding(image: Image.Image) -> list:
    model = app_state["model"]
    processor = app_state["processor"]
    device = app_state["device"]
    
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        embedding = model.get_image_features(pixel_values=inputs['pixel_values'])
    
    return embedding.cpu().numpy().flatten().tolist()

# --- API Endpoints ---
@app.post("/find-similar-products/")
async def find_similar_products(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    query_vector = get_image_embedding(query_image)
    
    product_ids = list(app_state["product_vectors"].keys())
    db_vectors = list(app_state["product_vectors"].values())
    
    similarities = cosine_similarity([query_vector], db_vectors)[0]
    
    results_with_scores = sorted(zip(product_ids, similarities), key=lambda item: item[1], reverse=True)
    
    top_results = []
    for product_id, score in results_with_scores[:10]:
        product_info = app_state["product_metadata"].get(product_id)
        if product_info:
            top_results.append({
                "product": product_info,
                "similarity": float(score)
            })
            
    return {"results": top_results}

