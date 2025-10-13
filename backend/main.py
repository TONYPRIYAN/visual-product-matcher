import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles # <-- 1. IMPORT THIS
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import io

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
VECTORS_FILE = "data/product_vectors.json"
METADATA_FILE = "data/metadata.json"

# --- Application State ---
app_state = {}

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Static Files Directory ---
# 2. THIS NEW LINE FIXES THE BROKEN IMAGES.
# It tells FastAPI to serve any file in the 'data' folder.
app.mount("/data", StaticFiles(directory="data"), name="data")

# --- Lifespan Events (Startup) ---
@app.on_event("startup")
async def startup_event():
    print("Server starting up...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model: '{MODEL_NAME}' on device: {device}")
    app_state["model"] = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    app_state["processor"] = CLIPProcessor.from_pretrained(MODEL_NAME)
    app_state["device"] = device
    
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

# --- Helper Functions ---
def get_image_embedding(image: Image.Image):
    model = app_state["model"]
    processor = app_state["processor"]
    device = app_state["device"]
    
    # Ensure image is RGB before processing
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).cpu().numpy()
    return embedding

# --- API Endpoint ---
@app.post("/find-similar-products/")
async def find_similar_products(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        query_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    query_vector = get_image_embedding(query_image)
    
    product_vectors = app_state["product_vectors"]
    product_ids = list(product_vectors.keys())
    db_vectors = [product_vectors[pid] for pid in product_ids]
    
    similarities = cosine_similarity(query_vector, db_vectors)[0]
    
    results = sorted(zip(product_ids, similarities), key=lambda x: x[1], reverse=True)
    
    top_results = []
    for product_id, sim in results[:10]:
        product_info = app_state["product_metadata"].get(product_id)
        if product_info:
            top_results.append({
                "product": product_info,
                "similarity": float(sim)
            })

    return {"results": top_results}

