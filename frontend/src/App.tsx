import { useState, useCallback } from 'react';
import axios from 'axios';
import './App.css'; // Import our new stylish CSS file

// --- Type Definitions for our data structures ---
interface Product {
  id: string;
  name: string;
  category: string;
  image_path: string;
}

interface SearchResult {
  product: Product;
  similarity: number;
}

// --- Placeholder Component for Initial State ---
const ResultsPlaceholder = () => (
  <div className="placeholder-container">
    <div>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '0.5rem' }}>Your results will appear here</h2>
      <p>Upload an image and click search to begin.</p>
    </div>
  </div>
);

// --- Product Modal (Pop-up) Component ---
const ProductModal = ({ product, onClose }: { product: Product; onClose: () => void }) => {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close-button" onClick={onClose}>&times;</button>
        <img src={`http://127.0.0.1:8000/${product.image_path}`} alt={product.name} className="modal-image" />
        <div className="modal-details">
          <h2>{product.name}</h2>
          <p>{product.category}</p>
        </div>
      </div>
    </div>
  );
};


// --- Main Application Component ---
function App() {
  // --- State Management ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null); // State for the pop-up

  // --- Handlers ---
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]);
      setError(null);
      setHasSearched(false);
    }
  };

  const handleSearch = async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResults([]);
    setHasSearched(true);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post<{ results: SearchResult[] }>('http://127.0.0.1:8000/find-similar-products/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults(response.data.results);
    } catch (err) {
      console.error("API call failed:", err);
      setError("Failed to fetch results. Please ensure the backend server is running and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // --- Drag and Drop Handlers ---
  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
  }, []);

  const handleDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    if (event.dataTransfer.files && event.dataTransfer.files[0]) {
      const file = event.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setResults([]);
        setError(null);
        setHasSearched(false);
      } else {
        setError("Please drop an image file (e.g., jpg, png).");
      }
    }
  }, []);

  // --- UI Rendering ---
  return (
    <>
      <div className="background-wrapper"></div>
      <div className="container">
        <header className="header">
          <h1 className="title-gradient">Visual Product Matcher</h1>
          <p>Upload an image to find visually similar products from our database.</p>
        </header>

        <main className="main-grid">
          <div className="left-column">
            <div className="uploader-box">
              <h2>1. Upload Your Image</h2>
              <div
                className="drop-zone"
                onDragOver={handleDragOver}
                onDrop={handleDrop}
                onClick={() => document.getElementById('fileInput')?.click()}
              >
                <input
                  type="file"
                  id="fileInput"
                  onChange={handleFileChange}
                  accept="image/*"
                  style={{ display: 'none' }}
                />
                <p>Drag & drop an image here, or click to select a file.</p>
              </div>
              {previewUrl && (
                <div className="image-preview">
                  <h3>Your Image:</h3>
                  <img src={previewUrl} alt="Selected preview" />
                </div>
              )}
              <button
                onClick={handleSearch}
                disabled={isLoading || !selectedFile}
                className="search-button"
              >
                {isLoading ? 'Searching...' : '2. Find Similar Products'}
              </button>
            </div>
          </div>

          <div className="right-column">
            {isLoading && <div className="loader-container"><div className="loader"></div></div>}
            {error && <div className="error-message">{error}</div>}
            
            {!isLoading && !error && !hasSearched && <ResultsPlaceholder />}
            
            {!isLoading && !error && hasSearched && results.length > 0 && (
              <div className="results-grid">
                {results.map((result, index) => (
                  <div 
                    key={result.product.id} 
                    className="result-card" 
                    style={{ animationDelay: `${index * 100}ms` }}
                    onClick={() => setSelectedProduct(result.product)} // <-- This makes the card clickable
                  >
                    <img src={`http://127.0.0.1:8000/${result.product.image_path}`} alt={result.product.name} />
                    <div className="card-content">
                      <h3>{index === 0 ? "Your Upload (Best Match)" : result.product.name}</h3>
                      <p className="category">{result.product.category}</p>
                      <div className="similarity-badge">
                        Similarity: {(result.similarity * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {!isLoading && !error && hasSearched && results.length === 0 && <ResultsPlaceholder />}
          </div>
        </main>
      </div>

      {/* --- Render the Modal Pop-up --- */}
      {selectedProduct && (
        <ProductModal product={selectedProduct} onClose={() => setSelectedProduct(null)} />
      )}
    </>
  );
}

export default App;

