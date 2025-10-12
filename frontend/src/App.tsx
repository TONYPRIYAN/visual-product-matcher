import { useState, useCallback } from 'react';
import axios from 'axios';
import './App.css'; // Import our new CSS file

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

// --- Main Application Component ---
function App() {
  // --- State Management ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // --- Handlers ---
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResults([]);
      setError(null);
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
      } else {
        setError("Please drop an image file (e.g., jpg, png).");
      }
    }
  }, []);

  // --- UI Rendering ---
  return (
    <div className="container">
      <header className="header">
        <h1>Visual Product Matcher</h1>
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
          {isLoading && (
            <div className="loader-container">
              <div className="loader"></div>
            </div>
          )}
          {error && <div className="error-message">{error}</div>}
          {results.length > 0 && (
            <div className="results-grid">
              {results.map((result, index) => (
                <div key={result.product.id} className="result-card">
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
        </div>
      </main>
    </div>
  );
}

export default App;

