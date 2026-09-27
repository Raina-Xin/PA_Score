import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import pickle
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")


def get_sbert_model(name: str, device: str = "auto") -> SentenceTransformer:
    """Load SBERT model with error handling."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
    
    try:
        model = SentenceTransformer(name)
        if device == "auto":
            device = "cuda" if hasattr(model, "device") else "cpu"
        if hasattr(model, "to"):
            model = model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load SBERT model '{name}': {e}")


def compute_embeddings(model: SentenceTransformer, texts: List[str], 
                      batch_size: int = 64, normalize: bool = False, 
                      show_progress: bool = False) -> np.ndarray:
    """Compute SBERT embeddings for a list of texts."""
    if not texts:
        return np.array([]).reshape(0, 384)  # Default dimension for all-MiniLM-L6-v2
    
    embeddings = model.encode(
        texts, 
        convert_to_numpy=True, 
        batch_size=batch_size, 
        show_progress_bar=show_progress
    )
    
    if normalize:
        # L2 normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
    
    return embeddings


def load_or_compute_embeddings(dataset_name: str, ids: List[str], texts: List[str], 
                              model_name: str, cache_dir: str, normalize: bool = False,
                              force_recompute: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load embeddings from cache or compute and cache them."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filenames
    cache_key = f"{dataset_name}_{model_name.replace('/', '_')}"
    if normalize:
        cache_key += "_normalized"
    
    embeddings_file = os.path.join(cache_dir, f"{cache_key}.npy")
    metadata_file = os.path.join(cache_dir, f"{cache_key}_metadata.json")
    
    # Check if cache exists and is valid
    if not force_recompute and os.path.exists(embeddings_file) and os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if IDs match exactly
            if metadata.get('ids') == ids and metadata.get('normalize') == normalize:
                embeddings = np.load(embeddings_file)
                print(f"Loaded cached embeddings for {dataset_name} ({len(embeddings)} vectors)")
                return embeddings, metadata
        except Exception as e:
            print(f"Warning: Failed to load cache, recomputing: {e}")
    
    # Compute embeddings
    print(f"Computing embeddings for {dataset_name} ({len(texts)} texts)...")
    model = get_sbert_model(model_name)
    embeddings = compute_embeddings(model, texts, normalize=normalize, show_progress=True)
    
    # Save to cache
    metadata = {
        'ids': ids,
        'model_name': model_name,
        'normalize': normalize,
        'n_texts': len(texts),
        'embedding_dim': embeddings.shape[1]
    }
    
    np.save(embeddings_file, embeddings)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Cached embeddings for {dataset_name} to {embeddings_file}")
    return embeddings, metadata


def fit_kmeans(emb_cal: np.ndarray, K: int, seed: int, use_minibatch: bool = False) -> Tuple[np.ndarray, Any]:
    """Fit KMeans on calibration embeddings."""
    if use_minibatch or len(emb_cal) > 100000:
        kmeans = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=1000)
    else:
        kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10)
    
    kmeans.fit(emb_cal)
    return kmeans.cluster_centers_, kmeans


def auto_select_k(emb_cal: np.ndarray, candidates: List[int], seed: int, 
                 use_minibatch: bool = False) -> int:
    """Choose K by silhouette score on calibration embeddings."""
    if len(emb_cal) < 2:
        return candidates[0]
    
    best_k = candidates[0]
    best_silhouette = -1
    
    for k in candidates:
        if k >= len(emb_cal):
            continue
            
        try:
            if use_minibatch or len(emb_cal) > 100000:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1000)
            else:
                kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
            
            labels = kmeans.fit_predict(emb_cal)
            
            # Check if we have at least 2 clusters with multiple points
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                continue
                
            # Check if any cluster has only one point (silhouette requires at least 2 points per cluster)
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            if min(cluster_sizes) < 2:
                continue
            
            silhouette = silhouette_score(emb_cal, labels)
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
                
        except Exception as e:
            print(f"Warning: Silhouette computation failed for K={k}: {e}")
            continue
    
    print(f"Auto-selected K={best_k} with silhouette={best_silhouette:.3f}")
    return best_k


def assign_clusters(emb: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Assign cluster labels and compute distances to centroids."""
    # Compute distances to all centroids
    distances = np.linalg.norm(emb[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    
    # Assign to nearest centroid
    labels = np.argmin(distances, axis=1)
    
    return labels, distances


def build_phi_features(labels: np.ndarray, distances: np.ndarray, 
                      phi_rep: str, n_clusters: int) -> np.ndarray:
    """Build phi features from cluster assignments and distances."""
    n_samples = len(labels)
    
    if phi_rep == "onehot":
        # One-hot encoding with bias column
        phi = np.zeros((n_samples, n_clusters + 1))
        phi[:, 0] = 1  # Bias column
        phi[np.arange(n_samples), labels + 1] = 1  # One-hot cluster indicators
        
    elif phi_rep == "dist3":
        # Top-3 distances with bias column
        phi = np.zeros((n_samples, 4))  # bias + 3 distances
        phi[:, 0] = 1  # Bias column
        
        # Sort distances and take top-3 smallest
        sorted_indices = np.argsort(distances, axis=1)
        for i in range(n_samples):
            phi[i, 1:4] = distances[i, sorted_indices[i, :3]]
    
    else:
        raise ValueError(f"Unknown phi_rep: {phi_rep}")
    
    return phi


def extract_question_text(raw_data: List[Dict], question_key: str) -> List[str]:
    """Extract question text from raw data with fallback options."""
    texts = []
    fallback_keys = ["input", "question_text", "text", "prompt"]
    
    for i, row in enumerate(raw_data):
        text = None
        
        # Try primary key first
        if question_key in row and row[question_key]:
            text = str(row[question_key]).strip()
        
        # Try fallback keys
        if not text:
            for key in fallback_keys:
                if key in row and row[key]:
                    text = str(row[key]).strip()
                    break
        
        if not text:
            available_keys = list(row.keys())[:5]  # Show first 5 keys
            raise ValueError(
                f"No question text found in row {i} (ID: {row.get('id', 'unknown')}). "
                f"Tried keys: {[question_key] + fallback_keys}. "
                f"Available keys: {available_keys}. "
                f"Use --question_key to specify the correct field name."
            )
        
        texts.append(text)
    
    return texts
