"""
Core utilities for working with embeddings.
Contains basic vector operations and analysis functions.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class EmbeddingStats:
    """Statistics about an embedding vector"""
    magnitude: float
    mean: float
    std_dev: float
    max_val: float
    max_dim: int
    min_val: float
    min_dim: int

def normalize_l2(x: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length. Required when manually reducing dimensions.
    OpenAI embeddings are already normalized, but this is useful when manipulating them.
    """
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        return x / norm if norm != 0 else x
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Since OpenAI embeddings are normalized, this is equivalent to dot product.
    """
    return np.dot(v1, v2)

def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angular distance in degrees between two vectors"""
    sim = min(1.0, cosine_similarity(v1, v2))  # Clamp to avoid float errors
    return np.arccos(sim) * 180 / np.pi

def analyze_vector(embed: np.ndarray) -> EmbeddingStats:
    """Calculate basic statistics about an embedding vector"""
    return EmbeddingStats(
        magnitude=np.linalg.norm(embed),
        mean=np.mean(embed),
        std_dev=np.std(embed),
        max_val=np.max(embed),
        max_dim=np.argmax(embed),
        min_val=np.min(embed),
        min_dim=np.argmin(embed)
    )

def find_largest_differences(v1: np.ndarray, v2: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """Find dimensions with largest absolute differences between vectors"""
    diff = v1 - v2
    top_dims = np.argsort(np.abs(diff))[-top_k:][::-1]
    return [(int(dim), float(diff[dim])) for dim in top_dims]