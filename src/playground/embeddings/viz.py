"""
Visualization utilities for embeddings analysis.
Provides various ways to visualize embedding relationships and properties.
"""

import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.figure import Figure

def plot_2d_pca(
    embeddings: np.ndarray,
    labels: List[str],
    title: str = "PCA Projection of Embeddings"
) -> Figure:
    """
    Create a 2D PCA plot of embeddings.
    
    Args:
        embeddings: Array of embedding vectors
        labels: Text labels for each point
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Compute PCA
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    ax.scatter(proj[:, 0], proj[:, 1], alpha=0.5)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (proj[i, 0], proj[i, 1]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add explained variance
    var_explained = pca.explained_variance_ratio_
    ax.set_title(f"{title}\nVariance explained: {var_explained[0]:.2%}, {var_explained[1]:.2%}")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def plot_component_distribution(
    embedding: np.ndarray,
    label: str,
    bins: int = 50,
    title: str = "Component Distribution"
) -> Figure:
    """
    Create a histogram of embedding components.
    
    Args:
        embedding: Single embedding vector
        label: Text label for the embedding
        bins: Number of histogram bins
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(embedding, bins=bins, alpha=0.7)
    ax.set_title(f"{title}: {label}")
    ax.set_xlabel("Component Value")
    ax.set_ylabel("Count")
    
    # Add mean and std dev lines
    mean = np.mean(embedding)
    std = np.std(embedding)
    ylim = ax.get_ylim()
    ax.vlines(mean, 0, ylim[1], color='red', linestyle='--', label=f'Mean: {mean:.4f}')
    ax.vlines([mean - std, mean + std], 0, ylim[1], color='green', linestyle=':', label=f'Std Dev: {std:.4f}')
    ax.legend()
    
    return fig

def plot_similarity_matrix(
    embeddings: np.ndarray,
    labels: List[str],
    title: str = "Similarity Matrix"
) -> Figure:
    """
    Create a heatmap of cosine similarities between embeddings.
    
    Args:
        embeddings: Array of embedding vectors
        labels: Text labels for each embedding
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    # Compute similarity matrix
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i,j] = np.dot(embeddings[i], embeddings[j])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    
    # Add labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    ax.set_title(title)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return fig