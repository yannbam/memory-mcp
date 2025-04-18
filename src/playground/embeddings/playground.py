#!/usr/bin/env python3
"""
Embeddings Playground
-------------------

Interactive tool for exploring embedding vector properties and relationships.
Supports various analysis modes and visualization options.
"""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Optional
from src.APIClient import get_api_client
from src.playground.embeddings.utils import (
    normalize_l2, cosine_similarity, angular_distance,
    analyze_vector, find_largest_differences
)
from src.playground.embeddings.viz import (
    plot_2d_pca, plot_component_distribution, plot_similarity_matrix
)

async def get_embedding(client, text: str, dimensions: Optional[int] = None) -> np.ndarray:
    """Get embedding for a piece of text"""
    args = {
        "model": "text-embedding-3-large",
        "input_text": text
    }
    if dimensions is not None:
        args["dimensions"] = dimensions
    embed = await client.create_embeddings(**args)
    return np.array(embed)

async def analyze_embeddings(
    client,
    texts: List[str],
    dimensions: Optional[int] = None,
    output_dir: Optional[str] = None
):
    """
    Analyze embeddings with various metrics and visualizations
    
    Args:
        client: API client for getting embeddings
        texts: List of text strings to analyze
        dimensions: Optional dimension reduction
        output_dir: Directory to save plots (if None, displays interactively)
    """
    # Get embeddings
    print("\nGetting embeddings...")
    embeddings = [await get_embedding(client, text, dimensions) for text in texts]
    embeddings = np.array(embeddings)
    
    # Basic vector analysis
    print("\nVector Analysis:")
    print("-" * 50)
    for i, (text, embed) in enumerate(zip(texts, embeddings)):
        stats = analyze_vector(embed)
        print(f"\nText {i+1}: '{text}'")
        print(f"Magnitude: {stats.magnitude:.4f}")
        print(f"Mean: {stats.mean:.4f}")
        print(f"Std Dev: {stats.std_dev:.4f}")
        print(f"Max: {stats.max_val:.4f} at dim {stats.max_dim}")
        print(f"Min: {stats.min_val:.4f} at dim {stats.min_dim}")
        
        # Plot component distribution
        fig = plot_component_distribution(embed, text)
        if output_dir:
            fig.savefig(f"{output_dir}/distribution_{i}.png")
        else:
            plt.show()
        plt.close(fig)
    
    # Pairwise analysis
    if len(texts) >= 2:
        print("\nPairwise Analysis:")
        print("-" * 50)
        n = len(texts)
        for i in range(n):
            for j in range(i+1, n):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                angle = angular_distance(embeddings[i], embeddings[j])
                print(f"\n'{texts[i]}' vs '{texts[j]}':")
                print(f"Cosine similarity: {sim:.4f}")
                print(f"Angular distance: {angle:.1f}°")
                
                # Find most different dimensions
                diffs = find_largest_differences(embeddings[i], embeddings[j])
                print("\nLargest differences in dimensions:")
                for dim, diff in diffs:
                    print(f"Dimension {dim}: {diff:.4f}")
    
    # PCA visualization for 3+ vectors
    if len(texts) >= 3:
        print("\nPCA Analysis:")
        print("-" * 50)
        fig = plot_2d_pca(embeddings, texts)
        if output_dir:
            fig.savefig(f"{output_dir}/pca.png")
        else:
            plt.show()
        plt.close(fig)
        
        # Similarity matrix
        fig = plot_similarity_matrix(embeddings, texts)
        if output_dir:
            fig.savefig(f"{output_dir}/similarity_matrix.png")
        else:
            plt.show()
        plt.close(fig)

def show_help():
    print("""
Embeddings Playground
-------------------
Analyze and visualize relationships between embedding vectors.

Usage:
  python embedding_playground.py [options] text1 [text2 text3 ...]

Options:
  --dimensions N    Reduce embedding dimensions to N
  --output-dir DIR  Save plots to directory instead of displaying
  --help           Show this help message

Examples:
  # Basic comparison
  python embedding_playground.py "happy" "joyful" "sad"
  
  # With dimension reduction
  python embedding_playground.py --dimensions 512 "king" "man" "woman" "queen"
  
  # Save plots
  python embedding_playground.py --output-dir plots "light" "dark" "brightness" "darkness"
""")

async def main():
    if len(sys.argv) < 2 or "--help" in sys.argv:
        show_help()
        sys.exit(1)
    
    # Parse arguments
    texts = []
    dimensions = None
    output_dir = None
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--dimensions":
            dimensions = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--output-dir":
            output_dir = sys.argv[i+1]
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            i += 2
        else:
            texts.append(sys.argv[i])
            i += 1
    
    # Initialize client
    client = get_api_client()
    
    # Run analysis
    await analyze_embeddings(client, texts, dimensions, output_dir)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())