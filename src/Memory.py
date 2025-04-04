"""
Memory Systems Module
--------------------

This module provides various memory systems for maintaining and retrieving conversation context.

Available memory systems:
- VectorMemory: Uses embeddings for semantic search
- SummaryMemory: Creates and retrieves conversation summaries
- TimeWindowMemory: Combines recent and important long-term memories
- KeywordMemory: Lightweight keyword-based search
- HierarchicalMemory: Three-tier system with immediate, short-term, and long-term memory

Import the specific memory system you need or use the factory function to create them.
"""

from src.memory import (
    VectorMemory,
    SummaryMemory,
    TimeWindowMemory,
    KeywordMemory,
    HierarchicalMemory
)

def create_memory(memory_type="vector", api_type=None, api_key=None):
    """
    Factory function to create a memory system of the specified type.
    
    Args:
        memory_type (str): The type of memory system to create.
            Options: "vector", "summary", "timewindow", "keyword", "hierarchical"
        api_type (str, optional): The API provider to use ("openai" or "openrouter").
            If None, will be determined from environment variables.
        api_key (str, optional): The API key to use with the specified provider.
            If None, will be read from environment variables.
    
    Returns:
        An instance of the specified memory system
    
    Raises:
        ValueError: If an invalid memory_type is specified
    """
    memory_type = memory_type.lower()
    
    # KeywordMemory doesn't use API calls, so doesn't need api_type or api_key
    if memory_type == "keyword":
        return KeywordMemory()
    
    # For API-dependent memory systems, pass the api configuration
    if memory_type == "vector":
        return VectorMemory(api_type=api_type, api_key=api_key)
    elif memory_type == "summary":
        return SummaryMemory(api_type=api_type, api_key=api_key)
    elif memory_type == "timewindow":
        return TimeWindowMemory(api_type=api_type, api_key=api_key)
    elif memory_type == "hierarchical":
        return HierarchicalMemory(api_type=api_type, api_key=api_key)
    else:
        raise ValueError(f"Invalid memory type: {memory_type}")

__all__ = [
    "VectorMemory",
    "SummaryMemory",
    "TimeWindowMemory",
    "KeywordMemory",
    "HierarchicalMemory",
    "create_memory"
]