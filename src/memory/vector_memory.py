import os
import numpy as np
from termcolor import colored
from openai import AsyncOpenAI

# Constants for VectorMemory
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.6
MAX_MEMORIES = 100

class VectorMemory:
    """
    Vector Memory System
    --------------------

    This memory system uses OpenAI's embeddings to create vector representations of messages
    and enables semantic search through the conversation history.

    How it works:
    1. Each message is converted into a high-dimensional vector (embedding) using OpenAI's embedding
    2. When retrieving memories, the query is also converted to an embedding
    3. Cosine similarity is used to find the most semantically similar messages
    4. Messages above the similarity threshold are returned, sorted by relevance

    Key features:
    - Semantic search capability (finds conceptually similar content, not just exact matches)
    - Maintains a fixed-size memory buffer (MAX_MEMORIES)
    - Uses similarity threshold to ensure quality matches
    - Returns top-k most relevant memories

    Use cases:
    - Finding semantically related previous conversations
    - Answering questions about past discussions
    - Maintaining context across long conversations
    """
    
    def __init__(self):
        try:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.memories = []
            self.embeddings = []
            print(colored("Vector memory system initialized", "green"))
        except Exception as e:
            print(colored(f"Error initializing vector memory: {str(e)}", "red"))
            raise e
    
    async def add_memory(self, message):
        try:
            embedding_response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=message['content']
            )
            embedding = embedding_response.data[0].embedding
            
            self.memories.append(message)
            self.embeddings.append(embedding)
            
            # Keep only MAX_MEMORIES most recent memories
            if len(self.memories) > MAX_MEMORIES:
                self.memories.pop(0)
                self.embeddings.pop(0)
                
            print(colored("Memory added successfully", "green"))
        except Exception as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))
    
    async def get_relevant_memories(self, query, top_k=3):
        try:
            query_embedding_response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            )
            
            query_embedding = query_embedding_response.data[0].embedding
            
            # Calculate similarities
            similarities = [
                np.dot(query_embedding, mem_embedding) / 
                (np.linalg.norm(query_embedding) * np.linalg.norm(mem_embedding))
                for mem_embedding in self.embeddings
            ]
            
            # Filter by similarity threshold
            relevant_indices = [
                i for i, sim in enumerate(similarities)
                if sim > SIMILARITY_THRESHOLD
            ]
            
            # Sort by similarity and get top-k
            relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
            relevant_indices = relevant_indices[:top_k]
            
            return [self.memories[i] for i in relevant_indices]
        except Exception as e:
            print(colored(f"Error retrieving memories: {str(e)}", "red"))
            return []
    
    def get_memory_stats(self):
        return {
            "total_memories": len(self.memories),
            "embedding_dimension": len(self.embeddings[0]) if self.embeddings else 0
        }