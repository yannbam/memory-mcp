import os
from termcolor import colored
from openai import AsyncOpenAI
from datetime import datetime
import numpy as np

# Constants for HierarchicalMemory
HIERARCHICAL_MODEL = "gpt-4o"
HIERARCHICAL_EMBEDDING_MODEL = "text-embedding-3-small"
IMMEDIATE_CONTEXT_SIZE = 5
SHORT_TERM_SIZE = 20
LONG_TERM_SIZE = 100
HIERARCHICAL_IMPORTANCE_THRESHOLD = 0.7

class HierarchicalMemory:
    """
    Hierarchical Memory System
    --------------------------

    This is the most sophisticated memory system, implementing a three-tier approach
    that combines immediate context, short-term summaries, and long-term embeddings.

    How it works:
    1. Three-tier memory structure:
       - Immediate context: Last few messages (IMMEDIATE_CONTEXT_SIZE)
       - Short-term memory: Summaries of recent conversations (SHORT_TERM_SIZE)
       - Long-term memory: Important embedded memories (LONG_TERM_SIZE)

    2. Memory flow:
       - New messages go to immediate context
       - Overflow from immediate context is summarized into short-term memory
       - Important information is embedded and stored in long-term memory

    3. Retrieval process:
       - Always includes immediate context
       - Uses embeddings to find relevant long-term memories
       - Uses GPT to select relevant short-term summaries
       - Combines all relevant information with proper context markers

    Key features:
    - Comprehensive memory management
    - Multiple retrieval strategies
    - Automatic memory flow between tiers
    - Importance-based filtering
    - Semantic search capabilities

    Use cases:
    - Complex, long-running conversations
    - Discussions requiring both detailed recent context and historical information
    - Situations where memory organization is critical
    """
    
    def __init__(self):
        try:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Three-tier memory system
            self.immediate_context = []  # Last few messages
            self.short_term_memory = []  # Recent summaries
            self.long_term_memory = []   # Important embedded memories
            
            print(colored("Hierarchical memory system initialized", "green"))
        except Exception as e:
            print(colored(f"Error initializing hierarchical memory: {str(e)}", "red"))
            raise e
    
    async def _create_embedding(self, text):
        try:
            response = await self.client.embeddings.create(
                model=HIERARCHICAL_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(colored(f"Error creating embedding: {str(e)}", "red"))
            return None
    
    async def _assess_importance(self, message):
        try:
            response = await self.client.chat.completions.create(
                model=HIERARCHICAL_MODEL,
                messages=[
                    {"role": "system", "content": "Rate the importance of this message for long-term memory on a scale of 0 to 1. Respond with only the number."},
                    {"role": "user", "content": message["content"]}
                ]
            )
            return float(response.choices[0].message.content.strip())
        except Exception as e:
            print(colored(f"Error assessing importance: {str(e)}", "red"))
            return 0
    
    async def _create_summary(self, messages):
        try:
            messages_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            response = await self.client.chat.completions.create(
                model=HIERARCHICAL_MODEL,
                messages=[
                    {"role": "system", "content": "Create a concise summary of this conversation chunk."},
                    {"role": "user", "content": messages_text}
                ]
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(colored(f"Error creating summary: {str(e)}", "red"))
            return None
    
    async def add_memory(self, message):
        try:
            # Add timestamp
            message["timestamp"] = datetime.now().isoformat()
            
            # Update immediate context
            self.immediate_context.append(message)
            
            # Overflow management
            if len(self.immediate_context) > IMMEDIATE_CONTEXT_SIZE:
                overflow_messages = self.immediate_context[:IMMEDIATE_CONTEXT_SIZE//2]
                self.immediate_context = self.immediate_context[IMMEDIATE_CONTEXT_SIZE//2:]
                
                # Create summary for overflow messages
                if overflow_messages:
                    summary = await self._create_summary(overflow_messages)
                    if summary:
                        self.short_term_memory.append({
                            "summary": summary,
                            "messages": overflow_messages,
                            "timestamp": overflow_messages[-1]["timestamp"]
                        })
            
            # Regulate short-term memory size
            if len(self.short_term_memory) > SHORT_TERM_SIZE:
                overflow_summaries = self.short_term_memory[:SHORT_TERM_SIZE//2]
                self.short_term_memory = self.short_term_memory[SHORT_TERM_SIZE//2:]
                
                # Process important memories for long-term storage
                for summary_item in overflow_summaries:
                    importance = await self._assess_importance({"content": summary_item["summary"]})
                    
                    if importance >= HIERARCHICAL_IMPORTANCE_THRESHOLD:
                        embedding = await self._create_embedding(summary_item["summary"])
                        if embedding:
                            self.long_term_memory.append({
                                "summary": summary_item["summary"],
                                "embedding": embedding,
                                "importance": importance,
                                "timestamp": summary_item["timestamp"]
                            })
            
            # Regulate long-term memory
            if len(self.long_term_memory) > LONG_TERM_SIZE:
                self.long_term_memory.sort(key=lambda x: x["importance"], reverse=True)
                self.long_term_memory = self.long_term_memory[:LONG_TERM_SIZE]
            
            print(colored("Memory added successfully", "green"))
        except Exception as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))
    
    async def get_relevant_memories(self, query):
        try:
            relevant_memories = []
            
            # Always include immediate context
            relevant_memories.extend(self.immediate_context)
            
            # Get query embedding for long-term memory search
            query_embedding = await self._create_embedding(query)
            if query_embedding:
                # Find relevant long-term memories
                similarities = []
                for memory in self.long_term_memory:
                    similarity = np.dot(query_embedding, memory["embedding"]) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(memory["embedding"])
                    )
                    similarities.append((similarity, memory))
                
                # Get top relevant long-term memories
                similarities.sort(reverse=True)
                for sim, memory in similarities[:3]:  # Top 3 most relevant
                    if sim > 0.7:  # Similarity threshold
                        relevant_memories.append({
                            "role": "assistant",
                            "content": f"Relevant past context: {memory['summary']}"
                        })
            
            # Get relevant short-term memories using GPT
            if self.short_term_memory:
                response = await self.client.chat.completions.create(
                    model=HIERARCHICAL_MODEL,
                    messages=[
                        {"role": "system", "content": "Select indices of relevant summaries for the query. Return space-separated numbers only."},
                        {"role": "user", "content": f"Query: {query}\n\nSummaries:\n" + 
                         "\n".join([f"{i}: {m['summary']}" for i, m in enumerate(self.short_term_memory)])}
                    ]
                )
                
                try:
                    indices = [int(i) for i in response.choices[0].message.content.split() if i.isdigit()]
                    for idx in indices[:2]:  # Top 2 most relevant
                        if 0 <= idx < len(self.short_term_memory):
                            relevant_memories.append({
                                "role": "assistant",
                                "content": f"Recent context: {self.short_term_memory[idx]['summary']}"
                            })
                except ValueError:
                    print(colored("Error parsing relevant summary indices", "yellow"))
            
            return relevant_memories
        except Exception as e:
            print(colored(f"Error retrieving memories: {str(e)}", "red"))
            return self.immediate_context  # Fallback to immediate context only
    
    def get_memory_stats(self):
        return {
            "immediate_context": len(self.immediate_context),
            "short_term_memory": len(self.short_term_memory),
            "long_term_memory": len(self.long_term_memory)
        }