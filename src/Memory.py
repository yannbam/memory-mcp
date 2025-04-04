import os
import numpy as np
from termcolor import colored
from openai import AsyncOpenAI
"""
Vector Memory System
----------------------

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

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.6
MAX_MEMORIES = 100

class VectorMemory:
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

"""
Summary Memory System
---------------------

This memory system creates concise summaries of conversation chunks to maintain
context while reducing memory usage.

How it works:
1. Messages are collected into chunks of size SUMMARY_THRESHOLD
2. When a chunk is full, GPT creates a summary of the conversation chunk
3. Summaries are stored along with their original messages
4. When retrieving memories, GPT identifies relevant summaries based on the query
5. Original messages from relevant summaries are returned

Key features:
- Automatic conversation chunking
- GPT-powered summarization
- Maintains original messages for context
- Limited number of summaries (MAX_SUMMARIES)
- Query-based retrieval of relevant chunks

Use cases:
- Long conversations where full context is important
- Reducing memory usage while preserving meaning
- Quick access to conversation highlights
"""

# Constants
SUMMARY_MODEL = "gpt-4o"
MAX_SUMMARIES = 10
SUMMARY_THRESHOLD = 5  # Number of messages before creating a new summary

class SummaryMemory:
    def __init__(self):
        try:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.current_chunk = []
            self.summaries = []
            print(colored("Summary memory system initialized", "green"))
        except Exception as e:
            print(colored(f"Error initializing summary memory: {str(e)}", "red"))
            raise e
    
    async def add_memory(self, message):
        try:
            self.current_chunk.append(message)
            
            if len(self.current_chunk) >= SUMMARY_THRESHOLD:
                await self._create_summary()
                
            print(colored("Memory added successfully", "green"))
        except Exception as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))
    
    async def _create_summary(self):
        try:
            chunk_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.current_chunk])
            
            response = await self.client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": "Create a brief, informative summary of this conversation chunk."},
                    {"role": "user", "content": chunk_text}
                ]
            )
            
            summary = {
                "content": response.choices[0].message.content,
                "original_messages": self.current_chunk.copy()
            }
            
            self.summaries.append(summary)
            self.current_chunk = []
            
            # Keep only MAX_SUMMARIES most recent summaries
            if len(self.summaries) > MAX_SUMMARIES:
                self.summaries.pop(0)
                
            print(colored("Created new memory summary", "green"))
        except Exception as e:
            print(colored(f"Error creating summary: {str(e)}", "red"))
    
    async def get_relevant_memories(self, query):
        try:
            # Get relevant summaries based on query
            response = await self.client.chat.completions.create(
                model=SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": "Select the most relevant summaries for the given query. Return indices of summaries only."},
                    {"role": "user", "content": f"Query: {query}\n\nSummaries:\n" + "\n".join([f"{i}: {s['content']}" for i, s in enumerate(self.summaries)])}
                ]
            )
            
            # Parse indices from response
            try:
                indices = [int(i) for i in response.choices[0].message.content.split() if i.isdigit()]
                relevant_memories = []
                
                for idx in indices:
                    if 0 <= idx < len(self.summaries):
                        relevant_memories.extend(self.summaries[idx]["original_messages"])
                
                return relevant_memories
            except ValueError:
                print(colored("Error parsing relevant summary indices", "yellow"))
                return []
                
        except Exception as e:
            print(colored(f"Error retrieving memories: {str(e)}", "red"))
            return []
    
    def get_memory_stats(self):
        return {
            "current_chunk_size": len(self.current_chunk),
            "total_summaries": len(self.summaries)
        }

"""
Time Window Memory System
-------------------------

This memory system combines recent messages with important long-term memories,
using a dual-storage approach based on time and importance.

How it works:
1. Recent messages are kept in a sliding window of size WINDOW_SIZE
2. Each message is evaluated for importance using GPT
3. Messages above IMPORTANCE_THRESHOLD are stored in long-term memory
4. When retrieving memories:
   - Recent messages are always included
   - Relevant important memories are selected based on the query

Key features:
- Maintains recent context with sliding window
- GPT-powered importance assessment
- Dual storage: recent and important memories
- Timestamp tracking for temporal context
- Importance-based sorting of long-term memories

Use cases:
- Balancing recent context with important historical information
- Conversations requiring both immediate and long-term context
- Prioritizing critical information while maintaining flow
"""

import os
from termcolor import colored
from openai import AsyncOpenAI
from datetime import datetime

# Constants
IMPORTANCE_MODEL = "gpt-4o"
WINDOW_SIZE = 10  # Number of recent messages to keep
MAX_IMPORTANT_MEMORIES = 50
IMPORTANCE_THRESHOLD = 0.7

class TimeWindowMemory:
    def __init__(self):
        try:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.recent_memories = []
            self.important_memories = []  # Long-term memory
            print(colored("Time-Window memory system initialized", "green"))
        except Exception as e:
            print(colored(f"Error initializing time-window memory: {str(e)}", "red"))
            raise e
    
    async def _assess_importance(self, message):
        try:
            response = await self.client.chat.completions.create(
                model=IMPORTANCE_MODEL,
                messages=[
                    {"role": "system", "content": "Rate the importance of this message for long-term memory on a scale of 0 to 1. Respond with only the number."},
                    {"role": "user", "content": message["content"]}
                ]
            )
            
            importance = float(response.choices[0].message.content.strip())
            return min(max(importance, 0), 1)  # Ensure between 0 and 1
        except Exception as e:
            print(colored(f"Error assessing importance: {str(e)}", "red"))
            return 0
    
    async def add_memory(self, message):
        try:
            # Add timestamp to message
            message["timestamp"] = datetime.now().isoformat()
            
            # Add to recent memories
            self.recent_memories.append(message)
            if len(self.recent_memories) > WINDOW_SIZE:
                self.recent_memories.pop(0)
            
            # Assess importance for long-term storage
            importance = await self._assess_importance(message)
            if importance >= IMPORTANCE_THRESHOLD:
                self.important_memories.append({
                    "message": message,
                    "importance": importance
                })
            
            # Keep only top important memories
            self.important_memories = sorted(self.important_memories, key=lambda x: x["importance"], reverse=True)
            self.important_memories = self.important_memories[:MAX_IMPORTANT_MEMORIES]
            
            print(colored("Memory added successfully", "green"))
        except Exception as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))
    
    async def get_relevant_memories(self, query):
        try:
            # Always include recent memories
            memories = self.recent_memories.copy()
            
            # Get relevant important memories
            response = await self.client.chat.completions.create(
                model=IMPORTANCE_MODEL,
                messages=[
                    {"role": "system", "content": "Select indices of important memories relevant to the query. Return space-separated numbers only."},
                    {"role": "user", "content": f"Query: {query}\n\nMemories:\n" + 
                     "\n".join([f"{i}: {m['message']['content']}" for i, m in enumerate(self.important_memories)])}
                ]
            )
            
            try:
                indices = [int(i) for i in response.choices[0].message.content.split() if i.isdigit()]
                for idx in indices:
                    if 0 <= idx < len(self.important_memories):
                        memories.append(self.important_memories[idx]["message"])
            except ValueError:
                pass
                
            return memories
        except Exception as e:
            print(colored(f"Error retrieving memories: {str(e)}", "red"))
            return self.recent_memories  # Fallback to recent memories only
    
    def get_memory_stats(self):
        return {
            "recent_memories": len(self.recent_memories),
            "important_memories": len(self.important_memories),
            "oldest_recent": self.recent_memories[0]["timestamp"] if self.recent_memories else None,
            "newest_recent": self.recent_memories[-1]["timestamp"] if self.recent_memories else None
        }

"""
Keyword Memory System
---------------------

This memory system uses natural language processing techniques to index and
retrieve memories based on keyword matching, without requiring API calls.

How it works:
1. Text processing:
   - Tokenization splits text into words
   - Stop words removal eliminates common words
   - Lemmatization reduces words to their base form
   
2. Memory indexing:
   - Keywords are extracted from each message
   - Messages are indexed by their keywords
   - Each keyword maintains a list of message IDs
   
3. Retrieval process:
   - Keywords are extracted from the query
   - Messages containing query keywords are scored
   - Results are ranked by keyword match frequency

Key features:
- Lightweight, no API dependencies
- Efficient text processing
- Efficient keyword-based indexing
- Configurable keyword parameters
- Score-based relevance ranking

Use cases:
- Fast, local memory searching
- Keyword-focused retrieval
- Situations where API access is limited
- Simple pattern matching needs
"""

import os
from termcolor import colored
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Constants
MAX_MEMORIES_PER_KEYWORD = 50
MIN_KEYWORD_LENGTH = 3

class KeywordMemory:
    def __init__(self):
        try:
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Memory storage
            self.keyword_index = defaultdict(list)
            self.all_memories = []
            
            print(colored("Keyword memory system initialized", "green"))
        except Exception as e:
            print(colored(f"Error initializing keyword memory: {str(e)}", "red"))
            raise e
    
    def _extract_keywords(self, text):
        try:
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and short words, lemmatize
            keywords = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words
                and len(token) >= MIN_KEYWORD_LENGTH
                and token.isalnum()
            ]
            
            return set(keywords)  # Remove duplicates
        except Exception as e:
            print(colored(f"Error extracting keywords: {str(e)}", "red"))
            return set()
    
    def add_memory(self, message):
        try:
            memory_id = len(self.all_memories)
            self.all_memories.append(message)
            
            # Extract keywords from message content
            keywords = self._extract_keywords(message["content"])
            
            # Index memory by keywords
            for keyword in keywords:
                if len(self.keyword_index[keyword]) < MAX_MEMORIES_PER_KEYWORD:
                    self.keyword_index[keyword].append(memory_id)
            
            print(colored("Memory added successfully", "green"))
        except Exception as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))
    
    def get_relevant_memories(self, query, max_results=5):
        try:
            # Extract keywords from query
            query_keywords = self._extract_keywords(query)
            
            # Find memories that match query keywords
            memory_scores = defaultdict(int)
            for keyword in query_keywords:
                for memory_id in self.keyword_index.get(keyword, []):
                    memory_scores[memory_id] += 1
            
            # Sort memories by relevance score
            relevant_ids = sorted(
                memory_scores.keys(),
                key=lambda x: memory_scores[x],
                reverse=True
            )[:max_results]
            
            return [self.all_memories[i] for i in relevant_ids]
        except Exception as e:
            print(colored(f"Error retrieving memories: {str(e)}", "red"))
            return []
    
    def get_memory_stats(self):
        return {
            "total_memories": len(self.all_memories),
            "total_keywords": len(self.keyword_index),
            "avg_memories_per_keyword": sum(len(v) for v in self.keyword_index.values()) / len(self.keyword_index) if self.keyword_index else 0
        }

"""
Hierarchical Memory System
--------------------------

This is the most sophisticated memory system, implementing a three-tier approach
that combines semantic context, short-term memory, and long-term memory.

How it works:
1. Three-tier memory structure:
   - Short-term memory: Recent messages (IMMEDIATE_CONTEXT_SIZE)
   - Medium-term memory: Important content from recent conversations
   - Long-term memory: Important semantic memories (LONG_TERM_SIZE)

2. Memory management:
   - Automatic filtering of irrelevant content
   - Importance-based filtering
   - Semantic search capabilities

3. Retrieval process:
   - Combines top memories from all tiers
   - Results ranked by relevance and recency
   - Dynamic weighting based on query context

Key features:
- Multiple context timeframes
- Semantic + importance-based ranking
- Intelligent memory management
- Fine-tuned importance thresholds
- Semantic search capabilities
- Enhanced context awareness

Use cases:
- Complex, long-running conversations
- Discussions requiring both recent context and historical information
- Situations where nuanced understanding is critical
"""

import os
from termcolor import colored
from openai import AsyncOpenAI
import numpy as np

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
IMMEDIATE_CONTEXT_SIZE = 5
MEDIUM_TERM_SIZE = 10
LONG_TERM_SIZE = 50

class HierarchicalMemory:
    def __init__(self):
        try:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.immediate_context = []  # Short-term memory
            self.medium_term_memory = []  # Recent important context
            self.long_term_memory = []  # Long-term memory (important memories)
            print(colored("Hierarchical memory system initialized", "green"))
        except Exception as e:
            print(colored(f"Error initializing hierarchical memory: {str(e)}", "red"))
            raise e
    
    async def create_embedding(self, text):
        try:
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(colored(f"Error creating embedding: {str(e)}", "red"))
            return None
    
    async def rate_importance(self, message):
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Rate the importance of this message for long-term memory on a scale of 0 to 1. Consider how crucial this information might be for future context. Respond with only the number."},
                    {"role": "user", "content": message["content"]}
                ]
            )
            return float(response.choices[0].message.content.strip())
        except Exception as e:
            print(colored(f"Error assessing importance: {str(e)}", "red"))
            return 0
    
    async def add_memory(self, message):
        try:
            # Add to immediate context (short-term memory)
            self.immediate_context.append(message)
            if len(self.immediate_context) > IMMEDIATE_CONTEXT_SIZE:
                # Move oldest message to medium-term if important enough
                oldest = self.immediate_context.pop(0)
                importance = await self.rate_importance(oldest)
                
                if importance > 0.5:  # Threshold for medium-term memory
                    embedding = await self.create_embedding(oldest["content"])
                    if embedding:
                        self.medium_term_memory.append({
                            "message": oldest,
                            "embedding": embedding,
                            "importance": importance
                        })
            
            # Manage medium-term memory
            if len(self.medium_term_memory) > MEDIUM_TERM_SIZE:
                # Move most important to long-term memory
                self.medium_term_memory.sort(key=lambda x: x["importance"], reverse=True)
                to_long_term = self.medium_term_memory.pop()
                
                if to_long_term["importance"] > 0.7:  # Threshold for long-term memory
                    self.long_term_memory.append(to_long_term)
                    
                    # Keep long-term memory within size limit
                    if len(self.long_term_memory) > LONG_TERM_SIZE:
                        self.long_term_memory.sort(key=lambda x: x["importance"])
                        self.long_term_memory.pop(0)  # Remove least important
            
            print(colored("Memory added successfully", "green"))
        except Exception as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))
    
    async def get_relevant_memories(self, query, context_size=5):
        # [TRUNCATED!]
