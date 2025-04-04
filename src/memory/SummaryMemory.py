import os
from termcolor import colored
from openai import AsyncOpenAI

# Constants for SummaryMemory
SUMMARY_MODEL = "gpt-4o"
MAX_SUMMARIES = 10
SUMMARY_THRESHOLD = 5  # Number of messages before creating a new summary

class SummaryMemory:
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