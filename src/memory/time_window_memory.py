import os
from termcolor import colored
from openai import AsyncOpenAI
from datetime import datetime

# Constants for TimeWindowMemory
IMPORTANCE_MODEL = "gpt-4o"
WINDOW_SIZE = 10  # Number of recent messages to keep
MAX_IMPORTANT_MEMORIES = 50
IMPORTANCE_THRESHOLD = 0.7

class TimeWindowMemory:
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