"""
OpenRouter API Example
---------------------

This example demonstrates how to use the memory systems with OpenRouter API.
It creates a VectorMemory instance and adds a few sample messages.
"""

import asyncio
import os
from dotenv import load_dotenv
from src.Memory import create_memory
from termcolor import colored

# Load environment variables from .env file
load_dotenv()

async def main():
    # Check if OpenRouter API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        print(colored("Error: OPENROUTER_API_KEY environment variable is not set.", "red"))
        print(colored("Please create a .env file with your OpenRouter API key or set it in your environment.", "red"))
        print(colored("Example .env file:", "yellow"))
        print(colored("OPENROUTER_API_KEY=your_openrouter_api_key_here", "yellow"))
        return
    
    print(colored("Creating VectorMemory with OpenRouter API...", "cyan"))
    
    # Create a VectorMemory instance with OpenRouter API
    memory = create_memory(
        memory_type="vector",
        api_type="openrouter"
    )
    
    # Sample messages
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me more about Paris."},
        {"role": "assistant", "content": "Paris is the capital and most populous city of France. It is located on the Seine River, in the north of the country. Paris is known worldwide for its rich history, art, fashion, gastronomy, and culture. Famous landmarks include the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which houses the Mona Lisa."}
    ]
    
    # Add messages to memory
    print(colored("\nAdding messages to memory...", "cyan"))
    for message in messages:
        print(colored(f"Adding message: {message['role']}: {message['content'][:50]}...", "blue"))
        await memory.add_memory(message)
    
    # Retrieve relevant memories
    print(colored("\nRetrieving relevant memories for query...", "cyan"))
    query = "What do you know about the Eiffel Tower?"
    print(colored(f"Query: {query}", "blue"))
    
    relevant_memories = await memory.get_relevant_memories(query)
    
    print(colored("\nRelevant memories:", "green"))
    for i, memory in enumerate(relevant_memories):
        print(colored(f"{i+1}. {memory['role']}: {memory['content']}", "yellow"))
    
    # Print memory stats
    print(colored("\nMemory stats:", "cyan"))
    stats = memory.get_memory_stats()
    for key, value in stats.items():
        print(colored(f"{key}: {value}", "blue"))

if __name__ == "__main__":
    asyncio.run(main())
