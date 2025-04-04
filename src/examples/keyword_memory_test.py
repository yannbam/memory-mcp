import sys
import os
from termcolor import colored

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.KeywordMemory import KeywordMemory

def test_keyword_memory():
    print(colored("\n=== Testing Keyword Memory ===", "blue"))
    keyword_memory = KeywordMemory()
    
    # Test adding memories
    test_messages = [
        {"content": "Python is an amazing programming language for data science and machine learning."},
        {"content": "Machine learning algorithms can solve complex problems in various domains."},
        {"content": "Data science requires strong statistical and programming skills."}
    ]
    
    for message in test_messages:
        keyword_memory.add_memory(message)
    
    # Test memory retrieval
    query = "programming machine learning"
    relevant_memories = keyword_memory.get_relevant_memories(query)
    
    print(colored("\nRelevant Memories:", "green"))
    for memory in relevant_memories:
        print(memory["content"])
    
    # Test memory stats
    stats = keyword_memory.get_memory_stats()
    print(colored("\nMemory Stats:", "green"))
    print(stats)

if __name__ == "__main__":
    print(colored("Keyword Memory Test", "cyan", attrs=['bold']))
    test_keyword_memory()
