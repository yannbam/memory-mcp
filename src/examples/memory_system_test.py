import asyncio
import sys
import os
from termcolor import colored

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.memory.VectorMemory import VectorMemory
from src.memory.HierarchicalMemory import HierarchicalMemory
from src.memory.KeywordMemory import KeywordMemory
from src.memory.SummaryMemory import SummaryMemory
from src.memory.TimeWindowMemory import TimeWindowMemory

async def test_vector_memory():
    print(colored("\n=== Testing Vector Memory ===", "blue"))
    vector_memory = VectorMemory()
    
    # Test adding memories
    test_messages = [
        {"content": "Machine learning is revolutionizing artificial intelligence."},
        {"content": "Python is a powerful programming language for data science."},
        {"content": "Neural networks can simulate complex human-like reasoning."}
    ]
    
    for message in test_messages:
        await vector_memory.add_memory(message)
    
    # Test memory retrieval
    query = "AI and data science"
    relevant_memories = await vector_memory.get_relevant_memories(query)
    
    print(colored("\nRelevant Memories:", "green"))
    for memory in relevant_memories:
        print(memory["content"])
    
    # Test memory stats
    stats = vector_memory.get_memory_stats()
    print(colored("\nMemory Stats:", "green"))
    print(stats)

async def test_hierarchical_memory():
    print(colored("\n=== Testing Hierarchical Memory ===", "blue"))
    hierarchical_memory = HierarchicalMemory()
    
    # Test adding memories
    test_messages = [
        {"role": "user", "content": "Tell me about quantum computing."},
        {"role": "assistant", "content": "Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform computation."},
        {"role": "user", "content": "How does that differ from classical computing?"},
        {"role": "assistant", "content": "Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously, allowing for much more complex computations."}
    ]
    
    for message in test_messages:
        await hierarchical_memory.add_memory(message)
    
    # Test memory retrieval
    query = "advanced computing technologies"
    relevant_memories = await hierarchical_memory.get_relevant_memories(query)
    
    print(colored("\nRelevant Memories:", "green"))
    for memory in relevant_memories:
        print(memory.get("content", "No content"))
    
    # Test memory stats
    stats = hierarchical_memory.get_memory_stats()
    print(colored("\nMemory Stats:", "green"))
    print(stats)

async def test_keyword_memory():
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

async def test_summary_memory():
    print(colored("\n=== Testing Summary Memory ===", "blue"))
    summary_memory = SummaryMemory()
    
    # Test adding memories
    test_messages = [
        {"role": "user", "content": "Let's discuss renewable energy."},
        {"role": "assistant", "content": "Renewable energy sources include solar, wind, hydroelectric, and geothermal power."},
        {"role": "user", "content": "Which one is the most promising?"},
        {"role": "assistant", "content": "Solar and wind energy are currently showing the most potential for widespread adoption."},
        {"role": "user", "content": "What about their environmental impact?"},
        {"role": "assistant", "content": "These sources have significantly lower carbon emissions compared to fossil fuels."}
    ]
    
    for message in test_messages:
        await summary_memory.add_memory(message)
    
    # Test memory retrieval
    query = "green energy technologies"
    relevant_memories = await summary_memory.get_relevant_memories(query)
    
    print(colored("\nRelevant Memories:", "green"))
    for memory in relevant_memories:
        print(f"{memory['role']}: {memory['content']}")
    
    # Test memory stats
    stats = summary_memory.get_memory_stats()
    print(colored("\nMemory Stats:", "green"))
    print(stats)

async def test_time_window_memory():
    print(colored("\n=== Testing Time Window Memory ===", "blue"))
    time_window_memory = TimeWindowMemory()
    
    # Test adding memories
    test_messages = [
        {"content": "Artificial intelligence is transforming multiple industries."},
        {"content": "Blockchain technology offers secure and transparent transaction systems."},
        {"content": "Cybersecurity is becoming increasingly critical in the digital age."},
        {"content": "Quantum computing might revolutionize cryptography and data processing."}
    ]
    
    for message in test_messages:
        await time_window_memory.add_memory(message)
    
    # Test memory retrieval
    query = "technology innovations"
    relevant_memories = await time_window_memory.get_relevant_memories(query)
    
    print(colored("\nRelevant Memories:", "green"))
    for memory in relevant_memories:
        print(memory["content"])
    
    # Test memory stats
    stats = time_window_memory.get_memory_stats()
    print(colored("\nMemory Stats:", "green"))
    print(stats)

async def main():
    print(colored("Memory System Comprehensive Test", "cyan", attrs=['bold']))
    
    # Run tests for each memory system
    await test_vector_memory()
    await test_hierarchical_memory()
    await test_keyword_memory()
    await test_summary_memory()
    await test_time_window_memory()

if __name__ == "__main__":
    asyncio.run(main())