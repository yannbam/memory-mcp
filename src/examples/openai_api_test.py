"""
OpenAI API Direct Test Script
----------------------------

This script directly tests the OpenAI API using our APIClient abstraction,
testing both embedding and chat completion functionality.
"""

import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from termcolor import colored
from ..APIClient import get_api_client

# Load environment variables from .env file
load_dotenv()

async def main():
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print(colored("Error: OPENAI_API_KEY environment variable is not set.", "red"))
        print(colored("Please create a .env file with your OpenAI API key or set it in your environment.", "red"))
        print(colored("Example .env file:", "yellow"))
        print(colored("OPENAI_API_KEY=your_openai_api_key_here", "yellow"))
        return
    
    print(colored("Testing OpenAI API with APIClient abstraction", "cyan"))
    print(colored("------------------------------------------", "cyan"))
    
    # Initialize OpenAI API client
    client = get_api_client(api_type="openai")
    
    # Test embedding creation
    print(colored("\nTesting embedding creation with OpenAI API...", "cyan"))
    
    embedding_model = "text-embedding-3-small"
    input_text = "This is a test text for creating embeddings."
    
    print(colored(f"Using model: {embedding_model}", "blue"))
    print(colored(f"Input text: {input_text}", "blue"))
    
    try:
        embedding = await client.create_embeddings(
            model=embedding_model,
            input_text=input_text
        )
        
        print(colored("\nEmbedding response:", "green"))
        print(colored(f"Embedding dimension: {len(embedding)}", "yellow"))
        print(colored(f"Embedding norm: {np.linalg.norm(embedding):.4f}", "yellow"))
        print(colored(f"First 5 values: {embedding[:5]}", "yellow"))
        
    except Exception as e:
        print(colored(f"Error testing embedding creation: {str(e)}", "red"))
    
    # Test chat completion
    print(colored("\nTesting chat completion with OpenAI API...", "cyan"))
    
    chat_model = "gpt-4o"
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise in your answers."},
        {"role": "user", "content": "What's the capital of France and what is it known for?"}
    ]
    
    print(colored(f"Using model: {chat_model}", "blue"))
    print(colored(f"Sending messages: {messages}", "blue"))
    
    try:
        response = await client.create_chat_completion(
            model=chat_model,
            messages=messages,
            temperature=0.7
        )
        
        print(colored("\nChat completion response:", "green"))
        print(colored(f"Model: {response.model}", "yellow"))
        print(colored(f"Response: {response.choices[0].message.content}", "yellow"))
        
    except Exception as e:
        print(colored(f"Error testing chat completion: {str(e)}", "red"))
    
    # Optional: Test similarity between embeddings
    print(colored("\nTesting embedding similarity...", "cyan"))
    
    try:
        # Create embeddings for two related texts
        text1 = "Paris is the capital of France."
        text2 = "The Eiffel Tower is in Paris."
        text3 = "Quantum physics studies subatomic particles."
        
        print(colored(f"Text 1: {text1}", "blue"))
        print(colored(f"Text 2: {text2}", "blue"))
        print(colored(f"Text 3: {text3}", "blue"))
        
        embedding1 = await client.create_embeddings(model=embedding_model, input_text=text1)
        embedding2 = await client.create_embeddings(model=embedding_model, input_text=text2)
        embedding3 = await client.create_embeddings(model=embedding_model, input_text=text3)
        
        # Calculate cosine similarity
        similarity12 = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity13 = np.dot(embedding1, embedding3) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding3))
        similarity23 = np.dot(embedding2, embedding3) / (np.linalg.norm(embedding2) * np.linalg.norm(embedding3))
        
        print(colored("\nSimilarity results:", "green"))
        print(colored(f"Similarity between text1 and text2: {similarity12:.4f}", "yellow"))
        print(colored(f"Similarity between text1 and text3: {similarity13:.4f}", "yellow"))
        print(colored(f"Similarity between text2 and text3: {similarity23:.4f}", "yellow"))
        
    except Exception as e:
        print(colored(f"Error testing embedding similarity: {str(e)}", "red"))

if __name__ == "__main__":
    asyncio.run(main())
