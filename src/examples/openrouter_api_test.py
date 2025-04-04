"""
OpenRouter API Direct Test Script
--------------------------------

This script directly tests the OpenRouter API using our APIClient abstraction,
focusing only on chat completion functionality since OpenRouter doesn't support embeddings.
"""

import asyncio
import os
from dotenv import load_dotenv
from termcolor import colored
from ..APIClient import get_api_client

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
    
    print(colored("Testing OpenRouter API with APIClient abstraction", "cyan"))
    print(colored("------------------------------------------------", "cyan"))
    
    # Initialize OpenRouter API client
    client = get_api_client(
        api_type="openrouter",
        http_referer="https://example.com",
        title="APIClient Test"
    )
    
    # Test chat completion
    print(colored("\nTesting chat completion with OpenRouter API...", "cyan"))
    
    # The model name for OpenRouter requires the provider prefix
    # model = "openai/gpt-4o"
    model = "mistralai/mistral-small-3.1-24b-instruct"
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise in your answers."},
        {"role": "user", "content": "What's the capital of France and what is it known for?"}
    ]
    
    print(colored(f"Using model: {model}", "blue"))
    print(colored(f"Sending messages: {messages}", "blue"))
    
    try:
        response = await client.create_chat_completion(
            model=model,
            messages=messages,
            temperature=0.7
        )
        
        print(colored("\nChat completion response:", "green"))
        print(colored(f"Model: {response.model}", "yellow"))
        print(colored(f"Response: {response.choices[0].message.content}", "yellow"))
        
    except Exception as e:
        print(colored(f"Error testing chat completion: {str(e)}", "red"))

if __name__ == "__main__":
    asyncio.run(main())
