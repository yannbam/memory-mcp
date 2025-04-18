"""
API Client Module
---------------

This module provides a unified interface for interacting with different LLM APIs.
Currently supports:
- OpenAI API
- OpenRouter API (which provides access to 300+ models)

The APIClient class is an abstract base class that defines the interface,
and specific implementations are provided for each supported API.
"""

import os
import abc
from typing import Dict, List, Any, Optional, Union, Literal
from dotenv import load_dotenv
from openai import AsyncOpenAI
from termcolor import colored

# Load environment variables from .env file if it exists
load_dotenv()

class APIClient(abc.ABC):
    """
    Abstract base class for API clients.
    Defines the interface for interacting with LLM APIs.
    """
    
    @abc.abstractmethod
    async def create_embeddings(
        self, 
        model: str, 
        input_text: str,
        dimensions: Optional[int] = None
    ) -> List[float]:
        """
        Create embeddings for the given input text.
        
        Args:
            model: The name of the embedding model to use
            input_text: The text to embed
            dimensions: Optional number of dimensions to reduce the embedding to
            
        Returns:
            A list of floats representing the embedding
        """
        pass
    
    @abc.abstractmethod
    async def create_chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion for the given messages.
        
        Args:
            model: The name of the model to use
            messages: A list of message dictionaries with 'role' and 'content' keys
            temperature: The temperature to use for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The API response as a dictionary
        """
        pass


class OpenAIClient(APIClient):
    """
    Client for the OpenAI API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: The OpenAI API key. If not provided, will be read from OPENAI_API_KEY environment variable.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Provide it as an argument or set the OPENAI_API_KEY environment variable.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        print(colored("OpenAI API client initialized", "green"))
    
    async def create_embeddings(
        self, 
        model: str, 
        input_text: str,
        dimensions: Optional[int] = None
    ) -> List[float]:
        """
        Create embeddings using the OpenAI API.
        
        Args:
            model: The name of the embedding model to use (e.g., "text-embedding-3-small")
            input_text: The text to embed
            dimensions: Optional number of dimensions to reduce the embedding to
            
        Returns:
            A list of floats representing the embedding
        """
        params = {
            "model": model,
            "input": input_text
        }
        if dimensions is not None:
            params["dimensions"] = dimensions
            
        response = await self.client.embeddings.create(**params)
        return response.data[0].embedding
    
    async def create_chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion using the OpenAI API.
        
        Args:
            model: The name of the model to use (e.g., "gpt-4o")
            messages: A list of message dictionaries with 'role' and 'content' keys
            temperature: The temperature to use for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The API response
        """
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        return await self.client.chat.completions.create(**params)


class OpenRouterClient(APIClient):
    """
    Client for the OpenRouter API, which provides access to 300+ models
    through a unified API interface compatible with the OpenAI API.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        http_referer: Optional[str] = None,
        title: Optional[str] = None
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: The OpenRouter API key. If not provided, will be read from OPENROUTER_API_KEY environment variable.
            http_referer: The HTTP Referer header for OpenRouter analytics (optional)
            title: The X-Title header for OpenRouter analytics (optional)
        """
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key is required. Provide it as an argument or set the OPENROUTER_API_KEY environment variable.")
        
        # Optional headers for OpenRouter analytics
        default_headers = {}
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if title:
            default_headers["X-Title"] = title
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=default_headers
        )
        print(colored("OpenRouter API client initialized", "green"))
    
    async def create_embeddings(
        self, 
        model: str, 
        input_text: str,
        dimensions: Optional[int] = None
    ) -> List[float]:
        """
        Create embeddings using the OpenRouter API.
        
        Args:
            model: The name of the embedding model to use 
                  (e.g., "text-embedding-3-small" or "openai/text-embedding-3-small")
            input_text: The text to embed
            dimensions: Optional number of dimensions to reduce the embedding to
            
        Returns:
            A list of floats representing the embedding
        """
        params = {
            "model": model,
            "input": input_text
        }
        if dimensions is not None:
            params["dimensions"] = dimensions
            
        response = await self.client.embeddings.create(**params)
        return response.data[0].embedding
    
    async def create_chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a chat completion using the OpenRouter API.
        
        Args:
            model: The name of the model to use (e.g., "openai/gpt-4o", "anthropic/claude-3-opus")
            messages: A list of message dictionaries with 'role' and 'content' keys
            temperature: The temperature to use for generation
            max_tokens: The maximum number of tokens to generate
            
        Returns:
            The API response
        """
        # Create the completion parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            # Disable the transform parameter to prevent truncating the middle of prompts
            "extra_body": {"transforms": []}
        }
        
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        return await self.client.chat.completions.create(**params)


def get_api_client(
    api_type: Literal["openai", "openrouter"] = None,
    api_key: Optional[str] = None,
    http_referer: Optional[str] = None,
    title: Optional[str] = None
) -> APIClient:
    """
    Factory function to get the appropriate API client.
    
    Args:
        api_type: The type of API client to get ("openai" or "openrouter").
                 If not provided, it will attempt to determine based on environment variables.
        api_key: The API key to use. If not provided, it will be read from the appropriate environment variable.
        http_referer: The HTTP Referer header for OpenRouter analytics (only applicable for OpenRouter)
        title: The X-Title header for OpenRouter analytics (only applicable for OpenRouter)
        
    Returns:
        An instance of the appropriate APIClient subclass
        
    Raises:
        ValueError: If the api_type is invalid or cannot be determined
    """
    # If api_type is not provided, try to determine from environment variables
    if api_type is None:
        if os.getenv("OPENAI_API_KEY"):
            api_type = "openai"
        elif os.getenv("OPENROUTER_API_KEY"):
            api_type = "openrouter"
        else:
            raise ValueError(
                "API type not specified and could not be determined from environment variables. "
                "Please provide api_type or set either OPENAI_API_KEY or OPENROUTER_API_KEY environment variables."
            )
    
    if api_type == "openai":
        return OpenAIClient(api_key=api_key)
    elif api_type == "openrouter":
        return OpenRouterClient(api_key=api_key, http_referer=http_referer, title=title)
    else:
        raise ValueError(f"Invalid API type: {api_type}. Supported types are 'openai' and 'openrouter'.")
