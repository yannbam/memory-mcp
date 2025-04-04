# Memory MCP

A collection of sophisticated memory systems for large language models.

## Memory Systems

- **VectorMemory**: Uses embeddings for semantic search
- **SummaryMemory**: Creates and retrieves conversation summaries
- **TimeWindowMemory**: Combines recent and important long-term memories
- **KeywordMemory**: Lightweight keyword-based search
- **HierarchicalMemory**: Three-tier system with immediate, short-term, and long-term memory

## Installation

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# Or install with dev dependencies
uv pip install -e ".[dev]"

# Download required NLTK data
python -m nltk.downloader punkt stopwords wordnet
```

## Usage

### Basic Usage

```python
from src.Memory import create_memory

# Create a memory system
memory = create_memory("vector")  # Options: "vector", "summary", "timewindow", "keyword", "hierarchical"

# Add a memory
await memory.add_memory({
    "role": "user",
    "content": "Hello, how are you?"
})

# Retrieve relevant memories
relevant_memories = await memory.get_relevant_memories("How are you doing today?")

# Get memory stats
stats = memory.get_memory_stats()
```

### API Provider Configuration

```python
# Create memory with OpenAI API
memory = create_memory(
    memory_type="vector",
    api_type="openai",
    api_key="your_openai_api_key"  # Optional, will use OPENAI_API_KEY from environment if not provided
)

# Create memory with OpenRouter API 
memory = create_memory(
    memory_type="vector",
    api_type="openrouter",
    api_key="your_openrouter_api_key"  # Optional, will use OPENROUTER_API_KEY from environment if not provided
)
```

### Running Examples

Navigate to the project root and run the examples using `uv`:

```bash
# Test OpenAI API functionality
uv run src/examples/openai_api_test.py

# Test OpenRouter API functionality
uv run src/examples/openrouter_api_test.py

# Run the OpenRouter memory example
uv run src/examples/openrouter_example.py
```

## Credits

Based on code by echo.hive
https://x.com/hive_echo/status/1880895879231721877
https://www.echohive.live
https://github.com/echohive42
