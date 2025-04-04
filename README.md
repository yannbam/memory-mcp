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

# Download required NLTK data
python -m nltk.downloader punkt stopwords wordnet
```

## Usage

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

## Credits

Based on code by echo.hive
https://x.com/hive_echo/status/1880895879231721877
https://www.echohive.live
https://github.com/echohive42
