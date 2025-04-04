from termcolor import colored
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Constants for KeywordMemory
MAX_MEMORIES_PER_KEYWORD = 50
MIN_KEYWORD_LENGTH = 3

class KeywordMemory:
    """
    Keyword Memory System
    ---------------------

    This memory system uses natural language processing techniques to index and
    retrieve memories based on keyword matching, without requiring API calls.

    How it works:
    1. Text processing:
       - Tokenization splits text into words
       - Stop words removal eliminates common words
       - Lemmatization reduces words to their base form
       
    2. Memory indexing:
       - Keywords are extracted from each message
       - Messages are indexed by their keywords
       - Each keyword maintains a list of message IDs
       
    3. Retrieval process:
       - Keywords are extracted from the query
       - Messages containing query keywords are scored
       - Results are ranked by keyword match frequency

    Key features:
    - Lightweight, no API dependencies
    - Efficient text processing
    - Efficient keyword-based indexing
    - Configurable keyword parameters
    - Score-based relevance ranking

    Use cases:
    - Fast, local memory searching
    - Keyword-focused retrieval
    - Situations where API access is limited
    - Simple pattern matching needs
    """
    
    def __init__(self):
        try:
            # Initialize NLTK components
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Memory storage
            self.keyword_index = defaultdict(list)
            self.all_memories = []
            
            print(colored("Keyword memory system initialized", "green"))
        except Exception as e:
            print(colored(f"Error initializing keyword memory: {str(e)}", "red"))
            raise e
    
    def _extract_keywords(self, text):
        try:
            # Tokenize and clean text
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and short words, lemmatize
            keywords = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token not in self.stop_words
                and len(token) >= MIN_KEYWORD_LENGTH
                and token.isalnum()
            ]
            
            return set(keywords)  # Remove duplicates
        except Exception as e:
            print(colored(f"Error extracting keywords: {str(e)}", "red"))
            return set()
    
    def add_memory(self, message):
        try:
            memory_id = len(self.all_memories)
            self.all_memories.append(message)
            
            # Extract keywords from message content
            keywords = self._extract_keywords(message["content"])
            
            # Index memory by keywords
            for keyword in keywords:
                if len(self.keyword_index[keyword]) < MAX_MEMORIES_PER_KEYWORD:
                    self.keyword_index[keyword].append(memory_id)
            
            print(colored("Memory added successfully", "green"))
        except Exception as e:
            print(colored(f"Error adding memory: {str(e)}", "red"))
    
    def get_relevant_memories(self, query, max_results=5):
        try:
            # Extract keywords from query
            query_keywords = self._extract_keywords(query)
            
            # Find memories that match query keywords
            memory_scores = defaultdict(int)
            for keyword in query_keywords:
                for memory_id in self.keyword_index.get(keyword, []):
                    memory_scores[memory_id] += 1
            
            # Sort memories by relevance score
            relevant_ids = sorted(
                memory_scores.keys(),
                key=lambda x: memory_scores[x],
                reverse=True
            )[:max_results]
            
            return [self.all_memories[i] for i in relevant_ids]
        except Exception as e:
            print(colored(f"Error retrieving memories: {str(e)}", "red"))
            return []
    
    def get_memory_stats(self):
        return {
            "total_memories": len(self.all_memories),
            "total_keywords": len(self.keyword_index),
            "avg_memories_per_keyword": sum(len(v) for v in self.keyword_index.values()) / len(self.keyword_index) if self.keyword_index else 0
        }