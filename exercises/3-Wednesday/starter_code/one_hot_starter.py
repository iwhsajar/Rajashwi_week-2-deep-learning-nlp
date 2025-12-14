"""
Exercise 02: One-Hot Encoding - Starter Code

Implement one-hot encoding and analyze its limitations.

Prerequisites:
- Reading: 03-one-hot-encoding.md
- Demo: demo_02_one_hot_encoding.py (REFERENCE FOR IMPLEMENTATION)
"""

import numpy as np
from collections import Counter

# ============================================================================
# TASK 2.1: Basic One-Hot Encoding
# ============================================================================

class OneHotEncoder:
    """
    One-hot encoder for text data.
    
    CONCEPT: Each word becomes a vector of length vocab_size
    - All zeros except a single 1 at that word's index
    - "cat" in vocab [cat, dog, fish] -> [1, 0, 0]
    """
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
    
    def fit(self, texts):
        """
        Build vocabulary from list of texts.
        
        STEPS:
        1. Split each text into lowercase words
        2. Collect all unique words
        3. Sort alphabetically (for reproducibility)
        4. Create word_to_idx: {word: index} mapping
        5. Create idx_to_word: {index: word} mapping
        """
        # YOUR CODE HERE
        pass
    
    def encode_word(self, word):
        """
        Encode single word to one-hot vector.
        
        OUTPUT: numpy array of shape (vocab_size,)
        - All zeros, except 1 at word's index
        - Return zero vector for unknown words
        
        HINT: np.zeros(self.vocab_size), then set one element to 1
        """
        # YOUR CODE HERE
        return np.zeros(1)
    
    def encode_sentence(self, sentence):
        """
        Encode sentence to matrix of one-hot vectors.
        
        OUTPUT: numpy array of shape (num_words, vocab_size)
        Each row is a word's one-hot vector.
        """
        # YOUR CODE HERE
        return np.zeros((1, 1))


# ============================================================================
# TASK 2.2: Memory Analysis
# ============================================================================

def analyze_memory_usage():
    """
    Analyze one-hot encoding memory requirements.
    
    FORMULA:
    - Each word = vocab_size floats = vocab_size * 4 bytes (float32)
    - Sentence of N words = N * vocab_size * 4 bytes
    
    TEST THESE VOCAB SIZES: 100, 1000, 10000, 50000, 100000
    
    QUESTION: Why is this problematic for real NLP (vocab > 100k)?
    """
    vocab_sizes = [100, 1000, 10000, 50000, 100000]
    
    print("One-Hot Encoding Memory Analysis")
    print(f"{'Vocab Size':>12} {'Bytes/Word':>12} {'KB/100 Words':>15}")
    print("-" * 45)
    
    # YOUR CODE: Calculate and print memory for each vocab size
    pass


# ============================================================================
# TASK 2.3: Sparsity Visualization
# ============================================================================

def visualize_sparsity():
    """
    Visualize how sparse one-hot vectors are.
    
    CREATE A HEATMAP showing one-hot encodings for a sentence.
    - Each row = a word
    - Each column = vocab index
    - Color: 0 = white, 1 = black
    
    Save to: one_hot_sparsity.png
    """
    import matplotlib.pyplot as plt
    
    # YOUR CODE: Encode a sentence and visualize as heatmap
    # Use plt.imshow() with cmap='binary'
    pass


# ============================================================================
# TASK 2.4: Limitations Analysis
# ============================================================================

def demonstrate_limitations():
    """
    Demonstrate the key limitations of one-hot encoding.
    
    LIMITATION 1: No Semantic Similarity
    - Compute cosine similarity between "dog" and "cat"
    - Compute cosine similarity between "dog" and "computer"
    - They should be the same (0)! One-hot captures no meaning.
    
    LIMITATION 2: Memory Explosion
    - Real vocabularies are 100k+ words
    - Calculate memory for a batch of 32 sentences, 100 words each
    
    FORMULA for cosine similarity:
    cosine_sim(a, b) = (a . b) / (||a|| * ||b||)
    """
    print("One-Hot Encoding Limitations")
    
    # YOUR CODE HERE
    pass


def write_analysis():
    """
    Write analysis.txt answering:
    
    1. Why can't one-hot capture word meaning?
    2. What is the memory cost for a 100k vocab, 100-word sentence?
    3. How do embeddings solve both problems? (preview for Exercise 03)
    """
    # YOUR CODE: Write your analysis
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02: One-Hot Encoding")
    print("=" * 60)
    
    # Test basic encoding
    corpus = ["the cat sat on the mat", "the dog ran in the park"]
    # encoder = OneHotEncoder()
    # encoder.fit(corpus)
    # print(f"Vocab size: {encoder.vocab_size}")
    # print(f"'cat' encoded: {encoder.encode_word('cat')}")
    
    # Uncomment as you complete:
    # analyze_memory_usage()
    # visualize_sparsity()
    # demonstrate_limitations()
    # write_analysis()
