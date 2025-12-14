"""
Exercise 02 Solution: One-Hot Encoding

Complete implementation of one-hot encoding for text with analysis.
"""

import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: One-Hot Encoding Implementation
# =============================================================================

class OneHotEncoder:
    """
    One-hot encoding for text data.
    
    Features:
    - Build vocabulary from corpus
    - Encode words as one-hot vectors
    - Encode sentences as matrices
    - Analyze sparsity and memory usage
    """
    
    def __init__(self):
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size: int = 0
    
    def build_vocabulary(self, texts: List[str], lowercase: bool = True) -> None:
        """
        Build vocabulary from corpus.
        
        Args:
            texts: List of text strings
            lowercase: Whether to lowercase
        """
        # Collect unique words
        words = set()
        for text in texts:
            if lowercase:
                text = text.lower()
            # Simple word tokenization
            tokens = text.split()
            words.update(tokens)
        
        # Sort for reproducibility
        sorted_words = sorted(words)
        
        # Build mappings
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(sorted_words)
        
        print(f"Vocabulary size: {self.vocab_size}")
    
    def encode_word(self, word: str) -> np.ndarray:
        """
        Encode a single word as one-hot vector.
        
        Args:
            word: Word to encode
            
        Returns:
            One-hot vector of shape (vocab_size,)
        """
        word = word.lower()
        vector = np.zeros(self.vocab_size, dtype=np.float32)
        
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            vector[idx] = 1.0
        
        return vector
    
    def encode_sentence(self, sentence: str) -> np.ndarray:
        """
        Encode a sentence as matrix of one-hot vectors.
        
        Args:
            sentence: Sentence to encode
            
        Returns:
            Matrix of shape (num_words, vocab_size)
        """
        words = sentence.lower().split()
        matrix = np.zeros((len(words), self.vocab_size), dtype=np.float32)
        
        for i, word in enumerate(words):
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                matrix[i, idx] = 1.0
        
        return matrix
    
    def decode_vector(self, vector: np.ndarray) -> str:
        """Decode one-hot vector back to word."""
        idx = np.argmax(vector)
        if vector[idx] == 0:
            return '<UNKNOWN>'
        return self.idx_to_word.get(idx, '<UNKNOWN>')
    
    def decode_matrix(self, matrix: np.ndarray) -> List[str]:
        """Decode matrix back to list of words."""
        return [self.decode_vector(row) for row in matrix]


# =============================================================================
# PART 2: Analysis Functions
# =============================================================================

def analyze_sparsity(matrix: np.ndarray) -> Dict:
    """
    Analyze sparsity of one-hot encoded matrix.
    
    Returns:
        Dictionary with sparsity metrics
    """
    total_elements = matrix.size
    non_zero = np.count_nonzero(matrix)
    zero = total_elements - non_zero
    
    sparsity = zero / total_elements
    
    return {
        'total_elements': total_elements,
        'non_zero': non_zero,
        'zero': zero,
        'sparsity': sparsity,
        'memory_bytes': matrix.nbytes,
        'memory_mb': matrix.nbytes / (1024 * 1024)
    }


def compare_with_dense(vocab_size: int, embedding_dim: int = 100):
    """
    Compare memory usage of one-hot vs dense embeddings.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of dense embeddings
    """
    # One-hot: each word is a vector of size vocab_size
    one_hot_per_word = vocab_size * 4  # float32 = 4 bytes
    
    # Dense embedding: each word is a vector of size embedding_dim
    dense_per_word = embedding_dim * 4  # float32 = 4 bytes
    
    # Embedding matrix storage
    embedding_matrix = vocab_size * embedding_dim * 4
    
    print(f"\nMemory Comparison (vocab_size={vocab_size}, embedding_dim={embedding_dim}):")
    print("-" * 60)
    print(f"One-hot vector per word: {one_hot_per_word:,} bytes ({one_hot_per_word/1024:.2f} KB)")
    print(f"Dense vector per word:   {dense_per_word:,} bytes ({dense_per_word/1024:.2f} KB)")
    print(f"Ratio: {one_hot_per_word/dense_per_word:.1f}x more memory for one-hot")
    print(f"\nFor 1000 word sequence:")
    print(f"  One-hot total: {1000 * one_hot_per_word / (1024*1024):.2f} MB")
    print(f"  Dense total:   {1000 * dense_per_word / (1024*1024):.2f} MB")


def visualize_one_hot(encoder: OneHotEncoder, words: List[str]):
    """Visualize one-hot vectors as heatmap."""
    # Encode words
    vectors = np.array([encoder.encode_word(word) for word in words])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(encoder.word_to_idx) * 0.3), len(words) * 0.5))
    
    # Plot heatmap
    im = ax.imshow(vectors, cmap='Blues', aspect='auto')
    
    # Labels
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('Vocabulary Index')
    ax.set_ylabel('Words')
    ax.set_title('One-Hot Encoding Visualization')
    
    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    plt.savefig('one_hot_visualization.png', dpi=150)
    plt.show()


# =============================================================================
# PART 3: Limitations Demonstration
# =============================================================================

def demonstrate_limitations(encoder: OneHotEncoder):
    """Demonstrate limitations of one-hot encoding."""
    print("\n" + "=" * 60)
    print("Limitations of One-Hot Encoding")
    print("=" * 60)
    
    # 1. No semantic similarity
    print("\n1. No Semantic Similarity:")
    word_pairs = [('king', 'queen'), ('cat', 'dog'), ('king', 'computer')]
    
    for word1, word2 in word_pairs:
        vec1 = encoder.encode_word(word1)
        vec2 = encoder.encode_word(word2)
        
        # Dot product (similarity)
        similarity = np.dot(vec1, vec2)
        
        print(f"   Similarity({word1}, {word2}) = {similarity}")
    
    print("\n   All different words have 0 similarity, regardless of meaning!")
    
    # 2. Dimensionality problem
    print("\n2. Dimensionality Problem:")
    print(f"   Vocabulary size: {encoder.vocab_size}")
    print(f"   Each word vector has {encoder.vocab_size} dimensions")
    print(f"   For 50,000 word vocab: each word needs 50,000 floats!")
    
    # 3. Memory usage
    print("\n3. Memory Usage:")
    sample_sentence = "the quick brown fox"
    encoded = encoder.encode_sentence(sample_sentence)
    stats = analyze_sparsity(encoded)
    
    print(f"   Sentence: '{sample_sentence}'")
    print(f"   Encoded shape: {encoded.shape}")
    print(f"   Sparsity: {stats['sparsity']:.2%}")
    print(f"   Memory: {stats['memory_bytes']:,} bytes for just 4 words!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02 Solution: One-Hot Encoding")
    print("=" * 60)
    
    # Sample corpus
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the cat sat on the mat",
        "the king and queen ruled the kingdom",
        "machine learning uses computers to learn patterns",
        "natural language processing helps computers understand text"
    ]
    
    # Build encoder
    encoder = OneHotEncoder()
    encoder.build_vocabulary(corpus)
    
    # Test encoding
    print("\n" + "=" * 60)
    print("Part 1: Basic One-Hot Encoding")
    print("=" * 60)
    
    test_word = "fox"
    one_hot = encoder.encode_word(test_word)
    print(f"\nWord: '{test_word}'")
    print(f"One-hot vector shape: {one_hot.shape}")
    print(f"Non-zero index: {np.argmax(one_hot)}")
    print(f"Decoded: '{encoder.decode_vector(one_hot)}'")
    
    # Encode sentence
    print("\n" + "=" * 60)
    print("Part 2: Sentence Encoding")
    print("=" * 60)
    
    test_sentence = "the quick fox"
    encoded_matrix = encoder.encode_sentence(test_sentence)
    print(f"\nSentence: '{test_sentence}'")
    print(f"Encoded matrix shape: {encoded_matrix.shape}")
    print(f"Decoded: {encoder.decode_matrix(encoded_matrix)}")
    
    # Analyze sparsity
    stats = analyze_sparsity(encoded_matrix)
    print(f"\nSparsity Analysis:")
    print(f"  Total elements: {stats['total_elements']:,}")
    print(f"  Non-zero elements: {stats['non_zero']}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    # Compare with dense
    print("\n" + "=" * 60)
    print("Part 3: Memory Comparison")
    print("=" * 60)
    compare_with_dense(vocab_size=10000, embedding_dim=100)
    compare_with_dense(vocab_size=50000, embedding_dim=300)
    
    # Demonstrate limitations
    demonstrate_limitations(encoder)
    
    # Visualize
    print("\n" + "=" * 60)
    print("Part 4: Visualization")
    print("=" * 60)
    visualize_one_hot(encoder, ['the', 'quick', 'fox', 'cat', 'dog'])
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. One-hot encoding is simple but inefficient")
    print("2. Vectors are extremely sparse (mostly zeros)")
    print("3. No semantic information is captured")
    print("4. Dense embeddings solve these problems!")

