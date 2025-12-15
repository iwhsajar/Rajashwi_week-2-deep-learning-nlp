"""
Exercise 03 Solution: Embedding Exploration

Complete implementation exploring word embeddings and semantic relationships.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import datetime
import os

os.makedirs('logs/exercise_embeddings', exist_ok=True)

# =============================================================================
# PART 1: Keras Embedding Layer Exploration
# =============================================================================

def explore_embedding_dimensions():
    """
    Experiment with different embedding dimensions.
    """
    print("=" * 60)
    print("Part 1: Embedding Dimension Comparison")
    print("=" * 60)
    
    vocab_size = 10000
    sequence_length = 100
    
    dimensions = [16, 32, 64, 128, 256]
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Sequence length: {sequence_length}")
    print("\n" + "-" * 50)
    print(f"{'Dimension':<12} {'Parameters':<15} {'Output Shape':<20}")
    print("-" * 50)
    
    for dim in dimensions:
        # Create embedding layer
        embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=dim,
            input_length=sequence_length
        )
        
        # Build to get parameter count
        embedding.build((None, sequence_length))
        params = embedding.count_params()
        
        # Output shape
        output_shape = (sequence_length, dim)
        
        print(f"{dim:<12} {params:<15,} {str(output_shape):<20}")
    
    print("-" * 50)
    print("\nNote: Parameters = vocab_size * embedding_dim")


# =============================================================================
# PART 2: Semantic Similarity with Embeddings
# =============================================================================

class SimpleEmbeddingModel:
    """
    Simple model to demonstrate embedding semantics.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Create embedding layer
        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )
        
        # Build the layer
        self.embedding.build((None,))
    
    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get embedding vector for a word index."""
        weights = self.embedding.get_weights()[0]
        return weights[word_idx]
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def find_similar(self, word_idx: int, top_k: int = 5) -> list:
        """Find most similar words to given word."""
        target_vec = self.get_embedding(word_idx)
        weights = self.embedding.get_weights()[0]
        
        similarities = []
        for idx in range(self.vocab_size):
            if idx != word_idx:
                sim = self.cosine_similarity(target_vec, weights[idx])
                similarities.append((idx, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


def demonstrate_semantic_similarity():
    """Demonstrate how trained embeddings capture semantics."""
    print("\n" + "=" * 60)
    print("Part 2: Semantic Similarity")
    print("=" * 60)
    
    # Use pretrained embeddings (simulated for demo)
    # In practice, you'd load GloVe or Word2Vec
    
    # Create a small vocabulary
    words = ['king', 'queen', 'man', 'woman', 'prince', 'princess', 
             'cat', 'dog', 'animal', 'pet',
             'computer', 'keyboard', 'mouse', 'screen']
    
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Create model with random embeddings
    model = SimpleEmbeddingModel(vocab_size=len(words), embedding_dim=50)
    
    print("\nWith RANDOM embeddings (before training):")
    print("-" * 40)
    
    # Check similarity between related words
    pairs = [('king', 'queen'), ('cat', 'dog'), ('king', 'computer')]
    
    for word1, word2 in pairs:
        vec1 = model.get_embedding(word_to_idx[word1])
        vec2 = model.get_embedding(word_to_idx[word2])
        sim = model.cosine_similarity(vec1, vec2)
        print(f"  similarity({word1}, {word2}) = {sim:.4f}")
    
    print("\nNote: With random embeddings, similarities are random!")
    print("After training, related words would have higher similarity.")


# =============================================================================
# PART 3: Vector Arithmetic
# =============================================================================

def demonstrate_vector_arithmetic():
    """
    Demonstrate word embedding vector arithmetic.
    
    Famous example: king - man + woman = queen
    """
    print("\n" + "=" * 60)
    print("Part 3: Vector Arithmetic")
    print("=" * 60)
    
    # Simulated pretrained embeddings that capture relationships
    # In practice, load real pretrained embeddings
    
    np.random.seed(42)
    
    # Create embeddings that have some structure
    # (This is a simplified simulation of what real embeddings look like)
    
    embedding_dim = 50
    
    # Base concepts
    royalty = np.random.randn(embedding_dim) * 0.5
    gender_male = np.random.randn(embedding_dim) * 0.5
    gender_female = np.random.randn(embedding_dim) * 0.5
    
    # Create word vectors as combinations
    embeddings = {
        'king': royalty + gender_male,
        'queen': royalty + gender_female,
        'man': gender_male + np.random.randn(embedding_dim) * 0.1,
        'woman': gender_female + np.random.randn(embedding_dim) * 0.1,
    }
    
    # Normalize
    for word in embeddings:
        embeddings[word] = embeddings[word] / np.linalg.norm(embeddings[word])
    
    # Vector arithmetic: king - man + woman should be close to queen
    result = embeddings['king'] - embeddings['man'] + embeddings['woman']
    result = result / np.linalg.norm(result)
    
    print("\nVector Arithmetic: king - man + woman = ?")
    print("-" * 40)
    
    # Find closest word
    similarities = {}
    for word, vec in embeddings.items():
        sim = np.dot(result, vec)
        similarities[word] = sim
        print(f"  Similarity with '{word}': {sim:.4f}")
    
    closest = max(similarities, key=similarities.get)
    print(f"\nClosest word: '{closest}'")
    
    if closest == 'queen':
        print("The analogy holds: king - man + woman = queen")


# =============================================================================
# PART 4: Embedding Visualization
# =============================================================================

def visualize_embeddings():
    """Visualize embeddings using dimensionality reduction."""
    print("\n" + "=" * 60)
    print("Part 4: Embedding Visualization")
    print("=" * 60)
    
    # Create sample word embeddings
    np.random.seed(42)
    
    words = [
        # Animals
        'cat', 'dog', 'bird', 'fish', 'lion',
        # Countries
        'france', 'germany', 'italy', 'spain', 'japan',
        # Colors
        'red', 'blue', 'green', 'yellow', 'purple'
    ]
    
    categories = {
        'Animals': ['cat', 'dog', 'bird', 'fish', 'lion'],
        'Countries': ['france', 'germany', 'italy', 'spain', 'japan'],
        'Colors': ['red', 'blue', 'green', 'yellow', 'purple']
    }
    
    # Create structured embeddings (simulated)
    embedding_dim = 50
    embeddings = {}
    
    # Add category-specific component
    for category, category_words in categories.items():
        category_vec = np.random.randn(embedding_dim)
        for word in category_words:
            word_specific = np.random.randn(embedding_dim) * 0.3
            embeddings[word] = category_vec + word_specific
    
    # Convert to matrix
    word_list = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[w] for w in word_list])
    
    # Reduce to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embedding_matrix)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Animals': 'red', 'Countries': 'blue', 'Colors': 'green'}
    
    for category, category_words in categories.items():
        indices = [word_list.index(w) for w in category_words]
        x = embeddings_2d[indices, 0]
        y = embeddings_2d[indices, 1]
        
        ax.scatter(x, y, c=colors[category], label=category, s=100, alpha=0.7)
        
        for i, word in zip(indices, category_words):
            ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                       fontsize=10, ha='center', va='bottom')
    
    ax.legend()
    ax.set_title('Word Embeddings Visualization (t-SNE)')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('embedding_visualization.png', dpi=150)
    plt.show()
    print("\nSaved visualization to 'embedding_visualization.png'")
    print("Note: Words in the same category cluster together!")


# =============================================================================
# PART 5: Training Embeddings
# =============================================================================

def train_embeddings_for_classification():
    """
    Train embeddings as part of a classification model.
    """
    print("\n" + "=" * 60)
    print("Part 5: Training Embeddings")
    print("=" * 60)
    
    # Create synthetic data
    vocab_size = 1000
    max_length = 50
    num_samples = 1000
    
    # Random sequences (in practice, these would be real tokenized text)
    np.random.seed(42)
    X = np.random.randint(1, vocab_size, size=(num_samples, max_length))
    y = np.random.randint(0, 2, size=(num_samples,))
    
    # Split
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # Create model with embedding layer
    embedding_dims = [16, 64, 128]
    results = {}
    
    for dim in embedding_dims:
        print(f"\nTraining with embedding_dim={dim}...")
        
        model = keras.Sequential([
            layers.Embedding(vocab_size, dim, input_length=max_length),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        log_dir = f"logs/exercise_embeddings/dim_{dim}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
        
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[tb_callback],
            verbose=0
        )
        
        val_acc = history.history['val_accuracy'][-1]
        results[dim] = val_acc
        print(f"  Validation accuracy: {val_acc:.4f}")
    
    print("\nResults Summary:")
    print("-" * 30)
    for dim, acc in results.items():
        print(f"  dim={dim}: acc={acc:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03 Solution: Embedding Exploration")
    print("=" * 60)
    
    # Part 1: Dimension comparison
    explore_embedding_dimensions()
    
    # Part 2: Semantic similarity
    demonstrate_semantic_similarity()
    
    # Part 3: Vector arithmetic
    demonstrate_vector_arithmetic()
    
    # Part 4: Visualization
    visualize_embeddings()
    
    # Part 5: Training
    train_embeddings_for_classification()
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Embedding dimension is a tradeoff: more dims = more capacity but more params")
    print("2. Trained embeddings capture semantic relationships")
    print("3. Vector arithmetic reveals learned relationships (king - man + woman = queen)")
    print("4. t-SNE/PCA visualization shows semantic clustering")

