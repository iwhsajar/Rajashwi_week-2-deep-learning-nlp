"""
Exercise 03: Embedding Exploration - Starter Code

Experiment with word embeddings and analyze semantic relationships.

Prerequisites:
- Reading: 04-introduction-to-embeddings.md
- Reading: 05-word2vec-intuition.md
- Demo: demo_03_word2vec_visualization.py (REFERENCE FOR VISUALIZATION)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# TASK 3.1: Keras Embedding Basics
# ============================================================================

def explore_embedding_layer():
    """
    Understand the Keras Embedding layer.
    
    KEY CONCEPT:
    - Embedding(vocab_size, embed_dim) creates a lookup table
    - Input: integer indices (word IDs)
    - Output: dense vectors of size embed_dim
    
    SHAPE TRANSFORMATION:
    - Input: (batch_size, sequence_length) - integers
    - Output: (batch_size, sequence_length, embed_dim) - floats
    
    PARAMETERS:
    - Total params = vocab_size * embed_dim
    
    TEST THESE EMBEDDING DIMS: 16, 32, 64, 128, 256
    with vocab_size = 1000
    """
    print("=" * 60)
    print("Task 3.1: Embedding Layer Basics")
    print("=" * 60)
    
    vocab_size = 1000
    embedding_dims = [16, 32, 64, 128, 256]
    
    # YOUR CODE:
    # 1. Create Embedding layer for each dim
    # 2. Calculate params (vocab_size * embed_dim)
    # 3. Print table showing dim vs params vs memory
    pass


# ============================================================================
# TASK 3.2: Training Embeddings
# ============================================================================

def train_sentiment_classifier():
    """
    Train embeddings through a simple sentiment task.
    
    SAMPLE DATA (provided):
    - Positive: "love", "great", "amazing", "wonderful"
    - Negative: "terrible", "awful", "hate", "bad"
    
    MODEL ARCHITECTURE:
    1. Embedding(vocab_size, 16)
    2. GlobalAveragePooling1D()  # Average all word vectors
    3. Dense(1, sigmoid)
    
    AFTER TRAINING:
    - Extract embedding weights: model.layers[0].get_weights()[0]
    - Words with similar sentiment should have similar vectors
    
    SEE: demo_03_word2vec_visualization.py for embedding extraction
    """
    print("=" * 60)
    print("Task 3.2: Training Embeddings")
    print("=" * 60)
    
    # Sample data
    texts = [
        "I love this movie it is amazing",
        "Great film excellent acting wonderful",
        "This movie is beautiful and amazing",
        "Terrible movie waste of time awful",
        "Bad film I hated it horrible",
        "Disappointing and boring terrible waste"
    ]
    labels = [1, 1, 1, 0, 0, 0]  # 1=positive, 0=negative
    
    # YOUR CODE:
    # 1. Tokenize texts (use keras.preprocessing.text.Tokenizer)
    # 2. Pad sequences to same length
    # 3. Build model with Embedding layer
    # 4. Train model
    # 5. Extract embedding matrix
    pass


# ============================================================================
# TASK 3.3: Semantic Similarity
# ============================================================================

def analyze_semantic_similarity(embeddings, word_to_idx):
    """
    Analyze semantic relationships in learned embeddings.
    
    COSINE SIMILARITY FORMULA:
    cos_sim(a, b) = (a . b) / (||a|| * ||b||)
    
    TEST THESE PAIRS:
    - ("love", "amazing") - should be similar (both positive)
    - ("terrible", "awful") - should be similar (both negative)
    - ("love", "hate") - should be dissimilar (opposite sentiment)
    
    COMPARE TO ONE-HOT:
    - One-hot similarity was always 0 between different words
    - Embeddings should capture meaning!
    """
    print("=" * 60)
    print("Task 3.3: Semantic Similarity")
    print("=" * 60)
    
    # YOUR CODE:
    # 1. Get embedding vectors for word pairs
    # 2. Compute cosine similarity
    # 3. Print results and analyze
    pass


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    Returns value in [-1, 1], higher = more similar.
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot / (norm_a * norm_b + 1e-8)


# ============================================================================
# TASK 3.4: Visualization
# ============================================================================

def visualize_embeddings(embeddings, idx_to_word):
    """
    Visualize embeddings in 2D using t-SNE or PCA.
    
    STEPS:
    1. Reduce dimensions: sklearn.manifold.TSNE or sklearn.decomposition.PCA
    2. Plot words as scatter points
    3. Color by sentiment (if known)
    4. Label each point with the word
    
    EXPECTED: Positive words cluster together, negative words cluster together
    
    SEE: demo_03_word2vec_visualization.py for complete visualization code
    
    Save to: embedding_visualization.png
    """
    print("=" * 60)
    print("Task 3.4: Visualization")
    print("=" * 60)
    
    # YOUR CODE:
    # 1. Apply PCA or t-SNE to reduce to 2D
    # 2. Plot with plt.scatter
    # 3. Add word labels with plt.annotate
    pass


# ============================================================================
# TASK 3.5: Embedding Dimension Experiment
# ============================================================================

def experiment_embedding_dims():
    """
    Test how embedding dimension affects model performance.
    
    TEST DIMS: 8, 16, 32, 64, 128
    
    HYPOTHESIS: Higher dims = more capacity but more overfitting risk
    
    RECORD:
    - Training accuracy for each dim
    - Validation accuracy for each dim
    - Training time
    """
    print("=" * 60)
    print("Task 3.5: Embedding Dimension Experiment")
    print("=" * 60)
    
    dims = [8, 16, 32, 64, 128]
    
    # YOUR CODE: Train model with each dim, compare results
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03: Embedding Exploration")
    print("=" * 60)
    
    # Uncomment as you complete:
    # explore_embedding_layer()
    # embeddings, word_to_idx, idx_to_word = train_sentiment_classifier()
    # analyze_semantic_similarity(embeddings, word_to_idx)
    # visualize_embeddings(embeddings, idx_to_word)
    # experiment_embedding_dims()
