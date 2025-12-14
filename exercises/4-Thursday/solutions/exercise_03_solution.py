"""
Exercise 03 Solution: Sequence Processing

Complete implementation of handling variable-length sequences with padding and masking.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# PART 1: Understanding Sequence Lengths
# =============================================================================

def analyze_sequence_lengths(sequences):
    """
    Analyze the distribution of sequence lengths.
    
    Args:
        sequences: List of sequences
    """
    lengths = [len(seq) for seq in sequences]
    
    print("Sequence Length Analysis:")
    print("-" * 40)
    print(f"Number of sequences: {len(sequences)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.1f}")
    print(f"Median length: {np.median(lengths):.1f}")
    print(f"Std length: {np.std(lengths):.1f}")
    
    # Percentiles
    for p in [50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(lengths, p):.0f}")
    
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean: {np.mean(lengths):.0f}')
    plt.axvline(np.median(lengths), color='green', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Distribution of Sequence Lengths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sequence_length_distribution.png', dpi=150)
    plt.show()
    
    return lengths


# =============================================================================
# PART 2: Padding Strategies
# =============================================================================

def demonstrate_padding():
    """
    Demonstrate different padding strategies.
    """
    print("\n" + "=" * 50)
    print("Padding Strategies")
    print("=" * 50)
    
    # Sample sequences of different lengths
    sequences = [
        [1, 2, 3],
        [4, 5, 6, 7, 8],
        [9, 10]
    ]
    
    print("\nOriginal sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i}: {seq} (length {len(seq)})")
    
    # Post-padding (add zeros at the end)
    post_padded = pad_sequences(sequences, maxlen=5, padding='post', truncating='post')
    print("\nPost-padding (maxlen=5):")
    for i, seq in enumerate(post_padded):
        print(f"  Sequence {i}: {list(seq)}")
    
    # Pre-padding (add zeros at the beginning)
    pre_padded = pad_sequences(sequences, maxlen=5, padding='pre', truncating='pre')
    print("\nPre-padding (maxlen=5):")
    for i, seq in enumerate(pre_padded):
        print(f"  Sequence {i}: {list(seq)}")
    
    # Truncation
    long_seq = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    
    truncated_post = pad_sequences(long_seq, maxlen=5, truncating='post')
    print("\nTruncating from end (keeps beginning):")
    print(f"  Original: {long_seq[0]}")
    print(f"  Truncated: {list(truncated_post[0])}")
    
    truncated_pre = pad_sequences(long_seq, maxlen=5, truncating='pre')
    print("\nTruncating from beginning (keeps end):")
    print(f"  Original: {long_seq[0]}")
    print(f"  Truncated: {list(truncated_pre[0])}")
    
    return post_padded, pre_padded


# =============================================================================
# PART 3: Masking
# =============================================================================

def demonstrate_masking():
    """
    Demonstrate how masking works with padded sequences.
    """
    print("\n" + "=" * 50)
    print("Masking with Padded Sequences")
    print("=" * 50)
    
    # Create sample data
    sequences = [
        [1, 2, 3, 4, 5],
        [6, 7, 8],
        [9, 10]
    ]
    
    # Pad sequences
    padded = pad_sequences(sequences, maxlen=5, padding='post')
    print("\nPadded sequences:")
    for i, seq in enumerate(padded):
        print(f"  {seq}")
    
    # Create model with masking
    vocab_size = 20
    embedding_dim = 8
    
    # Model WITHOUT masking
    model_no_mask = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=5),
        layers.LSTM(16),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Model WITH masking
    model_with_mask = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=5, mask_zero=True),
        layers.LSTM(16),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Check if masking is supported
    print("\nMasking support check:")
    print(f"  LSTM supports masking: {model_with_mask.layers[1].supports_masking}")
    
    # Create mask manually
    mask = padded != 0
    print("\nComputed mask (True = real token, False = padding):")
    for i, m in enumerate(mask):
        print(f"  Sequence {i}: {m}")
    
    return padded, mask


def compare_with_without_masking():
    """
    Compare model performance with and without masking.
    """
    print("\n" + "=" * 50)
    print("Comparing Models With/Without Masking")
    print("=" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Create sequences of varying lengths
    num_samples = 1000
    max_len = 50
    vocab_size = 100
    
    sequences = []
    labels = []
    
    for _ in range(num_samples):
        # Random length between 10 and 50
        length = np.random.randint(10, max_len + 1)
        # Random sequence
        seq = np.random.randint(1, vocab_size, size=length).tolist()
        sequences.append(seq)
        # Label based on sequence sum (arbitrary rule)
        labels.append(1 if sum(seq) > length * vocab_size / 2 else 0)
    
    labels = np.array(labels)
    
    # Pad sequences
    X = pad_sequences(sequences, maxlen=max_len, padding='post')
    
    # Split data
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = labels[:800], labels[800:]
    
    # Model without masking
    print("\nTraining without masking...")
    model_no_mask = keras.Sequential([
        layers.Embedding(vocab_size, 32, input_length=max_len),
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ])
    model_no_mask.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history_no_mask = model_no_mask.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Model with masking
    print("Training with masking...")
    model_mask = keras.Sequential([
        layers.Embedding(vocab_size, 32, input_length=max_len, mask_zero=True),
        layers.LSTM(32),
        layers.Dense(1, activation='sigmoid')
    ])
    model_mask.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    history_mask = model_mask.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # Compare results
    print("\nResults:")
    print(f"  Without masking - Val Accuracy: {history_no_mask.history['val_accuracy'][-1]:.4f}")
    print(f"  With masking    - Val Accuracy: {history_mask.history['val_accuracy'][-1]:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history_no_mask.history['val_loss'], label='No Masking')
    axes[0].plot(history_mask.history['val_loss'], label='With Masking')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history_no_mask.history['val_accuracy'], label='No Masking')
    axes[1].plot(history_mask.history['val_accuracy'], label='With Masking')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('masking_comparison.png', dpi=150)
    plt.show()
    
    return history_no_mask, history_mask


# =============================================================================
# PART 4: Custom Masking Layer
# =============================================================================

class CustomMasking(layers.Layer):
    """
    Custom masking layer that creates mask based on padding value.
    """
    
    def __init__(self, mask_value=0, **kwargs):
        super().__init__(**kwargs)
        self.mask_value = mask_value
        self.supports_masking = True
    
    def compute_mask(self, inputs, mask=None):
        """Compute mask: True where input != mask_value."""
        return keras.backend.not_equal(inputs, self.mask_value)
    
    def call(self, inputs):
        """Pass through input unchanged."""
        return inputs


def test_custom_masking():
    """Test custom masking layer."""
    print("\n" + "=" * 50)
    print("Custom Masking Layer")
    print("=" * 50)
    
    # Test data
    sequences = np.array([
        [1, 2, 3, 0, 0],
        [4, 5, 0, 0, 0],
        [6, 7, 8, 9, 10]
    ])
    
    # Apply custom masking
    custom_mask = CustomMasking(mask_value=0)
    mask = custom_mask.compute_mask(sequences)
    
    print("\nInput sequences:")
    print(sequences)
    print("\nComputed mask:")
    print(mask.numpy())


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03 Solution: Sequence Processing")
    print("=" * 60)
    
    # Generate sample sequences
    np.random.seed(42)
    sample_sequences = [
        np.random.randint(1, 100, size=np.random.randint(10, 200)).tolist()
        for _ in range(1000)
    ]
    
    # Part 1: Analyze lengths
    print("\n" + "=" * 60)
    print("Part 1: Analyzing Sequence Lengths")
    print("=" * 60)
    lengths = analyze_sequence_lengths(sample_sequences)
    
    # Part 2: Padding strategies
    demonstrate_padding()
    
    # Part 3: Masking
    demonstrate_masking()
    
    # Part 4: Comparison
    compare_with_without_masking()
    
    # Part 5: Custom masking
    test_custom_masking()
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Analyze sequence lengths to choose appropriate maxlen")
    print("2. Post-padding is common for RNNs (keeps recent context at end)")
    print("3. Masking prevents padded values from affecting model")
    print("4. Use mask_zero=True in Embedding layer for automatic masking")
    print("5. Masking can improve model accuracy and training efficiency")

