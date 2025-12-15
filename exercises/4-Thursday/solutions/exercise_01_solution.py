"""
Exercise 01 Solution: RNN Text Generation

Complete implementation of character-level RNN for text generation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import datetime
import os

os.makedirs('logs/exercise_rnn', exist_ok=True)

# =============================================================================
# PART 1: Data Preparation
# =============================================================================

def prepare_text_data(text: str, seq_length: int = 40):
    """
    Prepare text data for character-level RNN training.
    
    Args:
        text: Input text
        seq_length: Length of input sequences
        
    Returns:
        X, y arrays and character mappings
    """
    # Create character mappings
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    vocab_size = len(chars)
    print(f"Text length: {len(text)}")
    print(f"Vocabulary size: {vocab_size}")
    
    # Create sequences
    sequences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_length):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])
    
    print(f"Number of sequences: {len(sequences)}")
    
    # Vectorize
    X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.float32)
    y = np.zeros((len(sequences), vocab_size), dtype=np.float32)
    
    for i, (seq, next_char) in enumerate(zip(sequences, next_chars)):
        for t, char in enumerate(seq):
            X[i, t, char_to_idx[char]] = 1
        y[i, char_to_idx[next_char]] = 1
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y, char_to_idx, idx_to_char, vocab_size


# =============================================================================
# PART 2: RNN Model
# =============================================================================

def create_rnn_model(vocab_size: int, seq_length: int, hidden_units: int = 128):
    """
    Create character-level RNN model.
    
    Args:
        vocab_size: Number of unique characters
        seq_length: Length of input sequences
        hidden_units: Number of RNN units
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # RNN layer
        layers.SimpleRNN(
            hidden_units,
            input_shape=(seq_length, vocab_size),
            return_sequences=False
        ),
        
        # Output layer
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(vocab_size: int, seq_length: int, hidden_units: int = 128):
    """
    Create character-level LSTM model (typically better for text).
    """
    model = keras.Sequential([
        layers.LSTM(
            hidden_units,
            input_shape=(seq_length, vocab_size),
            return_sequences=True
        ),
        layers.Dropout(0.2),
        layers.LSTM(hidden_units),
        layers.Dropout(0.2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# PART 3: Text Generation
# =============================================================================

def sample_with_temperature(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample from probability distribution with temperature.
    
    Args:
        predictions: Probability distribution
        temperature: Controls randomness (higher = more random)
        
    Returns:
        Sampled index
    """
    predictions = np.asarray(predictions).astype('float64')
    
    # Apply temperature
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    
    # Sample from distribution
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)


def generate_text(model, seed_text: str, char_to_idx: dict, idx_to_char: dict,
                  seq_length: int, num_chars: int = 200, temperature: float = 1.0) -> str:
    """
    Generate text using trained model.
    
    Args:
        model: Trained RNN model
        seed_text: Starting text (must be seq_length characters)
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        seq_length: Sequence length model was trained on
        num_chars: Number of characters to generate
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    vocab_size = len(char_to_idx)
    generated = seed_text
    
    for _ in range(num_chars):
        # Prepare input
        x = np.zeros((1, seq_length, vocab_size))
        for t, char in enumerate(generated[-seq_length:]):
            if char in char_to_idx:
                x[0, t, char_to_idx[char]] = 1
        
        # Predict next character
        predictions = model.predict(x, verbose=0)[0]
        
        # Sample with temperature
        next_idx = sample_with_temperature(predictions, temperature)
        next_char = idx_to_char[next_idx]
        
        generated += next_char
    
    return generated


# =============================================================================
# PART 4: Training and Evaluation
# =============================================================================

def train_and_generate():
    """Train model and generate sample text."""
    
    # Sample text (in practice, use a larger corpus)
    text = """
    The quick brown fox jumps over the lazy dog.
    Machine learning is a subset of artificial intelligence.
    Deep learning uses neural networks with many layers.
    Natural language processing enables computers to understand text.
    Recurrent neural networks are designed for sequential data.
    LSTM networks can remember long-term dependencies.
    Text generation is a popular application of language models.
    The neural network learns patterns from the training data.
    Each character is predicted based on the previous characters.
    Temperature controls the randomness of the generated text.
    """ * 20  # Repeat for more training data
    
    # Prepare data
    seq_length = 40
    X, y, char_to_idx, idx_to_char, vocab_size = prepare_text_data(text, seq_length)
    
    # Create model
    print("\n" + "=" * 50)
    print("Creating LSTM Model")
    print("=" * 50)
    
    model = create_lstm_model(vocab_size, seq_length, hidden_units=128)
    model.summary()
    
    # Train
    print("\n" + "=" * 50)
    print("Training")
    print("=" * 50)
    
    # TensorBoard callback
    log_dir = "logs/exercise_rnn/lstm_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        X, y,
        batch_size=128,
        epochs=20,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
            tb_callback
        ]
    )
    
    # Generate with different temperatures
    print("\n" + "=" * 50)
    print("Text Generation")
    print("=" * 50)
    
    seed = text[:seq_length]
    
    for temperature in [0.5, 1.0, 1.5]:
        print(f"\n--- Temperature: {temperature} ---")
        generated = generate_text(
            model, seed, char_to_idx, idx_to_char,
            seq_length, num_chars=200, temperature=temperature
        )
        print(generated)
    
    return model, history


# =============================================================================
# PART 5: Compare RNN vs LSTM
# =============================================================================

def compare_rnn_lstm():
    """Compare RNN and LSTM on text generation task."""
    print("\n" + "=" * 60)
    print("Comparing RNN vs LSTM")
    print("=" * 60)
    
    # Simple text
    text = "abcdefghijklmnopqrstuvwxyz " * 100
    seq_length = 20
    
    X, y, char_to_idx, idx_to_char, vocab_size = prepare_text_data(text, seq_length)
    
    # Train RNN
    print("\nTraining SimpleRNN...")
    rnn_model = create_rnn_model(vocab_size, seq_length, hidden_units=64)
    rnn_history = rnn_model.fit(X, y, epochs=10, batch_size=64, verbose=0)
    
    # Train LSTM
    print("Training LSTM...")
    lstm_model = create_lstm_model(vocab_size, seq_length, hidden_units=64)
    lstm_history = lstm_model.fit(X, y, epochs=10, batch_size=64, verbose=0)
    
    # Compare
    print(f"\nFinal Training Loss:")
    print(f"  RNN:  {rnn_history.history['loss'][-1]:.4f}")
    print(f"  LSTM: {lstm_history.history['loss'][-1]:.4f}")
    
    print(f"\nFinal Training Accuracy:")
    print(f"  RNN:  {rnn_history.history['accuracy'][-1]:.4f}")
    print(f"  LSTM: {lstm_history.history['accuracy'][-1]:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01 Solution: RNN Text Generation")
    print("=" * 60)
    
    # Compare RNN vs LSTM
    compare_rnn_lstm()
    
    # Train and generate
    print("\n" + "=" * 60)
    print("Full Training and Generation")
    print("=" * 60)
    
    model, history = train_and_generate()
    
    # TensorBoard instructions
    print("\n" + "=" * 50)
    print("View Training in TensorBoard")
    print("=" * 50)
    print("Run: tensorboard --logdir=logs/exercise_rnn")
    print("Navigate to http://localhost:6006")
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)

