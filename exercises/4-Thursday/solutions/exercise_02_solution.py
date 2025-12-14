"""
Exercise 02 Solution: Pair Programming - RNN/LSTM Sentiment Classifier

Complete implementation of sentiment classification with RNN and LSTM.
This solution includes both the preprocessing pipeline and model architectures.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =============================================================================
# PART 1: Data Preprocessing (Navigator Role)
# =============================================================================

def load_imdb_data():
    """
    Load IMDB sentiment dataset.
    
    Returns:
        Training and test data with labels
    """
    print("Loading IMDB dataset...")
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(
        num_words=10000  # Only keep top 10,000 words
    )
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Sample sequence: {x_train[0][:20]}...")
    
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, x_test, max_length=200):
    """
    Preprocess sequences with padding.
    
    Args:
        x_train: Training sequences
        x_test: Test sequences
        max_length: Maximum sequence length
        
    Returns:
        Padded sequences
    """
    print(f"\nPadding sequences to length {max_length}...")
    
    # Analyze sequence lengths
    train_lengths = [len(x) for x in x_train]
    print(f"Original length stats: min={min(train_lengths)}, max={max(train_lengths)}, "
          f"mean={np.mean(train_lengths):.0f}")
    
    # Pad sequences
    x_train_padded = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
    x_test_padded = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
    
    print(f"Padded shape: {x_train_padded.shape}")
    
    return x_train_padded, x_test_padded


# =============================================================================
# PART 2: Model Architectures (Driver Role)
# =============================================================================

def create_simple_rnn(vocab_size=10000, embedding_dim=128, max_length=200):
    """
    Create simple RNN for sentiment classification.
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.SimpleRNN(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_lstm_model(vocab_size=10000, embedding_dim=128, max_length=200):
    """
    Create LSTM for sentiment classification.
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        layers.LSTM(32, dropout=0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_bidirectional_lstm(vocab_size=10000, embedding_dim=128, max_length=200):
    """
    Create Bidirectional LSTM for sentiment classification.
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
        layers.Bidirectional(layers.LSTM(32, dropout=0.2)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_gru_model(vocab_size=10000, embedding_dim=128, max_length=200):
    """
    Create GRU for sentiment classification (alternative to LSTM).
    """
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length, mask_zero=True),
        layers.GRU(64, return_sequences=True, dropout=0.2),
        layers.GRU(32, dropout=0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# PART 3: Training and Comparison
# =============================================================================

def train_and_compare():
    """
    Train multiple models and compare performance.
    """
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_imdb_data()
    x_train, x_test = preprocess_data(x_train, x_test, max_length=200)
    
    # Use subset for faster training
    x_train_sub = x_train[:5000]
    y_train_sub = y_train[:5000]
    x_val = x_train[5000:7000]
    y_val = y_train[5000:7000]
    
    # Models to compare
    models = {
        'SimpleRNN': create_simple_rnn(),
        'LSTM': create_lstm_model(),
        'BiLSTM': create_bidirectional_lstm(),
        'GRU': create_gru_model()
    }
    
    histories = {}
    results = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print('='*50)
        
        model.summary()
        
        history = model.fit(
            x_train_sub, y_train_sub,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
            ],
            verbose=1
        )
        
        histories[name] = history.history
        
        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[name] = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'params': model.count_params()
        }
        
        print(f"{name} Test Accuracy: {test_acc:.4f}")
    
    return histories, results


def plot_comparison(histories: dict, results: dict):
    """Plot training comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'SimpleRNN': 'blue', 'LSTM': 'red', 'BiLSTM': 'green', 'GRU': 'orange'}
    
    # Training Loss
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(history['loss'], label=name, color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for name, history in histories.items():
        ax.plot(history['val_loss'], label=name, color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax = axes[1, 0]
    for name, history in histories.items():
        ax.plot(history['val_accuracy'], label=name, color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test Results Bar Chart
    ax = axes[1, 1]
    names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in names]
    bar_colors = [colors[name] for name in names]
    
    bars = ax.bar(names, accuracies, color=bar_colors, alpha=0.7)
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Test Accuracy Comparison')
    ax.set_ylim(0.5, 1.0)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sentiment_model_comparison.png', dpi=150)
    plt.show()


def print_summary(results: dict):
    """Print comparison summary."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Test Accuracy':<15} {'Parameters':<15}")
    print("-" * 45)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['test_accuracy']:<15.4f} {metrics['params']:<15,}")
    
    # Find best model
    best = max(results, key=lambda x: results[x]['test_accuracy'])
    print("-" * 45)
    print(f"Best model: {best} with accuracy {results[best]['test_accuracy']:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02 Solution: RNN/LSTM Sentiment Classification")
    print("=" * 60)
    print("\nThis is a pair programming exercise:")
    print("- Navigator: Handles data preprocessing")
    print("- Driver: Implements model architectures")
    print("- Both: Train and compare models")
    
    # Train and compare models
    histories, results = train_and_compare()
    
    # Plot comparison
    plot_comparison(histories, results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 60)
    print("Pair Programming Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. LSTM typically outperforms SimpleRNN on longer sequences")
    print("2. Bidirectional models can capture context from both directions")
    print("3. GRU is faster than LSTM with similar performance")
    print("4. Dropout is essential for preventing overfitting")

