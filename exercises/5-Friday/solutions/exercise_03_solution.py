"""
Exercise 03 Solution: Combat Overfitting

Complete implementation of regularization techniques to combat overfitting.
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from tensorflow import keras
from tensorflow.keras import layers, regularizers

os.makedirs('logs/exercise_overfitting', exist_ok=True)

# =============================================================================
# PART 1: Create Overfitting Scenario
# =============================================================================

def create_overfitting_scenario():
    """
    Create a scenario that guarantees overfitting:
    - Small dataset
    - Large model
    """
    print("=" * 50)
    print("Creating Overfitting Scenario")
    print("=" * 50)
    
    # Load MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Use only 500 training samples (very small)
    x_train_small = x_train[:500].reshape(-1, 784).astype('float32') / 255.0
    y_train_small = y_train[:500]
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    print(f"Training samples: {len(x_train_small)} (intentionally small)")
    print(f"Test samples: {len(x_test)}")
    
    return (x_train_small, y_train_small), (x_test, y_test)


def create_overfit_model():
    """Create overly complex model (too many parameters for small data)."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model parameters: {model.count_params():,}")
    return model


# =============================================================================
# PART 2: Regularization Techniques
# =============================================================================

def create_dropout_model(dropout_rate=0.4):
    """Model with dropout regularization."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_l2_model(l2_lambda=0.001):
    """Model with L2 regularization."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,),
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_combined_model(dropout_rate=0.3, l2_lambda=0.001):
    """Model with both dropout and L2 regularization."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,),
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# PART 3: Training and Comparison
# =============================================================================

def train_model(model, train_data, test_data, epochs=100, name="Model"):
    """Train model and return history."""
    print(f"\nTraining {name}...")
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    log_dir = f"logs/exercise_overfitting/{name.lower()}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
    
    history = model.fit(
        train_data[0], train_data[1],
        epochs=epochs,
        batch_size=64,
        validation_data=test_data,
        callbacks=[early_stop, tb_callback],
        verbose=0
    )
    
    # Final evaluation
    train_loss, train_acc = model.evaluate(train_data[0], train_data[1], verbose=0)
    val_loss, val_acc = model.evaluate(test_data[0], test_data[1], verbose=0)
    
    print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
    print(f"  Gap:   {train_acc - val_acc:.4f}")
    print(f"  Epochs: {len(history.history['loss'])}")
    
    return history


def compare_all_techniques():
    """Compare all regularization techniques."""
    train_data, test_data = create_overfitting_scenario()
    
    results = {}
    
    # 1. Baseline (no regularization)
    print("\n" + "=" * 50)
    print("1. Baseline (No Regularization)")
    print("=" * 50)
    model = create_overfit_model()
    results['Baseline'] = train_model(model, train_data, test_data, name="Baseline")
    
    # 2. Dropout
    print("\n" + "=" * 50)
    print("2. Dropout Regularization")
    print("=" * 50)
    model = create_dropout_model(dropout_rate=0.4)
    results['Dropout'] = train_model(model, train_data, test_data, name="Dropout")
    
    # 3. L2 Regularization
    print("\n" + "=" * 50)
    print("3. L2 Regularization")
    print("=" * 50)
    model = create_l2_model(l2_lambda=0.001)
    results['L2'] = train_model(model, train_data, test_data, name="L2")
    
    # 4. Combined
    print("\n" + "=" * 50)
    print("4. Combined (Dropout + L2)")
    print("=" * 50)
    model = create_combined_model(dropout_rate=0.3, l2_lambda=0.001)
    results['Combined'] = train_model(model, train_data, test_data, name="Combined")
    
    return results


def plot_comparison(results):
    """Plot comparison of different techniques."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'Baseline': 'red', 'Dropout': 'blue', 'L2': 'green', 'Combined': 'purple'}
    
    # Training Loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history.history['loss'], label=name, color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history.history['val_loss'], label=name, color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Training Accuracy
    ax = axes[1, 0]
    for name, history in results.items():
        ax.plot(history.history['accuracy'], label=name, color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Accuracy
    ax = axes[1, 1]
    for name, history in results.items():
        ax.plot(history.history['val_accuracy'], label=name, color=colors[name])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regularization_comparison.png', dpi=150)
    plt.show()


def print_summary(results):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("REGULARIZATION COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Technique':<15} {'Final Train Acc':<18} {'Final Val Acc':<18} {'Gap':<10} {'Epochs':<10}")
    print("-" * 70)
    
    for name, history in results.items():
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        gap = train_acc - val_acc
        epochs = len(history.history['loss'])
        print(f"{name:<15} {train_acc:<18.4f} {val_acc:<18.4f} {gap:<10.4f} {epochs:<10}")
    
    # Find best
    best = max(results, key=lambda x: results[x].history['val_accuracy'][-1])
    print("-" * 70)
    print(f"Best technique: {best}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03 Solution: Combat Overfitting")
    print("=" * 60)
    
    # Compare all techniques
    results = compare_all_techniques()
    
    # Plot comparison
    plot_comparison(results)
    
    # Print summary
    print_summary(results)
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. Baseline model severely overfits (train acc >> val acc)")
    print("2. Dropout forces redundant feature learning")
    print("3. L2 regularization penalizes large weights")
    print("4. Combined approach often works best")
    print("5. Early stopping prevents over-training")

