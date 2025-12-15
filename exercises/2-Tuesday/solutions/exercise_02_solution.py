"""
Exercise 02 Solution: Optimizer Comparison

Complete implementation comparing SGD, Adam, and RMSprop optimizers.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import os

os.makedirs('logs/exercise_optimizer', exist_ok=True)

# =============================================================================
# PART 1: Data Preparation
# =============================================================================

def load_and_prepare_data():
    """Load MNIST and prepare for training."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Flatten and normalize
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Use subset for faster comparison
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    
    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# PART 2: Model Creation
# =============================================================================

def create_model():
    """Create consistent model architecture for fair comparison."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


# =============================================================================
# PART 3: Optimizer Training
# =============================================================================

def train_with_optimizer(optimizer_name, optimizer, x_train, y_train, x_test, y_test,
                         epochs=20, batch_size=128):
    """Train model with specified optimizer and return history."""
    # Create fresh model
    model = create_model()
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nTraining with {optimizer_name}...")
    
    # TensorBoard callback
    log_dir = f"logs/exercise_optimizer/{optimizer_name.replace(' ', '_').replace('+', 'plus')}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=[tb_callback],
        verbose=0
    )
    
    # Final evaluation
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Final test accuracy: {test_acc:.4f}")
    
    return history.history


# =============================================================================
# PART 4: Comparison Analysis
# =============================================================================

def compare_optimizers(results):
    """Create visualizations comparing optimizer performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {
        'SGD': 'blue',
        'SGD + Momentum': 'cyan',
        'RMSprop': 'green',
        'Adam': 'red'
    }
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for name, history in results.items():
        ax1.plot(history['loss'], label=name, color=colors[name])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss
    ax2 = axes[0, 1]
    for name, history in results.items():
        ax2.plot(history['val_loss'], label=name, color=colors[name])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax3 = axes[1, 0]
    for name, history in results.items():
        ax3.plot(history['accuracy'], label=name, color=colors[name])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Training Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation Accuracy
    ax4 = axes[1, 1]
    for name, history in results.items():
        ax4.plot(history['val_accuracy'], label=name, color=colors[name])
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Validation Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150)
    plt.show()
    print("\nSaved comparison plot to 'optimizer_comparison.png'")


def print_summary_table(results):
    """Print summary statistics for each optimizer."""
    print("\n" + "=" * 70)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Optimizer':<20} {'Final Train Acc':<18} {'Final Val Acc':<18} {'Best Val Acc':<15}")
    print("-" * 70)
    
    for name, history in results.items():
        train_acc = history['accuracy'][-1]
        val_acc = history['val_accuracy'][-1]
        best_val_acc = max(history['val_accuracy'])
        print(f"{name:<20} {train_acc:<18.4f} {val_acc:<18.4f} {best_val_acc:<15.4f}")
    
    print("-" * 70)


# =============================================================================
# PART 5: Learning Rate Sensitivity Analysis
# =============================================================================

def learning_rate_sensitivity():
    """Analyze how each optimizer handles different learning rates."""
    print("\n" + "=" * 60)
    print("Learning Rate Sensitivity Analysis")
    print("=" * 60)
    
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()
    
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    results = {}
    
    for lr in learning_rates:
        results[lr] = {}
        
        # SGD
        model = create_model()
        model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        try:
            history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)
            results[lr]['SGD'] = max(history.history['val_accuracy'])
        except:
            results[lr]['SGD'] = 0.0
        
        # Adam
        model = create_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        try:
            history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)
            results[lr]['Adam'] = max(history.history['val_accuracy'])
        except:
            results[lr]['Adam'] = 0.0
        
        # RMSprop
        model = create_model()
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=lr),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        try:
            history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=0)
            results[lr]['RMSprop'] = max(history.history['val_accuracy'])
        except:
            results[lr]['RMSprop'] = 0.0
    
    # Print results
    print(f"\n{'Learning Rate':<15} {'SGD':<12} {'Adam':<12} {'RMSprop':<12}")
    print("-" * 50)
    for lr in learning_rates:
        print(f"{lr:<15} {results[lr]['SGD']:<12.4f} {results[lr]['Adam']:<12.4f} {results[lr]['RMSprop']:<12.4f}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02 Solution: Optimizer Comparison")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()
    print(f"\nData loaded: {x_train.shape[0]} training samples")
    
    # Define optimizers to compare
    optimizers = {
        'SGD': keras.optimizers.SGD(learning_rate=0.01),
        'SGD + Momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
        'Adam': keras.optimizers.Adam(learning_rate=0.001)
    }
    
    # Train with each optimizer
    results = {}
    for name, optimizer in optimizers.items():
        results[name] = train_with_optimizer(
            name, optimizer,
            x_train, y_train, x_test, y_test,
            epochs=20, batch_size=128
        )
    
    # Compare results
    compare_optimizers(results)
    print_summary_table(results)
    
    # Learning rate sensitivity
    lr_results = learning_rate_sensitivity()
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)
    print("\nKey Insights:")
    print("1. Adam typically converges faster than SGD")
    print("2. SGD with momentum closes the gap significantly")
    print("3. Adam is more robust to learning rate choice")
    print("4. RMSprop performs well for this task")

