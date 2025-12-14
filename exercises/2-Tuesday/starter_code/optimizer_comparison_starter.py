"""
Exercise 02: Optimizer Comparison - Starter Code

Compare SGD, Momentum, RMSprop, and Adam on Fashion-MNIST.

Prerequisites:
- Reading: 02-gradient-descent-intuition.md
- Demo: demo_02_gradient_descent_comparison.py (REFERENCE FOR OPTIMIZER SETUP)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# DATA (PROVIDED)
# ============================================================================

def load_fashion_mnist():
    """Load Fashion-MNIST data"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)


# ============================================================================
# TASK 2.1: Create Optimizer Suite
# ============================================================================

def get_optimizers():
    """
    Create dictionary of optimizers to compare.
    
    REQUIRED OPTIMIZERS:
    - 'SGD': keras.optimizers.SGD(learning_rate=0.01)
    - 'SGD+Momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    - 'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001)
    - 'Adam': keras.optimizers.Adam(learning_rate=0.001)
    
    NOTE: Different LRs because adaptive methods work better with smaller LR
    """
    # YOUR CODE: Return dict of name -> optimizer
    pass


def create_model():
    """Create identical model for fair comparison (PROVIDED)"""
    return keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])


# ============================================================================
# TASK 2.2: Training Loop
# ============================================================================

def train_with_optimizer(optimizer, name, x_train, y_train, x_test, y_test, epochs=30):
    """
    Train model with given optimizer.
    
    IMPORTANT: Create a NEW model for each optimizer!
    (Otherwise you're continuing training, not starting fresh)
    
    PATTERN:
    model = create_model()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0)
    return history
    """
    print(f"Training with {name}...")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 2.3: Full Comparison
# ============================================================================

def run_comparison():
    """
    Run all optimizers and compare results.
    
    STEPS:
    1. Load data
    2. Get optimizer dictionary
    3. Train with each optimizer, store histories
    4. Plot loss curves on same graph
    5. Print final accuracies
    
    SEE: demo_02_gradient_descent_comparison.py for plotting pattern
    """
    print("=" * 60)
    print("Optimizer Comparison")
    print("=" * 60)
    
    # YOUR CODE:
    # 1. Load data
    # 2. Loop through optimizers, train each
    # 3. Plot all loss curves together
    pass


def plot_comparison(histories):
    """
    Plot training curves for all optimizers.
    
    PLOT REQUIREMENTS:
    - X-axis: Epoch
    - Y-axis: Validation Loss
    - One line per optimizer
    - Legend with optimizer names
    - Title: "Optimizer Comparison on Fashion-MNIST"
    
    Save to: optimizer_comparison.png
    """
    plt.figure(figsize=(10, 6))
    
    # YOUR CODE: Plot each history
    # for name, history in histories.items():
    #     plt.plot(history.history['val_loss'], label=name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Optimizer Comparison on Fashion-MNIST')
    plt.legend()
    plt.savefig('optimizer_comparison.png')
    print("Saved: optimizer_comparison.png")


def write_analysis(histories):
    """
    Write analysis.txt answering:
    1. Which optimizer converged fastest (fewest epochs to plateau)?
    2. Which achieved lowest final loss?
    3. Why does Adam often work well "out of the box"?
    4. When might SGD+Momentum beat Adam?
    """
    # YOUR CODE: Calculate metrics and write analysis
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02: Optimizer Comparison")
    print("=" * 60)
    
    # Uncomment to run:
    # run_comparison()
    
    print("\nComplete the functions and analyze the results!")
