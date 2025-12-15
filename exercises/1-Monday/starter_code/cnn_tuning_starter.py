"""
Exercise 03 Starter: CNN Tuning with TensorBoard

Your task: Complete the train_cnn function and run multiple experiments
with different configurations. Use descriptive log names so you can compare
runs in TensorBoard.

Tips:
- Each experiment should have a unique, descriptive log name
- Run: tensorboard --logdir=logs/cnn_tuning to see all runs overlaid
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime
import os

os.makedirs('logs/cnn_tuning', exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and preprocess MNIST for CNN."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape for CNN: (28, 28) -> (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# TODO: Complete this function!
# =============================================================================

def train_cnn(
    filters=[32, 64],
    kernel_size=3,
    pool_size=2,
    dense_units=64,
    experiment_name="baseline"
):
    """
    Train a CNN with given configuration and log to TensorBoard.
    
    Args:
        filters: List of filter counts for each conv block, e.g., [32, 64]
        kernel_size: Kernel size for Conv2D layers (3 or 5)
        pool_size: MaxPool size (2 or 3)
        dense_units: Number of neurons in dense layer
        experiment_name: Descriptive name for TensorBoard logs
    
    Returns:
        history: Training history
    """
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # TODO: Build model
    # Hint: Use layers.Conv2D, layers.MaxPooling2D, layers.Flatten, layers.Dense
    model = keras.Sequential([
        # Conv Block 1
        # YOUR CODE HERE
        
        # Conv Block 2 (if filters has 2 elements)
        # YOUR CODE HERE
        
        # Flatten
        # YOUR CODE HERE
        
        # Dense
        # YOUR CODE HERE
        
        # Output
        # YOUR CODE HERE
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # TODO: Create TensorBoard callback with experiment_name
    # The log directory should be: f"logs/cnn_tuning/{experiment_name}"
    log_dir = None  # YOUR CODE HERE
    tb_callback = None  # YOUR CODE HERE
    
    # Train
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        callbacks=[tb_callback],
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n[{experiment_name}] Test accuracy: {test_acc*100:.2f}%")
    
    return history


# =============================================================================
# EXPERIMENTS - Run these after completing train_cnn
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03: CNN Tuning with TensorBoard")
    print("=" * 60)
    
    # TODO: Run experiments with different configurations
    # Use descriptive experiment names so you can compare in TensorBoard!
    
    # Example experiments to run (uncomment after implementing train_cnn):
    
    # Experiment 1: Baseline
    # train_cnn(filters=[32, 64], kernel_size=3, experiment_name="baseline_32-64_k3")
    
    # Experiment 2: Fewer filters
    # train_cnn(filters=[16, 32], kernel_size=3, experiment_name="filters_16-32_k3")
    
    # Experiment 3: More filters
    # train_cnn(filters=[64, 128], kernel_size=3, experiment_name="filters_64-128_k3")
    
    # Experiment 4: Larger kernel
    # train_cnn(filters=[32, 64], kernel_size=5, experiment_name="kernel_32-64_k5")
    
    # Experiment 5: Different dense size
    # train_cnn(filters=[32, 64], dense_units=128, experiment_name="dense128_32-64_k3")
    
    print("\n" + "=" * 60)
    print("After running experiments, view in TensorBoard:")
    print("  tensorboard --logdir=logs/cnn_tuning")
    print("=" * 60)
