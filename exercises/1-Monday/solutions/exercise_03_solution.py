"""
Exercise 03 Solution: CNN Tuning with TensorBoard

This solution demonstrates how to:
1. Create a flexible CNN training function
2. Use descriptive TensorBoard log names
3. Compare multiple runs in TensorBoard
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
    
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)


# =============================================================================
# CNN TRAINING FUNCTION
# =============================================================================

def train_cnn(
    filters=[32, 64],
    kernel_size=3,
    pool_size=2,
    dense_units=64,
    experiment_name="baseline"
):
    """Train a CNN with given configuration and log to TensorBoard."""
    
    print(f"\n{'='*60}")
    print(f"Training: {experiment_name}")
    print(f"  Filters: {filters}, Kernel: {kernel_size}x{kernel_size}")
    print(f"  Pool: {pool_size}x{pool_size}, Dense: {dense_units}")
    print('='*60)
    
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Build model dynamically based on filters list
    model_layers = []
    
    for i, f in enumerate(filters):
        if i == 0:
            model_layers.append(
                layers.Conv2D(f, (kernel_size, kernel_size), padding='same',
                             activation='relu', input_shape=(28, 28, 1))
            )
        else:
            model_layers.append(
                layers.Conv2D(f, (kernel_size, kernel_size), padding='same',
                             activation='relu')
            )
        model_layers.append(layers.MaxPooling2D((pool_size, pool_size)))
    
    model_layers.extend([
        layers.Flatten(),
        layers.Dense(dense_units, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model = keras.Sequential(model_layers, name=experiment_name)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Parameters: {model.count_params():,}")
    
    # TensorBoard callback with descriptive name
    log_dir = f"logs/cnn_tuning/{experiment_name}"
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train
    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.2,
        callbacks=[tb_callback],
        verbose=0
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    return history, test_acc


# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03 Solution: CNN Tuning with TensorBoard")
    print("=" * 60)
    
    results = {}
    
    # Experiment 1: Baseline
    _, acc = train_cnn(
        filters=[32, 64],
        kernel_size=3,
        experiment_name="baseline_32-64_k3"
    )
    results['baseline'] = acc
    
    # Experiment 2: Fewer filters
    _, acc = train_cnn(
        filters=[16, 32],
        kernel_size=3,
        experiment_name="filters_16-32_k3"
    )
    results['fewer_filters'] = acc
    
    # Experiment 3: More filters
    _, acc = train_cnn(
        filters=[64, 128],
        kernel_size=3,
        experiment_name="filters_64-128_k3"
    )
    results['more_filters'] = acc
    
    # Experiment 4: Larger kernel
    _, acc = train_cnn(
        filters=[32, 64],
        kernel_size=5,
        experiment_name="kernel_32-64_k5"
    )
    results['larger_kernel'] = acc
    
    # Experiment 5: More dense units
    _, acc = train_cnn(
        filters=[32, 64],
        dense_units=128,
        experiment_name="dense128_32-64_k3"
    )
    results['more_dense'] = acc
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {acc*100:.2f}%")
    
    best = max(results, key=results.get)
    print(f"\nBest configuration: {best} ({results[best]*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("View all runs in TensorBoard:")
    print("  tensorboard --logdir=logs/cnn_tuning")
    print("=" * 60)
