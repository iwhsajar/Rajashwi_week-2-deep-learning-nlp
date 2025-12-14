"""
Exercise 03: Batch Normalization Experiment - Starter Code

Analyze the impact of batch normalization on training dynamics.

Prerequisites:
- Reading: 04-batch-normalization.md
- Demo: demo_03_batch_norm_impact.py (REFERENCE FOR BN PLACEMENT)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# DATA (PROVIDED)
# ============================================================================

def load_mnist():
    """Load subset of MNIST for faster experiments"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return (x_train[:10000], y_train[:10000]), (x_test, y_test)


# ============================================================================
# TASK 3.1: Baseline Model (No Batch Norm)
# ============================================================================

def create_baseline_model():
    """Deep model WITHOUT batch normalization (PROVIDED)"""
    return keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name='baseline_no_bn')


def train_baseline():
    """
    Train baseline and record: training time, final accuracy, loss curve.
    Use epochs=20, batch_size=128, Adam optimizer.
    """
    print("Task 3.1: Baseline (No BatchNorm)")
    
    # YOUR CODE: Load data, create model, train, return history + time
    pass


# ============================================================================
# TASK 3.2: BatchNorm Model
# ============================================================================

def create_batchnorm_model():
    """
    Create deep model WITH batch normalization.
    
    PATTERN (from demo_03_batch_norm_impact.py):
    
    layers.Dense(256),                    # Linear transform
    layers.BatchNormalization(),          # Normalize
    layers.Activation('relu'),            # Then activate
    
    WHY THIS ORDER:
    - BN normalizes the linear output before activation
    - This keeps activations in a good range for ReLU
    - Alternative: BN after activation also works (debated in literature)
    """
    # YOUR CODE: Build model with BN after each Dense layer
    # Architecture: 784 -> 256 -> 128 -> 64 -> 32 -> 10
    # Pattern: Dense -> BatchNorm -> Activation for each hidden layer
    pass


def train_batchnorm():
    """Train BatchNorm model with same settings as baseline."""
    print("Task 3.2: With BatchNorm")
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 3.3: Learning Rate Sensitivity
# ============================================================================

def test_learning_rates():
    """
    Test how each model handles higher learning rates.
    
    TEST THESE LRs: [0.001, 0.01, 0.1]
    
    HYPOTHESIS: BatchNorm models should be more robust to high LR.
    
    RECORD:
    - Final accuracy for each LR
    - Whether training was stable (no NaN loss)
    """
    print("Task 3.3: Learning Rate Sensitivity")
    
    learning_rates = [0.001, 0.01, 0.1]
    
    # YOUR CODE:
    # For each LR:
    #   - Train baseline model
    #   - Train batchnorm model
    #   - Record final val_accuracy for both
    pass


# ============================================================================
# TASK 3.4: Analysis
# ============================================================================

def plot_comparison(history_baseline, history_bn):
    """
    Plot training curves side by side.
    
    SUBPLOT 1: Validation Loss (both models)
    SUBPLOT 2: Validation Accuracy (both models)
    
    Save to: batchnorm_comparison.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # YOUR CODE: Plot val_loss and val_accuracy for both
    
    plt.tight_layout()
    plt.savefig('batchnorm_comparison.png')
    print("Saved: batchnorm_comparison.png")


def write_analysis():
    """
    Write analysis.txt answering:
    
    1. How did BatchNorm affect convergence speed?
    2. What happened with LR=0.1 for each model?
    3. Look at training time - was there overhead from BatchNorm?
    4. When would you recommend using BatchNorm?
    """
    # YOUR CODE: Write your analysis based on experimental results
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03: Batch Normalization Experiment")
    print("=" * 60)
    
    # Uncomment as you complete:
    # h_baseline = train_baseline()
    # h_bn = train_batchnorm()
    # plot_comparison(h_baseline, h_bn)
    # test_learning_rates()
    # write_analysis()
