"""
Exercise 03: MLP Architecture Tuning - Starter Code

Systematically optimize MLP architecture for MNIST.

Prerequisites:
- Reading: 03-mlp-digit-classifier.md, 05-training-vs-validation-data.md
- Demo: demo_03_mnist_mlp_classifier.py (REFERENCE FOR MODEL PATTERNS)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# DATA PREPARATION (PROVIDED - DO NOT MODIFY)
# ============================================================================

def load_mnist():
    """Load and preprocess MNIST with train/val/test split"""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_full = x_train_full.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Split: first 10k for validation
    x_train, y_train = x_train_full[10000:], y_train_full[10000:]
    x_val, y_val = x_train_full[:10000], y_train_full[:10000]
    
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ============================================================================
# BASELINE MODEL (PROVIDED - USE AS REFERENCE)
# ============================================================================

def create_baseline_model():
    """Baseline: 2 hidden layers, 128 neurons, ReLU"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# TASK 3.1: Baseline Evaluation
# ============================================================================

def evaluate_baseline():
    """
    Train baseline model and record metrics.
    
    RECORD THESE METRICS:
    - Training time (use time.time() before/after)
    - Final train accuracy
    - Final val accuracy
    - Gap = train_acc - val_acc (overfitting indicator)
    
    SEE: demo_03_mnist_mlp_classifier.py for training pattern
    """
    print("\n" + "=" * 50)
    print("Task 3.1: Baseline Evaluation")
    print("=" * 50)
    
    # YOUR CODE: Load data, train baseline for 10 epochs, record metrics
    pass


# ============================================================================
# TASK 3.2: Depth Experiment
# ============================================================================

def create_model_with_depth(num_layers, neurons_per_layer):
    """
    Create MLP with configurable depth.
    
    HINTS:
    - Use keras.Sequential()
    - First layer needs input_shape=(784,)
    - Add num_layers-1 more hidden layers in a loop
    - Final layer: Dense(10, 'softmax')
    """
    # YOUR CODE HERE
    pass


def experiment_depth():
    """
    Test depths while keeping total neurons similar.
    
    TEST THESE CONFIGURATIONS:
    | Layers | Neurons/Layer | Total Hidden Neurons |
    |--------|---------------|---------------------|
    | 1      | 256           | 256                 |
    | 2      | 128           | 256                 |
    | 3      | 85            | 255                 |
    | 4      | 64            | 256                 |
    
    QUESTION TO ANSWER: Does depth help when total capacity is similar?
    """
    print("\n" + "=" * 50)
    print("Task 3.2: Depth Experiment")
    print("=" * 50)
    
    configs = [(1, 256), (2, 128), (3, 85), (4, 64)]
    
    # YOUR CODE: Train each config, compare val_accuracy
    pass


# ============================================================================
# TASK 3.3: Width Experiment
# ============================================================================

def experiment_width():
    """
    Test widths with fixed depth (2 layers).
    
    TEST THESE WIDTHS: 32, 64, 128, 256, 512
    
    QUESTIONS TO ANSWER:
    1. How does accuracy scale with width?
    2. At what point do diminishing returns begin?
    3. How does parameter count affect training time?
    """
    print("\n" + "=" * 50)
    print("Task 3.3: Width Experiment")
    print("=" * 50)
    
    widths = [32, 64, 128, 256, 512]
    
    # YOUR CODE: Use create_model_with_depth(2, width) for each
    pass


# ============================================================================
# TASK 3.4: Activation Comparison
# ============================================================================

def create_model_with_activation(activation_name):
    """
    Create MLP with specified activation.
    
    SUPPORT THESE: 'relu', 'leaky_relu', 'elu', 'tanh'
    
    HINT FOR LEAKY RELU:
    - It's a layer, not a string activation
    - Use: layers.LeakyReLU(alpha=0.1) after Dense layer
    - Pattern: Dense(128) -> LeakyReLU() -> Dense(128) -> LeakyReLU() -> Dense(10)
    """
    # YOUR CODE HERE
    pass


def experiment_activations():
    """
    Compare activation functions.
    
    TEST: relu, leaky_relu, elu, tanh
    
    EXPECTED INSIGHTS:
    - ReLU: Fast, but can have "dead neurons"
    - LeakyReLU: Prevents dead neurons
    - ELU: Smooth, can speed up learning
    - tanh: Centered at 0, but vanishing gradients in deep nets
    """
    print("\n" + "=" * 50)
    print("Task 3.4: Activation Comparison")
    print("=" * 50)
    
    activations = ['relu', 'leaky_relu', 'elu', 'tanh']
    
    # YOUR CODE: Train each, compare results
    pass


# ============================================================================
# RESULTS SUMMARY
# ============================================================================

def generate_results_summary():
    """Generate results.md template - fill in after experiments"""
    template = """# MLP Architecture Tuning Results

## Depth Experiment
| Layers | Neurons | Val Accuracy |
|--------|---------|--------------|
| 1      | 256     | ???          |
| 2      | 128     | ???          |
| 3      | 85      | ???          |
| 4      | 64      | ???          |

## Width Experiment  
| Width | Params | Val Accuracy |
|-------|--------|--------------|
| 32    | ???    | ???          |
| 64    | ???    | ???          |
| 128   | ???    | ???          |
| 256   | ???    | ???          |
| 512   | ???    | ???          |

## Activation Experiment
| Activation | Val Accuracy |
|------------|--------------|
| ReLU       | ???          |
| LeakyReLU  | ???          |
| ELU        | ???          |
| tanh       | ???          |

## Final Recommendation
- Architecture: ??? layers x ??? neurons
- Activation: ???
- Justification: (explain based on your results)
"""
    with open('results.md', 'w') as f:
        f.write(template)
    print("Template saved to results.md - fill in your results!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03: MLP Architecture Tuning")
    print("=" * 60)
    
    # Uncomment as you complete:
    # evaluate_baseline()
    # experiment_depth()
    # experiment_width()
    # experiment_activations()
    
    generate_results_summary()
