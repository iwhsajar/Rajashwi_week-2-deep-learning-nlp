"""
Exercise 02: Autoencoder Challenge - Starter Code

Build three autoencoder variants for Fashion-MNIST.

Prerequisites:
- Reading: 02-autoencoders-concepts.md
- Demo: demo_02_autoencoder_build.py (SEE THIS FOR ARCHITECTURE EXAMPLES)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('reconstructions', exist_ok=True)

# ============================================================================
# DATA PREPARATION (PROVIDED - DO NOT MODIFY)
# ============================================================================

def load_fashion_mnist():
    """Load and preprocess Fashion-MNIST data"""
    (x_train, _), (x_test, _) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)
    return (x_train, x_train_flat), (x_test, x_test_flat)

def visualize_reconstructions(original, reconstructed, n=10, title="Reconstructions", filename="reconstruction.png"):
    """Visualize original vs reconstructed images (PROVIDED)"""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'reconstructions/{filename}', dpi=150)
    plt.close()
    print(f"Saved: reconstructions/{filename}")


# ============================================================================
# TASK 2.1: Simple Autoencoder
# ============================================================================

def build_simple_autoencoder():
    """
    Build a simple (shallow) autoencoder.
    
    Target Architecture: 784 -> 128 -> 32 -> 128 -> 784
    
    HINTS:
    - Use keras.Input(shape=(784,)) for input layer
    - Encoder: two Dense layers with 'relu' activation
    - Decoder: Dense(128, relu) then Dense(784, sigmoid)
    - sigmoid on output keeps values in [0, 1] like our normalized pixels
    - Compile with optimizer='adam', loss='mse'
    
    SEE: demo_02_autoencoder_build.py lines 45-70 for similar pattern
    
    Returns:
        tuple: (encoder, decoder, autoencoder) models
    """
    # YOUR CODE HERE
    # Step 1: Define input layer
    # Step 2: Build encoder (compress 784 -> 32)
    # Step 3: Build decoder (expand 32 -> 784)
    # Step 4: Create Model and compile
    pass


def train_simple_autoencoder():
    """
    Train the simple autoencoder and visualize results.
    
    HINTS:
    - autoencoder.fit(x_train, x_train, ...) - input = target for reconstruction
    - Use epochs=20, batch_size=256
    - validation_data=(x_test, x_test)
    
    SEE: demo_02_autoencoder_build.py lines 85-95 for training pattern
    """
    print("\n" + "=" * 50)
    print("Task 2.1: Simple Autoencoder")
    print("=" * 50)
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 2.2: Deep Autoencoder
# ============================================================================

def build_deep_autoencoder():
    """
    Build a deeper autoencoder with BatchNormalization.
    
    Target Architecture: 784 -> 256 -> 128 -> 64 -> 32 -> 64 -> 128 -> 256 -> 784
    
    HINTS:
    - Add BatchNormalization() after each Dense layer (before activation)
    - Pattern: Dense(n) -> BatchNormalization() -> Activation('relu')
    - BatchNorm helps with training stability in deeper networks
    
    SEE: demo_03_batch_norm_impact.py for BatchNorm usage patterns
    
    Returns:
        tuple: (encoder, decoder, autoencoder) models
    """
    # YOUR CODE HERE
    pass


def train_deep_autoencoder():
    """Train the deep autoencoder and compare with simple version."""
    print("\n" + "=" * 50)
    print("Task 2.2: Deep Autoencoder")
    print("=" * 50)
    
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 2.3: Denoising Autoencoder
# ============================================================================

def add_noise(images, noise_factor=0.3):
    """
    Add Gaussian noise to images.
    
    HINTS:
    - Use np.random.normal(size=images.shape) to generate noise
    - Multiply noise by noise_factor and add to images
    - Use np.clip(noisy, 0.0, 1.0) to keep valid pixel range
    
    Formula: noisy = clip(images + noise_factor * random_noise, 0, 1)
    """
    # YOUR CODE HERE
    pass


def build_denoising_autoencoder():
    """
    Build a denoising autoencoder.
    
    HINT: Use same architecture as deep autoencoder - the "denoising" 
    comes from HOW we train it, not the architecture itself.
    """
    # YOUR CODE HERE - reuse deep autoencoder architecture
    pass


def train_denoising_autoencoder():
    """
    Train the denoising autoencoder.
    
    KEY INSIGHT: Training differs from regular autoencoder!
    - Input: NOISY images
    - Target: CLEAN images
    - The network learns to remove noise
    
    SEE: demo_02_autoencoder_build.py lines 180-220 for denoising example
    """
    print("\n" + "=" * 50)
    print("Task 2.3: Denoising Autoencoder")
    print("=" * 50)
    
    # YOUR CODE HERE
    # 1. Load data
    # 2. Create noisy versions: add_noise(x_train), add_noise(x_test)
    # 3. Train: fit(noisy_input, clean_target)
    pass


# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

def compare_autoencoders():
    """
    Train all three and compare performance.
    
    DELIVERABLE: Create analysis.txt with your observations on:
    1. Which achieved lowest reconstruction error?
    2. How did training curves differ?
    3. What's the trade-off between depth and training time?
    """
    print("\n" + "=" * 60)
    print("Autoencoder Comparison")
    print("=" * 60)
    
    # YOUR CODE HERE
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02: Autoencoder Challenge")
    print("=" * 60)
    
    # Uncomment as you complete each task:
    # train_simple_autoencoder()
    # train_deep_autoencoder()
    # train_denoising_autoencoder()
    # compare_autoencoders()
    
    print("\nCheck 'reconstructions/' folder for visualizations.")
