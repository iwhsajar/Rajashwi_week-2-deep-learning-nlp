"""
Exercise 02: Autoencoder Challenge - Complete Solution

Three autoencoder variants for Fashion-MNIST.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('reconstructions', exist_ok=True)

# ============================================================================
# DATA PREPARATION
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
    """Visualize original vs reconstructed images."""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Original', fontsize=12)
        
        ax = plt.subplot(2, n, n + i + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Reconstructed', fontsize=12)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'reconstructions/{filename}', dpi=150)
    plt.close()
    print(f"Saved: reconstructions/{filename}")

# ============================================================================
# TASK 2.1: Simple Autoencoder - SOLUTION
# ============================================================================

def build_simple_autoencoder():
    """Build a simple (shallow) autoencoder: 784 -> 128 -> 32 -> 128 -> 784"""
    input_img = keras.Input(shape=(784,))
    
    # Encoder
    encoded = layers.Dense(128, activation='relu')(input_img)
    encoded = layers.Dense(32, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.Dense(784, activation='sigmoid')(decoded)
    
    # Models
    autoencoder = keras.Model(input_img, decoded, name='simple_autoencoder')
    encoder = keras.Model(input_img, encoded, name='simple_encoder')
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return encoder, autoencoder

def train_simple_autoencoder():
    """Train the simple autoencoder."""
    print("\n" + "=" * 50)
    print("Task 2.1: Simple Autoencoder")
    print("=" * 50)
    
    (_, x_train), (_, x_test) = load_fashion_mnist()
    encoder, autoencoder = build_simple_autoencoder()
    
    print("\nModel Summary:")
    autoencoder.summary()
    
    history = autoencoder.fit(
        x_train, x_train,
        epochs=20,
        batch_size=256,
        validation_data=(x_test, x_test),
        verbose=1
    )
    
    reconstructed = autoencoder.predict(x_test[:10])
    visualize_reconstructions(x_test[:10], reconstructed, 
                             title=f"Simple Autoencoder (Final Loss: {history.history['val_loss'][-1]:.4f})",
                             filename="simple_autoencoder.png")
    
    return history

# ============================================================================
# TASK 2.2: Deep Autoencoder - SOLUTION
# ============================================================================

def build_deep_autoencoder():
    """Build deep autoencoder with BatchNormalization."""
    input_img = keras.Input(shape=(784,))
    
    # Encoder with BatchNorm
    x = layers.Dense(256)(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    encoded = layers.Dense(32, activation='relu')(x)
    
    # Decoder with BatchNorm
    x = layers.Dense(64)(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    decoded = layers.Dense(784, activation='sigmoid')(x)
    
    autoencoder = keras.Model(input_img, decoded, name='deep_autoencoder')
    encoder = keras.Model(input_img, encoded, name='deep_encoder')
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return encoder, autoencoder

def train_deep_autoencoder():
    """Train the deep autoencoder."""
    print("\n" + "=" * 50)
    print("Task 2.2: Deep Autoencoder")
    print("=" * 50)
    
    (_, x_train), (_, x_test) = load_fashion_mnist()
    encoder, autoencoder = build_deep_autoencoder()
    
    print("\nModel Summary:")
    autoencoder.summary()
    
    history = autoencoder.fit(
        x_train, x_train,
        epochs=20,
        batch_size=256,
        validation_data=(x_test, x_test),
        verbose=1
    )
    
    reconstructed = autoencoder.predict(x_test[:10])
    visualize_reconstructions(x_test[:10], reconstructed,
                             title=f"Deep Autoencoder (Final Loss: {history.history['val_loss'][-1]:.4f})",
                             filename="deep_autoencoder.png")
    
    return history

# ============================================================================
# TASK 2.3: Denoising Autoencoder - SOLUTION
# ============================================================================

def add_noise(images, noise_factor=0.3):
    """Add Gaussian noise to images."""
    noisy = images + noise_factor * np.random.normal(size=images.shape)
    return np.clip(noisy, 0.0, 1.0)

def train_denoising_autoencoder():
    """Train the denoising autoencoder."""
    print("\n" + "=" * 50)
    print("Task 2.3: Denoising Autoencoder")
    print("=" * 50)
    
    (_, x_train), (_, x_test) = load_fashion_mnist()
    
    # Create noisy data
    x_train_noisy = add_noise(x_train, 0.3)
    x_test_noisy = add_noise(x_test, 0.3)
    
    # Use deep autoencoder architecture
    _, autoencoder = build_deep_autoencoder()
    autoencoder._name = 'denoising_autoencoder'
    
    history = autoencoder.fit(
        x_train_noisy, x_train,  # Input: noisy, Target: clean
        epochs=20,
        batch_size=256,
        validation_data=(x_test_noisy, x_test),
        verbose=1
    )
    
    # Visualize denoising
    reconstructed = autoencoder.predict(x_test_noisy[:10])
    
    # Three-row visualization
    plt.figure(figsize=(20, 6))
    for i in range(10):
        # Noisy input
        ax = plt.subplot(3, 10, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Noisy', fontsize=12)
        
        # Reconstruction
        ax = plt.subplot(3, 10, 10 + i + 1)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Denoised', fontsize=12)
        
        # Original clean
        ax = plt.subplot(3, 10, 20 + i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.ylabel('Original', fontsize=12)
    
    plt.suptitle(f"Denoising Autoencoder (Final Loss: {history.history['val_loss'][-1]:.4f})", fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstructions/denoising_autoencoder.png', dpi=150)
    plt.close()
    print("Saved: reconstructions/denoising_autoencoder.png")
    
    return history

# ============================================================================
# COMPARISON AND ANALYSIS
# ============================================================================

def compare_autoencoders():
    """Train all three and compare."""
    history_simple = train_simple_autoencoder()
    history_deep = train_deep_autoencoder()
    history_denoising = train_denoising_autoencoder()
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_simple.history['loss'], label='Simple')
    plt.plot(history_deep.history['loss'], label='Deep')
    plt.plot(history_denoising.history['loss'], label='Denoising')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history_simple.history['val_loss'], label='Simple')
    plt.plot(history_deep.history['val_loss'], label='Deep')
    plt.plot(history_denoising.history['val_loss'], label='Denoising')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Autoencoder Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstructions/comparison.png', dpi=150)
    plt.close()
    
    # Write analysis
    analysis = f"""Autoencoder Analysis
{'='*40}

Final Validation Losses:
- Simple Autoencoder: {history_simple.history['val_loss'][-1]:.4f}
- Deep Autoencoder: {history_deep.history['val_loss'][-1]:.4f}
- Denoising Autoencoder: {history_denoising.history['val_loss'][-1]:.4f}

Analysis:
The Deep Autoencoder typically achieves the lowest reconstruction loss due to its
increased capacity and BatchNormalization layers which stabilize training. The
additional layers allow it to learn more complex representations.

The Denoising Autoencoder shows a different loss pattern because it's trained
on a harder task (reconstructing clean images from noisy inputs). Despite higher
loss values, it learns more robust features that are useful for downstream tasks.

The Simple Autoencoder is fastest to train but has limited representational
capacity, resulting in blurrier reconstructions.

Recommendation: Use Deep Autoencoder for best reconstruction quality, or
Denoising Autoencoder when robust features are needed.
"""
    
    with open('analysis.txt', 'w') as f:
        f.write(analysis)
    
    print(analysis)

if __name__ == "__main__":
    compare_autoencoders()

