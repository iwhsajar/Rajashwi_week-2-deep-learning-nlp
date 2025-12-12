# Batch Normalization

## Learning Objectives
- Understand the problem of internal covariate shift in deep networks
- Learn how batch normalization normalizes activations during training
- Master the trainable parameters (gamma and beta) in batch normalization layers
- Recognize the benefits: faster training, higher learning rates, and regularization

## Why This Matters

Batch normalization is one of the most impactful innovations in deep learning, enabling the training of much deeper networks. Understanding it is critical because:

- **Training stability**: Batch norm prevents exploding/vanishing activations that destabilize training
- **Speed**: Networks with batch norm train 2-10x faster
- **Performance**: Often improves final model accuracy
- **Architecture flexibility**: Enables successful training of very deep networks (50+ layers)
- **Industry standard**: Used in most production deep learning systems

As you master neural network optimization this week, batch normalization reveals how normalizing internal activations addresses one of deep learning's fundamental challenges. This technique is ubiquitous in modern architectures—understanding it is essential for building effective models.

## The Problem: Internal Covariate Shift

### What is Covariate Shift?

**Covariate shift** occurs when the distribution of inputs changes between training and deployment:
- Train on data from 2020
- Deploy in 2025 with different data distribution
- Model performance degrades

### Internal Covariate Shift

Within a neural network, **each layer's input distribution changes** during training:

```python
# Iteration 1
Layer 1 weights → produces activations with mean=0.5, std=1.0
Layer 2 receives inputs with mean=0.5, std=1.0

# Iteration 2 (after weight update)
Layer 1 weights → produces activations with mean=2.3, std=3.5
Layer 2 receives inputs with mean=2.3, std=3.5
# Layer 2 must constantly adapt to changing input distributions!
```

**The challenge**: Each layer must learn its task while its inputs keep shifting. This is like trying to hit a moving target while the ground underneath you is also moving.

### Consequences

**1. Slow Training**
- Each layer must continuously adapt to input distribution changes
- Lower layers' updates ripple through upper layers
- Convergence takes many more iterations

**2. Vanishing/Exploding Activations**

```python
# Without normalization, activations can grow unbounded
# Example: 10 layers with ReLU activation

x = input_data
for layer in layers:
    x = relu(layer(x))
    # If layer outputs have mean > 1, activations explode
    # If layer outputs have mean < 1, activations vanish
```

**3. Sensitivity to Initialization**
- Poor weight initialization can cause training to fail
- Must use careful initialization schemes (Xavier, He)

**4. Lower Learning Rates Required**
- Large learning rates cause instability
- Forced to use small learning rates, slowing training

## The Solution: Batch Normalization

### Core Idea

Normalize layer inputs to have **consistent mean and variance** across the mini-batch:

```
For each feature in a layer's output:
1. Compute mean and variance across the mini-batch
2. Normalize to mean=0, variance=1
3. Apply learned scale (gamma) and shift (beta)
```

This stabilizes the distribution of inputs to subsequent layers.

### The Algorithm

For a mini-batch of activations `{x₁, x₂, ..., xₘ}`:

**Step 1: Compute batch statistics**

```
μ_batch = (1/m) * Σ xᵢ              # Batch mean
σ²_batch = (1/m) * Σ (xᵢ - μ_batch)²  # Batch variance
```

**Step 2: Normalize**

```
x̂ᵢ = (xᵢ - μ_batch) / √(σ²_batch + ε)
```

Where `ε` (epsilon) is a small constant (e.g., 1e-5) to prevent division by zero.

**Step 3: Scale and shift (learnable parameters)**

```
yᵢ = γ * x̂ᵢ + β
```

Where:
- `γ` (gamma): Learnable scale parameter
- `β` (beta): Learnable shift parameter

**Why learnable parameters?** Without them, normalization could harm the network's expressive power. Gamma and beta allow the network to undo normalization if needed (e.g., if the optimal activation distribution isn't mean=0, variance=1).

### Implementation in Keras

```python
from tensorflow import keras
from tensorflow.keras import layers

# Model with batch normalization
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.BatchNormalization(),  # Add after dense layer
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Common placement:**
- **After linear transformation, before activation**: `Dense → BatchNorm → Activation`
- **Or after activation**: `Dense → Activation → BatchNorm` (less common)

### Training vs. Inference Modes

**During Training:**
- Use batch statistics (mean and variance computed from current mini-batch)
- Update running averages of mean and variance

**During Inference:**
- Use running averages (population statistics) instead of batch statistics
- This ensures consistent behavior regardless of batch size

```python
# Keras handles this automatically
model.fit(X_train, y_train)  # Training mode: batch statistics

predictions = model.predict(X_test)  # Inference mode: population statistics
```

## Benefits of Batch Normalization

### 1. Faster Training (2-10x Speedup)

```python
import tensorflow as tf
from tensorflow import keras
import time

# Model WITHOUT batch normalization
model_no_bn = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Model WITH batch normalization
model_with_bn = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

# Compare training time to reach 95% accuracy
# model_with_bn typically reaches target in fewer epochs
```

### 2. Higher Learning Rates

Without batch norm:

```python
# Must use small learning rate to avoid instability
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
```

With batch norm:

```python
# Can use 10-100x higher learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.01)
```

### 3. Less Sensitivity to Initialization

Batch norm reduces dependence on careful weight initialization:

```python
# Without batch norm: Must use Xavier/He initialization carefully
# With batch norm: More forgiving to initialization choices
```

### 4. Regularization Effect

Batch norm adds noise (because it uses batch statistics, which vary):
- Acts as a mild regularizer
- Can reduce need for dropout (though combining both is still common)

### 5. Enables Deeper Networks

Before batch norm: Training 20+ layer networks was challenging

After batch norm: ResNet, DenseNet, and other very deep architectures became feasible

## Practical Example: Comparing with and without Batch Norm

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Model WITHOUT batch normalization
model_no_bn = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model_no_bn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model WITH batch normalization
model_with_bn = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation='softmax')
])

model_with_bn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train both models
print("Training without Batch Norm...")
history_no_bn = model_no_bn.fit(
    x_train, y_train,
    epochs=20,
    validation_split=0.2,
    verbose=0
)

print("Training with Batch Norm...")
history_with_bn = model_with_bn.fit(
    x_train, y_train,
    epochs=20,
    validation_split=0.2,
    verbose=0
)

# Plot comparison
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_no_bn.history['loss'], label='Without BN')
plt.plot(history_with_bn.history['loss'], label='With BN')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Training Loss Comparison')

plt.subplot(1, 2, 2)
plt.plot(history_no_bn.history['val_accuracy'], label='Without BN')
plt.plot(history_with_bn.history['val_accuracy'], label='With BN')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy Comparison')

plt.show()

# Typically: With BN converges faster and reaches higher accuracy
```

## Understanding Trainable Parameters

### Gamma and Beta

```python
# Create a batch normalization layer
bn_layer = keras.layers.BatchNormalization()

# After building the model, inspect parameters
model.summary()

# BatchNormalization layer shows trainable parameters:
# - gamma (scale): One parameter per feature
# - beta (shift): One parameter per feature
# Plus non-trainable parameters:
# - moving_mean: Running average of batch means
# - moving_variance: Running average of batch variances
```

**Example output:**

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)               (None, 128)               100480    
batch_normalization (BN)    (None, 128)               512       
  # 128 gammas + 128 betas + 128 moving_mean + 128 moving_variance
  # Trainable: 256 (gamma + beta)
  # Non-trainable: 256 (moving stats)
```

### Why Separate Scale and Shift?

Allows the network to represent any mean and variance:

```
# If optimal activation has mean=5, std=2:
# Network learns: gamma ≈ 2, beta ≈ 5

# If normalization hurts (rare), network can undo it:
# gamma ≈ batch_std, beta ≈ batch_mean
# Result: y ≈ x (identity transformation)
```

## Best Practices

### 1. Placement in Architecture

```python
# Common pattern: Dense → BatchNorm → Activation
model = keras.Sequential([
    keras.layers.Dense(128, use_bias=False),  # Bias redundant with BN
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.Dense(64, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    
    keras.layers.Dense(10, activation='softmax')
])
```

**Note**: When using batch norm, the bias in the Dense layer is redundant (beta serves the same purpose), so you can set `use_bias=False`.

### 2. Batch Size Considerations

Batch normalization works best with moderate to large batch sizes (32+):
- Very small batches: Batch statistics are noisy
- Batch size of 1: Batch norm fails (undefined variance)

For small batch sizes, consider **Group Normalization** or **Layer Normalization** instead.

### 3. Combining with Dropout

```python
# Batch norm + dropout can both be used
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),  # Dropout after batch norm
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(10, activation='softmax')
])
```

## Key Takeaways

1. **Internal covariate shift** occurs when layer input distributions change during training, slowing convergence
2. **Batch normalization normalizes activations** to have consistent mean and variance across mini-batches
3. **The algorithm**: Compute batch statistics → normalize → scale (gamma) and shift (beta)
4. **Trainable parameters** (gamma and beta) allow the network to learn optimal activation distributions
5. **Benefits**: 2-10x faster training, higher learning rates, less initialization sensitivity, regularization effect
6. **Training vs. inference**: Use batch statistics during training, population statistics during inference
7. **Batch norm enabled very deep networks** (50-1000+ layers) like ResNet and DenseNet
8. **Best practices**: Place after dense layers, use moderate batch sizes (32+), can combine with dropout

## External Resources

- [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167) - Original paper by Ioffe and Szegedy
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) - Research on why batch norm works
- [Group Normalization](https://arxiv.org/abs/1803.08494) - Alternative for small batch sizes

