# Gradient Descent: Intuition and Optimization

## Learning Objectives
- Understand the geometric intuition behind gradient descent
- Learn the role of learning rate in controlling convergence
- Explore the challenge of local minima and saddle points
- Master optimization variants: SGD, SGD with momentum, and Adam

## Why This Matters

Gradient descent is the fundamental optimization algorithm that powers neural network training. While backpropagation tells us **which direction** to adjust weights, gradient descent determines **how to take steps** in that direction. Understanding gradient descent is crucial because:

- **Training stability**: Choosing the right learning rate and optimizer prevents divergence or slow convergence
- **Time and cost**: Better optimization means faster training, reducing computational expenses
- **Model performance**: Advanced optimizers (Adam, RMSprop) can find better solutions than basic gradient descent
- **Hyperparameter tuning**: Understanding optimization dynamics guides learning rate schedules and batch size choices

This week's focus on neural network optimization reveals that training success depends as much on the optimization algorithm as it does on architecture. Gradient descent is your vehicle for navigating the loss landscape—choosing the right variant determines whether you find the best solution quickly or get stuck along the way.

## The Core Intuition

### The Mountain Analogy

Imagine you're blindfolded on a mountain and need to reach the lowest valley:
- **Gradient**: Steepness and direction of the slope under your feet
- **Gradient descent**: Walk downhill (opposite the gradient)
- **Learning rate**: Size of your steps
- **Loss landscape**: The mountain terrain itself

Your goal: reach the lowest point (minimum loss) by iteratively moving downhill.

### Mathematical Formulation

For a loss function `L(w)` with parameters `w`:

```
w_new = w_old - learning_rate * ∇L(w_old)

Where:
- ∇L(w) is the gradient (direction of steepest ascent)
- We subtract to move in the opposite direction (descent)
- learning_rate controls step size
```

### Simple Example: 1D Optimization

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple quadratic function: L(w) = (w - 3)²
def loss(w):
    return (w - 3) ** 2

def gradient(w):
    return 2 * (w - 3)

# Gradient descent
w = 0.0           # Starting point
learning_rate = 0.1
history = [w]

for step in range(20):
    grad = gradient(w)
    w = w - learning_rate * grad
    history.append(w)
    print(f"Step {step}: w = {w:.4f}, loss = {loss(w):.4f}")

# Converges to w = 3 (minimum of the parabola)
```

## The Learning Rate: Critical Hyperparameter

### Too Small: Slow Convergence

```python
learning_rate = 0.01  # Very small

# Will eventually reach minimum, but takes many iterations
# Analogy: Taking tiny steps down the mountain
```

**Problems:**
- Training takes excessively long
- May get stuck in plateaus
- Wastes computational resources

### Too Large: Divergence

```python
learning_rate = 2.0  # Too large

# Steps overshoot the minimum and bounce back and forth
# May diverge to infinity
# Analogy: Taking giant leaps that overshoot the valley
```

**Problems:**
- Loss oscillates or increases
- Training becomes unstable
- Never converges to minimum

### Just Right: Goldilocks Zone

```python
learning_rate = 0.1  # Appropriate for this problem

# Converges smoothly to minimum in reasonable time
```

**Practical guidance:**
- Start with common defaults: 0.001 (Adam), 0.01 (SGD)
- Use learning rate schedules to decrease over time
- Monitor loss curves and adjust if oscillating or plateauing

## Batch Gradient Descent vs. Variants

### Batch Gradient Descent

Computes gradient using **all training examples** before updating:

```python
# Compute gradient on entire dataset
gradient = compute_gradient(all_training_data)
weights = weights - learning_rate * gradient
```

**Advantages:**
- Stable gradient estimates
- Smooth convergence

**Disadvantages:**
- Very slow for large datasets
- Requires entire dataset in memory
- Updates infrequently

### Stochastic Gradient Descent (SGD)

Updates weights after **each individual example**:

```python
for example in training_data:
    gradient = compute_gradient(example)
    weights = weights - learning_rate * gradient
```

**Advantages:**
- Fast updates
- Can escape local minima (noisier gradients)
- Works with streaming data

**Disadvantages:**
- Very noisy gradients
- Loss curve oscillates significantly
- May not converge precisely to minimum

### Mini-Batch Gradient Descent (Most Common)

Computes gradient using a **small batch** of examples (e.g., 32, 64, 128):

```python
for batch in batches(training_data, batch_size=32):
    gradient = compute_gradient(batch)
    weights = weights - learning_rate * gradient
```

**Advantages:**
- Balances speed and stability
- Leverages GPU/TPU parallelism efficiently
- Generalizes better than full-batch

**Disadvantages:**
- Batch size is another hyperparameter
- Still has some gradient noise

**In Keras:**

```python
model.fit(
    X_train, y_train,
    batch_size=32,  # Mini-batch size
    epochs=10
)
```

## Local Minima and Saddle Points

### The Non-Convex Landscape

Unlike simple quadratic functions, neural network loss landscapes are highly non-convex:

```
     Global Min
         ↓
    Local Min  Saddle Point
        ↓          ↓
  ╱╲  ╱─╲    ╱──╲ ╱──
 ╱  ╲╱   ╲  ╱    ╲    ╲
╱                      ╲
```

**Local Minimum**: A valley that's not the deepest—gradient is zero, but better solutions exist elsewhere.

**Saddle Point**: Gradient is zero, but it's a minimum in some directions and maximum in others (like a mountain pass).

**Global Minimum**: The absolute lowest point—the best solution.

### How Gradient Descent Handles This

**Classic concern**: Gradient descent gets stuck in local minima.

**Modern understanding**: In high dimensions (millions of parameters), local minima are rare. Saddle points are the real challenge.

**Solutions:**
- **Momentum**: Helps escape shallow local minima and navigate saddle points
- **Stochastic noise**: SGD's randomness helps jump out of poor local minima
- **Advanced optimizers**: Adam, RMSprop use adaptive learning rates per parameter

## Momentum: Accelerating Convergence

### The Intuition

Imagine rolling a ball down a hill:
- **Without momentum**: Ball stops at every bump
- **With momentum**: Ball accumulates speed and rolls over small bumps

### Mathematical Formulation

```python
velocity = 0

for step in range(num_steps):
    gradient = compute_gradient(weights)
    
    # Update velocity (exponential moving average of gradients)
    velocity = momentum * velocity - learning_rate * gradient
    
    # Update weights
    weights = weights + velocity
```

**Typical momentum value**: 0.9

### Benefits

1. **Faster convergence**: Accumulates velocity in consistent gradient directions
2. **Smooths oscillations**: Averages out noisy gradients
3. **Escapes shallow minima**: Momentum carries optimization through small bumps

**In Keras:**

```python
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9
)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

## Advanced Optimizers

### RMSprop (Root Mean Square Propagation)

Adapts learning rate for each parameter based on recent gradient magnitudes:

```python
# Maintains moving average of squared gradients
squared_gradients = decay * squared_gradients + (1 - decay) * gradient**2

# Adaptive learning rate per parameter
adaptive_gradient = gradient / sqrt(squared_gradients + epsilon)
weights = weights - learning_rate * adaptive_gradient
```

**Benefits:**
- Automatically adjusts learning rates
- Works well with non-stationary objectives
- Handles sparse gradients effectively

**In Keras:**

```python
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
```

### Adam (Adaptive Moment Estimation)

Combines momentum and RMSprop—the most popular optimizer in deep learning:

```python
# First moment (momentum)
m = beta1 * m + (1 - beta1) * gradient

# Second moment (RMSprop)
v = beta2 * v + (1 - beta2) * gradient**2

# Bias correction
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)

# Update
weights = weights - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

**Default hyperparameters:**
- `learning_rate = 0.001`
- `beta1 = 0.9` (momentum decay)
- `beta2 = 0.999` (RMSprop decay)

**Why Adam is popular:**
- Works well out-of-the-box with default settings
- Adapts learning rates per parameter
- Combines benefits of momentum and RMSprop
- Generally robust across different problems

**In Keras:**

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
```

## Comparing Optimizers: Practical Example

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Build same model with different optimizers
def create_model(optimizer_name):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    optimizers = {
        'sgd': keras.optimizers.SGD(learning_rate=0.01),
        'sgd_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'rmsprop': keras.optimizers.RMSprop(learning_rate=0.001),
        'adam': keras.optimizers.Adam(learning_rate=0.001)
    }
    
    model.compile(
        optimizer=optimizers[optimizer_name],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train and compare
results = {}
for opt_name in ['sgd', 'sgd_momentum', 'rmsprop', 'adam']:
    print(f"\nTraining with {opt_name}...")
    model = create_model(opt_name)
    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=0
    )
    results[opt_name] = history.history

# Typically: Adam > RMSprop > SGD+Momentum > SGD (for convergence speed)
```

## Learning Rate Schedules

Rather than using a fixed learning rate, decrease it over time:

### Step Decay

```python
def step_decay_schedule(epoch):
    initial_lr = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * (drop ** np.floor(epoch / epochs_drop))
    return lr

lr_scheduler = keras.callbacks.LearningRateScheduler(step_decay_schedule)
model.fit(X_train, y_train, epochs=50, callbacks=[lr_scheduler])
```

### Exponential Decay

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.1,
    decay_steps=10000,
    decay_rate=0.96
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
```

### Reduce on Plateau

```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Multiply LR by 0.5
    patience=5,        # Wait 5 epochs before reducing
    min_lr=1e-7
)
model.fit(X_train, y_train, callbacks=[reduce_lr])
```

## Key Takeaways

1. **Gradient descent is the optimization algorithm** that iteratively moves weights toward lower loss
2. **Learning rate is critical**: Too small = slow, too large = unstable
3. **Mini-batch gradient descent** balances computation speed and gradient stability (most common in practice)
4. **Local minima** are less of a problem than previously thought; **saddle points** are the real challenge in high dimensions
5. **Momentum accelerates convergence** by accumulating velocity in consistent gradient directions
6. **Adam optimizer** combines momentum and adaptive learning rates—the default choice for most deep learning tasks
7. **Learning rate schedules** decrease learning rate over time, helping fine-tune convergence
8. **Experiment with optimizers**: Adam is a great default, but SGD with momentum sometimes generalizes better

## External Resources

- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/) - Comprehensive comparison by Sebastian Ruder
- [CS231n: Optimization](http://cs231n.stanford.edu/slides/2023/lecture_6.pdf) - Stanford's visual lecture on optimization
- [Adam: A Method for Stochastic Optimization (Paper)](https://arxiv.org/abs/1412.6980) - Original Adam paper by Kingma and Ba

