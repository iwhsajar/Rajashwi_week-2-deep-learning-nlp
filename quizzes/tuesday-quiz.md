# Weekly Knowledge Check: Backpropagation, Gradient Descent, and Batch Normalization (Tuesday)

## Part 1: Multiple Choice

### 1. What mathematical concept is fundamental to backpropagation?

- [ ] A) The chain rule from calculus
- [ ] B) Matrix inversion
- [ ] C) Eigenvalue decomposition
- [ ] D) Fourier transform

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) The chain rule from calculus

**Explanation:** Backpropagation applies the chain rule to compute gradients of the loss with respect to each weight. For nested functions `f(g(x))`, the derivative is `df/dg * dg/dx`. Neural networks are deeply nested functions, and the chain rule lets us propagate gradients backward layer by layer.

- **Why others are wrong:**
  - B) Matrix inversion is used in linear algebra, not core to backprop
  - C) Eigenvalue decomposition is for matrix analysis
  - D) Fourier transform is for signal processing
</details>

---

### 2. Which problem occurs when gradients become extremely small in deep networks?

- [ ] A) Exploding gradients
- [ ] B) Dead neurons
- [ ] C) Vanishing gradients
- [ ] D) Overfitting

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Vanishing gradients

**Explanation:** Vanishing gradients occur when gradients multiply through many layers and shrink to near-zero values. This prevents early layers from learning because their weight updates become negligible. Sigmoid/tanh activations exacerbate this problem.

- **Why others are wrong:**
  - A) Exploding gradients are when values become extremely large
  - C) Dead neurons are ReLU-specific (always output zero)
  - D) Overfitting is about memorizing training data, unrelated to gradient magnitude
</details>

---

### 3. What problem does batch normalization primarily solve?

- [ ] A) Internal covariate shift
- [ ] B) Underfitting
- [ ] C) Overfitting
- [ ] D) Class imbalance

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Internal covariate shift

**Explanation:** Internal covariate shift occurs when each layer's input distribution changes during training because earlier layers' weights are being updated. Batch normalization normalizes layer inputs to have consistent mean and variance, stabilizing training.

- **Why others are wrong:**
  - B) Underfitting requires more model capacity or longer training
  - C) Overfitting requires regularization (dropout, etc.)
  - D) Class imbalance needs sampling strategies or weighted loss
</details>

---

### 4. What are gamma and beta in batch normalization?

- [ ] A) Fixed hyperparameters set before training
- [ ] B) Running averages of mean and variance
- [ ] C) The learning rate and momentum
- [ ] D) Learnable scale and shift parameters

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Learnable scale and shift parameters

**Explanation:** After normalizing to mean=0 and variance=1, batch norm applies `y = gamma * x_normalized + beta`. Gamma (scale) and beta (shift) are trainable parameters that allow the network to learn the optimal output distribution.

- **Why others are wrong:**
  - A) They are learned during training, not fixed
  - B) Running averages are separate non-trainable parameters
  - C) Learning rate and momentum are optimizer hyperparameters
</details>

---

### 5. With batch normalization, you can typically use learning rates that are:

- [ ] A) 10-100x smaller
- [ ] B) About the same
- [ ] C) Always exactly 0.001
- [ ] D) 10-100x larger

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) 10-100x larger

**Explanation:** Batch normalization stabilizes training by normalizing activations, making the network less sensitive to the scale of weights. This allows much higher learning rates without causing instability, significantly speeding up training.

- **Why others are wrong:**
  - A) Smaller learning rates would slow training needlessly
  - B) Same rate wastes the stability benefit
  - D) Learning rate depends on the problem, not a fixed value
</details>

---

### 6. Which optimizer combines momentum and adaptive learning rates per parameter?

- [ ] A) SGD
- [ ] B) Adam
- [ ] C) RMSprop
- [ ] D) Adagrad

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Adam

**Explanation:** Adam (Adaptive Moment Estimation) combines the benefits of momentum (first moment) and RMSprop (second moment), adapting learning rates per parameter. It's typically the recommended default optimizer.

- **Why others are wrong:**
  - A) SGD only uses fixed learning rate (optionally with momentum)
  - C) RMSprop has adaptive rates but no momentum
  - D) Adagrad has adaptive rates but they monotonically decrease
</details>

---

### 7. What does a negative gradient indicate about the direction to update weights?

- [ ] A) Decrease the weight to reduce loss
- [ ] B) Increase the weight to reduce loss
- [ ] C) Keep the weight unchanged
- [ ] D) Reset the weight to zero

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Increase the weight to reduce loss

**Explanation:** Gradient descent updates: `w = w - learning_rate * gradient`. If gradient is negative, subtracting a negative number means ADDING to the weight. A negative gradient means the loss decreases as the weight increases.

- **Why others are wrong:**
  - A) That would increase the loss
  - C) Zero gradient means no change, not negative
  - D) Resetting is not a gradient-based operation
</details>

---

### 8. In the weight update formula `w_new = w_old - learning_rate * gradient`, what happens if the learning rate is too high?

- [ ] A) Training becomes more stable
- [ ] B) Model trains slower but more accurately
- [ ] C) Gradients vanish faster
- [ ] D) Loss may oscillate or diverge

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Loss may oscillate or diverge

**Explanation:** If the learning rate is too high, weight updates overshoot the minimum. This causes the loss to oscillate wildly or even increase (diverge). The model "bounces around" instead of converging.

- **Why others are wrong:**
  - A) High learning rate causes instability, not stability
  - B) Slow training results from LOW learning rate
  - C) Vanishing gradients are related to activation functions, not learning rate
</details>

---

### 9. What is the typical batch size range for mini-batch gradient descent?

- [ ] A) 32-256
- [ ] B) 1 (single sample)
- [ ] C) 10,000+
- [ ] D) Equal to entire dataset

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) 32-256

**Explanation:** Mini-batch sizes of 32, 64, 128, or 256 are most common. They balance gradient stability (larger = more stable) with computational efficiency and GPU memory constraints.

- **Why others are wrong:**
  - A) Batch size 1 is stochastic gradient descent (very noisy)
  - C) 10,000+ is impractical for most GPUs
  - D) Full batch is slow and doesn't generalize as well
</details>

---

### 10. During inference (prediction), batch normalization uses:

- [ ] A) Batch statistics from the current mini-batch
- [ ] B) Random statistics
- [ ] C) Population (running) statistics computed during training
- [ ] D) No statistics (batch norm is disabled)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Population (running) statistics computed during training

**Explanation:** During inference, batch statistics would be noisy (especially for small batches). Batch normalization uses running averages of mean and variance computed during training, called population statistics, for consistent behavior.

- **Why others are wrong:**
  - A) Batch statistics are only used during training
  - C) Random statistics would produce inconsistent predictions
  - D) Batch norm is still active, just with different statistics
</details>

---

### 11. What is the default learning rate for the Adam optimizer in Keras?

- [ ] A) 1.0
- [ ] B) 0.1
- [ ] C) 0.0001
- [ ] D) 0.001

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) 0.001

**Explanation:** Adam's default learning rate is 0.001 in most frameworks including Keras/TensorFlow. This works well for many problems out-of-the-box, making Adam a popular "set and forget" optimizer.

- **Why others are wrong:**
  - A) 1.0 is far too high for most problems
  - C) 0.1 is typical for SGD, not Adam
  - D) 0.0001 is lower than the default
</details>

---

### 12. What is a saddle point in the context of neural network optimization?

- [ ] A) The global minimum of the loss function
- [ ] B) A point where gradient is zero but it's a minimum in some directions and maximum in others
- [ ] C) A local minimum
- [ ] D) The starting point of training

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) A point where gradient is zero but it's a minimum in some directions and maximum in others

**Explanation:** A saddle point is like a mountain pass - going one direction you're at a minimum (valley), going perpendicular you're at a maximum (ridge). Gradient is zero, but it's not a minimum. In high dimensions, saddle points are more common than local minima.

- **Why others are wrong:**
  - A) Global minimum is the absolute lowest point
  - C) Local minimum is a minimum in all directions
  - D) Starting point typically has non-zero gradient
</details>

---

### 13. Why is ReLU activation preferred over sigmoid for hidden layers in deep networks?

- [ ] A) ReLU has a constant gradient of 1 for positive inputs, preventing vanishing gradients
- [ ] B) ReLU outputs values between 0 and 1
- [ ] C) ReLU is slower to compute
- [ ] D) ReLU always produces non-zero outputs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) ReLU has a constant gradient of 1 for positive inputs, preventing vanishing gradients

**Explanation:** ReLU activation (`max(0, x)`) has a gradient of 1 for positive inputs and 0 for negative inputs. Unlike sigmoid (max gradient 0.25), ReLU doesn't shrink the gradient, preventing vanishing gradient problems in deep networks.

- **Why others are wrong:**
  - A) Sigmoid outputs 0-1; ReLU outputs 0 to infinity
  - C) ReLU is actually faster (simple max operation)
  - D) ReLU outputs 0 for negative inputs (that's the "dying ReLU" problem)
</details>

---

### 14. What does momentum help with in gradient descent optimization?

- [ ] A) Slowing down convergence
- [ ] B) Accelerating convergence and escaping shallow local minima
- [ ] C) Increasing the learning rate automatically
- [ ] D) Reducing the number of parameters

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Accelerating convergence and escaping shallow local minima

**Explanation:** Momentum accumulates velocity in directions where gradients are consistent, helping roll over small bumps (shallow local minima) and accelerating convergence in low-curvature directions. A typical momentum value is 0.9.

- **Why others are wrong:**
  - A) Momentum speeds up, not slows down
  - B) Learning rate is separate; momentum modifies gradient direction
  - D) Momentum doesn't change model architecture
</details>

---

### 15. When using batch normalization, what should you set `use_bias` to in the preceding Dense layer?

- [ ] A) True (always)
- [ ] B) A random value
- [ ] C) False (redundant with batch norm's beta)
- [ ] D) Equal to the learning rate

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) False (redundant with batch norm's beta)

**Explanation:** The beta parameter in batch normalization serves the same purpose as the bias in the Dense layer - it shifts the output. Since batch normalization already has a trainable shift (beta), the Dense layer's bias is redundant.

- **Why others are wrong:**
  - A) Using bias wastes parameters (works but inefficient)
  - C) use_bias is a boolean, not a random value
  - D) use_bias is unrelated to learning rate
</details>

---

## Part 2: True/False

### 16. The forward pass and backward pass happen simultaneously during training.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Forward and backward passes happen sequentially: (1) Forward pass computes predictions and caches intermediate values, (2) Backward pass uses cached values to compute gradients. The backward pass requires forward pass results.
</details>

---

### 17. ReLU activation helps prevent vanishing gradients because its gradient is always 1 for positive inputs.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** ReLU activation (`max(0, x)`) has a gradient of 1 for positive inputs and 0 for negative inputs. Unlike sigmoid (max gradient 0.25), ReLU doesn't shrink the gradient, preventing vanishing gradient problems.
</details>

---

### 18. Batch normalization can only be used with fully connected (Dense) layers.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Batch normalization works with many layer types. For CNNs, there's BatchNormalization for Conv2D layers. For RNNs, Layer Normalization is often preferred, but batch norm principles apply broadly across architectures.
</details>

---

### 19. Gradient clipping is a technique to prevent exploding gradients.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** Gradient clipping limits the magnitude of gradients during training. If gradients exceed a threshold, they are scaled down. This prevents exploding gradients from destabilizing training.
</details>

---

### 20. Adam optimizer always produces better results than SGD with momentum.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** While Adam often converges faster, SGD with momentum sometimes achieves better final generalization, especially in computer vision tasks. The "best" optimizer depends on the specific problem, and experimentation is recommended.
</details>

---

## Part 3: Code Prediction

### 21. What does this code output?

```python
import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2

gradient = tape.gradient(y, x)
print(gradient.numpy())
```

- [ ] A) 3.0
- [ ] B) 9.0
- [ ] C) 6.0
- [ ] D) 2.0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 6.0

**Explanation:** The gradient of `y = x^2` with respect to `x` is `dy/dx = 2x`. When `x = 3`, the gradient is `2 * 3 = 6`. TensorFlow's `GradientTape` automatically computes this derivative.

- **Why others are wrong:**
  - A) 3.0 is the input value, not the gradient
  - B) 9.0 is x^2, the output value
  - D) 2.0 would be the gradient at x=1
</details>

---

### 22. After one gradient descent step with learning_rate=0.1 and gradient=-4, what is the new weight?

```python
w = 5.0
gradient = -4.0
learning_rate = 0.1
w = w - learning_rate * gradient
print(w)
```

- [ ] A) 4.6
- [ ] B) 5.4
- [ ] C) 5.0
- [ ] D) 0.9

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 5.4

**Explanation:** `w = 5.0 - 0.1 * (-4.0) = 5.0 - (-0.4) = 5.0 + 0.4 = 5.4`. Subtracting a negative gradient means adding to the weight.
</details>

---

### 23. How many trainable parameters does this BatchNormalization layer have if applied to a 128-dimensional input?

```python
keras.layers.BatchNormalization()
# Applied to output shape (None, 128)
```

- [ ] A) 128
- [ ] B) 256
- [ ] C) 512
- [ ] D) 0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 256

**Explanation:** BatchNormalization has trainable parameters: gamma (scale) and beta (shift), one per feature. For 128 features: 128 gammas + 128 betas = 256 trainable parameters. (Plus 256 non-trainable for moving mean/variance.)
</details>

---

### 24. What will happen during training with this configuration?

```python
optimizer = keras.optimizers.SGD(learning_rate=2.0)
model.compile(optimizer=optimizer, loss='mse')
```

- [ ] A) Training will converge quickly
- [ ] B) Training will likely diverge (loss increases or oscillates)
- [ ] C) Training will be very slow
- [ ] D) No effect on training

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Training will likely diverge (loss increases or oscillates)

**Explanation:** A learning rate of 2.0 is extremely high. Weight updates will overshoot the minimum, causing the loss to oscillate wildly or diverge to infinity. Typical SGD learning rates are 0.01-0.1.
</details>

---

### 25. What is the output of this gradient computation?

```python
import tensorflow as tf

x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x ** 3  # y = x^3

gradient = tape.gradient(y, x)
print(gradient.numpy())
```

- [ ] A) 8.0
- [ ] B) 6.0
- [ ] C) 12.0
- [ ] D) 3.0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 12.0

**Explanation:** The derivative of `y = x^3` is `dy/dx = 3x^2`. When `x = 2`, the gradient is `3 * 2^2 = 3 * 4 = 12`.
</details>

---

### 26. What does this code accomplish?

```python
keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5
)
```

- [ ] A) Increases learning rate when validation loss improves
- [ ] B) Reduces learning rate by half if validation loss doesn't improve for 5 epochs
- [ ] C) Stops training after 5 epochs
- [ ] D) Reduces batch size by half

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Reduces learning rate by half if validation loss doesn't improve for 5 epochs

**Explanation:** `ReduceLROnPlateau` monitors `val_loss`, waits `patience=5` epochs without improvement, then multiplies the learning rate by `factor=0.5` (halves it). This helps fine-tune convergence when training plateaus.
</details>

---

## Part 4: Fill-in-the-Blank

### 27. The backpropagation algorithm computes gradients by applying the _______ rule from calculus.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** chain

**Explanation:** The chain rule allows computing derivatives of nested functions: d[f(g(x))]/dx = df/dg * dg/dx. Neural networks are deeply nested functions, and the chain rule enables gradient computation through all layers.
</details>

---

### 28. To solve vanishing gradients, the most common solution is to use _______ activation instead of sigmoid.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** ReLU (or "relu", "Rectified Linear Unit")

**Explanation:** ReLU activation has a gradient of 1 for positive inputs, preventing gradient shrinkage. Sigmoid and tanh have maximum gradients of 0.25 and 1 respectively, which multiply to very small values over many layers.
</details>

---

### 29. The batch normalization formula normalizes activations to have mean _______ and variance _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** 0 (zero) and 1 (one)

**Explanation:** Batch normalization computes `x_normalized = (x - mean) / sqrt(variance + epsilon)`, resulting in normalized values with approximately mean=0 and variance=1 across each batch.
</details>

---

### 30. Exploding gradients can be controlled using a technique called gradient _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** clipping

**Explanation:** Gradient clipping limits the magnitude of gradients during training. If gradients exceed a threshold, they are scaled down. This prevents exploding gradients from destabilizing training.
</details>

---

### 31. The GradientTape context manager in TensorFlow is used for _______ differentiation.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** automatic

**Explanation:** TensorFlow's GradientTape records operations for automatic differentiation. It traces computations during the forward pass, then uses this record to compute gradients in the backward pass.
</details>

---

### 32. In mini-batch gradient descent, the _______ is the number of training examples used to compute each gradient update.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** batch size

**Explanation:** Batch size determines how many samples are used to compute the average gradient before updating weights. Common values are 32, 64, 128, or 256.
</details>

---

## Part 5: Scenario-Based Questions

### 33. Your model's training loss keeps decreasing, but validation loss starts increasing after epoch 10. What's happening and what should you do?

- [ ] A) Underfitting - add more layers
- [ ] B) Overfitting - add regularization (dropout, early stopping)
- [ ] C) Learning rate too high - increase it
- [ ] D) Vanishing gradients - switch to sigmoid activation

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Overfitting - add regularization (dropout, early stopping)

**Explanation:** Diverging training and validation loss is the classic signature of overfitting. The model is memorizing training data rather than learning generalizable patterns. Solutions include dropout, early stopping, or data augmentation.

- **Why others are wrong:**
  - A) Underfitting shows high loss on BOTH sets
  - C) High learning rate causes oscillation, not this pattern
  - D) Vanishing gradients prevent learning entirely
</details>

---

### 34. Your training loss oscillates wildly and sometimes increases. What should you adjust first?

- [ ] A) Add more training data
- [ ] B) Decrease the learning rate
- [ ] C) Add more layers to the model
- [ ] D) Increase batch size to 1

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Decrease the learning rate

**Explanation:** Oscillating or increasing loss typically indicates the learning rate is too high. Weight updates overshoot the minimum, causing the optimization to bounce around or diverge. Reducing the learning rate should stabilize training.

- **Why others are wrong:**
  - A) More data doesn't fix optimization instability
  - C) More layers could worsen the problem
  - D) Batch size 1 is very noisy, not the solution
</details>

---

### 35. Your deep network (20 layers) trains very slowly and early layer weights barely change. What's the likely cause?

- [ ] A) Learning rate too high
- [ ] B) Batch size too large
- [ ] C) Vanishing gradients
- [ ] D) Too many features

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Vanishing gradients

**Explanation:** In deep networks, gradients can shrink exponentially as they propagate backward. Early layers receive tiny gradients, so their weights barely update. Solutions include ReLU activation, batch normalization, or residual connections.

- **Why others are wrong:**
  - A) High learning rate would cause instability, not slow updates
  - B) Batch size affects noise, not gradient magnitude
  - D) Feature count doesn't cause gradient issues
</details>

---

### 36. You're comparing Adam and SGD for training. Adam converges faster but SGD achieves slightly better final accuracy. Which should you use for production?

- [ ] A) Always Adam (faster is better)
- [ ] B) Always SGD (higher accuracy is better)
- [ ] C) Depends on priorities: Adam for faster iteration, SGD for best generalization
- [ ] D) Neither - use Adagrad instead

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Depends on priorities: Adam for faster iteration, SGD for best generalization

**Explanation:** There's no universally "best" optimizer. Adam is great for rapid prototyping and most problems. SGD with momentum can achieve better generalization in some cases (especially computer vision). Choose based on your specific requirements.

- **Why others are wrong:**
  - A/B) Absolutist answers ignore problem-specific tradeoffs
  - D) Adagrad is rarely the best choice for deep learning
</details>

---

### 37. Adding batch normalization to your model doubled the number of parameters. Is this expected?

- [ ] A) Yes, this is normal behavior
- [ ] B) No, batch norm adds very few parameters (gamma + beta per feature)
- [ ] C) Yes, because batch norm duplicates all weights
- [ ] D) No, batch norm has no learnable parameters

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) No, batch norm adds very few parameters (gamma + beta per feature)

**Explanation:** BatchNormalization adds only 2 parameters per feature (gamma and beta), which is minimal compared to Dense layer weights. If parameters doubled, something else changed in the architecture.

- **Why others are wrong:**
  - A) Doubling parameters is not normal for batch norm
  - C) Batch norm doesn't duplicate weights
  - D) Batch norm has trainable gamma and beta
</details>

---

## Bonus Questions

### 38. What is the "dying ReLU" problem?

- [ ] A) ReLU neurons explode to infinity
- [ ] B) ReLU neurons output zero for all inputs, gradients become zero, learning stops
- [ ] C) ReLU causes training to take too long
- [ ] D) ReLU only works with certain optimizers

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) ReLU neurons output zero for all inputs, gradients become zero, learning stops

**Explanation:** If a ReLU neuron's input is always negative (due to large negative bias or poor initialization), it outputs zero for all inputs. With zero output, the gradient is zero, so the neuron never updates - it's "dead." Solutions include Leaky ReLU or careful initialization.

- **Why others are wrong:**
  - A) ReLU prevents explosion (gradient is 1 for positive)
  - C) ReLU actually speeds up training
  - D) ReLU works with all optimizers
</details>

---

### 39. Why is the loss function for neural networks often called non-convex?

- [ ] A) It has a single global minimum
- [ ] B) It has multiple local minima and saddle points
- [ ] C) It's always convex
- [ ] D) It doesn't depend on the weights

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It has multiple local minima and saddle points

**Explanation:** A convex function has a single global minimum (any local minimum is the global minimum). Neural network loss landscapes are non-convex: they have many local minima, saddle points, and plateaus due to the complex interaction of millions of parameters.

- **Why others are wrong:**
  - A) That would make it convex
  - C) It's non-convex for networks with hidden layers
  - D) Loss definitely depends on weights
</details>

---

### 40. In batch normalization, what happens if the batch size is 1?

- [ ] A) Batch norm works perfectly fine
- [ ] B) Batch norm fails because variance is undefined (zero variance)
- [ ] C) Batch norm uses the global mean
- [ ] D) Training becomes 10x faster

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Batch norm fails because variance is undefined (zero variance)

**Explanation:** With a single sample, variance is zero (or undefined), making normalization impossible. Batch norm requires multiple samples to compute meaningful statistics. For batch size 1, use Layer Normalization or Group Normalization instead.

- **Why others are wrong:**
  - A) Batch norm breaks with batch size 1
  - C) It can't compute meaningful batch statistics from 1 sample
  - D) Training would fail, not speed up
</details>

---

*Quiz generated by Practice Quiz Agent for Week 2: Deep Learning & NLP - Tuesday Content*

