"""
Exercise 01: Backpropagation by Hand - Starter Code

Verify your manual calculations with TensorFlow.

Prerequisites:
- Reading: 01-backpropagation-algorithm.md
- Demo: demo_01_backprop_visualization.py (REFERENCE FOR GRADIENT TAPE)
"""

import tensorflow as tf
import numpy as np

# ============================================================================
# NETWORK CONFIGURATION (PROVIDED - USE THESE EXACT VALUES)
# ============================================================================

# Network: Input(2) -> Hidden(2, sigmoid) -> Output(1, sigmoid)

X = np.array([[0.5, 0.3]])      # Input: shape (1, 2)
y_true = np.array([[0.8]])       # Target: shape (1, 1)

# Layer 1: Input -> Hidden
W1 = np.array([[0.1, 0.2],       # shape (2, 2)
               [0.3, 0.4]])
b1 = np.array([[0.1, 0.1]])      # shape (1, 2)

# Layer 2: Hidden -> Output
W2 = np.array([[0.5],            # shape (2, 1)
               [0.6]])
b2 = np.array([[0.1]])           # shape (1, 1)


# ============================================================================
# HELPER FUNCTIONS (PROVIDED)
# ============================================================================

def sigmoid(x):
    """Sigmoid: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """d(sigmoid)/dx = sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)


# ============================================================================
# TASK 1.1: Forward Pass
# ============================================================================

def forward_pass_manual():
    """
    Compute forward pass step by step.
    
    FORMULAS (implement these in order):
    1. z1 = X @ W1 + b1        (matrix multiply + bias)
    2. a1 = sigmoid(z1)        (activation)
    3. z2 = a1 @ W2 + b2       (matrix multiply + bias)
    4. y_pred = sigmoid(z2)    (output activation)
    5. loss = mean((y_pred - y_true)^2)   (MSE loss)
    
    Print each intermediate value to verify your hand calculations.
    
    Returns:
        dict with keys: 'z1', 'a1', 'z2', 'y_pred', 'loss'
    """
    print("=" * 50)
    print("FORWARD PASS")
    print("=" * 50)
    
    # YOUR CODE: Implement the 5 steps above
    # Print each result like: print(f"z1 = {z1}")
    pass


# ============================================================================
# TASK 1.2: Backward Pass (Chain Rule)
# ============================================================================

def backward_pass_manual(forward_results):
    """
    Compute gradients using chain rule.
    
    FORMULAS (chain rule from output to input):
    
    Output Layer:
    1. dL/dy_pred = 2*(y_pred - y_true) / n      # MSE gradient
    2. dy_pred/dz2 = sigmoid_derivative(z2)
    3. dL/dz2 = dL/dy_pred * dy_pred/dz2         # chain rule
    4. dL/dW2 = a1.T @ dL/dz2                    # gradient for W2
    5. dL/db2 = sum(dL/dz2)                      # gradient for b2
    
    Hidden Layer:
    6. dL/da1 = dL/dz2 @ W2.T                    # backprop error
    7. da1/dz1 = sigmoid_derivative(z1)
    8. dL/dz1 = dL/da1 * da1/dz1                 # chain rule
    9. dL/dW1 = X.T @ dL/dz1                     # gradient for W1
    10. dL/db1 = sum(dL/dz1)                     # gradient for b1
    
    Returns:
        dict with keys: 'dL_dW1', 'dL_db1', 'dL_dW2', 'dL_db2'
    """
    print("\n" + "=" * 50)
    print("BACKWARD PASS")
    print("=" * 50)
    
    # YOUR CODE: Implement the 10 steps above
    # Use forward_results to get z1, a1, z2, y_pred
    pass


# ============================================================================
# TASK 1.3: TensorFlow Verification
# ============================================================================

def verify_with_tensorflow():
    """
    Use TensorFlow GradientTape to compute gradients automatically.
    
    PATTERN (from demo_01_backprop_visualization.py):
    
    with tf.GradientTape() as tape:
        # Forward pass inside tape
        z1 = tf.matmul(X_tf, W1_tf) + b1_tf
        a1 = tf.sigmoid(z1)
        # ... continue forward pass ...
        loss = tf.reduce_mean(tf.square(y_pred - y_true_tf))
    
    # Get gradients
    gradients = tape.gradient(loss, [W1_tf, b1_tf, W2_tf, b2_tf])
    
    HINTS:
    - Convert numpy arrays to tf.constant for inputs
    - Use tf.Variable for weights (so tape can track them)
    - Compare with your manual gradients - they should match!
    """
    print("\n" + "=" * 50)
    print("TENSORFLOW VERIFICATION")
    print("=" * 50)
    
    # YOUR CODE: 
    # 1. Create TF variables for W1, b1, W2, b2
    # 2. Forward pass inside GradientTape
    # 3. Get gradients with tape.gradient()
    # 4. Print and compare with manual results
    pass


def compare_results(manual_grads, tf_grads):
    """Compare manual vs TF gradients and report match/mismatch"""
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    
    # YOUR CODE: For each gradient, compute max absolute difference
    # Report MATCH if diff < 1e-6, otherwise MISMATCH


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01: Backpropagation by Hand")
    print("=" * 60)
    
    print("\nNetwork: 2 inputs -> 2 hidden (sigmoid) -> 1 output (sigmoid)")
    print(f"Input X: {X}")
    print(f"Target y: {y_true}")
    
    # Uncomment as you complete:
    # forward_results = forward_pass_manual()
    # manual_grads = backward_pass_manual(forward_results)
    # tf_grads = verify_with_tensorflow()
    # compare_results(manual_grads, tf_grads)
    
    print("\nComplete the functions to verify your hand calculations!")
