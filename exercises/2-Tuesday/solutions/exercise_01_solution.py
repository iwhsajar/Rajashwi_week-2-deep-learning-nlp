"""
Exercise 01 Solution: Backpropagation by Hand

Complete implementation of manual backpropagation for a 2-layer network.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# =============================================================================
# PART 1: Manual Forward and Backward Pass
# =============================================================================

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid: a * (1 - a) where a = sigmoid(z)."""
    return a * (1 - a)

def relu(z):
    """ReLU activation function."""
    return np.maximum(0, z)

def relu_derivative(z):
    """Derivative of ReLU: 1 if z > 0, else 0."""
    return (z > 0).astype(float)


class ManualNeuralNetwork:
    """
    A 2-layer neural network with manual forward and backward pass.
    
    Architecture:
        Input (2) -> Hidden (3, ReLU) -> Output (1, Sigmoid)
    """
    
    def __init__(self, input_size=2, hidden_size=3, output_size=1, seed=42):
        np.random.seed(seed)
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Store intermediate values for backprop
        self.cache = {}
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Args:
            X: Input data, shape (batch_size, input_size)
        
        Returns:
            Output predictions, shape (batch_size, output_size)
        """
        # Layer 1: Linear transformation
        z1 = np.dot(X, self.W1) + self.b1
        
        # Layer 1: ReLU activation
        a1 = relu(z1)
        
        # Layer 2: Linear transformation
        z2 = np.dot(a1, self.W2) + self.b2
        
        # Layer 2: Sigmoid activation
        a2 = sigmoid(z2)
        
        # Cache for backprop
        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2
        }
        
        return a2
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute binary cross-entropy loss.
        
        Loss = -1/m * sum(y*log(y_pred) + (1-y)*log(1-y_pred))
        """
        m = y_true.shape[0]
        epsilon = 1e-15  # Prevent log(0)
        
        loss = -1/m * np.sum(
            y_true * np.log(y_pred + epsilon) + 
            (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        return loss
    
    def backward(self, y_true):
        """
        Backward pass: compute gradients using chain rule.
        
        Returns:
            Dictionary of gradients for all parameters
        """
        m = y_true.shape[0]
        
        # Retrieve cached values
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        a2 = self.cache['a2']
        
        # =====================================================================
        # STEP 1: Output layer gradients
        # =====================================================================
        
        # dL/da2: Gradient of loss w.r.t. output activation
        # For BCE loss with sigmoid: dL/da2 = (a2 - y) / (a2 * (1 - a2))
        # But when combined with sigmoid derivative, simplifies to:
        # dL/dz2 = a2 - y
        dz2 = a2 - y_true  # Shape: (m, 1)
        
        # dL/dW2 = dL/dz2 * dz2/dW2 = a1.T @ dz2
        dW2 = (1/m) * np.dot(a1.T, dz2)  # Shape: (hidden, 1)
        
        # dL/db2 = dL/dz2 * dz2/db2 = sum(dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)  # Shape: (1, 1)
        
        # =====================================================================
        # STEP 2: Hidden layer gradients (backpropagate through W2)
        # =====================================================================
        
        # dL/da1 = dL/dz2 * dz2/da1 = dz2 @ W2.T
        da1 = np.dot(dz2, self.W2.T)  # Shape: (m, hidden)
        
        # dL/dz1 = dL/da1 * da1/dz1 = da1 * relu'(z1)
        dz1 = da1 * relu_derivative(z1)  # Shape: (m, hidden)
        
        # dL/dW1 = dL/dz1 * dz1/dW1 = X.T @ dz1
        dW1 = (1/m) * np.dot(X.T, dz1)  # Shape: (input, hidden)
        
        # dL/db1 = dL/dz1 * dz1/db1 = sum(dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)  # Shape: (1, hidden)
        
        gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return gradients
    
    def update_weights(self, gradients, learning_rate=0.1):
        """Apply gradient descent update."""
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']
    
    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """Train the network."""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            # Backward pass
            gradients = self.backward(y)
            
            # Update weights
            self.update_weights(gradients, learning_rate)
            
            if verbose and epoch % 100 == 0:
                accuracy = np.mean((y_pred > 0.5) == y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return losses


# =============================================================================
# PART 2: Verification with TensorFlow
# =============================================================================

def verify_with_tensorflow(X, y, manual_nn):
    """
    Create equivalent TensorFlow model and verify gradients match.
    """
    # Create TensorFlow model with same weights
    model = keras.Sequential([
        keras.layers.Dense(3, activation='relu', input_shape=(2,)),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Copy weights from manual network
    model.layers[0].set_weights([manual_nn.W1, manual_nn.b1.flatten()])
    model.layers[1].set_weights([manual_nn.W2, manual_nn.b2.flatten()])
    
    # Compute gradients with TensorFlow
    X_tf = tf.constant(X, dtype=tf.float32)
    y_tf = tf.constant(y, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        y_pred = model(X_tf)
        loss = keras.losses.binary_crossentropy(y_tf, y_pred)
        loss = tf.reduce_mean(loss)
    
    tf_gradients = tape.gradient(loss, model.trainable_variables)
    
    # Compute gradients with manual network
    manual_nn.forward(X)
    manual_gradients = manual_nn.backward(y)
    
    # Compare
    print("\nGradient Comparison (Manual vs TensorFlow):")
    print("-" * 50)
    
    print(f"dW1 difference: {np.max(np.abs(manual_gradients['dW1'] - tf_gradients[0].numpy())):.8f}")
    print(f"db1 difference: {np.max(np.abs(manual_gradients['db1'].flatten() - tf_gradients[1].numpy())):.8f}")
    print(f"dW2 difference: {np.max(np.abs(manual_gradients['dW2'] - tf_gradients[2].numpy())):.8f}")
    print(f"db2 difference: {np.max(np.abs(manual_gradients['db2'].flatten() - tf_gradients[3].numpy())):.8f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01 Solution: Backpropagation by Hand")
    print("=" * 60)
    
    # XOR problem dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)
    
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)
    
    print("\nDataset (XOR problem):")
    print(f"X:\n{X}")
    print(f"y: {y.flatten()}")
    
    # Create and train manual network
    print("\n" + "=" * 60)
    print("Training Manual Neural Network")
    print("=" * 60)
    
    nn = ManualNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    losses = nn.train(X, y, epochs=1000, learning_rate=0.5)
    
    # Final predictions
    print("\nFinal Predictions:")
    predictions = nn.forward(X)
    for i in range(len(X)):
        print(f"  Input: {X[i]} -> Predicted: {predictions[i][0]:.4f}, Actual: {y[i][0]}")
    
    # Verify with TensorFlow
    print("\n" + "=" * 60)
    print("Verifying Gradients with TensorFlow")
    print("=" * 60)
    
    nn_verify = ManualNeuralNetwork(input_size=2, hidden_size=3, output_size=1)
    verify_with_tensorflow(X, y, nn_verify)
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)

