"""
Exercise 01: TensorBoard Exploration - Starter Code

Prerequisites:
- Reading: 01-tensorboard-visualization.md
- Demo: demo_01_tensorboard_setup.py (REFERENCE FOR ALL TASKS)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime
import os

# ============================================================================
# DATA & MODEL (PROVIDED - DO NOT MODIFY)
# ============================================================================

def load_mnist_data():
    """Load and preprocess MNIST data"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Create a simple MLP for MNIST classification"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# TASK 1.1: Basic TensorBoard Setup
# ============================================================================

def create_tensorboard_callback(experiment_name="default"):
    """
    Create a TensorBoard callback with proper log directory.
    
    REQUIREMENTS:
    - Log directory format: logs/{experiment_name}_{timestamp}
    - Enable histogram_freq=1 for weight histograms
    - Enable write_graph=True for model visualization
    
    HINTS:
    - Use datetime.datetime.now().strftime("%Y%m%d-%H%M%S") for timestamp
    - keras.callbacks.TensorBoard takes log_dir as first argument
    
    SEE: demo_01_tensorboard_setup.py lines 35-50
    """
    # YOUR CODE HERE
    pass


# ============================================================================
# TASK 1.2: Custom Metric Logging
# ============================================================================

class CustomMetricsCallback(keras.callbacks.Callback):
    """
    Custom callback to log additional metrics to TensorBoard.
    
    REQUIREMENTS:
    - Log current learning rate each epoch
    - Log loss ratio (train_loss / val_loss) as overfitting indicator
    
    HINTS:
    - Create file writer with: tf.summary.create_file_writer(log_dir + "/custom")
    - Get LR: self.model.optimizer.learning_rate (may need .numpy())
    - Write scalar: tf.summary.scalar('name', value, step=epoch)
    - Use "with self.file_writer.as_default():" context
    
    SEE: demo_01_tensorboard_setup.py lines 70-100 for custom callback pattern
    """
    
    def __init__(self, log_dir):
        super().__init__()
        # YOUR CODE: Create file writer
    
    def on_epoch_end(self, epoch, logs=None):
        """Log custom metrics at end of each epoch"""
        logs = logs or {}
        # YOUR CODE: 
        # 1. Get learning rate from optimizer
        # 2. Calculate loss_ratio = train_loss / val_loss
        # 3. Write both to TensorBoard with tf.summary.scalar


# ============================================================================
# TASK 1.3: Experiment Comparison
# ============================================================================

def run_experiment(learning_rate, experiment_name):
    """
    Run training experiment with specified learning rate.
    
    HINTS:
    - Create optimizer: keras.optimizers.Adam(learning_rate=learning_rate)
    - Train for 20 epochs with validation_split=0.2
    
    SEE: demo_01_tensorboard_setup.py "Part 4: Running Multiple Experiments"
    """
    # YOUR CODE HERE
    pass


def compare_learning_rates():
    """
    Compare learning rates: 0.001, 0.01, 0.1
    
    After running, view in TensorBoard:
      tensorboard --logdir=logs
    
    Use TensorBoard's "Runs" panel to compare experiments side-by-side.
    """
    learning_rates = [0.001, 0.01, 0.1]
    
    # YOUR CODE: Loop through LRs, call run_experiment with descriptive names
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01: TensorBoard Exploration")
    print("=" * 60)
    
    # Uncomment as you complete:
    # Task 1.1
    # tb_callback = create_tensorboard_callback("basic_test")
    # model = create_model()
    # (x_train, y_train), _ = load_mnist_data()
    # model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[tb_callback])
    
    # Task 1.2 - Add custom callback
    # custom_cb = CustomMetricsCallback("logs/custom_test")
    # model.fit(..., callbacks=[tb_callback, custom_cb])
    
    # Task 1.3
    # compare_learning_rates()
    
    print("\nTo view results: tensorboard --logdir=logs")
    print("Then open http://localhost:6006")
