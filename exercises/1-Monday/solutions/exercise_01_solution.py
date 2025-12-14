"""
Exercise 01: TensorBoard Exploration - Complete Solution

This is the instructor solution with all tasks implemented.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime
import os

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_mnist_data():
    """Load and preprocess MNIST data"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_model(learning_rate=0.001):
    """Create a simple MLP for MNIST classification"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================================================
# TASK 1.1: Basic TensorBoard Setup - SOLUTION
# ============================================================================

def create_tensorboard_callback(experiment_name="default"):
    """
    Create a TensorBoard callback with proper log directory.
    """
    # Create timestamp for unique run identification
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create log directory path
    log_dir = f"logs/{experiment_name}_{timestamp}"
    
    # Create TensorBoard callback with all options
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,      # Log weight histograms every epoch
        write_graph=True,      # Visualize model graph
        write_images=True,     # Log weight images
        update_freq='epoch',   # Update logs per epoch
        profile_batch=2        # Profile second batch for performance
    )
    
    print(f"TensorBoard logs will be saved to: {log_dir}")
    return tensorboard_callback

# ============================================================================
# TASK 1.2: Custom Metric Logging - SOLUTION
# ============================================================================

class CustomMetricsCallback(keras.callbacks.Callback):
    """
    Custom callback to log additional metrics to TensorBoard.
    """
    
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir + "/custom_metrics")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log custom metrics at end of each epoch."""
        logs = logs or {}
        
        # Get current learning rate
        lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        
        # Calculate loss ratio (overfitting indicator)
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 1)
        loss_ratio = train_loss / val_loss if val_loss > 0 else 0
        
        # Calculate accuracy gap
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        acc_gap = train_acc - val_acc
        
        # Write metrics to TensorBoard
        with self.file_writer.as_default():
            tf.summary.scalar('learning_rate', lr, step=epoch)
            tf.summary.scalar('loss_ratio', loss_ratio, step=epoch)
            tf.summary.scalar('accuracy_gap', acc_gap, step=epoch)
        
        self.file_writer.flush()

# ============================================================================
# TASK 1.3: Experiment Comparison - SOLUTION
# ============================================================================

def run_experiment(learning_rate, experiment_name):
    """
    Run a training experiment with specified learning rate.
    """
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Create model with custom learning rate
    model = create_model(learning_rate=learning_rate)
    
    # Create TensorBoard callback
    tb_callback = create_tensorboard_callback(experiment_name)
    
    # Create custom metrics callback
    log_dir = tb_callback.log_dir
    custom_callback = CustomMetricsCallback(log_dir)
    
    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.2,
        callbacks=[tb_callback, custom_callback],
        verbose=1
    )
    
    return history

def compare_learning_rates():
    """
    Compare three different learning rates and log to TensorBoard.
    """
    learning_rates = [0.001, 0.01, 0.1]
    histories = {}
    
    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"Running experiment with learning rate: {lr}")
        print('='*50)
        
        experiment_name = f"lr_{str(lr).replace('.', '_')}"
        history = run_experiment(lr, experiment_name)
        histories[lr] = history
        
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nLR={lr}: Final validation accuracy = {final_val_acc:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for lr, history in histories.items():
        final_acc = history.history['val_accuracy'][-1]
        best_acc = max(history.history['val_accuracy'])
        print(f"LR={lr}: Final={final_acc:.4f}, Best={best_acc:.4f}")
    
    return histories

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01: TensorBoard Exploration - SOLUTION")
    print("=" * 60)
    
    # Run all experiments
    compare_learning_rates()
    
    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("To view TensorBoard, run:")
    print("  tensorboard --logdir=logs")
    print("Then open http://localhost:6006 in your browser")
    print("=" * 60)

