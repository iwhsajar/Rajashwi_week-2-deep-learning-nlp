"""
Exercise 04: Custom Callbacks - Starter Code

Create custom Keras callbacks for training control and logging.

Prerequisites:
- Reading: 05-customizing-training-process.md
- Demo: demo_04_custom_training_loop.py (REFERENCE FOR CALLBACK STRUCTURE)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import datetime
import os

# ============================================================================
# TASK 4.1: Learning Rate Scheduler Callback
# ============================================================================

class StepLRScheduler(keras.callbacks.Callback):
    """
    Reduce learning rate by `factor` every `step_epochs` epochs.
    
    CALLBACK METHODS TO IMPLEMENT:
    - on_epoch_begin(self, epoch, logs=None): Called at start of each epoch
    
    HINTS:
    - Access optimizer: self.model.optimizer
    - Get LR: self.model.optimizer.learning_rate (use .numpy() if needed)
    - Set LR: self.model.optimizer.learning_rate.assign(new_value)
    
    SEE: demo_04_custom_training_loop.py for callback examples
    """
    
    def __init__(self, factor=0.5, step_epochs=10, verbose=True):
        super().__init__()
        self.factor = factor
        self.step_epochs = step_epochs
        self.verbose = verbose
        self.lr_history = []
    
    def on_epoch_begin(self, epoch, logs=None):
        """
        Reduce LR if epoch is a multiple of step_epochs.
        
        LOGIC:
        1. Get current LR from optimizer
        2. If epoch > 0 and epoch % step_epochs == 0: multiply LR by factor
        3. Update optimizer LR
        4. Append current LR to lr_history
        """
        # YOUR CODE HERE
        pass
    
    def get_lr_history(self):
        return self.lr_history


# ============================================================================
# TASK 4.2: JSON Training Logger
# ============================================================================

class JSONTrainingLogger(keras.callbacks.Callback):
    """
    Log training metrics to JSON file after each epoch.
    
    CALLBACK METHOD: on_epoch_end(self, epoch, logs=None)
    
    LOG ENTRY FORMAT:
    {
        "epoch": 1,
        "timestamp": "2024-01-15T10:30:00",
        "loss": 0.5,
        "accuracy": 0.85,
        "val_loss": 0.6,
        "val_accuracy": 0.82
    }
    """
    
    def __init__(self, filepath='training_log.json'):
        super().__init__()
        self.filepath = filepath
        self.logs_list = []
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Append metrics to log file.
        
        HINTS:
        - logs dict contains: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
        - Use datetime.datetime.now().isoformat() for timestamp
        - Append to self.logs_list, then write entire list to JSON
        """
        logs = logs or {}
        
        # YOUR CODE: Create entry dict, append to logs_list, write to file
        pass


# ============================================================================
# TASK 4.3: Early Stopping with Patience
# ============================================================================

class CustomEarlyStopping(keras.callbacks.Callback):
    """
    Stop training when validation loss hasn't improved for `patience` epochs.
    
    CALLBACK METHODS:
    - on_train_begin(self, logs=None): Reset state
    - on_epoch_end(self, epoch, logs=None): Check improvement
    
    ATTRIBUTES TO TRACK:
    - best_loss: Best val_loss seen so far
    - wait: Epochs since last improvement
    - stopped_epoch: Epoch when training stopped (or 0)
    """
    
    def __init__(self, patience=5, min_delta=0.001):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta  # Minimum improvement to count as better
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, logs=None):
        """Reset state at start of training"""
        # YOUR CODE: Reset best_loss, wait, stopped_epoch
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Check if we should stop.
        
        LOGIC:
        1. Get current val_loss
        2. If val_loss < best_loss - min_delta: improvement! Reset wait, update best
        3. Else: increment wait
        4. If wait >= patience: set self.model.stop_training = True
        """
        # YOUR CODE HERE
        pass


# ============================================================================
# TASK 4.4: Test Your Callbacks
# ============================================================================

def create_model():
    """Simple model for testing (PROVIDED)"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def test_callbacks():
    """
    Test all callbacks together.
    
    STEPS:
    1. Load MNIST data
    2. Create model
    3. Create all 3 callbacks
    4. Train with all callbacks
    5. Verify: LR decreased, JSON log exists, training stopped early
    """
    print("Testing Custom Callbacks")
    
    # YOUR CODE:
    # callbacks = [
    #     StepLRScheduler(factor=0.5, step_epochs=5),
    #     JSONTrainingLogger('test_log.json'),
    #     CustomEarlyStopping(patience=3)
    # ]
    # model.fit(..., callbacks=callbacks)
    pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 04: Custom Callbacks")
    print("=" * 60)
    
    # Uncomment to test:
    # test_callbacks()
    
    print("\nImplement the callback classes and run test_callbacks()!")
