"""
Exercise 02 Solution: Checkpoint Strategy

Complete implementation of optimal checkpoint strategies for training.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# =============================================================================
# PART 1: Create Model and Data
# =============================================================================

def create_model():
    """Create model for checkpoint testing."""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_data():
    """Load and prepare MNIST data."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Use subset
    return (x_train[:5000], y_train[:5000]), (x_test, y_test)


# =============================================================================
# PART 2: Different Checkpoint Strategies
# =============================================================================

def strategy_save_best_only(model, train_data, val_data, checkpoint_dir):
    """
    Strategy 1: Save only when validation loss improves.
    
    This is the most common strategy - keeps only the best model.
    """
    print("\n" + "=" * 50)
    print("Strategy 1: Save Best Only")
    print("=" * 50)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    history = model.fit(
        train_data[0], train_data[1],
        epochs=20,
        batch_size=128,
        validation_data=val_data,
        callbacks=[checkpoint],
        verbose=0
    )
    
    # Check what was saved
    saved_files = os.listdir(checkpoint_dir)
    print(f"\nSaved files: {saved_files}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    
    return history


def strategy_save_every_epoch(model, train_data, val_data, checkpoint_dir):
    """
    Strategy 2: Save checkpoint at every epoch.
    
    Useful for long training runs where you might want to resume.
    """
    print("\n" + "=" * 50)
    print("Strategy 2: Save Every Epoch")
    print("=" * 50)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.keras'),
        save_freq='epoch',
        verbose=0
    )
    
    history = model.fit(
        train_data[0], train_data[1],
        epochs=10,
        batch_size=128,
        validation_data=val_data,
        callbacks=[checkpoint],
        verbose=0
    )
    
    # Check what was saved
    saved_files = sorted(os.listdir(checkpoint_dir))
    print(f"\nSaved files: {saved_files}")
    print(f"Total checkpoints: {len(saved_files)}")
    
    return history


def strategy_save_best_with_early_stopping(model, train_data, val_data, checkpoint_dir):
    """
    Strategy 3: Save best model with early stopping.
    
    Most practical for production - stops when no improvement.
    """
    print("\n" + "=" * 50)
    print("Strategy 3: Best Model + Early Stopping")
    print("=" * 50)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, 'best_model.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_data[0], train_data[1],
        epochs=100,  # High limit - early stopping will kick in
        batch_size=128,
        validation_data=val_data,
        callbacks=callbacks,
        verbose=0
    )
    
    print(f"\nTraining stopped at epoch: {len(history.history['loss'])}")
    print(f"Best val_loss: {min(history.history['val_loss']):.4f}")
    
    return history


def strategy_save_periodic(model, train_data, val_data, checkpoint_dir, period=5):
    """
    Strategy 4: Save every N epochs.
    
    Balance between storage and recovery capability.
    """
    print("\n" + "=" * 50)
    print(f"Strategy 4: Save Every {period} Epochs")
    print("=" * 50)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    class PeriodicCheckpoint(keras.callbacks.Callback):
        def __init__(self, period, save_path):
            super().__init__()
            self.period = period
            self.save_path = save_path
        
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.period == 0:
                path = os.path.join(
                    self.save_path, 
                    f'checkpoint_epoch_{epoch+1:02d}.keras'
                )
                self.model.save(path)
                print(f"\nSaved checkpoint at epoch {epoch + 1}")
    
    checkpoint = PeriodicCheckpoint(period, checkpoint_dir)
    
    history = model.fit(
        train_data[0], train_data[1],
        epochs=20,
        batch_size=128,
        validation_data=val_data,
        callbacks=[checkpoint],
        verbose=0
    )
    
    saved_files = sorted(os.listdir(checkpoint_dir))
    print(f"\nSaved files: {saved_files}")
    
    return history


def strategy_save_top_k(model, train_data, val_data, checkpoint_dir, k=3):
    """
    Strategy 5: Keep only top K models.
    
    Saves best K models, automatically deleting older ones.
    """
    print("\n" + "=" * 50)
    print(f"Strategy 5: Keep Top {k} Models")
    print("=" * 50)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    class TopKCheckpoint(keras.callbacks.Callback):
        def __init__(self, save_path, k=3):
            super().__init__()
            self.save_path = save_path
            self.k = k
            self.best_models = []  # (val_loss, filepath)
        
        def on_epoch_end(self, epoch, logs=None):
            val_loss = logs.get('val_loss')
            filepath = os.path.join(
                self.save_path, 
                f'model_epoch_{epoch+1:02d}_loss_{val_loss:.4f}.keras'
            )
            
            # Save current model
            self.model.save(filepath)
            self.best_models.append((val_loss, filepath))
            
            # Sort by validation loss
            self.best_models.sort(key=lambda x: x[0])
            
            # Remove models beyond top k
            while len(self.best_models) > self.k:
                _, path_to_remove = self.best_models.pop()
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                    print(f"\nRemoved: {os.path.basename(path_to_remove)}")
    
    checkpoint = TopKCheckpoint(checkpoint_dir, k=k)
    
    history = model.fit(
        train_data[0], train_data[1],
        epochs=15,
        batch_size=128,
        validation_data=val_data,
        callbacks=[checkpoint],
        verbose=0
    )
    
    saved_files = sorted(os.listdir(checkpoint_dir))
    print(f"\nFinal saved files ({len(saved_files)} models): {saved_files}")
    
    return history


# =============================================================================
# PART 3: Compare Strategies
# =============================================================================

def compare_strategies():
    """Compare all checkpoint strategies."""
    train_data, val_data = load_data()
    
    strategies = {}
    
    # Strategy 1: Save best only
    model = create_model()
    strategies['best_only'] = strategy_save_best_only(
        model, train_data, val_data, 'checkpoints/strategy1'
    )
    
    # Strategy 2: Save every epoch
    model = create_model()
    strategies['every_epoch'] = strategy_save_every_epoch(
        model, train_data, val_data, 'checkpoints/strategy2'
    )
    
    # Strategy 3: Best + early stopping
    model = create_model()
    strategies['best_early_stop'] = strategy_save_best_with_early_stopping(
        model, train_data, val_data, 'checkpoints/strategy3'
    )
    
    # Strategy 4: Periodic
    model = create_model()
    strategies['periodic'] = strategy_save_periodic(
        model, train_data, val_data, 'checkpoints/strategy4', period=5
    )
    
    # Strategy 5: Top K
    model = create_model()
    strategies['top_k'] = strategy_save_top_k(
        model, train_data, val_data, 'checkpoints/strategy5', k=3
    )
    
    return strategies


def print_recommendations():
    """Print checkpoint strategy recommendations."""
    print("\n" + "=" * 60)
    print("CHECKPOINT STRATEGY RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = """
    | Strategy            | Use When                          | Pros                    | Cons                    |
    |---------------------|-----------------------------------|-------------------------|-------------------------|
    | Save Best Only      | Production training               | Minimal storage         | Can't recover if best   |
    |                     |                                   |                         | model overfits          |
    |---------------------|-----------------------------------|-------------------------|-------------------------|
    | Save Every Epoch    | Research, debugging               | Full history            | High storage            |
    |---------------------|-----------------------------------|-------------------------|-------------------------|
    | Best + Early Stop   | Standard production               | Optimal model, saves    | None                    |
    |                     |                                   | compute                 |                         |
    |---------------------|-----------------------------------|-------------------------|-------------------------|
    | Save Periodic       | Long training, limited storage    | Balance storage/safety  | May miss best           |
    |---------------------|-----------------------------------|-------------------------|-------------------------|
    | Top K Models        | When you want options             | Multiple good models    | Slightly complex        |
    
    RECOMMENDATION: For most cases, use "Best + Early Stopping" strategy.
    """
    print(recommendations)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 02 Solution: Checkpoint Strategy")
    print("=" * 60)
    
    # Compare all strategies
    strategies = compare_strategies()
    
    # Print recommendations
    print_recommendations()
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)

