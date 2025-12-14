# Exercise 02: Checkpoint Strategy

## Learning Objectives

- Implement ModelCheckpoint callback with various configurations
- Design checkpoint strategies for different scenarios
- Resume training from checkpoints after interruption
- Manage checkpoint storage efficiently

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_02_checkpoint_callback.py`, you saw how checkpoints save your progress during training. This exercise explores different checkpointing strategies for various scenarios.

## Checkpoint Configuration Reference

```python
keras.callbacks.ModelCheckpoint(
    filepath='path/model.h5',      # Where to save
    monitor='val_loss',            # Metric to track
    mode='min',                    # 'min' for loss, 'max' for accuracy
    save_best_only=True,           # Only save if metric improved
    save_weights_only=False,       # Save weights or full model
    verbose=1                      # Print when saving
)
```

---

## Part 1: Basic Checkpointing (15 min)

### Task 1.1: Save Every Epoch

Navigate to `starter_code/checkpoint_starter.py`:

```python
import numpy as np
import os
import glob
from tensorflow import keras
from tensorflow.keras import layers

# Create directory for checkpoints
os.makedirs('checkpoints', exist_ok=True)

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

def create_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Checkpoint every epoch
checkpoint_every = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_epoch_{epoch:02d}.h5',
    save_freq='epoch',
    verbose=1
)

model = create_model()
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[checkpoint_every],
    verbose=1
)

# List saved checkpoints
checkpoints = sorted(glob.glob('checkpoints/model_epoch_*.h5'))
print(f"\nSaved {len(checkpoints)} checkpoints:")
for cp in checkpoints:
    print(f"  {cp}")
```

### Task 1.2: Save Best Model Only

```python
# Only save when validation loss improves
checkpoint_best = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/best_model.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

model2 = create_model()
history2 = model2.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[checkpoint_best],
    verbose=1
)

# Load best model and evaluate
best_model = keras.models.load_model('checkpoints/best_model.h5')
best_loss, best_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Best model test accuracy: {best_acc:.4f}")
```

---

## Part 2: Advanced Checkpoint Patterns (15 min)

### Task 2.1: Include Metrics in Filename

Track which checkpoint has which performance:

```python
# Include epoch and validation loss in filename
checkpoint_detailed = keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/model_{epoch:02d}_valloss_{val_loss:.4f}.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=False,  # Save all to see progression
    verbose=1
)

model3 = create_model()
model3.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[checkpoint_detailed],
    verbose=0
)

# Parse checkpoints to find best
checkpoints = glob.glob('checkpoints/model_*_valloss_*.h5')
best_checkpoint = min(checkpoints, key=lambda x: float(x.split('valloss_')[1].replace('.h5', '')))
print(f"Best checkpoint: {best_checkpoint}")
```

### Task 2.2: Keep Only Top N Checkpoints

Prevent disk from filling up during long training:

```python
class KeepNBestCheckpoints(keras.callbacks.Callback):
    """
    Custom callback to keep only the N best checkpoints.
    """
    
    def __init__(self, filepath_template, monitor='val_loss', mode='min', n_best=3):
        super().__init__()
        self.filepath_template = filepath_template
        self.monitor = monitor
        self.mode = mode
        self.n_best = n_best
        self.best_checkpoints = []  # List of (metric_value, filepath)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_value = logs.get(self.monitor)
        
        if current_value is None:
            return
        
        filepath = self.filepath_template.format(epoch=epoch+1, **logs)
        self.model.save(filepath)
        
        self.best_checkpoints.append((current_value, filepath))
        
        # Sort by metric
        if self.mode == 'min':
            self.best_checkpoints.sort(key=lambda x: x[0])
        else:
            self.best_checkpoints.sort(key=lambda x: -x[0])
        
        # Remove excess checkpoints
        while len(self.best_checkpoints) > self.n_best:
            _, to_remove = self.best_checkpoints.pop()
            if os.path.exists(to_remove):
                os.remove(to_remove)
                print(f"Removed checkpoint: {to_remove}")

# Test custom callback
keep_best_3 = KeepNBestCheckpoints(
    filepath_template='checkpoints/top3_epoch_{epoch:02d}_loss_{val_loss:.4f}.h5',
    monitor='val_loss',
    mode='min',
    n_best=3
)

model4 = create_model()
model4.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[keep_best_3], verbose=1)
```

---

## Part 3: Resume Training (15 min)

### Task 3.1: Simulate Interruption and Resume

```python
def train_with_resume(initial_epochs=5, additional_epochs=5):
    """
    Demonstrate training interruption and resume.
    """
    # Phase 1: Initial training
    print("Phase 1: Initial training")
    model = create_model()
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath='checkpoints/resume_model.h5',
        save_best_only=False
    )
    
    history1 = model.fit(
        x_train, y_train,
        epochs=initial_epochs,
        validation_split=0.2,
        callbacks=[checkpoint],
        verbose=1
    )
    
    initial_acc = history1.history['val_accuracy'][-1]
    print(f"After Phase 1: val_accuracy = {initial_acc:.4f}")
    
    # Simulate interruption - delete model from memory
    del model
    
    # Phase 2: Resume training
    print("\nPhase 2: Resume from checkpoint")
    resumed_model = keras.models.load_model('checkpoints/resume_model.h5')
    
    # Continue training
    history2 = resumed_model.fit(
        x_train, y_train,
        epochs=additional_epochs,
        initial_epoch=initial_epochs,  # Continue epoch numbering
        validation_split=0.2,
        callbacks=[checkpoint],
        verbose=1
    )
    
    final_acc = history2.history['val_accuracy'][-1]
    print(f"After Phase 2: val_accuracy = {final_acc:.4f}")
    print(f"Improvement: {(final_acc - initial_acc) * 100:.2f}%")
    
    return resumed_model

final_model = train_with_resume()
```

### Task 3.2: Combine with Early Stopping

```python
def robust_training_pipeline(epochs=50, patience=5):
    """
    Production training with checkpoints and early stopping.
    """
    model = create_model()
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath='checkpoints/robust_best.h5',
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    stopped_epoch = len(history.history['loss'])
    print(f"\nTraining stopped at epoch {stopped_epoch}")
    
    # The model now has best weights (from restore_best_weights)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {test_acc:.4f}")
    
    return model, history

model, history = robust_training_pipeline()
```

---

## Checkpoint Strategy Guide

Based on your scenario, choose appropriate strategy:

| Scenario | save_best_only | Filename Pattern | Notes |
|----------|----------------|------------------|-------|
| Short training (<1hr) | True | best_model.h5 | Simple, just keep best |
| Long training (hrs-days) | False | model_{epoch}.h5 | Keep all for recovery |
| Limited disk space | Use KeepNBest | top3_*.h5 | Auto-delete old |
| Hyperparameter search | True | model_{params}.h5 | One per config |

---

## Reflection Questions

1. **What's the difference between save_best_only=True and EarlyStopping with restore_best_weights=True?**

2. **If training crashes at epoch 45 of 100, how do you resume?** What's the initial_epoch parameter for?

3. **Why might you want to keep multiple checkpoints instead of just the best?**

4. **For a training run expected to take 24 hours, what checkpoint strategy would you use?**

---

## Definition of Done

- [ ] Implemented epoch-based checkpointing
- [ ] Implemented best-model-only checkpointing
- [ ] Created descriptive checkpoint filenames with metrics
- [ ] Implemented KeepNBest callback
- [ ] Demonstrated training resume from checkpoint
- [ ] Combined checkpointing with early stopping
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_02_solution.py`. Only check after completing your attempt!

