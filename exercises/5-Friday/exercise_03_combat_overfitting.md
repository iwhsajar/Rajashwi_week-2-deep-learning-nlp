# Exercise 03: Combat Overfitting

## Learning Objectives

- Create and diagnose overfitting scenarios
- Apply multiple regularization techniques
- Compare effectiveness of different approaches
- Build a robust anti-overfitting training pipeline

## Duration

**Estimated Time:** 60 minutes

## Background

In `demo_04_regularization_techniques.py`, you saw how dropout and L2 regularization reduce overfitting. This exercise has you systematically combat overfitting on a challenging dataset.

---

## Part 1: Create Overfitting Scenario (15 min)

### Task 1.1: Setup Overfitting-Prone Data

Small dataset + large model = guaranteed overfitting:

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Use small subset of MNIST (overfitting territory)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Use only 500 training samples (very small)
x_train_small = x_train[:500].reshape(-1, 784).astype('float32') / 255.0
y_train_small = y_train[:500]
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Create overly complex model (too many parameters for small data)
def create_overfit_model():
    """Large model that will definitely overfit on small data."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train baseline (will overfit)
baseline_model = create_overfit_model()
baseline_history = baseline_model.fit(
    x_train_small, y_train_small,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=0
)

# Plot overfitting
def plot_training(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_training(baseline_history, 'Baseline (Overfitting)')
print(f"Train accuracy: {baseline_history.history['accuracy'][-1]:.4f}")
print(f"Val accuracy: {baseline_history.history['val_accuracy'][-1]:.4f}")
print(f"Gap: {baseline_history.history['accuracy'][-1] - baseline_history.history['val_accuracy'][-1]:.4f}")
```

---

## Part 2: Apply Regularization Techniques (25 min)

### Task 2.1: Dropout Regularization

```python
def create_dropout_model(dropout_rate=0.3):
    """Model with dropout after each hidden layer."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,)),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

dropout_model = create_dropout_model(dropout_rate=0.4)
dropout_history = dropout_model.fit(
    x_train_small, y_train_small,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=0
)

plot_training(dropout_history, 'With Dropout (0.4)')
```

### Task 2.2: L2 Regularization

```python
def create_l2_model(l2_lambda=0.01):
    """Model with L2 regularization on all layers."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,),
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

l2_model = create_l2_model(l2_lambda=0.001)
l2_history = l2_model.fit(
    x_train_small, y_train_small,
    epochs=100,
    validation_data=(x_test, y_test),
    verbose=0
)

plot_training(l2_history, 'With L2 Regularization')
```

### Task 2.3: Early Stopping

```python
def train_with_early_stopping(model, patience=10):
    """Train with early stopping callback."""
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        x_train_small, y_train_small,
        epochs=200,  # High limit, early stopping will kick in
        validation_data=(x_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    return history

early_stop_model = create_overfit_model()
early_stop_history = train_with_early_stopping(early_stop_model, patience=10)

plot_training(early_stop_history, 'With Early Stopping')
print(f"Stopped at epoch: {len(early_stop_history.history['loss'])}")
```

### Task 2.4: Combined Approach

```python
def create_regularized_model(dropout_rate=0.3, l2_lambda=0.001):
    """Model with both dropout and L2 regularization."""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_shape=(784,),
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

combined_model = create_regularized_model()
combined_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
]

combined_history = combined_model.fit(
    x_train_small, y_train_small,
    epochs=200,
    validation_data=(x_test, y_test),
    callbacks=combined_callbacks,
    verbose=0
)

plot_training(combined_history, 'Combined (Dropout + L2 + Early Stopping)')
```

---

## Part 3: Compare and Analyze (20 min)

### Task 3.1: Create Comparison Table

```python
def evaluate_model(history, model, x_test, y_test, name):
    """Evaluate model and return metrics."""
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    gap = train_acc - val_acc
    epochs = len(history.history['loss'])
    
    return {
        'name': name,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'gap': gap,
        'epochs': epochs
    }

results = [
    evaluate_model(baseline_history, baseline_model, x_test, y_test, 'Baseline'),
    evaluate_model(dropout_history, dropout_model, x_test, y_test, 'Dropout'),
    evaluate_model(l2_history, l2_model, x_test, y_test, 'L2'),
    evaluate_model(early_stop_history, early_stop_model, x_test, y_test, 'Early Stop'),
    evaluate_model(combined_history, combined_model, x_test, y_test, 'Combined')
]

# Print comparison table
print("\nComparison Table:")
print("-" * 80)
print(f"{'Method':<15} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Epochs':<8}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<15} {r['train_acc']:.4f}       {r['val_acc']:.4f}       {r['test_acc']:.4f}       {r['gap']:.4f}     {r['epochs']}")
```

### Task 3.2: Hyperparameter Search

Find optimal regularization strength:

```python
def grid_search_regularization():
    """Search for best dropout and L2 combination."""
    dropout_rates = [0.2, 0.3, 0.4, 0.5]
    l2_lambdas = [0.01, 0.001, 0.0001]
    
    best_val_acc = 0
    best_config = None
    results = []
    
    for dropout in dropout_rates:
        for l2 in l2_lambdas:
            model = create_regularized_model(dropout_rate=dropout, l2_lambda=l2)
            history = model.fit(
                x_train_small, y_train_small,
                epochs=50,
                validation_data=(x_test, y_test),
                callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            results.append({
                'dropout': dropout,
                'l2': l2,
                'val_acc': val_acc
            })
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_config = (dropout, l2)
            
            print(f"Dropout={dropout}, L2={l2}: val_acc={val_acc:.4f}")
    
    print(f"\nBest config: Dropout={best_config[0]}, L2={best_config[1]}")
    print(f"Best val_acc: {best_val_acc:.4f}")
    
    return results

# Run grid search (takes a few minutes)
# search_results = grid_search_regularization()
```

---

## Results Summary

Complete this table with your experimental results:

| Technique | Train Acc | Val Acc | Test Acc | Gap | Epochs |
|-----------|-----------|---------|----------|-----|--------|
| Baseline | | | | | |
| Dropout (0.4) | | | | | |
| L2 (0.001) | | | | | |
| Early Stopping | | | | | |
| Combined | | | | | |

---

## Reflection Questions

1. **Which single technique was most effective at reducing overfitting?** Why do you think that is?

2. **Did combining techniques work better than individual approaches?** Was there a point of diminishing returns?

3. **What is the relationship between the train-val gap and test accuracy?** Is a smaller gap always better?

4. **If you had more training data, would you use less regularization?** How do you decide how much is enough?

5. **For your final production model, what regularization strategy would you use?** Justify your choice.

---

## Definition of Done

- [ ] Created overfitting baseline with visible train-val gap
- [ ] Applied and evaluated dropout regularization
- [ ] Applied and evaluated L2 regularization
- [ ] Applied and evaluated early stopping
- [ ] Combined all techniques
- [ ] Created comparison table with all methods
- [ ] Identified best approach for this dataset
- [ ] Reflection questions answered

---

## Bonus Challenge

Implement data augmentation for image data (even though this is flattened MNIST):

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Reshape back to images for augmentation
x_train_images = x_train_small.reshape(-1, 28, 28, 1)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

# Train with augmented data
# This effectively increases your dataset size
```

---

## Solution

The solution is available in `solutions/exercise_03_solution.py`. Only check after completing your attempt!

