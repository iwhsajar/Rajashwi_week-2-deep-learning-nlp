# Training vs. Validation Data: Data Splitting Strategies

## Learning Objectives
- Understand why separating training and validation data is essential
- Learn data splitting strategies: holdout method, k-fold cross-validation, and stratified sampling
- Master techniques to prevent data leakage
- Apply best practices for evaluating model generalization

## Why This Matters

The difference between a model that performs well in development and one that succeeds in production often comes down to proper data splitting. Without careful separation of training and validation data:

- **Overfitting goes undetected**: Your model memorizes training examples but fails on new data
- **Misleading metrics**: You report 99% accuracy in testing, but production performance is 60%
- **Wasted resources**: You deploy a model that doesn't generalize, requiring expensive rework
- **Lost business value**: Customers receive poor predictions, damaging trust and revenue

In production machine learning, proper data splitting is the difference between a model that works in the lab and one that delivers business value. As you master deep learning fundamentals this week, understanding data splitting ensures your models will generalize beyond the training set.

## The Core Problem: Generalization

### What is Generalization?

A model **generalizes** when it performs well on data it has never seen before. This is the entire point of machine learning—we want to make predictions on future, unseen examples.

### The Memorization Trap

Without proper data separation, models can "cheat" by memorizing:

```python
# Example of overfitting
model.fit(data, labels, epochs=1000)
accuracy = model.evaluate(data, labels)  # Same data!
print(f"Accuracy: {accuracy}")  # Could be 100%... but meaningless
```

This is like giving students the exact exam questions during study time—perfect scores don't mean they understand the material.

## Data Splitting Fundamentals

### The Three Datasets

**1. Training Set (60-80% of data)**
- Used to train the model (update weights)
- Model sees these examples during backpropagation
- Largest portion of your data

**2. Validation Set (10-20% of data)**
- Used to tune hyperparameters and evaluate during training
- Model never trains on this data
- Helps detect overfitting
- Guides decisions about architecture, learning rate, regularization

**3. Test Set (10-20% of data)**
- Final evaluation of model performance
- Used only once after all development is complete
- Simulates real-world deployment
- Never influences any training decisions

### Why Three Splits?

**Training vs. Validation:**
- Training: Learn patterns
- Validation: Check if patterns generalize

**Why Not Just Training and Test?**
- If you tune hyperparameters based on test performance, test set becomes part of development
- You need a "held-out held-out" set for final unbiased evaluation

## Data Splitting Strategies

### 1. Holdout Method (Simple Split)

The most common approach: randomly split data once.

```python
from sklearn.model_selection import train_test_split

# Split into training and temporary set
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3,      # 30% for validation + test
    random_state=42     # Reproducibility
)

# Split temporary set into validation and test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,      # Half of 30% = 15% each
    random_state=42
)

# Result: 70% train, 15% validation, 15% test
```

**Advantages:**
- Simple and fast
- Works well with large datasets
- Easy to implement

**Disadvantages:**
- Single split may not be representative
- Small datasets may not have enough validation/test samples
- Performance estimates can be noisy

### 2. K-Fold Cross-Validation

Divide data into K equal parts (folds), train K times using different validation folds.

```python
from sklearn.model_selection import KFold
import numpy as np

# 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    # Split data
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Train model
    model = build_model()
    model.fit(X_train_fold, y_train_fold, epochs=10, verbose=0)
    
    # Evaluate
    score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    scores.append(score)
    print(f"Fold {fold + 1}: Accuracy = {score[1]:.4f}")

# Average performance across folds
print(f"Mean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
```

**Advantages:**
- Every data point used for validation exactly once
- More robust performance estimate
- Reduces variance in performance metrics
- Better for small datasets

**Disadvantages:**
- Computationally expensive (train K models)
- Not ideal for very large datasets
- Requires more time for deep learning models

**When to use:** Small to medium datasets where you need reliable performance estimates.

### 3. Stratified Sampling

Ensures class distribution is preserved across splits—critical for imbalanced datasets.

```python
from sklearn.model_selection import train_test_split

# Imbalanced dataset example
# Class 0: 90% of data
# Class 1: 10% of data

# Without stratification (BAD)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Risk: Validation set might have 0% or 20% of Class 1 (not representative)

# With stratification (GOOD)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,        # Preserve class distribution
    random_state=42
)
# Guarantee: Both sets have ~10% Class 1, ~90% Class 0
```

**When to use:** Classification problems with imbalanced classes.

### 4. Stratified K-Fold Cross-Validation

Combines stratification with cross-validation:

```python
from sklearn.model_selection import StratifiedKFold

# Stratified 5-fold cross-validation
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Each fold maintains class distribution
    # Train and evaluate...
```

**When to use:** Small imbalanced datasets.

## Preventing Data Leakage

### What is Data Leakage?

Information from validation/test sets "leaking" into training, causing overly optimistic performance estimates.

### Common Sources of Leakage

**1. Preprocessing Before Splitting**

```python
# WRONG: Leakage via normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit(X)  # Uses statistics from ALL data

X_train, X_val = train_test_split(X_scaled, test_size=0.2)
# Validation data influenced training data statistics!

# CORRECT: Fit on training data only
X_train, X_val = train_test_split(X, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training only
X_val_scaled = scaler.transform(X_val)          # Apply training statistics
```

**2. Feature Selection Leakage**

```python
# WRONG: Select features using all data
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=10)
X_selected = selector.fit_transform(X, y)  # Sees validation labels!

X_train, X_val = train_test_split(X_selected, test_size=0.2)

# CORRECT: Select features using training data only
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

selector = SelectKBest(k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
```

**3. Time Series Data**

```python
# WRONG: Random split for time series
X_train, X_val = train_test_split(time_series_data, test_size=0.2)
# Future data in training set!

# CORRECT: Chronological split
split_point = int(len(time_series_data) * 0.8)
X_train = time_series_data[:split_point]
X_val = time_series_data[split_point:]
# Training on past, validate on future
```

**4. Duplicate Records**

```python
# WRONG: Duplicates across splits
# If same customer appears in both training and validation, model can memorize

# CORRECT: Group by entity before splitting
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(X, y, groups=customer_ids):
    X_train, X_val = X[train_idx], X[val_idx]
    # No customer appears in both sets
```

## Best Practices in Keras/TensorFlow

### Using Validation Split

```python
# Automatic validation split (simplest)
model.fit(
    X_train, y_train,
    epochs=50,
    validation_split=0.2,  # Use last 20% of training data
    batch_size=32
)
```

**Note:** This takes the last 20% of data, so shuffle first if not time series.

### Using Separate Validation Data

```python
# Explicit validation set (more control)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),  # Explicit validation set
    batch_size=32
)

# Access training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

### Final Test Evaluation

```python
# After all development is complete
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# This is the number you report to stakeholders
```

## Key Takeaways

1. **Always split data into training, validation, and test sets** before any preprocessing or modeling
2. **Training set**: Learn patterns (model sees during training)
3. **Validation set**: Tune hyperparameters and detect overfitting (model evaluates but doesn't train on)
4. **Test set**: Final unbiased evaluation (use only once)
5. **Holdout method** (simple random split) works well for large datasets
6. **K-fold cross-validation** provides robust estimates for small/medium datasets
7. **Stratified sampling** is essential for imbalanced classification problems
8. **Prevent data leakage** by fitting preprocessing only on training data, then transforming validation/test data
9. **Special cases require special splits**: Time series need chronological splits, grouped data needs group-aware splits
10. **Monitor validation metrics during training** to detect overfitting early

## External Resources

- [Scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html) - Comprehensive documentation on data splitting techniques
- [Avoiding Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/) - Common pitfalls and solutions
- [Cross-Validation Strategies](https://scikit-learn.org/stable/modules/cross_validation.html) - Detailed guide to k-fold and stratified methods

