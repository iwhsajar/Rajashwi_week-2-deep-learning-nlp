# Exercise 01: Model Persistence Lab

## Learning Objectives

- Save models in different formats (H5, SavedModel, weights-only)
- Load models for inference and continued training
- Understand what each format contains
- Verify model integrity after save/load

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_01_save_load_models.py`, you saw different ways to persist models. This exercise gives you hands-on practice with each format.

## Saving Formats Reference

```
# Format 1: HDF5 (.h5) - Single file, widely compatible
model.save('model.h5')

# Format 2: SavedModel - Directory format, TensorFlow Serving compatible  
model.save('saved_model/')

# Format 3: Weights only - Just the learned parameters
model.save_weights('weights.h5')

# Format 4: Architecture only - JSON specification
json_config = model.to_json()
```

---

## Part 1: Save Complete Models (15 min)

### Task 1.1: Train a Model

Navigate to `starter_code/persistence_starter.py`:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

def train_sample_model():
    """Train a simple MNIST classifier."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=1)
    
    return model, x_test, y_test

model, x_test, y_test = train_sample_model()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Original model test accuracy: {test_acc:.4f}")
```

### Task 1.2: Save in H5 Format

```python
# Save complete model as HDF5
model.save('models/mnist_classifier.h5')

# Check file size
h5_size = os.path.getsize('models/mnist_classifier.h5')
print(f"H5 file size: {h5_size / 1024:.2f} KB")

# Verify contents
# H5 contains: architecture, weights, optimizer state, training config
```

### Task 1.3: Save in SavedModel Format

```python
# Save as SavedModel (directory)
model.save('models/mnist_savedmodel')

# Explore the directory structure
for root, dirs, files in os.walk('models/mnist_savedmodel'):
    level = root.replace('models/mnist_savedmodel', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f'{subindent}{file}')
```

---

## Part 2: Load and Verify Models (15 min)

### Task 2.1: Load H5 Model

```python
# Load complete model from H5
loaded_h5 = keras.models.load_model('models/mnist_classifier.h5')

# Verify architecture matches
loaded_h5.summary()

# Verify predictions match
original_preds = model.predict(x_test[:5])
loaded_preds = loaded_h5.predict(x_test[:5])

print("\nPrediction comparison (should be identical):")
print(f"Max difference: {np.max(np.abs(original_preds - loaded_preds)):.10f}")
```

### Task 2.2: Load SavedModel

```python
# Load from SavedModel directory
loaded_savedmodel = keras.models.load_model('models/mnist_savedmodel')

# Evaluate to verify
saved_loss, saved_acc = loaded_savedmodel.evaluate(x_test, y_test, verbose=0)
print(f"SavedModel test accuracy: {saved_acc:.4f}")
print(f"Matches original: {abs(saved_acc - test_acc) < 0.0001}")
```

### Task 2.3: Load Weights Only

```python
# Save weights only
model.save_weights('models/mnist_weights.h5')

# To load weights, you need to recreate the architecture first
def create_model_architecture():
    """Recreate the same architecture."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create new model and load weights
new_model = create_model_architecture()
new_model.load_weights('models/mnist_weights.h5')

# Compile for evaluation
new_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Verify
weights_loss, weights_acc = new_model.evaluate(x_test, y_test, verbose=0)
print(f"Weights-only model accuracy: {weights_acc:.4f}")
```

---

## Part 3: Advanced Persistence (15 min)

### Task 3.1: Save/Load Custom Objects

If your model has custom layers or loss functions:

```python
# Define a custom layer
class CustomDenseLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Build model with custom layer
custom_model = keras.Sequential([
    CustomDenseLayer(64, input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

custom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
custom_model.fit(x_train[:1000], y_train[:1000], epochs=2, verbose=0)

# Save
custom_model.save('models/custom_model.h5')

# Load with custom objects
loaded_custom = keras.models.load_model(
    'models/custom_model.h5',
    custom_objects={'CustomDenseLayer': CustomDenseLayer}
)
```

### Task 3.2: Continue Training from Saved Model

```python
# Load a saved model
model_to_continue = keras.models.load_model('models/mnist_classifier.h5')

# Continue training
print("Continuing training...")
history = model_to_continue.fit(
    x_train, y_train,
    epochs=3,
    validation_split=0.2,
    verbose=1
)

# Evaluate improvement
final_loss, final_acc = model_to_continue.evaluate(x_test, y_test, verbose=0)
print(f"After additional training: {final_acc:.4f}")
print(f"Improvement: {(final_acc - test_acc) * 100:.2f}%")
```

---

## Comparison Table

Fill in after completing exercises:

| Format | File Size | Contains | Best For |
|--------|-----------|----------|----------|
| H5 (.h5) | | | |
| SavedModel | | | |
| Weights only | | | |
| JSON (architecture) | | | |

---

## Reflection Questions

1. **When would you use H5 vs SavedModel format?** Consider deployment scenarios.

2. **Why might you save weights separately from architecture?** Think about experimentation workflows.

3. **What happens if you try to load weights into a model with different architecture?**

4. **How would you version control your models?** What about very large models?

---

## Definition of Done

- [ ] Saved model in H5 format
- [ ] Saved model in SavedModel format
- [ ] Saved and loaded weights only
- [ ] Verified predictions match after loading
- [ ] Continued training from saved checkpoint
- [ ] Comparison table completed
- [ ] Reflection questions answered

---

## Solution

The solution is available in `solutions/exercise_01_solution.py`. Only check after completing your attempt!

