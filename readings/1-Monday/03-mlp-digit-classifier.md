# Building a Multi-Layer Perceptron Digit Classifier

## Learning Objectives
- Understand the architecture of Multi-Layer Perceptrons (MLPs) for classification tasks
- Learn how to design layer configurations for the MNIST digit recognition problem
- Master the process of training, evaluating, and optimizing neural network classifiers
- Recognize the relationship between architecture choices and model performance

## Why This Matters

The MNIST digit classification problem is the "Hello World" of deep learning, but don't let its simplicity fool you. Building an effective MLP digit classifier teaches fundamental skills that transfer directly to production systems:

- **Architecture design**: How to choose the number of layers, neurons, and activation functions
- **Performance evaluation**: Understanding accuracy, loss, and the bias-variance tradeoff
- **Hyperparameter tuning**: Systematically improving model performance
- **Classification fundamentals**: Concepts that apply to fraud detection, medical diagnosis, customer segmentation, and countless other business problems

As you master deep learning fundamentals this week, the MLP digit classifier serves as your training ground for understanding how neural networks transform raw pixels into intelligent predictions.

## The MNIST Dataset

### What is MNIST?

MNIST (Modified National Institute of Standards and Technology) contains:
- 60,000 training images of handwritten digits (0-9)
- 10,000 test images
- Each image is 28×28 pixels in grayscale
- Labeled with the correct digit

This dataset is perfect for learning because:
- Small enough to train quickly on CPU
- Complex enough to require neural networks
- Well-established baseline performance for comparison
- Widely used in research and education

### Loading MNIST in Keras

```python
from tensorflow import keras
import numpy as np

# Load dataset (automatically downloads if needed)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Examine shapes
print(f"Training data: {x_train.shape}")    # (60000, 28, 28)
print(f"Training labels: {y_train.shape}")  # (60000,)
print(f"Test data: {x_test.shape}")         # (10000, 28, 28)
```

## MLP Architecture for MNIST

### Why Multi-Layer?

A single-layer perceptron can only learn linear decision boundaries. For complex patterns like handwritten digits, we need:
- **Hidden layers** to learn hierarchical features
- **Non-linear activations** to capture curved decision boundaries
- **Multiple neurons** to represent different aspects of digit shapes

### Architecture Design

**Input Layer:**
- 28×28 = 784 input neurons (one per pixel)
- Flatten 2D images into 1D vectors

**Hidden Layers:**
- First hidden layer: 128-512 neurons (learn basic features)
- Second hidden layer: 64-256 neurons (learn complex patterns)
- Activation function: ReLU (Rectified Linear Unit)

**Output Layer:**
- 10 neurons (one per digit class: 0-9)
- Activation function: Softmax (produces probability distribution)

### Building the Model

```python
from tensorflow.keras import layers, models

# Preprocessing: normalize and flatten
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# Convert labels to categorical (one-hot encoding)
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

# Build MLP architecture
model = models.Sequential([
    layers.Input(shape=(784,)),
    
    # First hidden layer
    layers.Dense(128, activation='relu', name='hidden_1'),
    
    # Second hidden layer
    layers.Dense(64, activation='relu', name='hidden_2'),
    
    # Output layer
    layers.Dense(10, activation='softmax', name='output')
])

# Display architecture
model.summary()
```

**Model Summary Output:**

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
hidden_1 (Dense)            (None, 128)                100480    
hidden_2 (Dense)            (None, 64)                 8256      
output (Dense)              (None, 10)                 650       
=================================================================
Total params: 109,386
Trainable params: 109,386
```

### Understanding the Parameter Count

- **Hidden Layer 1**: (784 inputs × 128 neurons) + 128 biases = 100,480 parameters
- **Hidden Layer 2**: (128 inputs × 64 neurons) + 64 biases = 8,256 parameters
- **Output Layer**: (64 inputs × 10 neurons) + 10 biases = 650 parameters

## Configuring Training

### Loss Function

For multi-class classification with one-hot encoded labels:

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # For one-hot encoded labels
    metrics=['accuracy']
)
```

Alternatively, with integer labels:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)
```

### Training the Model

```python
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.2,  # Use 20% of training data for validation
    verbose=1
)
```

**Training Output Example:**

```
Epoch 1/10
375/375 [==============================] - 2s 5ms/step - loss: 0.2714 - accuracy: 0.9223 - val_loss: 0.1345 - val_accuracy: 0.9595
Epoch 2/10
375/375 [==============================] - 2s 4ms/step - loss: 0.1145 - accuracy: 0.9661 - val_loss: 0.1012 - val_accuracy: 0.9697
...
```

## Performance Evaluation

### Test Set Evaluation

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Typical results: 97-98% accuracy
```

### Making Predictions

```python
# Predict on test images
predictions = model.predict(x_test[:10])

# Get predicted class (highest probability)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test[:10]

# Display results
for i in range(10):
    print(f"Image {i}: Predicted={predicted_classes[i]}, True={true_classes[i]}")
```

### Analyzing Errors

```python
# Find misclassified examples
predictions_all = model.predict(x_test)
predicted_classes_all = np.argmax(predictions_all, axis=1)
misclassified = np.where(predicted_classes_all != y_test)[0]

print(f"Misclassified: {len(misclassified)} out of {len(y_test)}")
print(f"Error rate: {len(misclassified) / len(y_test) * 100:.2f}%")
```

## Architecture Optimization

### Experimenting with Layer Configuration

**Deeper Network (More Layers):**

```python
model_deep = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# May achieve slightly better accuracy but trains slower
```

**Wider Network (More Neurons):**

```python
model_wide = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# More capacity but risk of overfitting
```

**Adding Regularization:**

```python
model_regularized = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),  # Prevent overfitting
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
```

### Key Design Considerations

1. **Layer depth**: 2-3 hidden layers usually sufficient for MNIST
2. **Layer width**: 64-256 neurons per layer balances capacity and efficiency
3. **Activation functions**: ReLU for hidden layers, softmax for output
4. **Regularization**: Add dropout if validation accuracy plateaus while training improves

## Key Takeaways

1. **MLPs transform raw pixels into predictions** through learned feature hierarchies in hidden layers
2. **Architecture matters**: Number of layers, neurons per layer, and activation functions significantly impact performance
3. **Typical MNIST MLP achieves 97-98% accuracy** with 2-3 hidden layers
4. **Input preprocessing is crucial**: Normalize pixel values (0-1 range) and flatten images
5. **Softmax output layer** produces probability distribution over 10 digit classes
6. **Validation data** helps detect overfitting—if validation accuracy stops improving while training accuracy increases, add regularization
7. **This architecture generalizes**: The same principles apply to classifying medical images, detecting defects, or categorizing customer behavior

## External Resources

- [MNIST Database Official Page](http://yann.lecun.com/exdb/mnist/) - Original dataset documentation and benchmarks
- [Neural Networks and Deep Learning - Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) - Excellent visual explanation of digit classification
- [Keras Sequential Model Guide](https://keras.io/guides/sequential_model/) - Official documentation on building MLPs

