# TensorBoard Visualization

## Learning Objectives
- Understand what TensorBoard is and why it's essential for deep learning development
- Learn how to set up TensorBoard in TensorFlow/Keras projects
- Master logging metrics, visualizing training graphs, and comparing experiments
- Apply TensorBoard to debug and optimize neural network training

## Why This Matters

When training neural networks, you're essentially navigating in a high-dimensional space trying to find optimal parameters. Without proper visualization tools, this process is like flying blind. TensorBoard is TensorFlow's visualization toolkit that transforms abstract training metrics into actionable insights.

In production machine learning environments, TensorBoard is indispensable for:
- **Debugging training issues**: Identify exploding/vanishing gradients, learning rate problems, or data pipeline bottlenecks
- **Experiment tracking**: Compare dozens of model configurations to find the best architecture
- **Team collaboration**: Share visual training progress with stakeholders who may not be data scientists
- **Model optimization**: Make data-driven decisions about when to stop training or adjust hyperparameters

This week's epic focuses on mastering deep learning fundamentals—TensorBoard is your window into understanding how your models actually learn.

## The Concept

### What is TensorBoard?

TensorBoard is a web-based application that visualizes machine learning experiments. It reads log files generated during model training and presents them through interactive dashboards. Think of it as your mission control center for neural network development.

### Core Components

**1. Scalars Dashboard**
Tracks single numerical values over time, such as:
- Training and validation loss
- Accuracy metrics
- Learning rate changes
- Custom metrics you define

**2. Graphs Dashboard**
Visualizes your model's computational graph, showing:
- Layer connections and data flow
- Tensor shapes at each operation
- Device placement (CPU/GPU)

**3. Distributions and Histograms**
Shows how weights, biases, and activations evolve during training:
- Weight distributions across layers
- Activation patterns
- Gradient magnitudes

**4. Images Dashboard**
Displays image data, useful for:
- Monitoring input data quality
- Visualizing model predictions
- Tracking image generation (autoencoders, GANs)

### How TensorBoard Works

The workflow follows three steps:

1. **Logging**: During training, your code writes summary data to log files
2. **Running TensorBoard**: Launch the TensorBoard server pointing to your log directory
3. **Visualization**: Access the web interface to explore your training metrics

## Setting Up TensorBoard

### Installation

TensorBoard comes bundled with TensorFlow:

```python
# TensorBoard is included with TensorFlow
import tensorflow as tf
from tensorflow import keras
```

### Basic Integration with Keras

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Create a unique log directory with timestamp
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Configure TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,        # Log weight histograms every epoch
    write_graph=True,        # Visualize the model graph
    write_images=False,      # Save model weights as images
    update_freq='epoch'      # Log metrics after each epoch
)

# Use the callback during training
model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[tensorboard_callback]
)
```

### Launching TensorBoard

```bash
# From command line, navigate to your project directory
tensorboard --logdir=logs/fit

# TensorBoard will start on http://localhost:6006
```

## Logging Custom Metrics

Beyond default metrics, you can log custom values:

```python
import tensorflow as tf

# Create a file writer
log_dir = "logs/custom/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)

# Log custom scalar values
with summary_writer.as_default():
    for step in range(100):
        # Your training logic here
        custom_metric = compute_custom_metric()
        
        tf.summary.scalar('custom_metric', custom_metric, step=step)
        summary_writer.flush()
```

## Comparing Experiments

One of TensorBoard's most powerful features is experiment comparison:

```python
# Experiment 1: Baseline model
log_dir_1 = "logs/experiment_baseline"
tensorboard_callback_1 = TensorBoard(log_dir=log_dir_1)

# Experiment 2: With dropout
log_dir_2 = "logs/experiment_dropout"
tensorboard_callback_2 = TensorBoard(log_dir=log_dir_2)

# Launch TensorBoard with both directories
# tensorboard --logdir=logs
```

TensorBoard will overlay multiple runs, allowing you to compare:
- Which architecture converges faster
- Which hyperparameters prevent overfitting
- How different optimizers affect training stability

## Practical Example: Training Visualization

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime

# Build a simple model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Set up TensorBoard
log_dir = "logs/mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True
)

# Train with TensorBoard logging
history = model.fit(
    x_train, y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[tensorboard_callback],
    verbose=1
)
```

After running this code, you can visualize:
- Training vs. validation loss curves (detecting overfitting)
- Accuracy improvements over epochs
- Weight distributions in the Dense layers
- The complete model architecture graph

## Key Takeaways

1. **TensorBoard is essential for understanding neural network training dynamics**—it turns abstract numbers into visual insights
2. **Integration is simple**: Just add the TensorBoard callback to your `model.fit()` call
3. **Use unique log directories** (with timestamps) to organize different experiments
4. **Compare experiments side-by-side** by pointing TensorBoard to a parent directory containing multiple runs
5. **Monitor multiple metrics simultaneously**: loss, accuracy, custom metrics, weight distributions
6. **TensorBoard runs independently** from your training code—you can explore visualizations while training continues

## External Resources

- [TensorBoard Official Documentation](https://www.tensorflow.org/tensorboard) - Comprehensive guide from TensorFlow team
- [TensorBoard: Visualizing Learning](https://www.tensorflow.org/tensorboard/get_started) - Getting started tutorial with examples
- [Advanced TensorBoard Usage](https://www.tensorflow.org/tensorboard/scalars_and_keras) - Custom logging and advanced features

