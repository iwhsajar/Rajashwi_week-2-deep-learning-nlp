# Customizing the Training Process

## Learning Objectives
- Understand when and why to move beyond `model.fit()` with default settings
- Learn to implement custom training loops with manual gradient computation
- Master callbacks for learning rate schedules, early stopping, and custom logging
- Explore advanced optimizers and their hyperparameters

## Why This Matters

While `model.fit()` with default settings works for many problems, production machine learning often requires fine-grained control over training. Understanding how to customize training is critical because:

- **Complex loss functions**: Multi-task learning, adversarial training, and reinforcement learning require custom loss computations
- **Non-standard architectures**: GANs, Siamese networks, and meta-learning models need specialized training loops
- **Performance optimization**: Learning rate schedules, gradient clipping, and mixed precision training accelerate convergence
- **Monitoring and debugging**: Custom metrics, intermediate visualizations, and gradient inspections reveal training dynamics
- **Production requirements**: Checkpointing, distributed training, and resource management demand control over the training process

As you deepen your understanding of neural network optimization this week, customizing training transforms you from a user of deep learning frameworks into an architect who can implement novel algorithms and optimize for specific production constraints.

## Beyond Basic Training

### The Standard Approach

```python
# Simple, but limited control
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)
```

**What's happening under the hood:**
1. Loop over epochs
2. Shuffle training data
3. Create mini-batches
4. For each batch: forward pass → compute loss → backpropagation → weight update
5. Evaluate on validation data
6. Repeat

### When to Customize

**Callbacks** (easier):
- Learning rate schedules
- Early stopping
- Model checkpointing
- TensorBoard logging
- Custom metrics

**Custom training loops** (more control):
- Non-standard loss functions (GAN discriminator/generator, contrastive loss)
- Multiple models training together
- Gradient accumulation for large effective batch sizes
- Custom regularization schemes
- Advanced debugging and visualization

## Callbacks: Extending `model.fit()`

### What are Callbacks?

Callbacks are functions called at specific points during training:
- Beginning/end of training
- Beginning/end of each epoch
- Beginning/end of each batch

### Built-in Callbacks

**1. ModelCheckpoint: Save Best Models**

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_loss',        # Metric to monitor
    save_best_only=True,       # Only save when metric improves
    mode='min',                # 'min' for loss, 'max' for accuracy
    verbose=1
)

model.fit(X_train, y_train, epochs=50, callbacks=[checkpoint])
# Automatically saves the best model based on validation loss
```

**2. EarlyStopping: Prevent Overfitting**

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,               # Wait 10 epochs for improvement
    restore_best_weights=True, # Restore weights from best epoch
    verbose=1
)

model.fit(X_train, y_train, epochs=100, callbacks=[early_stop])
# Stops training if validation loss doesn't improve for 10 epochs
```

**3. ReduceLROnPlateau: Adaptive Learning Rate**

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                # Multiply LR by 0.5
    patience=5,                # Wait 5 epochs before reducing
    min_lr=1e-7,               # Don't go below this LR
    verbose=1
)

model.fit(X_train, y_train, epochs=50, callbacks=[reduce_lr])
# Reduces learning rate when validation loss plateaus
```

**4. TensorBoard: Real-time Visualization**

```python
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,          # Log weight histograms every epoch
    write_graph=True
)

model.fit(X_train, y_train, epochs=50, callbacks=[tensorboard])
# Launch TensorBoard: tensorboard --logdir=logs/fit
```

**5. Combining Multiple Callbacks**

```python
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss'),
    TensorBoard(log_dir='logs/fit')
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)
```

### Custom Callbacks

Create your own callback for specialized logging or behavior:

```python
from tensorflow.keras.callbacks import Callback

class CustomLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        print(f"\nEpoch {epoch + 1} complete!")
        print(f"Training loss: {logs['loss']:.4f}")
        print(f"Validation loss: {logs['val_loss']:.4f}")
        
        # Custom logic: Save predictions on validation set
        if epoch % 5 == 0:  # Every 5 epochs
            val_predictions = self.model.predict(X_val)
            # Save or analyze predictions
    
    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch"""
        if batch % 100 == 0:
            print(f"Batch {batch} - Loss: {logs['loss']:.4f}")

custom_callback = CustomLoggingCallback()
model.fit(X_train, y_train, epochs=50, callbacks=[custom_callback])
```

**Common custom callback uses:**
- Log to external systems (MLflow, Weights & Biases)
- Send alerts when training stalls
- Dynamically adjust hyperparameters
- Visualize intermediate predictions

## Learning Rate Schedules

### Time-Based Decay

Learning rate decreases over time:

```python
from tensorflow.keras.optimizers.schedules import ExponentialDecay

initial_learning_rate = 0.1
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True  # Apply decay in discrete steps
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')
```

### Step Decay

Reduce learning rate at specific epochs:

```python
def step_decay(epoch):
    initial_lr = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))
    return lr

from tensorflow.keras.callbacks import LearningRateScheduler
lr_scheduler = LearningRateScheduler(step_decay, verbose=1)

model.fit(X_train, y_train, epochs=50, callbacks=[lr_scheduler])
```

### Cosine Annealing

Learning rate follows a cosine curve:

```python
from tensorflow.keras.optimizers.schedules import CosineDecay

lr_schedule = CosineDecay(
    initial_learning_rate=0.1,
    decay_steps=10000
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

### Warm-Up + Decay

Start with low learning rate, increase, then decay:

```python
class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, decay_steps):
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
    
    def __call__(self, step):
        # Linear warm-up
        warmup_lr = self.initial_lr * (step / self.warmup_steps)
        
        # Exponential decay after warm-up
        decay_lr = self.initial_lr * (0.96 ** ((step - self.warmup_steps) / self.decay_steps))
        
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: decay_lr
        )

lr_schedule = WarmUpSchedule(initial_lr=0.001, warmup_steps=1000, decay_steps=10000)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
```

## Custom Training Loops

### When to Use Custom Loops

- Training GANs (alternating discriminator/generator)
- Multi-task learning with complex loss combinations
- Reinforcement learning
- Meta-learning and few-shot learning
- Advanced debugging and gradient inspection

### Basic Custom Training Loop

```python
import tensorflow as tf

# Model, loss, optimizer
model = create_model()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# Training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss(loss)
    train_accuracy(y, predictions)

# Training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    # Reset metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # Iterate over batches
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    print(f"Epoch {epoch + 1}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}")
```

### Custom Training with Validation

```python
# Validation step
@tf.function
def val_step(x, y):
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    val_loss(loss)
    val_accuracy(y, predictions)

# Metrics for validation
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

# Training loop with validation
for epoch in range(EPOCHS):
    # Reset metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()
    
    # Training
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    # Validation
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)
    
    print(f"Epoch {epoch + 1}")
    print(f"  Train Loss: {train_loss.result():.4f}, Train Accuracy: {train_accuracy.result():.4f}")
    print(f"  Val Loss: {val_loss.result():.4f}, Val Accuracy: {val_accuracy.result():.4f}")
```

### Gradient Clipping

Prevent exploding gradients:

```python
@tf.function
def train_step_with_clipping(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Clip gradients by norm
    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=1.0)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, predictions)
```

### Gradient Accumulation (Simulate Larger Batch Size)

```python
accumulation_steps = 4  # Accumulate gradients over 4 batches

@tf.function
def train_step_with_accumulation(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions) / accumulation_steps  # Scale loss
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Accumulate gradients
    for i, grad in enumerate(gradients):
        accumulated_gradients[i].assign_add(grad)

# Initialize accumulated gradients
accumulated_gradients = [tf.Variable(tf.zeros_like(var), trainable=False) 
                         for var in model.trainable_variables]

# Training loop
for epoch in range(EPOCHS):
    for step, (x_batch, y_batch) in enumerate(train_dataset):
        train_step_with_accumulation(x_batch, y_batch)
        
        # Apply accumulated gradients every N steps
        if (step + 1) % accumulation_steps == 0:
            optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
            
            # Reset accumulated gradients
            for grad in accumulated_gradients:
                grad.assign(tf.zeros_like(grad))
```

## Advanced Optimizers

### Tuning Adam Hyperparameters

```python
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,          # Momentum decay (first moment)
    beta_2=0.999,        # RMSprop decay (second moment)
    epsilon=1e-7,        # Numerical stability
    amsgrad=False        # AMSGrad variant (more stable, slower)
)
```

### Comparing Optimizers

```python
optimizers_to_test = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'SGD+Momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
}

results = {}
for name, opt in optimizers_to_test.items():
    model = create_model()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=0)
    results[name] = history.history
    print(f"{name}: Final Val Accuracy = {history.history['val_accuracy'][-1]:.4f}")
```

## Key Takeaways

1. **Callbacks extend `model.fit()`** without requiring custom training loops—use for checkpointing, early stopping, and learning rate schedules
2. **ModelCheckpoint saves best models** based on validation metrics
3. **EarlyStopping prevents overfitting** by halting training when validation performance stops improving
4. **Learning rate schedules** (exponential decay, step decay, cosine annealing) improve convergence
5. **Custom callbacks** enable specialized logging, alerts, and dynamic hyperparameter adjustment
6. **Custom training loops** provide full control for complex architectures (GANs, multi-task learning)
7. **Gradient clipping prevents exploding gradients** in deep networks or RNNs
8. **Gradient accumulation simulates larger batch sizes** when GPU memory is limited
9. **Adam is the default optimizer** for most tasks, but experiment with SGD+Momentum for some problems
10. **`@tf.function` decorator** compiles training steps to graph mode for faster execution

## External Resources

- [Keras Callbacks API](https://keras.io/api/callbacks/) - Complete list of built-in callbacks
- [Custom Training Loops Guide](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) - Official TensorFlow tutorial
- [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186) - Research paper on advanced learning rate schedules

