# Exercise 01: TensorBoard Exploration

## Learning Objectives

- Set up TensorBoard callbacks for model training
- Log custom metrics beyond default loss and accuracy
- Compare multiple experiments using TensorBoard's interface
- Interpret training curves to diagnose model behavior

## Duration

**Estimated Time:** 45 minutes

## Background

In `demo_01_tensorboard_setup.py`, you saw how TensorBoard provides real-time visibility into model training. This exercise gives you hands-on practice configuring TensorBoard for your own experiments.

## The Concepts You Need

### TensorBoard Callback Configuration

```python
# TensorBoard callback requires a log directory
# Each experiment should have a unique subdirectory

keras.callbacks.TensorBoard(
    log_dir='logs/experiment_name',  # Where to save logs
    histogram_freq=1,                 # Log weight histograms every N epochs
    write_graph=True,                 # Visualize model architecture
    update_freq='epoch'               # When to write logs
)
```

### Custom Metric Logging

```python
# Use tf.summary to log additional metrics
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
    tf.summary.scalar('metric_name', value, step=epoch)
```

---

## Part 1: Basic TensorBoard Setup (15 min)

### Task 1.1: Create a TensorBoard Callback

Navigate to `starter_code/tensorboard_starter.py` and complete the `create_tensorboard_callback()` function.

**Your implementation should:**
1. Generate a timestamp for unique run identification
2. Create a log directory path using the experiment name and timestamp
3. Return a configured TensorBoard callback

**Hints:**
- Use `datetime.datetime.now().strftime('%Y%m%d-%H%M%S')` for timestamps
- Use `os.path.join('logs', f'{experiment_name}_{timestamp}')` for the path
- Set `histogram_freq=1` to log weight distributions

### Task 1.2: Verify TensorBoard Logs

After implementing, run this test:

```python
(x_train, y_train), (x_test, y_test) = load_mnist_data()
model = create_model()
tb_callback = create_tensorboard_callback("basic_test")
model.fit(x_train, y_train, epochs=5, validation_split=0.2, callbacks=[tb_callback])
```

**Verification:**
1. Check that a `logs/` directory was created
2. Launch TensorBoard: `tensorboard --logdir=logs`
3. Open `http://localhost:6006` and verify you see training curves

---

## Part 2: Custom Metric Logging (15 min)

### Task 2.1: Implement CustomMetricsCallback

Complete the `CustomMetricsCallback` class to log:
- Current learning rate at each epoch
- Loss ratio (training loss / validation loss) as an overfitting indicator

**The callback should:**
1. Create a file writer in `__init__`
2. Extract the learning rate from the optimizer in `on_epoch_end`
3. Calculate the loss ratio (handle division by zero!)
4. Write both metrics using `tf.summary.scalar`

**Hints for getting learning rate:**
```python
# Inside on_epoch_end:
lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
```

**Hints for loss ratio:**
```python
train_loss = logs.get('loss', 0)
val_loss = logs.get('val_loss', 1)  # Default to 1 to avoid division by zero
loss_ratio = train_loss / val_loss if val_loss > 0 else 0
```

### Task 2.2: Test Your Custom Callback

```python
tb_callback = create_tensorboard_callback("custom_metrics_test")
custom_callback = CustomMetricsCallback("logs/custom_metrics_test")
model.fit(x_train, y_train, epochs=10, validation_split=0.2, 
          callbacks=[tb_callback, custom_callback])
```

**Verification in TensorBoard:**
- You should see `learning_rate` and `loss_ratio` in the Scalars tab
- Loss ratio > 1 indicates potential underfitting
- Loss ratio < 1 indicates potential overfitting

---

## Part 3: Experiment Comparison (15 min)

### Task 3.1: Compare Learning Rates

Complete the `compare_learning_rates()` function to run three experiments with different learning rates: 0.001, 0.01, and 0.1.

**Each experiment should:**
1. Create a model with the specified learning rate
2. Use a descriptive experiment name like `lr_0.001`
3. Train for 20 epochs with validation
4. Print the final validation accuracy

### Task 3.2: Analyze in TensorBoard

After running all experiments:

1. Open TensorBoard
2. In the Scalars tab, enable all three runs
3. Compare the training curves

**Questions to Answer:**
- Which learning rate converges fastest?
- Which achieves the highest final accuracy?
- Does any learning rate cause unstable training?

---

## Reflection Questions

Answer these in comments at the end of your code:

1. **What happens to the training curve when learning rate is too high (0.1)?** Describe the behavior you observe.

2. **How does the loss ratio metric help you detect overfitting early?** At what value should you be concerned?

3. **Why do we use unique timestamps in log directories?** What happens if multiple runs share the same directory?

4. **Looking at the Distributions tab, what do healthy weight distributions look like?** What would "dead neurons" look like?

---

## Definition of Done

- [ ] `create_tensorboard_callback()` creates proper log directories
- [ ] `CustomMetricsCallback` logs learning rate and loss ratio
- [ ] Three learning rate experiments completed and logged
- [ ] TensorBoard shows all runs for comparison
- [ ] Reflection questions answered with observations

---

## Solution

The solution is available in `solutions/exercise_01_solution.py`. Only check after completing your attempt!

