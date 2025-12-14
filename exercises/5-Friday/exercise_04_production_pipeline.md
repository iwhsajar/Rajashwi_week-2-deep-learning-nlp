# Exercise 04: Production Training Pipeline

## Learning Objectives

- Build a complete, production-ready training pipeline
- Integrate checkpointing, early stopping, and logging
- Implement model versioning and experiment tracking
- Create reproducible training runs

## Duration

**Estimated Time:** 60 minutes

## Background

This exercise synthesizes everything from Week 2 Friday: model persistence, checkpointing, early stopping, and regularization. You'll build a reusable `ProductionTrainer` class that handles all aspects of training a production model.

---

## Part 1: Project Structure Setup (10 min)

### Task 1.1: Design the Output Directory Structure

A production pipeline needs organized outputs:

```
production_runs/
    mnist_classifier_20240115_143022/
        checkpoints/
            best_model.keras
            checkpoint_epoch_10.keras
        logs/
            tensorboard/
            training_log.csv
        models/
            final_model.keras
            final_model.weights.h5
        config.json
        summary.json
```

### Task 1.2: Implement Directory Setup

Open `starter_code/production_pipeline_starter.py` and complete:

```python
class ProductionTrainer:
    def __init__(self, model_name, output_dir='production_runs'):
        self.model_name = model_name
        self.output_dir = output_dir
        
        # TODO: Create timestamped run directory
        # self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # self.run_dir = os.path.join(output_dir, f'{model_name}_{self.timestamp}')
        
        # TODO: Define subdirectories
        # self.checkpoint_dir = ...
        # self.log_dir = ...
        # self.model_dir = ...
    
    def _setup_directories(self):
        # TODO: Create all directories with os.makedirs(exist_ok=True)
        pass
```

---

## Part 2: Model Building with Regularization (15 min)

### Task 2.1: Configurable Model Builder

Build models with configurable regularization:

```python
def build_model(self, input_shape, num_classes, 
                hidden_layers=[128, 64], 
                dropout_rate=0.3, 
                l2_lambda=0.001):
    """
    Build model with:
    - Configurable hidden layers
    - Dropout after each layer
    - L2 regularization on all weights
    - Appropriate output activation
    """
    # TODO: Store config for reproducibility
    self.config = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'l2_lambda': l2_lambda
    }
    
    # TODO: Build Sequential model
    # For each hidden layer:
    #   - Dense with L2 regularization
    #   - Dropout
    # Output layer:
    #   - sigmoid for binary (num_classes=2)
    #   - softmax for multi-class
    pass
```

### Task 2.2: Store Configuration

Ensure all hyperparameters are saved for reproducibility:

```python
def _save_config(self):
    """Save config to JSON for reproducibility."""
    config_path = os.path.join(self.run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(self.config, f, indent=2)
    print(f"Config saved to: {config_path}")
```

---

## Part 3: Callback Suite (15 min)

### Task 3.1: Create Production Callbacks

Implement a comprehensive callback suite:

```python
def _create_callbacks(self, patience=10):
    """
    Create production-grade callbacks:
    1. ModelCheckpoint - save best model
    2. ModelCheckpoint - periodic saves (every 5 epochs)
    3. EarlyStopping - prevent overfitting
    4. ReduceLROnPlateau - adaptive learning rate
    5. TensorBoard - visualization
    6. CSVLogger - tabular metrics log
    """
    callbacks = []
    
    # 1. Best model checkpoint
    # TODO: checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.keras')
    # callbacks.append(ModelCheckpoint(..., save_best_only=True))
    
    # 2. Periodic checkpoint (every 5 epochs)
    # TODO: periodic_path = os.path.join(self.checkpoint_dir, 'checkpoint_{epoch:02d}.keras')
    # callbacks.append(ModelCheckpoint(..., save_freq=5*steps_per_epoch))
    
    # 3. Early stopping
    # TODO: callbacks.append(EarlyStopping(patience=patience, restore_best_weights=True))
    
    # 4. Learning rate reduction
    # TODO: callbacks.append(ReduceLROnPlateau(factor=0.5, patience=patience//2))
    
    # 5. TensorBoard
    # TODO: callbacks.append(TensorBoard(log_dir=self.log_dir))
    
    # 6. CSV Logger
    # TODO: csv_path = os.path.join(self.log_dir, 'training_log.csv')
    # callbacks.append(CSVLogger(csv_path))
    
    return callbacks
```

---

## Part 4: Training and Saving (15 min)

### Task 4.1: Complete Training Method

```python
def train(self, x_train, y_train, x_val, y_val, 
          epochs=100, batch_size=32, patience=10):
    """
    Run complete training pipeline.
    """
    # Setup
    self._setup_directories()
    self._save_config()
    
    callbacks = self._create_callbacks(patience)
    
    # Train
    print(f"\n{'='*50}")
    print(f"Starting training: {self.model_name}")
    print(f"Run directory: {self.run_dir}")
    print(f"{'='*50}\n")
    
    self.history = self.model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return self.history
```

### Task 4.2: Save Final Artifacts

```python
def save_final_model(self):
    """
    Save all final artifacts:
    - Complete model (.keras)
    - Weights only (.weights.h5)
    - Training summary
    """
    # TODO: Save complete model
    # final_path = os.path.join(self.model_dir, 'final_model.keras')
    # self.model.save(final_path)
    
    # TODO: Save weights only
    # weights_path = os.path.join(self.model_dir, 'final_model.weights.h5')
    # self.model.save_weights(weights_path)
    
    # TODO: Save training summary
    # summary = {
    #     'final_train_acc': float(self.history.history['accuracy'][-1]),
    #     'final_val_acc': float(self.history.history['val_accuracy'][-1]),
    #     'best_val_acc': float(max(self.history.history['val_accuracy'])),
    #     'epochs_trained': len(self.history.history['loss']),
    #     'stopped_early': len(self.history.history['loss']) < epochs,
    #     'timestamp': self.timestamp
    # }
    # summary_path = os.path.join(self.run_dir, 'summary.json')
    # with open(summary_path, 'w') as f:
    #     json.dump(summary, f, indent=2)
    pass
```

---

## Part 5: Full Pipeline Test (5 min)

### Task 5.1: Run Complete Pipeline

```python
def test_production_pipeline():
    """Test the complete production pipeline."""
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Use subset for faster testing
    x_train, y_train = x_train[:5000], y_train[:5000]
    x_val, y_val = x_test[:1000], y_test[:1000]
    
    # Create trainer
    trainer = ProductionTrainer(model_name='mnist_classifier')
    
    # Build model
    trainer.build_model(
        input_shape=(784,),
        num_classes=10,
        hidden_layers=[256, 128, 64],
        dropout_rate=0.3,
        l2_lambda=0.001
    )
    
    # Compile
    trainer.compile_model(learning_rate=0.001)
    
    # Train
    trainer.train(
        x_train, y_train,
        x_val, y_val,
        epochs=50,
        batch_size=128,
        patience=10
    )
    
    # Save
    trainer.save_final_model()
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    trainer.load_best_model()
    loss, acc = trainer.model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    
    # List created files
    print("\nCreated files:")
    for root, dirs, files in os.walk(trainer.run_dir):
        level = root.replace(trainer.run_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")

if __name__ == "__main__":
    test_production_pipeline()
```

---

## Results Checklist

After running, verify these files exist:

```
[ ] production_runs/{model_name}_{timestamp}/
    [ ] config.json
    [ ] summary.json
    [ ] checkpoints/
        [ ] best_model.keras
    [ ] logs/
        [ ] training_log.csv
        [ ] (TensorBoard files)
    [ ] models/
        [ ] final_model.keras
        [ ] final_model.weights.h5
```

---

## Reflection Questions

1. **Why save both the complete model and weights separately?** When would you use each?

2. **What happens if training is interrupted mid-epoch?** How would you add crash recovery?

3. **How would you extend this pipeline for hyperparameter tuning?** (multiple runs with different configs)

4. **What additional monitoring would you add for production?** (GPU utilization, memory, etc.)

5. **How would you deploy the best model from this pipeline?** What format would you export?

---

## Definition of Done

- [ ] ProductionTrainer class fully implemented
- [ ] All directories created correctly
- [ ] Config saved as JSON
- [ ] Best model checkpoint saved
- [ ] CSV training log generated
- [ ] TensorBoard logs viewable
- [ ] Final model and weights saved
- [ ] Summary JSON with training stats
- [ ] Test accuracy reported
- [ ] Reflection questions answered

---

## Bonus Challenge

Add MLflow integration for experiment tracking:

```python
import mlflow

def train_with_mlflow(self, ...):
    """Train with MLflow tracking."""
    with mlflow.start_run(run_name=self.model_name):
        # Log parameters
        mlflow.log_params(self.config)
        
        # Train
        history = self.model.fit(...)
        
        # Log metrics
        mlflow.log_metrics({
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        })
        
        # Log model
        mlflow.keras.log_model(self.model, "model")
```

---

## Solution

The solution is available in `solutions/exercise_04_solution.py`. Only check after completing your attempt!

