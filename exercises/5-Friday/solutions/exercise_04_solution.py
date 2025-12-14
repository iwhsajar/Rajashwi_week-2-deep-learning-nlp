"""
Exercise 04 Solution: Production Training Pipeline

Complete implementation of a production-ready training pipeline.
"""

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger, TerminateOnNaN
)

# =============================================================================
# PRODUCTION TRAINER CLASS
# =============================================================================

class ProductionTrainer:
    """
    Production-ready training pipeline with:
    - Automatic checkpointing
    - Early stopping
    - TensorBoard logging
    - Model versioning
    - Regularization (dropout + L2)
    """
    
    def __init__(self, model_name, output_dir='production_runs'):
        """
        Initialize the trainer.
        
        Args:
            model_name: Name for this model/experiment
            output_dir: Base directory for all outputs
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Create versioned run directory
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(output_dir, f'{model_name}_{self.timestamp}')
        
        # Subdirectories
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.log_dir = os.path.join(self.run_dir, 'logs')
        self.model_dir = os.path.join(self.run_dir, 'models')
        
        self.model = None
        self.history = None
        self.config = {}
    
    def _setup_directories(self):
        """Create all necessary directories."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        print(f"Created run directory: {self.run_dir}")
    
    def build_model(self, input_shape, num_classes, 
                    hidden_layers=[128, 64], 
                    dropout_rate=0.3, 
                    l2_lambda=0.001):
        """
        Build model with regularization.
        
        Args:
            input_shape: Input shape tuple
            num_classes: Number of output classes
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout probability
            l2_lambda: L2 regularization strength
        """
        # Store config
        self.config = {
            'input_shape': input_shape,
            'num_classes': num_classes,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'l2_lambda': l2_lambda,
            'model_name': self.model_name,
            'timestamp': self.timestamp
        }
        
        # Build model
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        
        for units in hidden_layers:
            model.add(layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_lambda)
            ))
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(num_classes, activation='softmax'))
        
        self.model = model
        print(f"Model built with {model.count_params():,} parameters")
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model."""
        if self.model is None:
            raise ValueError("Must build model before compiling")
        
        num_classes = self.config.get('num_classes', 2)
        if num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'sparse_categorical_crossentropy'
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.config['learning_rate'] = learning_rate
        self.config['loss_function'] = loss
        print(f"Model compiled with {loss}")
    
    def _create_callbacks(self, patience=10):
        """Create production callbacks."""
        callbacks = []
        
        # 1. Checkpoint - save best model
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.keras')
        callbacks.append(ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ))
        
        # 2. Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        # 3. TensorBoard
        callbacks.append(TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,
            write_graph=True
        ))
        
        # 4. ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1
        ))
        
        # 5. CSV Logger
        csv_path = os.path.join(self.log_dir, 'training_log.csv')
        callbacks.append(CSVLogger(csv_path))
        
        # 6. Terminate on NaN (safety)
        callbacks.append(TerminateOnNaN())
        
        return callbacks
    
    def train(self, x_train, y_train, x_val, y_val, 
              epochs=100, batch_size=32, patience=10):
        """
        Train the model with all production features.
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
    
    def _save_config(self):
        """Save configuration to JSON."""
        config_path = os.path.join(self.run_dir, 'config.json')
        
        # Convert tuple to list for JSON serialization
        config_to_save = self.config.copy()
        if 'input_shape' in config_to_save:
            config_to_save['input_shape'] = list(config_to_save['input_shape'])
        
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=2)
        print(f"Config saved to: {config_path}")
    
    def save_final_model(self):
        """Save final model and training artifacts."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save final model (complete)
        final_path = os.path.join(self.model_dir, 'final_model.keras')
        self.model.save(final_path)
        print(f"Saved final model to: {final_path}")
        
        # Save weights only
        weights_path = os.path.join(self.model_dir, 'final_model.weights.h5')
        self.model.save_weights(weights_path)
        print(f"Saved weights to: {weights_path}")
        
        # Save training summary
        if self.history:
            summary = {
                'model_name': self.model_name,
                'timestamp': self.timestamp,
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_train_acc': float(self.history.history['accuracy'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'final_val_acc': float(self.history.history['val_accuracy'][-1]),
                'best_val_loss': float(min(self.history.history['val_loss'])),
                'best_val_acc': float(max(self.history.history['val_accuracy'])),
                'epochs_trained': len(self.history.history['loss']),
                'total_parameters': self.model.count_params()
            }
            
            summary_path = os.path.join(self.run_dir, 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved summary to: {summary_path}")
    
    def load_best_model(self):
        """Load the best checkpoint."""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.keras')
        if os.path.exists(best_path):
            self.model = keras.models.load_model(best_path)
            print(f"Loaded best model from: {best_path}")
            return self.model
        else:
            print("No checkpoint found!")
            return None
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Train')
        axes[0].plot(self.history.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Train')
        axes[1].plot(self.history.history['val_accuracy'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.run_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"Saved training plot to: {plot_path}")


# =============================================================================
# TEST PIPELINE
# =============================================================================

def test_production_pipeline():
    """Test the complete production pipeline."""
    print("=" * 60)
    print("Testing Production Pipeline")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Use subset
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
    
    # Show model summary
    trainer.model.summary()
    
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
    
    # Plot training history
    trainer.plot_training_history()
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)
    
    trainer.load_best_model()
    loss, acc = trainer.model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    
    # List created files
    print("\nCreated files:")
    for root, dirs, files in os.walk(trainer.run_dir):
        level = root.replace(trainer.run_dir, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = '  ' * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            print(f"{sub_indent}{file} ({size/1024:.1f} KB)")
    
    return trainer


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 04 Solution: Production Pipeline")
    print("=" * 60)
    
    trainer = test_production_pipeline()
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)
    print("\nProduction Pipeline Features:")
    print("- Versioned run directories with timestamps")
    print("- Configuration saved as JSON")
    print("- Best model checkpointing")
    print("- Early stopping with best weights restore")
    print("- Learning rate reduction on plateau")
    print("- TensorBoard logging")
    print("- CSV training log")
    print("- Training history visualization")
    print("- Final model and weights saved")
    print("- Summary JSON with metrics")

