"""
Exercise 01 Solution: Model Persistence

Complete implementation of saving and loading models in different formats.
"""

import numpy as np
import os
import tempfile
import datetime
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs('logs/exercise_persistence', exist_ok=True)

# =============================================================================
# PART 1: Create Sample Model
# =============================================================================

def create_sample_model():
    """Create a sample model for persistence testing."""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, epochs=5):
    """Train model on MNIST subset."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Use subset
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    
    # TensorBoard callback
    log_dir = "logs/exercise_persistence/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_split=0.2,
        callbacks=[tb_callback],
        verbose=1
    )
    
    return history, (x_test, y_test)


# =============================================================================
# PART 2: Different Saving Formats
# =============================================================================

def save_in_all_formats(model, base_path='saved_models'):
    """
    Save model in all available formats.
    
    Args:
        model: Trained Keras model
        base_path: Directory to save models
    """
    os.makedirs(base_path, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("Saving Model in Different Formats")
    print("=" * 50)
    
    # 1. Keras format (.keras) - Recommended
    keras_path = os.path.join(base_path, 'model.keras')
    model.save(keras_path)
    print(f"\n1. Keras format: {keras_path}")
    print(f"   Size: {os.path.getsize(keras_path) / 1024:.2f} KB")
    
    # 2. HDF5 format (.h5) - Legacy
    h5_path = os.path.join(base_path, 'model.h5')
    model.save(h5_path)
    print(f"\n2. HDF5 format: {h5_path}")
    print(f"   Size: {os.path.getsize(h5_path) / 1024:.2f} KB")
    
    # 3. SavedModel format (directory) - For TF Serving
    savedmodel_path = os.path.join(base_path, 'savedmodel')
    # Use export() for Keras 3 compatibility (TensorFlow Serving format)
    try:
        model.export(savedmodel_path)
    except AttributeError:
        # Fallback for older Keras versions
        model.save(savedmodel_path, save_format='tf')
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, _, filenames in os.walk(savedmodel_path)
        for filename in filenames
    )
    print(f"\n3. SavedModel format: {savedmodel_path}/")
    print(f"   Total size: {total_size / 1024:.2f} KB")
    
    # 4. Weights only (.weights.h5)
    weights_path = os.path.join(base_path, 'model.weights.h5')
    model.save_weights(weights_path)
    print(f"\n4. Weights only: {weights_path}")
    print(f"   Size: {os.path.getsize(weights_path) / 1024:.2f} KB")
    
    # 5. Architecture only (JSON)
    json_path = os.path.join(base_path, 'model_architecture.json')
    with open(json_path, 'w') as f:
        f.write(model.to_json())
    print(f"\n5. Architecture JSON: {json_path}")
    print(f"   Size: {os.path.getsize(json_path) / 1024:.2f} KB")
    
    return {
        'keras': keras_path,
        'h5': h5_path,
        'savedmodel': savedmodel_path,
        'weights': weights_path,
        'json': json_path
    }


# =============================================================================
# PART 3: Loading Models
# =============================================================================

def load_and_verify(paths, original_model, test_data):
    """
    Load models from different formats and verify they match.
    
    Args:
        paths: Dictionary of saved paths
        original_model: Original trained model
        test_data: (x_test, y_test) tuple
    """
    x_test, y_test = test_data
    
    print("\n" + "=" * 50)
    print("Loading and Verifying Models")
    print("=" * 50)
    
    # Get original predictions
    original_preds = original_model.predict(x_test[:100], verbose=0)
    original_loss, original_acc = original_model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nOriginal model - Loss: {original_loss:.4f}, Accuracy: {original_acc:.4f}")
    
    results = {}
    
    # 1. Load from Keras format
    print("\n1. Loading from .keras format...")
    loaded_keras = keras.models.load_model(paths['keras'])
    keras_preds = loaded_keras.predict(x_test[:100], verbose=0)
    keras_loss, keras_acc = loaded_keras.evaluate(x_test, y_test, verbose=0)
    matches = np.allclose(original_preds, keras_preds, rtol=1e-5)
    print(f"   Loss: {keras_loss:.4f}, Accuracy: {keras_acc:.4f}")
    print(f"   Predictions match original: {matches}")
    results['keras'] = {'matches': matches, 'acc': keras_acc}
    
    # 2. Load from HDF5
    print("\n2. Loading from .h5 format...")
    loaded_h5 = keras.models.load_model(paths['h5'])
    h5_preds = loaded_h5.predict(x_test[:100], verbose=0)
    h5_loss, h5_acc = loaded_h5.evaluate(x_test, y_test, verbose=0)
    matches = np.allclose(original_preds, h5_preds, rtol=1e-5)
    print(f"   Loss: {h5_loss:.4f}, Accuracy: {h5_acc:.4f}")
    print(f"   Predictions match original: {matches}")
    results['h5'] = {'matches': matches, 'acc': h5_acc}
    
    # 3. Load from SavedModel
    print("\n3. Loading from SavedModel format...")
    try:
        # Keras 3: Use TFSMLayer for SavedModel inference
        loaded_savedmodel = keras.layers.TFSMLayer(paths['savedmodel'], call_endpoint='serve')
        sm_preds = loaded_savedmodel(x_test[:100]).numpy()
        # Calculate accuracy manually for TFSMLayer
        sm_pred_classes = np.argmax(sm_preds, axis=1)
        sm_acc = np.mean(sm_pred_classes == y_test[:100])
        matches = np.allclose(original_preds, sm_preds, rtol=1e-5)
        print(f"   Accuracy (on subset): {sm_acc:.4f}")
        print(f"   Predictions match original: {matches}")
    except Exception as e:
        # Fallback for older Keras
        loaded_savedmodel = keras.models.load_model(paths['savedmodel'])
        sm_preds = loaded_savedmodel.predict(x_test[:100], verbose=0)
        sm_loss, sm_acc = loaded_savedmodel.evaluate(x_test, y_test, verbose=0)
        matches = np.allclose(original_preds, sm_preds, rtol=1e-5)
        print(f"   Loss: {sm_loss:.4f}, Accuracy: {sm_acc:.4f}")
        print(f"   Predictions match original: {matches}")
    results['savedmodel'] = {'matches': matches, 'acc': sm_acc}
    
    # 4. Load weights into new model
    print("\n4. Loading weights into new architecture...")
    new_model = create_sample_model()
    new_model.load_weights(paths['weights'])
    weights_preds = new_model.predict(x_test[:100], verbose=0)
    weights_loss, weights_acc = new_model.evaluate(x_test, y_test, verbose=0)
    matches = np.allclose(original_preds, weights_preds, rtol=1e-5)
    print(f"   Loss: {weights_loss:.4f}, Accuracy: {weights_acc:.4f}")
    print(f"   Predictions match original: {matches}")
    results['weights'] = {'matches': matches, 'acc': weights_acc}
    
    # 5. Load from JSON + weights
    print("\n5. Loading from JSON architecture + weights...")
    with open(paths['json'], 'r') as f:
        json_config = f.read()
    
    json_model = keras.models.model_from_json(json_config)
    json_model.load_weights(paths['weights'])
    json_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    json_preds = json_model.predict(x_test[:100], verbose=0)
    json_loss, json_acc = json_model.evaluate(x_test, y_test, verbose=0)
    matches = np.allclose(original_preds, json_preds, rtol=1e-5)
    print(f"   Loss: {json_loss:.4f}, Accuracy: {json_acc:.4f}")
    print(f"   Predictions match original: {matches}")
    results['json'] = {'matches': matches, 'acc': json_acc}
    
    return results


# =============================================================================
# PART 4: Summary
# =============================================================================

def print_format_comparison():
    """Print comparison of different formats."""
    print("\n" + "=" * 60)
    print("FORMAT COMPARISON")
    print("=" * 60)
    
    comparison = """
    | Format      | Extension    | Contents                   | Use Case                |
    |-------------|--------------|----------------------------|-------------------------|
    | Keras       | .keras       | Architecture + Weights +   | Recommended default     |
    |             |              | Optimizer + Config         |                         |
    | HDF5        | .h5          | Architecture + Weights +   | Legacy, compatibility   |
    |             |              | Optimizer + Config         |                         |
    | SavedModel  | directory    | Full TF graph + variables  | TF Serving, production  |
    | Weights     | .weights.h5  | Only weights               | Transfer learning       |
    | JSON        | .json        | Only architecture          | Version control         |
    """
    print(comparison)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 01 Solution: Model Persistence")
    print("=" * 60)
    
    # Create and train model
    print("\nCreating and training model...")
    model = create_sample_model()
    history, test_data = train_model(model, epochs=5)
    
    # Save in all formats
    paths = save_in_all_formats(model)
    
    # Load and verify
    results = load_and_verify(paths, model, test_data)
    
    # Print comparison
    print_format_comparison()
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    all_match = all(r['matches'] for r in results.values())
    print(f"All formats produce identical predictions: {all_match}")
    
    print("\n" + "=" * 60)
    print("Solution Complete!")
    print("=" * 60)

