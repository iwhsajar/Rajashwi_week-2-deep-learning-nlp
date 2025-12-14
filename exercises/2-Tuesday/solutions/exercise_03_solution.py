"""
Exercise 03 Solution: Batch Normalization Experiment

Complete implementation comparing training with and without batch normalization.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# =============================================================================
# PART 1: Model Definitions
# =============================================================================

def create_model_without_bn():
    """Create deep model WITHOUT batch normalization."""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(784,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def create_model_with_bn():
    """Create deep model WITH batch normalization after each layer."""
    model = keras.Sequential([
        layers.Dense(256, input_shape=(784,)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.Dense(10, activation='softmax')
    ])
    return model


# =============================================================================
# PART 2: Convergence Comparison
# =============================================================================

def compare_convergence(x_train, y_train, x_test, y_test, epochs=30, lr=0.001):
    """Compare convergence speed with and without batch normalization."""
    
    # Model without BN
    model_no_bn = create_model_without_bn()
    model_no_bn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training WITHOUT batch normalization...")
    history_no_bn = model_no_bn.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=0
    )
    
    # Model with BN
    model_bn = create_model_with_bn()
    model_bn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training WITH batch normalization...")
    history_bn = model_bn.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=0
    )
    
    return history_no_bn.history, history_bn.history


def plot_convergence(history_no_bn, history_bn):
    """Plot convergence comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss comparison
    axes[0].plot(history_no_bn['loss'], 'b-', label='Train (No BN)', alpha=0.7)
    axes[0].plot(history_no_bn['val_loss'], 'b--', label='Val (No BN)', alpha=0.7)
    axes[0].plot(history_bn['loss'], 'r-', label='Train (With BN)', alpha=0.7)
    axes[0].plot(history_bn['val_loss'], 'r--', label='Val (With BN)', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[1].plot(history_no_bn['accuracy'], 'b-', label='Train (No BN)', alpha=0.7)
    axes[1].plot(history_no_bn['val_accuracy'], 'b--', label='Val (No BN)', alpha=0.7)
    axes[1].plot(history_bn['accuracy'], 'r-', label='Train (With BN)', alpha=0.7)
    axes[1].plot(history_bn['val_accuracy'], 'r--', label='Val (With BN)', alpha=0.7)
    axes[1].axhline(y=0.95, color='gray', linestyle=':', label='95% Target')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('batch_norm_convergence.png', dpi=150)
    plt.show()


def find_epochs_to_target(history, target=0.95, metric='val_accuracy'):
    """Find number of epochs to reach target accuracy."""
    for i, acc in enumerate(history[metric]):
        if acc >= target:
            return i + 1
    return None  # Target not reached


# =============================================================================
# PART 3: Learning Rate Tolerance
# =============================================================================

def learning_rate_tolerance_experiment(x_train, y_train, x_test, y_test):
    """Test how different learning rates affect training stability."""
    
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    results = {'Without BN': {}, 'With BN': {}}
    
    for lr in learning_rates:
        print(f"\nTesting LR = {lr}")
        
        # Without BN
        try:
            model = create_model_without_bn()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            history = model.fit(x_train, y_train, epochs=10, 
                              validation_data=(x_test, y_test), verbose=0)
            
            # Check if training diverged (loss became NaN or very high)
            final_loss = history.history['loss'][-1]
            if np.isnan(final_loss) or final_loss > 10:
                results['Without BN'][lr] = 'Diverged'
            else:
                results['Without BN'][lr] = f"{history.history['val_accuracy'][-1]:.4f}"
        except Exception as e:
            results['Without BN'][lr] = 'Error'
        
        # With BN
        try:
            model = create_model_with_bn()
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            history = model.fit(x_train, y_train, epochs=10,
                              validation_data=(x_test, y_test), verbose=0)
            
            final_loss = history.history['loss'][-1]
            if np.isnan(final_loss) or final_loss > 10:
                results['With BN'][lr] = 'Diverged'
            else:
                results['With BN'][lr] = f"{history.history['val_accuracy'][-1]:.4f}"
        except Exception as e:
            results['With BN'][lr] = 'Error'
    
    # Print results
    print("\n" + "=" * 60)
    print("Learning Rate Tolerance Results")
    print("=" * 60)
    print(f"{'LR':<10} {'Without BN':<15} {'With BN':<15}")
    print("-" * 40)
    for lr in learning_rates:
        print(f"{lr:<10} {results['Without BN'][lr]:<15} {results['With BN'][lr]:<15}")
    
    return results


# =============================================================================
# PART 4: Inspect BN Parameters
# =============================================================================

def inspect_bn_parameters(model):
    """Inspect learned batch normalization parameters."""
    print("\n" + "=" * 60)
    print("Batch Normalization Parameters")
    print("=" * 60)
    
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            gamma = layer.gamma.numpy()
            beta = layer.beta.numpy()
            moving_mean = layer.moving_mean.numpy()
            moving_var = layer.moving_variance.numpy()
            
            print(f"\nLayer: {layer.name}")
            print(f"  gamma (scale):    mean={gamma.mean():.4f}, std={gamma.std():.4f}, "
                  f"min={gamma.min():.4f}, max={gamma.max():.4f}")
            print(f"  beta (shift):     mean={beta.mean():.4f}, std={beta.std():.4f}, "
                  f"min={beta.min():.4f}, max={beta.max():.4f}")
            print(f"  moving_mean:      mean={moving_mean.mean():.4f}")
            print(f"  moving_variance:  mean={moving_var.mean():.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Exercise 03 Solution: Batch Normalization Experiment")
    print("=" * 60)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Use subset
    x_train = x_train[:10000]
    y_train = y_train[:10000]
    
    print(f"\nData loaded: {x_train.shape[0]} training samples")
    
    # Part 1: Convergence comparison
    print("\n" + "=" * 60)
    print("Part 1: Convergence Comparison")
    print("=" * 60)
    
    history_no_bn, history_bn = compare_convergence(
        x_train, y_train, x_test, y_test, epochs=30
    )
    
    plot_convergence(history_no_bn, history_bn)
    
    # Calculate speedup
    epochs_no_bn = find_epochs_to_target(history_no_bn, target=0.95)
    epochs_bn = find_epochs_to_target(history_bn, target=0.95)
    
    print(f"\nEpochs to 95% validation accuracy:")
    print(f"  Without BN: {epochs_no_bn if epochs_no_bn else 'Not reached'}")
    print(f"  With BN:    {epochs_bn if epochs_bn else 'Not reached'}")
    
    if epochs_no_bn and epochs_bn:
        speedup = epochs_no_bn / epochs_bn
        print(f"  Speedup:    {speedup:.2f}x faster with BN")
    
    # Part 2: Learning rate tolerance
    print("\n" + "=" * 60)
    print("Part 2: Learning Rate Tolerance")
    print("=" * 60)
    
    lr_results = learning_rate_tolerance_experiment(x_train, y_train, x_test, y_test)
    
    # Part 3: Inspect BN parameters
    print("\n" + "=" * 60)
    print("Part 3: Inspect BN Parameters")
    print("=" * 60)
    
    # Train a fresh model and inspect
    model_bn = create_model_with_bn()
    model_bn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_bn.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=0)
    
    inspect_bn_parameters(model_bn)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final val accuracy (no BN):  {history_no_bn['val_accuracy'][-1]:.4f}")
    print(f"Final val accuracy (with BN): {history_bn['val_accuracy'][-1]:.4f}")
    print("\nKey Insights:")
    print("1. Batch normalization accelerates convergence")
    print("2. BN allows higher learning rates without divergence")
    print("3. Gamma and beta learn to scale/shift as needed")
    print("4. Moving statistics track activation distributions")

