# Exercise 03: MLP Architecture Tuning

## Learning Objectives

- Design MLP architectures for classification tasks
- Experiment with network depth and width
- Analyze the impact of different hyperparameters
- Use systematic experimentation to find optimal configurations

## Duration

**Estimated Time:** 75 minutes

## Background

In `demo_03_mnist_mlp_classifier.py`, you saw a 3-layer MLP achieve ~98% accuracy on MNIST. But was that architecture optimal? In this exercise, you'll systematically explore the architecture space to understand how design choices affect performance.

## Your Challenge

Build the best possible MLP for MNIST digit classification by experimenting with:
- Number of hidden layers (1-4)
- Number of neurons per layer (32, 64, 128, 256)
- Activation functions (ReLU, tanh, LeakyReLU)
- Learning rates and optimizers

**Target:** Achieve >98% test accuracy with the smallest possible model.

---

## Part 1: Baseline Model (15 min)

### Task 1.1: Implement Your Training Pipeline

Navigate to `starter_code/mlp_tuning_starter.py`. You'll need to create a flexible model builder that accepts configuration parameters.

Create a function that builds an MLP with configurable architecture. The function should:
- Accept a list of hidden layer sizes
- Accept an activation function name
- Return a compiled model

### Task 1.2: Establish Baseline

Train the baseline architecture from the demo:
- Hidden layers: [128, 64]
- Activation: ReLU
- Optimizer: Adam (lr=0.001)
- Epochs: 20

Record:
- Training time
- Final training accuracy
- Final validation accuracy
- Total parameters

---

## Part 2: Depth Experiments (20 min)

### Task 2.1: Vary Network Depth

Keep total neurons approximately constant (~200) while varying depth:

| Config | Layers | Architecture | Approx Params |
|--------|--------|--------------|---------------|
| Shallow | 1 | [200] | ~160K |
| Medium | 2 | [128, 64] | ~110K |
| Deep | 3 | [100, 64, 32] | ~90K |
| Very Deep | 4 | [64, 64, 32, 32] | ~70K |

Train each configuration and record:
- Validation accuracy at epoch 10
- Final validation accuracy at epoch 20
- Any signs of training difficulty (slow convergence, oscillation)

### Task 2.2: Analyze Results

Create a table or plot comparing the configurations. Consider:
- Does adding more layers always help?
- At what depth do diminishing returns appear?
- Does a deeper network train slower per epoch?

---

## Part 3: Width Experiments (20 min)

### Task 3.1: Vary Layer Width

Fix depth at 2 layers, vary width:

| Config | Architecture | Total Params |
|--------|--------------|--------------|
| Narrow | [32, 16] | ~27K |
| Medium | [64, 32] | ~55K |
| Wide | [128, 64] | ~110K |
| Very Wide | [256, 128] | ~235K |

### Task 3.2: Efficiency Analysis

Calculate efficiency: `accuracy / (parameters / 1000)`

Which configuration gives the best accuracy per parameter?

---

## Part 4: Activation Function Comparison (10 min)

### Task 4.1: Compare Activations

Using your best architecture from Parts 2-3, compare:
- ReLU: `activation='relu'`
- Tanh: `activation='tanh'`
- LeakyReLU: `layers.LeakyReLU(alpha=0.1)`

Note: LeakyReLU requires a separate layer:
```python
model.add(layers.Dense(128))
model.add(layers.LeakyReLU(alpha=0.1))
```

Record training curves and final accuracy for each.

---

## Part 5: Final Optimization (10 min)

### Task 5.1: Combine Best Findings

Based on your experiments, design your optimal architecture.

Consider:
- Best depth from Part 2
- Best width from Part 3
- Best activation from Part 4
- Add dropout if you observe overfitting

### Task 5.2: Final Evaluation

Train your optimal model and evaluate on the test set:

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
```

---

## Results Summary

Complete this table with your findings:

| Experiment | Best Config | Val Accuracy | Parameters |
|------------|-------------|--------------|------------|
| Depth | | | |
| Width | | | |
| Activation | | | |
| **Final** | | | |

---

## Reflection Questions

1. **What is the relationship between model size and accuracy?** Is bigger always better?

2. **Did deeper networks always outperform shallow ones?** Why or why not?

3. **How did training time scale with model size?** Was the additional accuracy worth the extra compute?

4. **If you had to deploy this model on a mobile device with limited memory, which configuration would you choose?**

---

## Definition of Done

- [ ] Baseline model trained and metrics recorded
- [ ] At least 4 depth configurations tested
- [ ] At least 4 width configurations tested
- [ ] Activation function comparison completed
- [ ] Final optimized model achieves >98% test accuracy
- [ ] Results table completed
- [ ] Reflection questions answered

---

## Bonus Challenge

Implement a simple hyperparameter search:

```python
from itertools import product

depths = [[64], [64, 32], [64, 32, 16]]
activations = ['relu', 'tanh']
learning_rates = [0.01, 0.001, 0.0001]

results = []
for depth, act, lr in product(depths, activations, learning_rates):
    # Train model with these hyperparameters
    # Record results
    pass

# Find best configuration
```

---

## Solution

The solution is available in `solutions/exercise_03_solution.py`. Only check after completing your attempt!

