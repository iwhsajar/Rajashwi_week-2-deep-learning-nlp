# LSTM: Long Short-Term Memory Networks

## Learning Objectives
- Understand how LSTMs solve the vanishing gradient problem
- Learn the architecture: forget gate, input gate, output gate, and cell state
- Master the difference between cell state and hidden state
- Apply LSTMs to sequence modeling tasks in NLP

## Why This Matters

LSTMs revolutionized sequence modeling by solving RNNs' fundamental limitation—the inability to learn long-term dependencies. Understanding LSTMs is critical because:

- **Industry standard**: Before transformers, LSTMs were the default for NLP, speech, and time-series tasks
- **Production systems**: Still widely deployed in translation, speech recognition, and forecasting
- **Long-term memory**: Captures dependencies across hundreds of time steps (vs. ~10 for basic RNNs)
- **Foundation for GRUs and attention**: Understanding LSTMs clarifies these advanced mechanisms
- **Interviews**: LSTM architecture is a common technical interview question

As you deepen your sequential modeling knowledge this week, LSTMs represent the breakthrough that made deep learning for language practical. The gating mechanisms introduced here influenced all subsequent sequential architectures.

## The Problem: Vanishing Gradients in RNNs

### Quick Recap

Basic RNNs struggle with long sequences:

```
"The cat, which was sitting on the mat near the window overlooking the garden, was hungry."

RNN processes word by word:
"The" → "cat" → "which" → "was" → ... (12 words) ... → "hungry"

By the time RNN reaches "hungry," gradient signal from "cat" has vanished
Model can't learn that "cat" (subject) connects to "was hungry" (predicate)
```

**Root cause**: Gradients multiply through many time steps, shrinking exponentially.

## LSTM Solution: Gating Mechanisms

### Core Innovation

LSTMs introduce **gates**—neural network layers that learn to:
- **Forget** irrelevant information
- **Store** important information
- **Output** relevant information

Think of gates as **valves** controlling information flow.

### LSTM Architecture Overview

LSTMs maintain two states:
1. **Cell state** (C_t): Long-term memory (highway for information)
2. **Hidden state** (h_t): Short-term memory (output at each time step)

**Three gates control information flow:**
1. **Forget gate**: What to remove from cell state
2. **Input gate**: What new information to add to cell state
3. **Output gate**: What to output from cell state

## LSTM Architecture in Detail

### Visual Structure

```
                Input xₜ and previous hidden state hₜ₋₁
                              ↓
        ┌─────────────────────────────────────────┐
        │                                         │
        │  Forget Gate    Input Gate    Output   │
        │      ↓              ↓          Gate     │
        │   [sigmoid]     [sigmoid]       ↓       │
        │      ↓              ↓        [sigmoid]  │
        │      fₜ             iₜ           oₜ     │
        │      │              │            │      │
        │      │         [tanh] → C̃ₜ       │      │
        │      │              │            │      │
        │      ↓              ↓            │      │
        │  Cₜ₋₁ ×──fₜ──→  +  ←──iₜ×C̃ₜ    │      │
        │      │             │             │      │
        │      └─────────────┘             │      │
        │           Cₜ (cell state)        │      │
        │            │                     │      │
        │            └────→ tanh ←─────────┘      │
        │                    │                    │
        │                    × oₜ                 │
        │                    ↓                    │
        └────────────────────────────────────────┘
                             hₜ (hidden state)
```

### Step-by-Step: How LSTM Processes Information

**Input at time t:**
- xₜ: Current input (e.g., word embedding)
- hₜ₋₁: Previous hidden state
- Cₜ₋₁: Previous cell state

**Step 1: Forget Gate**

Decide what information to discard from cell state:

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)

Where:
- σ = sigmoid function (output between 0 and 1)
- fₜ = forget gate activation
- 0 = "completely forget this"
- 1 = "completely keep this"
```

**Example:**
```
Text: "The cat was hungry. The dog was sleeping."

Processing "dog":
Forget gate might output fₜ ≈ 0 for "cat" information
(New subject, forget old subject)
```

**Step 2: Input Gate**

Decide what new information to add to cell state:

```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)       # What to update
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)   # Candidate values

Where:
- iₜ = input gate activation (0 to 1)
- C̃ₜ = candidate cell state (-1 to 1)
```

**Example:**
```
Processing "dog":
Input gate might activate (iₜ ≈ 1) for new subject information
Candidate C̃ₜ contains "dog" representation
```

**Step 3: Update Cell State**

Combine forget and input gates to update cell state:

```
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ

Where:
- fₜ * Cₜ₋₁ = forget some old information
- iₜ * C̃ₜ = add some new information
```

**Example:**
```
Cₜ = 0.1 * (cat info) + 0.9 * (dog info)
# Mostly forget cat, remember dog
```

**Step 4: Output Gate**

Decide what to output based on cell state:

```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)  # What parts to output
hₜ = oₜ * tanh(Cₜ)             # Final hidden state

Where:
- oₜ = output gate activation
- hₜ = new hidden state (LSTM output)
```

**Example:**
```
Output gate filters cell state
hₜ contains relevant information for current prediction
```

## Mathematical Summary

```python
# LSTM forward pass (one time step)

# Forget gate
fₜ = sigmoid(Wf @ [hₜ₋₁, xₜ] + bf)

# Input gate
iₜ = sigmoid(Wi @ [hₜ₋₁, xₜ] + bi)
C̃ₜ = tanh(Wc @ [hₜ₋₁, xₜ] + bc)

# Update cell state
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ

# Output gate
oₜ = sigmoid(Wo @ [hₜ₋₁, xₜ] + bo)
hₜ = oₜ * tanh(Cₜ)

# hₜ is the output (used for predictions and next time step)
# Cₜ carries long-term memory to next time step
```

## Cell State vs. Hidden State

### Cell State (Cₜ): The Memory Highway

- **Long-term memory**: Information can flow unchanged across many time steps
- **Additive updates**: New info added, old info subtracted (gradients flow easily)
- **Not directly output**: Internal memory only

**Why it solves vanishing gradients:**
```
∂Cₜ/∂Cₜ₋₁ = fₜ (forget gate)

Unlike RNNs where gradients multiply through tanh/sigmoid repeatedly,
LSTM cell state uses addition → gradients flow backward without vanishing!
```

### Hidden State (hₜ): The Output

- **Short-term memory**: Filtered version of cell state
- **Used for predictions**: Output layer uses hₜ
- **Passed to next time step**: Influences next computation

**Analogy:**
- Cell state = Long-term memory (everything you've learned)
- Hidden state = Working memory (what you're currently thinking about)

## LSTM in Keras/TensorFlow

### Basic LSTM Layer

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=100),
    layers.LSTM(
        units=64,              # Number of LSTM cells (hidden state dimension)
        activation='tanh',     # Activation for cell state candidate
        recurrent_activation='sigmoid',  # Activation for gates
        return_sequences=False,  # Return only final hidden state
        dropout=0.2,           # Input dropout
        recurrent_dropout=0.2  # Recurrent connection dropout
    ),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### Stacked LSTMs (Multiple Layers)

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    
    # First LSTM layer (return sequences for next LSTM)
    layers.LSTM(128, return_sequences=True),
    
    # Second LSTM layer (return final state only)
    layers.LSTM(64),
    
    layers.Dense(10, activation='softmax')
])
```

**Why stack LSTMs?**
- First layer: Learns low-level patterns
- Second layer: Learns high-level abstractions
- Enables hierarchical feature learning

### Bidirectional LSTM

Process sequence in both directions:

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    
    layers.Bidirectional(layers.LSTM(64)),
    # Forward LSTM: processes left-to-right
    # Backward LSTM: processes right-to-left
    # Concatenates outputs → 128-dim hidden state
    
    layers.Dense(1, activation='sigmoid')
])
```

**When to use bidirectional:**
- Text classification: Context from both directions helps
- Named entity recognition: "New York" benefits from seeing both words
- Not suitable for: Text generation (can't see future during generation)

## Practical Example: Sentiment Analysis with LSTM

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dataset
texts = [
    "I absolutely loved this movie, best film ever",
    "Terrible waste of time, do not watch",
    "Amazing performance by the actors",
    "Boring and predictable plot",
    "Highly recommend, brilliant storytelling"
]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

# Tokenize and pad
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=20)
y = np.array(labels)

# Build LSTM model
model = keras.Sequential([
    layers.Embedding(input_dim=5000, output_dim=64, input_length=20),
    layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(X, y, epochs=50, verbose=0)

# Predict
test_text = ["This movie was absolutely wonderful"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_X = pad_sequences(test_seq, maxlen=20)
prediction = model.predict(test_X, verbose=0)
print(f"Sentiment: {'Positive' if prediction > 0.5 else 'Negative'} ({prediction[0][0]:.2f})")
```

## LSTM vs. Basic RNN

| Aspect | RNN | LSTM |
|--------|-----|------|
| **Architecture** | Single hidden state | Cell state + hidden state |
| **Long-term memory** | ~10 time steps | 100+ time steps |
| **Vanishing gradients** | Severe problem | Largely solved |
| **Parameters** | Fewer (faster) | More (4x gates) |
| **Training time** | Faster | Slower |
| **Performance** | Poor on long sequences | Excellent |
| **Use case** | Short sequences, simple patterns | Long sequences, complex dependencies |

**When to use RNN:** Very simple tasks, short sequences, speed critical

**When to use LSTM:** Most NLP tasks, long sequences, complex patterns

## How LSTMs Solve Vanishing Gradients

### Gradient Flow Through Cell State

```
Basic RNN:
gradient propagates through: hₜ → hₜ₋₁ → hₜ₋₂ → ...
Each step multiplies by tanh'(·) ≤ 1 → gradient vanishes

LSTM:
gradient propagates through: Cₜ → Cₜ₋₁ → Cₜ₋₂ → ...
Cell state uses addition (Cₜ = fₜ * Cₜ₋₁ + ...), not multiplication
Gradient flows backward without repeated multiplication → no vanishing!
```

### Constant Error Carousel

The cell state acts as a **constant error carousel**:
- Gradients can flow backward hundreds of time steps
- Forget gate controls how much information to retain
- Learned gates prevent both vanishing and exploding gradients

## Variants and Extensions

### GRU (Gated Recurrent Unit)

Simplified version of LSTM (fewer gates):

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.GRU(64),  # Fewer parameters than LSTM, often similar performance
    layers.Dense(1, activation='sigmoid')
])
```

**GRU vs. LSTM:**
- **GRU**: 2 gates (reset, update), faster training, fewer parameters
- **LSTM**: 3 gates, more expressive, better for complex tasks
- **Rule of thumb**: Try both, use whichever performs better

### Peephole Connections

Let gates look at cell state directly:

```
fₜ = σ(Wf · [Cₜ₋₁, hₜ₋₁, xₜ] + bf)  # Forget gate sees Cₜ₋₁

Not available in standard Keras, but exists in research
```

## Practical Tips for Training LSTMs

### 1. Regularization

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    # dropout: Drop input connections
    # recurrent_dropout: Drop recurrent connections
    layers.Dense(1, activation='sigmoid')
])
```

### 2. Gradient Clipping

```python
optimizer = keras.optimizers.Adam(clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy')
```

### 3. Batch Normalization

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.LSTM(64, return_sequences=True),
    layers.BatchNormalization(),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])
```

### 4. Learning Rate Scheduling

```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.96
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

## Common Pitfalls

### 1. Forgetting `return_sequences=True`

```python
# WRONG: Stacking LSTMs without return_sequences
model = keras.Sequential([
    layers.LSTM(64),              # Returns only final state
    layers.LSTM(32),              # ERROR: Expects sequences, gets single state
])

# CORRECT: return_sequences=True for all but last LSTM
model = keras.Sequential([
    layers.LSTM(64, return_sequences=True),  # Returns full sequence
    layers.LSTM(32),                         # Processes sequence
])
```

### 2. Too Many LSTM Layers

```python
# Usually sufficient
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.LSTM(64),
    layers.Dense(10, activation='softmax')
])

# 2 layers for complex tasks
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    layers.Dense(10, activation='softmax')
])

# 3+ layers: Usually overkill, consider transformers instead
```

### 3. Not Enough Data

LSTMs have many parameters (4x gates × weights):
- Require substantial training data
- Use pre-trained embeddings to reduce data needs
- Consider simpler models if data is scarce

## Key Takeaways

1. **LSTMs solve vanishing gradients** through gated architecture with separate cell state
2. **Three gates**: Forget (remove info), input (add info), output (filter info)
3. **Cell state** = long-term memory (highway for gradients), **hidden state** = short-term memory (output)
4. **Additive cell state updates** allow gradients to flow backward without vanishing
5. **Handles 100+ time steps** vs. ~10 for basic RNNs
6. **4x more parameters** than RNNs (slower training, better performance)
7. **Industry standard** for sequential modeling before transformers
8. **GRU** is a simpler alternative with competitive performance

## External Resources

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah's illustrated guide (essential reading)
- [Long Short-Term Memory (Original Paper)](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber, 1997
- [LSTM Tutorial in TensorFlow](https://www.tensorflow.org/text/tutorials/text_generation) - Official text generation guide

