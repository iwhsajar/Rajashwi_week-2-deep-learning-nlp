# Recurrent Neural Networks (RNNs)

## Learning Objectives
- Understand why sequential data requires specialized architectures
- Learn RNN architecture: hidden states, recurrent connections, and sequential processing
- Recognize the vanishing gradient problem in RNNs
- Apply RNNs to text classification and sequence prediction tasks

## Why This Matters

Recurrent Neural Networks revolutionized how we process sequential data—text, time series, speech, and video. Understanding RNNs is critical because:

- **Sequential modeling foundation**: RNNs introduced the concept of memory in neural networks
- **Text processing**: Before transformers, RNNs (and LSTMs) were the standard for NLP tasks
- **Real-world applications**: Language translation, speech recognition, music generation, stock prediction
- **Conceptual stepping stone**: Understanding RNNs is essential for grasping LSTMs, GRUs, and attention mechanisms
- **Production systems**: Still used in many deployed systems, especially for time-series prediction

As you transition from static embeddings to sequential models this week, RNNs represent the breakthrough that enabled neural networks to process language as a sequence—capturing word order and context that simple feedforward networks cannot.

## The Problem with Feedforward Networks

### Why Standard Neural Networks Fail for Sequences

**Feedforward networks** process inputs independently:

```python
# Feedforward network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

# Each input processed in isolation
# No memory of previous inputs
```

**Problems for sequential data:**

1. **Fixed input size**: Can't handle variable-length sequences
```
"Hi" (2 words) vs. "The quick brown fox jumps" (5 words)
# Feedforward network requires same input size!
```

2. **No temporal information**: Ignores order
```
"The dog bit the man" ≠ "The man bit the dog"
# Same words, different meanings—order matters!
```

3. **No context retention**:
```
"The movie was not good" 
# Can't capture that "not" negates "good"
```

## RNN Architecture: The Core Idea

### Sequential Processing with Memory

**Key innovation**: RNNs maintain a **hidden state** that carries information from previous time steps.

```
Input sequence: x₁, x₂, x₃, x₄, x₅ (e.g., words in a sentence)

RNN processing:
t=1: h₁ = f(x₁, h₀)      # h₀ is initial hidden state (often zeros)
t=2: h₂ = f(x₂, h₁)      # h₁ carries information from x₁
t=3: h₃ = f(x₃, h₂)      # h₂ carries information from x₁, x₂
t=4: h₄ = f(x₄, h₃)      # h₃ carries information from x₁, x₂, x₃
t=5: h₅ = f(x₅, h₄)      # h₄ carries information from all previous inputs

Each hidden state hₜ depends on current input xₜ and previous hidden state hₜ₋₁
```

### Visual Representation

**Folded (compact) view:**

```
    Input (xₜ)
       ↓
    ┌─────┐ ←── hₜ₋₁ (previous hidden state)
    │ RNN │
    └─────┘
       ↓
    hₜ (current hidden state) → Output (yₜ)
       ↓
    (feeds to next time step)
```

**Unfolded (sequential) view:**

```
x₁      x₂      x₃      x₄      x₅
 ↓       ↓       ↓       ↓       ↓
[RNN] → [RNN] → [RNN] → [RNN] → [RNN]
 ↓       ↓       ↓       ↓       ↓
h₁      h₂      h₃      h₄      h₅
 ↓       ↓       ↓       ↓       ↓
y₁      y₂      y₃      y₄      y₅
```

The same RNN cell is applied at each time step, but it maintains state across the sequence.

## Mathematical Formulation

### RNN Update Equations

At each time step `t`:

```
hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
yₜ = Wₕᵧ · hₜ + bᵧ

Where:
- xₜ: Input at time t
- hₜ: Hidden state at time t
- hₜ₋₁: Previous hidden state
- yₜ: Output at time t
- Wₓₕ: Input-to-hidden weights
- Wₕₕ: Hidden-to-hidden weights (recurrent connection)
- Wₕᵧ: Hidden-to-output weights
- bₕ, bᵧ: Biases
- tanh: Activation function (could also be ReLU, sigmoid)
```

### Example Calculation

```python
import numpy as np

# Dimensions
input_size = 3      # e.g., 3-dimensional word embedding
hidden_size = 2     # 2 hidden units
output_size = 1     # 1 output (e.g., sentiment score)

# Weights (randomly initialized)
W_xh = np.random.randn(hidden_size, input_size)   # (2, 3)
W_hh = np.random.randn(hidden_size, hidden_size)  # (2, 2)
W_hy = np.random.randn(output_size, hidden_size)  # (1, 2)
b_h = np.zeros((hidden_size, 1))                  # (2, 1)
b_y = np.zeros((output_size, 1))                  # (1, 1)

# Initialize hidden state
h_prev = np.zeros((hidden_size, 1))  # (2, 1)

# Input at time step t
x_t = np.array([[0.5], [0.2], [0.8]])  # (3, 1)

# Compute new hidden state
h_t = np.tanh(np.dot(W_xh, x_t) + np.dot(W_hh, h_prev) + b_h)
print(f"Hidden state h_t:\n{h_t}")
# [[0.45], [-0.32]]

# Compute output
y_t = np.dot(W_hy, h_t) + b_y
print(f"Output y_t:\n{y_t}")
# [[0.18]]
```

## RNN in Keras/TensorFlow

### Simple RNN Layer

```python
from tensorflow import keras
from tensorflow.keras import layers

# Define model with RNN
model = keras.Sequential([
    layers.SimpleRNN(
        units=64,              # 64 hidden units
        activation='tanh',     # Activation function
        return_sequences=False, # Return only last hidden state
        input_shape=(None, 100) # (sequence_length, feature_dim)
    ),
    layers.Dense(1, activation='sigmoid')  # Output layer
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

**Parameters:**
- `units`: Number of hidden units (hidden state dimensionality)
- `activation`: Activation function (tanh, relu)
- `return_sequences`: 
  - `False`: Return only final hidden state (for classification)
  - `True`: Return hidden states at all time steps (for sequence-to-sequence)
- `input_shape`: `(sequence_length, feature_dimension)`
  - `None` allows variable-length sequences

### Example: Sentiment Analysis with RNN

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Sample data
texts = [
    "I love this movie",
    "Terrible film, waste of time",
    "Amazing experience, highly recommend",
    "Boring and predictable",
    "Best movie I've ever seen"
]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

# Tokenize
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)
y = np.array(labels)

print(f"Input shape: {X.shape}")  # (5, 10)

# Build RNN model
model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    # Embedding layer: Maps word indices to 32-dim vectors
    
    layers.SimpleRNN(units=16, activation='tanh'),
    # RNN layer: Processes sequence, returns final hidden state
    
    layers.Dense(1, activation='sigmoid')
    # Output layer: Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X, y, epochs=50, verbose=0)

# Predict
test_text = ["I love this"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_X = pad_sequences(test_seq, maxlen=10)
prediction = model.predict(test_X, verbose=0)
print(f"Sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")
```

### Return Sequences: Many-to-Many Tasks

```python
# For sequence-to-sequence tasks (e.g., translation, text generation)
model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    layers.SimpleRNN(units=64, return_sequences=True),
    # Returns hidden state at each time step
    # Output shape: (batch_size, sequence_length, 64)
    
    layers.Dense(vocab_size, activation='softmax')
    # Predict next word at each time step
])
```

## Types of RNN Architectures

### 1. Many-to-One (Sequence Classification)

**Input**: Sequence (e.g., sentence)  
**Output**: Single value (e.g., sentiment label)

```
x₁ → x₂ → x₃ → x₄ → x₅
 ↓    ↓    ↓    ↓    ↓
[RNN][RNN][RNN][RNN][RNN]
                      ↓
                     y (final output only)
```

**Example**: Sentiment analysis, spam detection, topic classification

### 2. One-to-Many (Sequence Generation)

**Input**: Single value (e.g., image encoding)  
**Output**: Sequence (e.g., caption)

```
x
 ↓
[RNN] → [RNN] → [RNN] → [RNN] → [RNN]
 ↓       ↓       ↓       ↓       ↓
y₁      y₂      y₃      y₄      y₅
```

**Example**: Image captioning, music generation

### 3. Many-to-Many (Sequence-to-Sequence)

**Input**: Sequence  
**Output**: Sequence (same length)

```
x₁ → x₂ → x₃ → x₄ → x₅
 ↓    ↓    ↓    ↓    ↓
[RNN][RNN][RNN][RNN][RNN]
 ↓    ↓    ↓    ↓    ↓
y₁   y₂   y₃   y₄   y₅
```

**Example**: Part-of-speech tagging, named entity recognition

### 4. Encoder-Decoder (Seq2Seq)

**Input**: Variable-length sequence  
**Output**: Different-length sequence

```
Encoder:
x₁ → x₂ → x₃ → x₄
 ↓    ↓    ↓    ↓
[RNN][RNN][RNN][RNN] → context vector

Decoder:
context vector → [RNN] → [RNN] → [RNN] → [RNN] → [RNN]
                   ↓       ↓       ↓       ↓       ↓
                  y₁      y₂      y₃      y₄      y₅
```

**Example**: Machine translation, text summarization

## The Vanishing Gradient Problem

### The Challenge

During backpropagation through time (BPTT), gradients can **vanish** (become extremely small) as they propagate backward through many time steps.

**Why it happens:**

```
Gradient at time t=1 depends on gradients from all future time steps
For a 100-word sentence:
gradient₁ ∝ ∂L/∂h₁₀₀ × ∂h₁₀₀/∂h₉₉ × ... × ∂h₂/∂h₁

If each ∂hₜ/∂hₜ₋₁ < 1 (e.g., 0.9):
gradient₁ ∝ 0.9¹⁰⁰ ≈ 0.00003  (vanishes!)
```

**Consequences:**

1. **Long-term dependencies not learned**: Model can't remember information from early in sequence
```
"The cat, which was sitting on the mat near the window in the living room, was hungry."
# By the time model processes "hungry," information about "cat" has vanished
```

2. **Early layers train slowly**: Gradients near input are tiny, weights barely update

3. **Short-term memory only**: RNN effectively only remembers last 5-10 time steps

### Activation Function Impact

```python
# tanh activation (common in RNNs)
# tanh'(x) ≤ 1 for all x
# Repeated multiplication: gradient shrinks exponentially

# ReLU activation (sometimes used)
# ReLU'(x) = 1 for x > 0
# But can cause exploding gradients
```

### Exploding Gradient Problem

Opposite issue: gradients become extremely large

```
If each ∂hₜ/∂hₜ₋₁ > 1 (e.g., 1.1):
gradient₁ ∝ 1.1¹⁰⁰ ≈ 13,780  (explodes!)
```

**Solution**: Gradient clipping

```python
# In Keras
optimizer = keras.optimizers.Adam(clipnorm=1.0)  # Clip gradient norm to 1.0

# Or clip by value
optimizer = keras.optimizers.Adam(clipvalue=0.5)  # Clip each gradient component to ±0.5
```

## Limitations of Basic RNNs

1. **Vanishing gradients**: Can't learn long-term dependencies
2. **Sequential processing**: Can't parallelize (slower training)
3. **Limited context window**: Effectively only remembers last few time steps
4. **Sensitivity to sequence length**: Performance degrades for very long sequences

**Solution**: LSTMs (Long Short-Term Memory) - next topic!

## Practical Tips for Training RNNs

### 1. Gradient Clipping

```python
model.compile(
    optimizer=keras.optimizers.Adam(clipnorm=1.0),
    loss='categorical_crossentropy'
)
```

### 2. Batch Normalization

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.SimpleRNN(64, return_sequences=True),
    layers.BatchNormalization(),
    layers.SimpleRNN(64),
    layers.Dense(10, activation='softmax')
])
```

### 3. Dropout for Regularization

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.SimpleRNN(64, dropout=0.2, recurrent_dropout=0.2),
    # dropout: Input dropout
    # recurrent_dropout: Dropout on recurrent connections
    layers.Dense(1, activation='sigmoid')
])
```

### 4. Use Bidirectional RNNs

Process sequence in both directions (forward and backward):

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.Bidirectional(layers.SimpleRNN(64)),
    # Processes sequence left-to-right and right-to-left
    # Concatenates both hidden states → 128-dim output
    layers.Dense(1, activation='sigmoid')
])
```

## Key Takeaways

1. **RNNs process sequences** by maintaining a hidden state that carries information across time steps
2. **Recurrent connections** allow information from previous inputs to influence current processing
3. **Hidden state formula**: hₜ = tanh(Wₓₕ·xₜ + Wₕₕ·hₜ₋₁ + bₕ)
4. **Variable-length sequences**: RNNs naturally handle inputs of different lengths
5. **Vanishing gradient problem**: Limits RNNs to short-term dependencies (~10 time steps)
6. **Exploding gradients**: Solved with gradient clipping
7. **Basic RNNs are limited**: LSTMs and GRUs (next topics) address vanishing gradient issue
8. **Still relevant**: RNNs remain useful for simple sequential tasks and as conceptual foundation

## External Resources

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Christopher Olah's illustrated guide (preview for next topic)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Andrej Karpathy's famous blog post
- [Sequence Models Course (Coursera)](https://www.coursera.org/learn/nlp-sequence-models) - Andrew Ng's comprehensive RNN course

