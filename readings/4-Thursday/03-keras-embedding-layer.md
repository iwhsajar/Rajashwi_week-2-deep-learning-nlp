# Keras Embedding Layer

## Learning Objectives
- Understand how the Keras Embedding layer converts word indices to dense vectors
- Learn to create trainable embeddings that learn from your task
- Master integration of pre-trained embeddings (Word2Vec, GloVe) into Keras models
- Recognize when to freeze vs. fine-tune embedding weights

## Why This Matters

The Embedding layer is the bridge between tokenized text and neural networks. Understanding it is critical because:

- **Universal component**: Every text-based deep learning model starts with an embedding layer
- **Transfer learning**: Pre-trained embeddings enable strong performance with limited data
- **Memory efficiency**: Embedding layers handle sparse word indices without creating massive one-hot matrices
- **Task-specific learning**: Trainable embeddings adapt to your specific problem
- **Production systems**: Proper embedding configuration impacts model size, speed, and performance

As you build sequential NLP models this week, the Embedding layer is your first layer in virtually every architecture. Mastering its configuration unlocks both task-specific learning and transfer learning from massive pre-trained corpora.

## What is the Embedding Layer?

### The Problem It Solves

Neural networks need numerical inputs, but text consists of discrete symbols (words).

**Challenge:**
```python
# Tokenized sentence (word indices)
sentence = [45, 123, 7, 891]  # ["I", "love", "deep", "learning"]

# Can't feed indices directly to LSTM/RNN
# Need to convert to dense vectors (embeddings)
```

**Naive solution: One-hot encoding**
```python
vocab_size = 10000
one_hot_matrix = np.zeros((4, vocab_size))  # 4 words × 10000 dimensions
one_hot_matrix[0, 45] = 1   # "I"
one_hot_matrix[1, 123] = 1  # "love"
# ... extremely memory-intensive and sparse!
```

**Better solution: Embedding layer**
```python
from tensorflow.keras.layers import Embedding

embedding_layer = Embedding(input_dim=10000, output_dim=100)
# input_dim: vocabulary size (10,000 words)
# output_dim: embedding dimension (100-dimensional vectors)

# Input: [45, 123, 7, 891] (4 word indices)
# Output: (4, 100) matrix — 4 dense 100-dimensional vectors
```

### How It Works

The Embedding layer is essentially a **lookup table**:

```
Embedding matrix (weights): (vocab_size × embedding_dim)

┌─────────────────────────────────────┐
│ Word 0:  [0.1, -0.3, 0.5, ..., 0.2] │ ← 100-dim vector
│ Word 1:  [0.4, 0.2, -0.1, ..., 0.7] │
│ Word 2:  [-0.2, 0.6, 0.3, ..., -0.1]│
│ ...                                  │
│ Word 45: [0.25, -0.15, 0.8, ..., 0] │ ← "I"
│ ...                                  │
│ Word 123:[0.3, -0.2, 0.7, ..., 0.1] │ ← "love"
│ ...                                  │
└─────────────────────────────────────┘

Input word index 45 → look up row 45 → return 100-dim vector
```

**Key insight**: This lookup is differentiable—during backpropagation, the embedding vectors update based on task loss!

## Creating an Embedding Layer

### Basic Syntax

```python
from tensorflow.keras.layers import Embedding

embedding = Embedding(
    input_dim=10000,      # Vocabulary size
    output_dim=100,       # Embedding dimension (100-300 typical)
    input_length=50       # Optional: max sequence length
)
```

**Parameters:**
- `input_dim`: Vocabulary size (number of unique tokens)
- `output_dim`: Embedding dimensionality (100, 200, 300 common)
- `input_length`: Length of input sequences (optional, helpful for model summary)
- `embeddings_initializer`: How to initialize weights (default: 'uniform')
- `trainable`: Whether to update embeddings during training (default: True)

### Input and Output Shapes

```python
# Input shape: (batch_size, sequence_length)
# Example: 32 sentences, each with 50 words
input_shape = (32, 50)  # 32 sentences × 50 word indices

# Embedding layer
embedding = Embedding(input_dim=10000, output_dim=128, input_length=50)

# Output shape: (batch_size, sequence_length, embedding_dim)
output_shape = (32, 50, 128)  # 32 sentences × 50 words × 128-dim embeddings
```

## Trainable Embeddings (Learn from Task)

### Random Initialization

Default behavior: embeddings start random and learn during training.

```python
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Sample data
texts = ["I love NLP", "Deep learning is great", "Transformers are powerful"]
labels = [1, 1, 1]  # All positive

# Tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)
y = np.array(labels)

# Model with trainable embeddings
model = keras.Sequential([
    layers.Embedding(
        input_dim=1000,     # Vocabulary size
        output_dim=64,      # Learn 64-dim embeddings
        input_length=10     # Sequence length
    ),
    layers.LSTM(32),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50)

# After training, embeddings have learned task-specific representations
embedding_matrix = model.layers[0].get_weights()[0]  # Shape: (1000, 64)
print(f"Learned embedding for word 'love': {embedding_matrix[tokenizer.word_index['love']]}")
```

**When to use trainable embeddings:**
- Large labeled dataset for your task
- Domain-specific vocabulary (medical, legal)
- Task-specific semantic requirements

**Advantages:**
- Embeddings optimized for your specific task
- No dependency on external pre-trained models

**Disadvantages:**
- Requires substantial training data
- Cold start problem (no semantic information initially)

## Pre-trained Embeddings (Transfer Learning)

### Why Pre-trained Embeddings?

Pre-trained embeddings (Word2Vec, GloVe, FastText) are trained on billions of words:
- Capture general semantic relationships
- Enable training with limited data
- Provide warm start (semantic similarity from day one)

### Loading Pre-trained Embeddings

**Step 1: Download embeddings**

```bash
# GloVe (Stanford)
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
# Files: glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt
```

**Step 2: Load into dictionary**

```python
import numpy as np

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

# Load 100-dimensional GloVe embeddings
glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
print(f"Loaded {len(glove_embeddings)} word vectors")  # 400,000 words

# Check embedding
print(f"Embedding for 'king': {glove_embeddings['king'][:10]}...")  # First 10 dimensions
```

**Step 3: Create embedding matrix for your vocabulary**

```python
# Assume you have a tokenizer fitted on your corpus
vocab_size = len(tokenizer.word_index) + 1  # +1 for padding token (index 0)
embedding_dim = 100

# Initialize with zeros
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Fill with GloVe embeddings
for word, idx in tokenizer.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector
    # Words not in GloVe remain zeros (or could use random initialization)

print(f"Embedding matrix shape: {embedding_matrix.shape}")  # (vocab_size, 100)
```

**Step 4: Use in Keras model**

```python
model = keras.Sequential([
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],  # Initialize with pre-trained embeddings
        input_length=50,
        trainable=False              # Freeze weights (don't update during training)
    ),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Freezing vs. Fine-Tuning Embeddings

### Frozen Embeddings (`trainable=False`)

```python
layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=False  # Embeddings won't update during training
)
```

**When to freeze:**
- Small dataset (risk of overfitting if you fine-tune)
- Trust pre-trained embeddings (trained on billions of words)
- Want faster training (fewer parameters to update)

**Advantages:**
- Prevents overfitting on small datasets
- Faster training
- Leverages full power of pre-trained knowledge

**Disadvantages:**
- Embeddings not adapted to your task
- May not capture domain-specific semantics

### Fine-Tunable Embeddings (`trainable=True`)

```python
layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    trainable=True  # Embeddings will update during training
)
```

**When to fine-tune:**
- Moderate to large dataset
- Domain-specific language (medical, legal, social media)
- Want task-specific adaptations

**Advantages:**
- Adapts embeddings to your specific task
- Can learn domain-specific semantics
- Best of both worlds (pre-trained + task-specific)

**Disadvantages:**
- Risk of overfitting on small datasets
- Slower training

### Hybrid Approach: Freeze Then Fine-Tune

```python
# Phase 1: Train with frozen embeddings
model = create_model(trainable=False)
model.fit(X_train, y_train, epochs=10)

# Phase 2: Fine-tune embeddings
model.layers[0].trainable = True
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR
              loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)
```

## Complete Example: Sentiment Analysis with Pre-trained Embeddings

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset
texts = [
    "I absolutely loved this movie",
    "Terrible waste of time",
    "Amazing performance by actors",
    "Boring and predictable",
    "Highly recommend this film"
]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

# Tokenize
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=20)
y = np.array(labels)

# Load GloVe embeddings
glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Create embedding matrix
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in tokenizer.word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector

# Build model with pre-trained embeddings
model = keras.Sequential([
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=20,
        trainable=False  # Frozen GloVe embeddings
    ),
    layers.LSTM(64, dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X, y, epochs=50, verbose=0)

# Predict
test_text = ["This movie was wonderful"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_X = pad_sequences(test_seq, maxlen=20)
prediction = model.predict(test_X, verbose=0)
print(f"Sentiment: {'Positive' if prediction > 0.5 else 'Negative'}")
```

## Embedding Layer with Masking

### Handling Variable-Length Sequences

Padded sequences contain padding tokens (usually 0) that should be ignored:

```python
# Sentence 1: "I love NLP" → [45, 123, 7, 0, 0]  (padded to length 5)
# Sentence 2: "Deep learning is great" → [12, 89, 34, 56, 0]

# Padding tokens (0) should not influence RNN/LSTM computation
```

**Solution: Masking**

```python
model = keras.Sequential([
    layers.Embedding(
        input_dim=10000,
        output_dim=128,
        mask_zero=True  # Automatically mask padding (value 0)
    ),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

# LSTM will ignore time steps where input is 0 (padding)
```

**How it works:**
- Embedding layer creates a mask for positions with value 0
- Subsequent layers (LSTM, GRU) use this mask to skip computation for padding

## Visualizing Learned Embeddings

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Extract embedding weights
embedding_weights = model.layers[0].get_weights()[0]  # Shape: (vocab_size, embedding_dim)

# Select words to visualize
words = ['love', 'hate', 'good', 'bad', 'amazing', 'terrible', 'great', 'awful']
word_indices = [tokenizer.word_index[word] for word in words if word in tokenizer.word_index]
word_vectors = embedding_weights[word_indices]

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(word_vectors)

# Plot
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    if word in tokenizer.word_index:
        x, y = embeddings_2d[i]
        plt.scatter(x, y)
        plt.annotate(word, (x, y), fontsize=12)

plt.title('Learned Word Embeddings (2D projection)')
plt.show()

# Similar words (e.g., "love", "amazing", "great") should cluster together
```

## Practical Tips

### 1. Choosing Embedding Dimension

```python
# Guideline: Roughly vocab_size^0.25
vocab_size = 10000
embedding_dim = int(vocab_size ** 0.25)  # ~17.78 → use 32, 64, or 128

# Common choices: 50, 100, 128, 200, 256, 300
# Larger dim: More capacity, slower training
# Smaller dim: Faster, may lose semantic richness
```

### 2. Regularization

```python
from tensorflow.keras import regularizers

layers.Embedding(
    input_dim=10000,
    output_dim=128,
    embeddings_regularizer=regularizers.l2(1e-4)  # L2 regularization on embeddings
)
```

### 3. Embedding Dropout

```python
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),
    layers.SpatialDropout1D(0.2),  # Dropout entire embedding dimensions
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])
```

### 4. Saving and Loading Trained Embeddings

```python
# Save model (includes embeddings)
model.save('sentiment_model.h5')

# Load model
loaded_model = keras.models.load_model('sentiment_model.h5')

# Extract and save embedding weights separately
embedding_weights = model.layers[0].get_weights()[0]
np.save('trained_embeddings.npy', embedding_weights)

# Load embeddings into new model
loaded_embeddings = np.load('trained_embeddings.npy')
new_model = keras.Sequential([
    layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[loaded_embeddings]
    ),
    # ... rest of architecture
])
```

## Key Takeaways

1. **Embedding layer converts word indices to dense vectors**—the first layer in every text model
2. **Lookup table**: Embedding matrix is (vocab_size × embedding_dim), each word has a row
3. **Trainable embeddings** learn from scratch on your task (requires data)
4. **Pre-trained embeddings** (Word2Vec, GloVe) enable transfer learning with limited data
5. **Freeze vs. fine-tune**: Freeze for small datasets, fine-tune for large/domain-specific data
6. **`mask_zero=True`** automatically ignores padding tokens in sequences
7. **Common dimensions**: 50-300 for embeddings, depends on vocabulary size and task complexity
8. **Hybrid approach**: Start frozen, then fine-tune with lower learning rate

## External Resources

- [Keras Embedding Layer Documentation](https://keras.io/api/layers/core_layers/embedding/) - Official API reference
- [Using Pre-trained Word Embeddings](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) - François Chollet's tutorial
- [GloVe Pre-trained Embeddings](https://nlp.stanford.edu/projects/glove/) - Stanford's GloVe project

