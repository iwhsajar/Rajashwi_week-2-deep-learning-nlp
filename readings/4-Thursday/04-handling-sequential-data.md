# Handling Sequential Data: Padding, Masking, and Batch Processing

## Learning Objectives
- Understand the challenge of variable-length sequences in neural networks
- Master padding strategies: pre-padding vs. post-padding
- Learn masking to ignore padding tokens during computation
- Apply batch processing techniques for efficient training on variable-length data

## Why This Matters

Real-world text data varies wildly in length—tweets are short, reviews are medium, documents are long. Handling this variability correctly is critical because:

- **Model input requirements**: Neural networks expect fixed-size inputs
- **Memory efficiency**: Proper padding and batching prevent OOM (out-of-memory) errors
- **Training stability**: Masking ensures padding doesn't influence gradient updates
- **Production systems**: Inference batching requires consistent handling of variable-length inputs
- **Performance**: Incorrect padding/masking can degrade model accuracy by 10-30%

As you build sequential NLP models this week, proper sequence handling is the difference between a model that learns effectively and one that trains on noise. This is a fundamental skill for production-ready NLP systems.

## The Variable-Length Sequence Problem

### Real-World Text Varies in Length

```python
# Example corpus
sentences = [
    "Hi",                                           # 1 word
    "I love NLP",                                   # 3 words
    "Deep learning is revolutionizing AI",          # 5 words
    "The quick brown fox jumps over the lazy dog"   # 9 words
]

# After tokenization (word indices)
sequences = [
    [42],                           # Length 1
    [15, 89, 234],                  # Length 3
    [56, 123, 78, 456, 789],        # Length 5
    [12, 34, 56, 78, 90, 11, 22, 33, 44]  # Length 9
]
```

### Why Neural Networks Need Fixed Length

```python
# Neural network layers expect consistent input shapes
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=???),
    # What length to specify when sentences vary from 1 to 9 words?
    
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

# Solution: Pad all sequences to the same length
```

## Padding Sequences

### What is Padding?

Add special padding tokens (usually 0) to make all sequences the same length.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

sequences = [
    [42],
    [15, 89, 234],
    [56, 123, 78, 456, 789]
]

# Pad to length of longest sequence (5)
padded = pad_sequences(sequences)
print(padded)
# [[  0   0   0   0  42]     ← 4 padding tokens added
#  [  0   0  15  89 234]     ← 2 padding tokens added
#  [ 56 123  78 456 789]]    ← no padding needed
```

### Pre-Padding vs. Post-Padding

**Pre-padding** (default): Add padding at the beginning

```python
padded_pre = pad_sequences(sequences, padding='pre')
# [[  0   0   0   0  42]
#  [  0   0  15  89 234]
#  [ 56 123  78 456 789]]
```

**Post-padding**: Add padding at the end

```python
padded_post = pad_sequences(sequences, padding='post')
# [[ 42   0   0   0   0]
#  [ 15  89 234   0   0]
#  [ 56 123  78 456 789]]
```

**Which to use?**

| Task | Recommendation | Reason |
|------|----------------|--------|
| **Classification (many-to-one)** | Pre-padding | RNN's final hidden state (for classification) isn't diluted by padding |
| **Sequence-to-sequence** | Post-padding | More intuitive alignment with output |
| **Bidirectional models** | Either | Model sees both directions, less impact |

### Setting Maximum Length

**Truncate long sequences**, pad short sequences:

```python
sequences = [
    [1, 2, 3],                    # Length 3
    [4, 5, 6, 7, 8, 9, 10],      # Length 7 (will be truncated)
]

# Pad/truncate to length 5
padded = pad_sequences(sequences, maxlen=5)
print(padded)
# [[0 0 1 2 3]       ← padded to 5
#  [6 7 8 9 10]]     ← truncated to 5 (first 2 tokens removed)
```

**Truncation strategies:**

```python
# Truncate from beginning (default for pre-padding)
padded = pad_sequences(sequences, maxlen=5, truncating='pre')
# [[0 0 1 2 3]
#  [6 7 8 9 10]]  ← keeps last 5 tokens

# Truncate from end
padded = pad_sequences(sequences, maxlen=5, truncating='post')
# [[0 0 1 2 3]
#  [4 5 6 7 8]]   ← keeps first 5 tokens
```

### Choosing `maxlen`

**Option 1: Fixed maximum**

```python
maxlen = 100  # Fixed length (e.g., for tweets, reviews)
padded = pad_sequences(sequences, maxlen=maxlen)
```

**Option 2: Percentile-based**

```python
import numpy as np

# Compute sequence lengths
lengths = [len(seq) for seq in sequences]

# Use 95th percentile (covers 95% of data without excessive padding)
maxlen = int(np.percentile(lengths, 95))
print(f"95th percentile length: {maxlen}")

padded = pad_sequences(sequences, maxlen=maxlen)
```

**Option 3: Longest sequence (not recommended for large datasets)**

```python
maxlen = max(len(seq) for seq in sequences)
# Risk: One very long sequence causes excessive padding for all others
```

### Custom Padding Value

```python
# Use value other than 0 (if 0 is a valid token in your vocabulary)
padded = pad_sequences(sequences, maxlen=5, value=-1)
print(padded)
# [[-1 -1  1  2  3]
#  [ 6  7  8  9 10]]
```

## Masking: Ignoring Padding

### The Problem with Padding

Padding tokens are artificial—they shouldn't influence model training:

```python
sequence = [42, 0, 0, 0, 0]  # 1 real word + 4 padding

# Without masking:
# Model computes hidden states for all 5 positions
# Padding (0) incorrectly influences learned representations
```

### Solution: Masking

Tell the model to **ignore** certain positions (padding):

```python
from tensorflow.keras.layers import Embedding, LSTM

model = keras.Sequential([
    Embedding(
        input_dim=10000,
        output_dim=128,
        mask_zero=True  # Automatically mask positions where input = 0
    ),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Now LSTM only processes non-zero positions
# Padding (0) is skipped entirely
```

### How Masking Works

**Under the hood:**

1. Embedding layer creates a boolean mask:
```python
input = [42, 15, 89, 0, 0]
mask = [True, True, True, False, False]  # False = ignore
```

2. Subsequent layers (LSTM, GRU) receive the mask and skip computation for `False` positions:
```python
# LSTM only computes hidden states for positions 0, 1, 2
# Positions 3, 4 (padding) are skipped
```

### Manual Masking Layer

```python
from tensorflow.keras.layers import Masking

model = keras.Sequential([
    Masking(mask_value=0.0),  # Mask positions where all features are 0
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# Useful when not using Embedding layer (e.g., time series data)
```

### Masking with Custom Padding Value

```python
# If you padded with -1 instead of 0
model = keras.Sequential([
    Embedding(input_dim=10000, output_dim=128),
    Masking(mask_value=-1.0),  # Mask positions with value -1
    LSTM(64),
    Dense(1, activation='sigmoid')
])
```

## Batch Processing Variable-Length Sequences

### Batching Basics

Neural networks train on **batches** of examples for efficiency:

```python
# Batch of 3 sequences
batch = [
    [1, 2, 3, 4, 5],
    [6, 7, 8],
    [9, 10]
]

# Problem: Different lengths! Can't form a tensor
```

### Solution: Pad Within Batch

```python
# Pad to longest sequence in batch
batch_padded = pad_sequences(batch, padding='post')
print(batch_padded)
# [[1  2  3  4  5]
#  [6  7  8  0  0]
#  [9 10  0  0  0]]

# Now can be stacked into a tensor: (3, 5)
```

### Dynamic Batching (Minimize Padding)

**Problem**: Global `maxlen` causes excessive padding.

```
Dataset:
- 90% of sequences: 10-20 tokens
- 10% of sequences: 100+ tokens

Global maxlen = 100:
- 90% of data has 80-90 padding tokens (wasteful!)
```

**Solution: Batch by similar lengths**

```python
import numpy as np

def batch_by_length(sequences, batch_size=32):
    # Sort by length
    sorted_sequences = sorted(sequences, key=len)
    
    # Create batches of similar-length sequences
    batches = []
    for i in range(0, len(sorted_sequences), batch_size):
        batch = sorted_sequences[i:i+batch_size]
        # Pad only to longest in this batch
        padded_batch = pad_sequences(batch, padding='post')
        batches.append(padded_batch)
    
    return batches

# Example
sequences = [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10], [11, 12, 13]]
batches = batch_by_length(sequences, batch_size=2)

# Batch 1: [[6, 0], [1, 2]]           ← max_len = 2
# Batch 2: [[3, 4, 5, 0], [11, 12, 13, 0]]  ← max_len = 4
# Batch 3: [[7, 8, 9, 10]]            ← max_len = 4
```

**Benefits:**
- Less padding waste
- Faster training (fewer unnecessary computations)
- Better GPU utilization

## Practical Example: Text Classification with Proper Handling

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dataset with variable-length sentences
texts = [
    "Great movie!",
    "Terrible film, waste of time",
    "Amazing experience",
    "Boring and predictable plot with no character development",
    "Loved it"
]
labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative

# Tokenize
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

print("Original sequences:")
for seq in sequences:
    print(f"Length {len(seq)}: {seq}")

# Determine maxlen based on 95th percentile
lengths = [len(seq) for seq in sequences]
maxlen = int(np.percentile(lengths, 95))
print(f"\n95th percentile length: {maxlen}")

# Pad sequences
X = pad_sequences(sequences, maxlen=maxlen, padding='pre')
y = np.array(labels)

print(f"\nPadded sequences shape: {X.shape}")
print("Padded sequences:")
print(X)

# Build model with masking
model = keras.Sequential([
    layers.Embedding(
        input_dim=1000,
        output_dim=32,
        input_length=maxlen,
        mask_zero=True  # Ignore padding
    ),
    layers.LSTM(16),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X, y, epochs=50, verbose=0)

# Predict on new text (automatic padding)
test_text = ["This is great"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_X = pad_sequences(test_seq, maxlen=maxlen, padding='pre')
prediction = model.predict(test_X, verbose=0)
print(f"\nPrediction: {'Positive' if prediction > 0.5 else 'Negative'}")
```

## Handling Sequences in Production

### Inference on Single Examples

```python
# During training: batch processing
X_train = pad_sequences(train_sequences, maxlen=100)

# During inference: single examples
def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='pre')
    prediction = model.predict(padded, verbose=0)
    return "Positive" if prediction > 0.5 else "Negative"

result = predict_sentiment("I loved this movie!")
print(result)
```

### Batched Inference (Production)

```python
def predict_batch(texts, batch_size=32):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=100, padding='pre')
    
    predictions = []
    for i in range(0, len(padded), batch_size):
        batch = padded[i:i+batch_size]
        batch_preds = model.predict(batch, verbose=0)
        predictions.extend(batch_preds)
    
    return np.array(predictions)

# Efficient batch prediction
test_texts = ["Great!", "Terrible", "Amazing", "Awful"]
results = predict_batch(test_texts)
print(results)
```

## Advanced Techniques

### 1. Pack Padded Sequences (PyTorch-style)

In PyTorch, `pack_padded_sequence` avoids computing on padding:

```python
# TensorFlow/Keras handles this automatically with mask_zero=True
# No manual packing needed
```

### 2. Bucketing

Group sequences into length buckets:

```python
def create_buckets(sequences, bucket_boundaries=[10, 20, 50, 100]):
    buckets = {boundary: [] for boundary in bucket_boundaries}
    buckets['overflow'] = []
    
    for seq in sequences:
        length = len(seq)
        assigned = False
        for boundary in sorted(bucket_boundaries):
            if length <= boundary:
                buckets[boundary].append(seq)
                assigned = True
                break
        if not assigned:
            buckets['overflow'].append(seq)
    
    return buckets

# Process each bucket with its own maxlen
sequences = [[1, 2], [3, 4, 5, 6], [7] * 15, [8] * 60]
buckets = create_buckets(sequences)

for boundary, bucket_seqs in buckets.items():
    if bucket_seqs:
        padded = pad_sequences(bucket_seqs, maxlen=boundary if isinstance(boundary, int) else 200)
        print(f"Bucket {boundary}: {len(bucket_seqs)} sequences padded to {padded.shape[1]}")
```

### 3. Dynamic RNN (TensorFlow 1.x-style)

Not needed in TensorFlow 2.x / Keras—masking handles this automatically.

## Common Pitfalls

### 1. Forgetting to Pad Test Data

```python
# WRONG
X_train = pad_sequences(train_sequences, maxlen=100)
X_test = test_sequences  # Forgot to pad!

# CORRECT
X_train = pad_sequences(train_sequences, maxlen=100)
X_test = pad_sequences(test_sequences, maxlen=100)
```

### 2. Different Padding for Train and Test

```python
# WRONG
X_train = pad_sequences(train_sequences, maxlen=100, padding='pre')
X_test = pad_sequences(test_sequences, maxlen=100, padding='post')
# Inconsistent padding confuses the model!

# CORRECT
X_train = pad_sequences(train_sequences, maxlen=100, padding='pre')
X_test = pad_sequences(test_sequences, maxlen=100, padding='pre')
```

### 3. Not Using Masking

```python
# WRONG: Model learns from padding noise
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128),  # No mask_zero
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])

# CORRECT: Model ignores padding
model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, mask_zero=True),
    layers.LSTM(64),
    layers.Dense(1, activation='sigmoid')
])
```

### 4. Excessive `maxlen`

```python
# WRONG: One outlier causes massive padding
maxlen = max(len(seq) for seq in sequences)  # Could be 1000+ tokens

# CORRECT: Use percentile
maxlen = int(np.percentile([len(seq) for seq in sequences], 95))
```

## Key Takeaways

1. **Padding makes variable-length sequences uniform** for batch processing
2. **Pre-padding vs. post-padding**: Use pre-padding for classification, post-padding for seq2seq
3. **Choose `maxlen` wisely**: 95th percentile balances coverage and padding waste
4. **Masking with `mask_zero=True`** ensures padding doesn't influence training
5. **Batch by length** to minimize padding overhead (advanced optimization)
6. **Consistent padding** between train and test is critical
7. **Inference requires same padding strategy** as training
8. **Keras handles complexity**: `mask_zero=True` automatically propagates masks through layers

## External Resources

- [Keras Masking and Padding Guide](https://www.tensorflow.org/guide/keras/masking_and_padding) - Official TensorFlow documentation
- [Understanding Masking in Sequence Models](https://machinelearningmastery.com/how-to-use-masking-in-keras/) - Practical tutorial
- [Efficient Batching for Variable-Length Sequences](https://arxiv.org/abs/1506.06724) - Research on bucketing strategies

