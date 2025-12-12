# One-Hot Encoding for Text

## Learning Objectives
- Understand how one-hot encoding represents words as vectors
- Learn to construct one-hot vectors for text data
- Recognize the limitations of one-hot encoding: sparsity and lack of semantic similarity
- Appreciate why dense embeddings (next topic) are superior for most NLP tasks

## Why This Matters

One-hot encoding is the simplest way to convert words to numerical vectors, making it a natural starting point for understanding text representation. While it's rarely used in modern deep learning (dense embeddings have superseded it), understanding one-hot encoding is important because:

- **Conceptual foundation**: It clarifies why dense embeddings are necessary
- **Output layers**: One-hot encoding is still used for output labels in classification tasks
- **Debugging**: Understanding sparse representations helps interpret model behavior
- **Historical context**: Many classic NLP techniques relied on one-hot encoding
- **Interviews**: Explaining one-hot limitations demonstrates understanding of representation learning

As you build your NLP foundation this week, one-hot encoding represents the bridge between raw text and neural-network-ready representations. Its limitations motivate the embedding techniques you'll learn next.

## What is One-Hot Encoding?

### Definition

**One-hot encoding** represents each word as a vector where:
- Vector length equals vocabulary size
- One position is set to 1 (the position corresponding to the word)
- All other positions are 0

### Visual Example

```
Vocabulary: ["cat", "dog", "bird", "fish"]
Vocabulary size: 4

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"bird" → [0, 0, 1, 0]
"fish" → [0, 0, 0, 1]
```

Each word is represented by a unique vector with exactly one "hot" (1) position.

### Mathematical Formulation

For a vocabulary of size `V`, each word `w` is represented as a vector `v ∈ {0,1}^V` where:

```
v[i] = 1 if i corresponds to word w
v[i] = 0 otherwise
```

## Creating One-Hot Encodings

### Manual Implementation

```python
import numpy as np

# Define vocabulary
vocabulary = ["I", "love", "NLP", "and", "deep", "learning"]
vocab_size = len(vocabulary)

# Create word-to-index mapping
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
print(word_to_idx)
# {'I': 0, 'love': 1, 'NLP': 2, 'and': 3, 'deep': 4, 'learning': 5}

# One-hot encode a word
def one_hot_encode(word, word_to_idx, vocab_size):
    vector = np.zeros(vocab_size)
    if word in word_to_idx:
        vector[word_to_idx[word]] = 1
    return vector

# Encode "love"
love_vector = one_hot_encode("love", word_to_idx, vocab_size)
print(love_vector)
# [0. 1. 0. 0. 0. 0.]
#     ↑ Position 1 (index of "love")

# Encode "NLP"
nlp_vector = one_hot_encode("NLP", word_to_idx, vocab_size)
print(nlp_vector)
# [0. 0. 1. 0. 0. 0.]
#        ↑ Position 2 (index of "NLP")
```

### Using NumPy

```python
# More efficient with NumPy
def one_hot_encode_batch(words, word_to_idx, vocab_size):
    indices = [word_to_idx.get(word, -1) for word in words]
    # Filter out unknown words
    valid_indices = [idx for idx in indices if idx != -1]
    
    one_hot = np.zeros((len(valid_indices), vocab_size))
    one_hot[np.arange(len(valid_indices)), valid_indices] = 1
    return one_hot

words = ["I", "love", "NLP"]
one_hot_matrix = one_hot_encode_batch(words, word_to_idx, vocab_size)
print(one_hot_matrix)
# [[1. 0. 0. 0. 0. 0.]   ← "I"
#  [0. 1. 0. 0. 0. 0.]   ← "love"
#  [0. 0. 1. 0. 0. 0.]]  ← "NLP"
```

### Using Keras

```python
from tensorflow.keras.utils import to_categorical

# Word indices
word_indices = [0, 1, 2]  # ["I", "love", "NLP"]
vocab_size = 6

# One-hot encode
one_hot = to_categorical(word_indices, num_classes=vocab_size)
print(one_hot)
# [[1. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]]
```

### Using scikit-learn

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Word indices (as column vector)
word_indices = np.array([0, 1, 2, 3]).reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
one_hot = encoder.fit_transform(word_indices)
print(one_hot)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
```

## One-Hot Encoding for Sentences

To represent a sentence, one-hot encode each word:

```python
# Sentence: "I love NLP"
sentence = ["I", "love", "NLP"]

# One-hot encode each word
sentence_encoded = [one_hot_encode(word, word_to_idx, vocab_size) for word in sentence]
sentence_matrix = np.array(sentence_encoded)

print(sentence_matrix)
# [[1. 0. 0. 0. 0. 0.]   ← "I"
#  [0. 1. 0. 0. 0. 0.]   ← "love"
#  [0. 0. 1. 0. 0. 0.]]  ← "NLP"
# Shape: (3, 6) — 3 words, vocab size 6
```

**Bag of Words**: Sum or average one-hot vectors to get sentence representation:

```python
# Bag of words (sum)
bow = sentence_matrix.sum(axis=0)
print(bow)
# [1. 1. 1. 0. 0. 0.]
# Indicates presence of "I", "love", "NLP" (loses word order)

# Normalized bag of words (average)
bow_normalized = sentence_matrix.mean(axis=0)
print(bow_normalized)
# [0.333 0.333 0.333 0.0 0.0 0.0]
```

## Limitations of One-Hot Encoding

### 1. High Dimensionality (Curse of Dimensionality)

**Problem**: Vector size equals vocabulary size.

```python
# Small vocabulary
vocab_small = ["cat", "dog", "bird"]
vocab_size_small = 3
# One-hot vector length: 3 ✓ Manageable

# Realistic vocabulary
vocab_large = ["word1", "word2", ..., "word50000"]
vocab_size_large = 50000
# One-hot vector length: 50,000 ✗ Unwieldy
```

**Consequences:**
- Memory intensive (50,000 dimensions per word)
- Computationally expensive
- Difficult to learn patterns (extremely sparse input space)

### 2. Sparsity

**Problem**: 99.9%+ of vector elements are zeros.

```python
# Vocabulary size: 10,000
vocab_size = 10000

# One-hot vector for "cat"
cat_vector = np.zeros(10000)
cat_vector[42] = 1  # Only 1 non-zero value

# Sparsity
sparsity = (np.sum(cat_vector == 0) / len(cat_vector)) * 100
print(f"Sparsity: {sparsity}%")
# Sparsity: 99.99%
```

**Consequences:**
- Inefficient storage
- Neural networks struggle with extremely sparse inputs
- Gradient updates affect only one dimension per word

### 3. No Semantic Similarity

**Problem**: All words are equally different.

```python
# Calculate similarity (dot product)
cat_vec = np.array([1, 0, 0, 0, 0])
dog_vec = np.array([0, 1, 0, 0, 0])
bird_vec = np.array([0, 0, 1, 0, 0])
car_vec = np.array([0, 0, 0, 1, 0])

# Similarity: cat vs dog (both animals)
similarity_cat_dog = np.dot(cat_vec, dog_vec)
print(f"Cat-Dog similarity: {similarity_cat_dog}")  # 0

# Similarity: cat vs car (different categories)
similarity_cat_car = np.dot(cat_vec, car_vec)
print(f"Cat-Car similarity: {similarity_cat_car}")  # 0

# All pairs have similarity = 0 (orthogonal vectors)
# One-hot encoding doesn't capture that "cat" and "dog" are more similar than "cat" and "car"
```

**Key insight**: One-hot vectors are **orthogonal** (perpendicular in vector space), meaning:
- Every pair of words is equally dissimilar
- No notion of semantic or syntactic similarity
- Model can't leverage relatedness ("cat" and "dog" must be learned independently)

### 4. Cannot Handle Out-of-Vocabulary (OOV) Words

**Problem**: Unseen words have no representation.

```python
# Vocabulary
vocabulary = ["cat", "dog", "bird"]
word_to_idx = {"cat": 0, "dog": 1, "bird": 2}
vocab_size = 3

# Encode known word
cat_vector = one_hot_encode("cat", word_to_idx, vocab_size)
print(cat_vector)  # [1. 0. 0.]

# Encode unknown word
fish_vector = one_hot_encode("fish", word_to_idx, vocab_size)
print(fish_vector)  # [0. 0. 0.] — All zeros (no information!)

# Or map to special <UNK> token
word_to_idx_with_unk = {"<UNK>": 0, "cat": 1, "dog": 2, "bird": 3}
def encode_with_unk(word):
    idx = word_to_idx_with_unk.get(word, 0)  # Default to <UNK>
    vector = np.zeros(len(word_to_idx_with_unk))
    vector[idx] = 1
    return vector

fish_vector_unk = encode_with_unk("fish")
print(fish_vector_unk)  # [1. 0. 0. 0.] — Maps to <UNK>
# But "fish", "zebra", "elephant" all have same representation!
```

### 5. No Shared Information Between Related Words

**Problem**: Model treats "run", "running", "ran" as completely independent.

```python
# These are all forms of "run" but have no shared representation
run_vec = [1, 0, 0, 0, 0]
running_vec = [0, 1, 0, 0, 0]
ran_vec = [0, 0, 1, 0, 0]

# Model must learn their relationship from scratch
# With embeddings (next topic), these would have similar representations
```

## When One-Hot Encoding is Still Used

Despite limitations, one-hot encoding remains useful in specific contexts:

### 1. Output Layer (Classification)

```python
# Multi-class classification: Predict one of 10 classes
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')  # One-hot output
])

# True label: Class 3
y_true = to_categorical([3], num_classes=10)
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# Model prediction (probabilities over 10 classes)
y_pred = model.predict(X)
# [0.05, 0.02, 0.1, 0.7, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01]
#                      ↑ Highest probability for class 3
```

### 2. Small, Fixed Vocabulary (Feature Engineering)

```python
# Days of week (only 7 options)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# One-hot encoding makes sense (small, fixed vocabulary)
```

### 3. Categorical Features in Tabular Data

```python
# Country feature (e.g., for a machine learning model predicting sales)
countries = ["USA", "Canada", "Mexico", "UK", "France"]
# One-hot encode categorical features for traditional ML models
```

## Comparing One-Hot to Dense Embeddings (Preview)

**One-Hot Encoding:**

```
Vocabulary size: 10,000
Vector size: 10,000 (sparse, 99.99% zeros)
"cat" → [0, 0, ..., 1, ..., 0] (10,000 dimensions)
"dog" → [0, 0, ..., 0, 1, 0, ..., 0]
Similarity(cat, dog) = 0 (orthogonal)
```

**Dense Embeddings** (next topic):

```
Vocabulary size: 10,000
Vector size: 100-300 (dense, all values non-zero)
"cat" → [0.2, -0.5, 0.8, 0.1, ..., 0.3] (100 dimensions)
"dog" → [0.25, -0.48, 0.75, 0.15, ..., 0.28]
Similarity(cat, dog) = 0.92 (very similar!)
```

**Key differences:**

| Aspect | One-Hot | Dense Embeddings |
|--------|---------|------------------|
| **Dimensionality** | V (vocabulary size) | 100-300 (fixed, small) |
| **Sparsity** | 99.9%+ zeros | All values non-zero |
| **Semantic similarity** | No (all orthogonal) | Yes (similar words have similar vectors) |
| **OOV handling** | Poor (map to <UNK>) | Better (use subword embeddings) |
| **Learned** | No | Yes (from data) |
| **Memory** | High (VxV matrix) | Low (Vx100-300 matrix) |

## Practical Example: One-Hot for Simple Classification

```python
import numpy as np
from tensorflow import keras

# Sample dataset: Sentiment analysis
texts = ["I love this", "I hate this", "This is great", "This is terrible"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Build vocabulary
vocabulary = sorted(set(" ".join(texts).lower().split()))
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
vocab_size = len(vocabulary)

print(f"Vocabulary ({vocab_size} words): {vocabulary}")

# One-hot encode sentences (bag of words)
def text_to_bow(text, word_to_idx, vocab_size):
    words = text.lower().split()
    bow = np.zeros(vocab_size)
    for word in words:
        if word in word_to_idx:
            bow[word_to_idx[word]] += 1  # Count occurrences
    return bow

X = np.array([text_to_bow(text, word_to_idx, vocab_size) for text in texts])
y = np.array(labels)

print(f"Input shape: {X.shape}")  # (4, vocab_size)

# Simple model
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(vocab_size,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50, verbose=0)

# Predict
test_text = "I love great"
test_vec = text_to_bow(test_text, word_to_idx, vocab_size).reshape(1, -1)
prediction = model.predict(test_vec, verbose=0)
print(f"Prediction for '{test_text}': {'Positive' if prediction > 0.5 else 'Negative'}")
```

This works for toy examples but doesn't scale or capture semantic meaning.

## Key Takeaways

1. **One-hot encoding represents each word as a sparse binary vector**—one "hot" position, all others zero
2. **Vector length equals vocabulary size**—for 50,000 words, vectors are 50,000-dimensional
3. **Extremely sparse**: 99.9%+ of values are zero, leading to memory and computational inefficiency
4. **No semantic similarity**: All words are equally different (orthogonal vectors)
5. **Cannot capture relationships**: "run" and "running" are treated as completely unrelated
6. **Poor OOV handling**: Unseen words map to all-zeros or generic `<UNK>` token
7. **Still used for output layers** in classification tasks (one-hot labels)
8. **Motivation for embeddings**: All these limitations are addressed by dense word embeddings (next topic)

## External Resources

- [One-Hot Encoding Explained](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) - Comprehensive tutorial
- [Understanding Word Embeddings](https://arxiv.org/abs/1301.3781) - Word2Vec paper showing superiority over one-hot
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) - Original Word2Vec paper (preview for next topic)

