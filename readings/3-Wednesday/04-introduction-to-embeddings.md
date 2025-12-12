# Introduction to Word Embeddings

## Learning Objectives
- Understand dense vector representations of words and their advantages over one-hot encoding
- Learn the concept of embedding space and semantic similarity
- Explore key properties of embeddings: dimensionality, learned representations, and compositionality
- Recognize how embeddings capture meaning and relationships between words

## Why This Matters

Word embeddings revolutionized NLP and are the foundation of modern language models. Understanding embeddings is critical because:

- **Universal technique**: Used in virtually every modern NLP system (search, translation, chatbots, sentiment analysis)
- **Transfer learning**: Pre-trained embeddings (Word2Vec, GloVe, FastText) enable training with limited data
- **Semantic understanding**: Embeddings capture meaning, enabling models to understand that "dog" and "puppy" are related
- **Foundation for advanced models**: BERT, GPT, and transformer architectures build on embedding principles
- **Production systems**: Embeddings power recommendation systems, semantic search, and content understanding at scale

As you build your NLP foundation this week, embeddings represent a paradigm shift from symbolic (one-hot) to distributed representations. This is where NLP transitions from brittle rule-based systems to intelligent, data-driven models.

## What are Word Embeddings?

### Definition

**Word embeddings** are dense, low-dimensional vector representations of words where semantically similar words have similar vectors.

**Key characteristics:**
- **Dense**: All values are non-zero (unlike sparse one-hot vectors)
- **Low-dimensional**: Typically 100-300 dimensions (vs. 10,000+ for one-hot)
- **Learned**: Trained from large text corpora, not hand-crafted
- **Semantic**: Similar meanings → similar vectors

### Visual Comparison

**One-Hot Encoding (Sparse):**

```
Vocabulary size: 10,000

"king"  → [0, 0, ..., 1, 0, ..., 0]  (10,000 dimensions, 99.99% zeros)
"queen" → [0, 0, ..., 0, 1, 0, ..., 0]
"man"   → [0, 1, 0, ..., 0, 0, ..., 0]
"woman" → [0, 0, ..., 0, 0, 1, ..., 0]

Similarity(king, queen) = 0  (orthogonal)
Similarity(king, man) = 0    (orthogonal)
```

**Word Embeddings (Dense):**

```
Embedding size: 100 (much smaller!)

"king"  → [0.23, -0.15, 0.89, ..., 0.45]  (100 dimensions, all non-zero)
"queen" → [0.25, -0.14, 0.87, ..., 0.43]  (similar to "king")
"man"   → [0.18, -0.22, 0.65, ..., 0.51]
"woman" → [0.20, -0.21, 0.63, ..., 0.49]  (similar to "man")

Similarity(king, queen) = 0.87  (very similar!)
Similarity(king, man) = 0.72    (somewhat similar - both royalty/male)
Similarity(king, pizza) = 0.12  (dissimilar)
```

## The Embedding Space

### Geometric Intuition

Embeddings place words in a continuous vector space where:
- **Distance reflects semantic similarity**: Nearby words have related meanings
- **Directions capture relationships**: Vector arithmetic encodes analogies

**2D Visualization Example** (real embeddings are 100-300D, but we can project to 2D):

```
                    queen •
                          |
                    king •
    
    cat •               dog •
              bird •
    
    pizza •         burger •
                hamburger •
```

Similar words cluster together:
- Royal terms: king, queen
- Animals: cat, dog, bird
- Food: pizza, burger, hamburger

### Semantic Similarity

Measure similarity using **cosine similarity**:

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Example embeddings (simplified)
king = np.array([0.5, 0.8, 0.2])
queen = np.array([0.52, 0.79, 0.21])
pizza = np.array([-0.3, -0.1, 0.9])

sim_king_queen = cosine_similarity(king, queen)
sim_king_pizza = cosine_similarity(king, pizza)

print(f"Similarity(king, queen): {sim_king_queen:.2f}")  # ~0.99 (very similar)
print(f"Similarity(king, pizza): {sim_king_pizza:.2f}")  # ~0.15 (dissimilar)
```

**Cosine similarity ranges:**
- 1.0: Identical direction (very similar)
- 0.0: Orthogonal (unrelated)
- -1.0: Opposite direction (antonyms sometimes)

## Key Properties of Embeddings

### 1. Dimensionality Reduction

Compress sparse one-hot vectors into dense, low-dimensional representations:

```
One-hot:      10,000 dimensions, sparse
Embeddings:   100-300 dimensions, dense

Memory savings: ~99% reduction
Semantic richness: Vastly improved
```

**Why 100-300 dimensions?**
- Enough capacity to capture semantic nuances
- Computationally efficient
- Empirically validated across tasks

### 2. Learned from Data

Embeddings aren't manually designed—they're learned from text:

**Training process** (simplified):
1. Start with random vectors for each word
2. Train on a large corpus (Wikipedia, web crawl, books)
3. Adjust vectors so words appearing in similar contexts become similar
4. Result: Semantically meaningful representations

**Distributional Hypothesis**: "You shall know a word by the company it keeps."
- Words with similar contexts (surrounding words) have similar meanings
- "dog" and "puppy" appear in similar contexts → similar embeddings

### 3. Capturing Semantic Relationships

Embeddings encode analogies and relationships through vector arithmetic:

**Famous example: Word Analogies**

```
king - man + woman ≈ queen

In vector space:
vector(king) - vector(man) + vector(woman) ≈ vector(queen)
```

**How this works:**
- `vector(king) - vector(man)` ≈ "royalty" direction
- Adding `vector(woman)` ≈ female + royalty
- Result is close to `vector(queen)`

**More examples:**

```
Paris - France + Italy ≈ Rome
(capital - country + different country ≈ that country's capital)

bigger - big + small ≈ smaller
(comparative - positive + different adjective ≈ that adjective's comparative)

walking - walk + swim ≈ swimming
(gerund - base + different verb ≈ that verb's gerund)
```

### 4. Multi-Sense Representations (Limitation)

Traditional embeddings assign one vector per word, even if words have multiple meanings:

```
"bank" → [0.1, -0.3, 0.7, ...]

Contexts:
- "I deposited money at the bank" (financial institution)
- "We sat by the river bank" (land alongside water)

Problem: Same vector for both meanings!
```

**Solutions** (beyond this week's scope):
- **Contextualized embeddings** (ELMo, BERT): Different vectors for different contexts
- You'll explore these in future weeks

## Creating Embeddings

### 1. Random Initialization (Baseline)

```python
import numpy as np

# Random embeddings (no semantic meaning)
vocab_size = 10000
embedding_dim = 100

embeddings = np.random.randn(vocab_size, embedding_dim)
# Each word gets a random 100-dimensional vector
```

**Problem**: No semantic structure, purely random.

### 2. Trainable Embeddings (Learn During Task)

```python
from tensorflow import keras

# Embedding layer in Keras
model = keras.Sequential([
    keras.layers.Embedding(
        input_dim=10000,    # Vocabulary size
        output_dim=100,     # Embedding dimension
        input_length=50     # Max sequence length
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation='sigmoid')
])

# Embeddings start random, update during training
# Learn representations specific to your task
```

**When to use:**
- Task-specific data (embeddings optimized for sentiment, translation, etc.)
- Large labeled dataset for your task

**Limitation**: Requires substantial training data.

### 3. Pre-trained Embeddings (Transfer Learning)

Use embeddings trained on massive corpora:

**Popular pre-trained embeddings:**
- **Word2Vec** (Google): Trained on 100B words from Google News
- **GloVe** (Stanford): Trained on 6B tokens from Wikipedia + web crawl
- **FastText** (Facebook): Handles out-of-vocabulary words via subword embeddings

```python
# Load GloVe embeddings (example)
import numpy as np

# GloVe format: word vector_values
# Example line: "the 0.04656 0.21318 -0.0074364 ..."

def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings_index[word] = vector
    return embeddings_index

# Load GloVe 100-dimensional embeddings
glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')

# Get embedding for "king"
king_vec = glove_embeddings['king']
print(king_vec.shape)  # (100,)
```

**Using pre-trained embeddings in Keras:**

```python
# Create embedding matrix
vocab_size = 10000
embedding_dim = 100

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in word_to_idx.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector

# Use in model
model = keras.Sequential([
    keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],  # Initialize with GloVe
        trainable=False              # Freeze or allow fine-tuning
    ),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])
```

**Trainable vs. Frozen:**
- **Frozen** (`trainable=False`): Use pre-trained knowledge as-is
- **Fine-tunable** (`trainable=True`): Adapt embeddings to your task

## Embedding Space Properties

### Clustering by Semantic Category

Words of similar types cluster together:

```
Countries:  USA, Canada, France, Germany, Japan
Cities:     London, Paris, Tokyo, New York
Animals:    dog, cat, elephant, lion
Verbs:      run, walk, jump, swim
```

**Application: Semantic search**

```python
# Find most similar words to "king"
def find_similar(word, embeddings_index, top_n=5):
    if word not in embeddings_index:
        return []
    
    word_vec = embeddings_index[word]
    similarities = {}
    
    for other_word, other_vec in embeddings_index.items():
        if other_word == word:
            continue
        sim = cosine_similarity(word_vec, other_vec)
        similarities[other_word] = sim
    
    # Sort by similarity
    sorted_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similar[:top_n]

# Find words similar to "king"
similar_to_king = find_similar("king", glove_embeddings, top_n=5)
print(similar_to_king)
# [('queen', 0.87), ('monarch', 0.82), ('prince', 0.78), ('throne', 0.72), ('royal', 0.70)]
```

### Directions Encode Relationships

Consistent directions in embedding space represent relationships:

```
Gender direction:
king - queen ≈ man - woman ≈ actor - actress

Verb tense direction:
walk - walked ≈ run - ran ≈ jump - jumped

Comparative direction:
good - better ≈ bad - worse ≈ big - bigger
```

**Practical use: Analogy completion**

```python
def analogy(word_a, word_b, word_c, embeddings_index, top_n=1):
    """
    Solve: word_a is to word_b as word_c is to ?
    Example: king is to queen as man is to ?
    """
    if word_a not in embeddings_index or word_b not in embeddings_index or word_c not in embeddings_index:
        return None
    
    # Compute: word_b - word_a + word_c
    vec_a = embeddings_index[word_a]
    vec_b = embeddings_index[word_b]
    vec_c = embeddings_index[word_c]
    
    result_vec = vec_b - vec_a + vec_c
    
    # Find closest word to result_vec
    similarities = {}
    for word, vec in embeddings_index.items():
        if word in [word_a, word_b, word_c]:
            continue
        sim = cosine_similarity(result_vec, vec)
        similarities[word] = sim
    
    sorted_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similar[:top_n]

# king - man + woman ≈ ?
result = analogy("king", "man", "woman", glove_embeddings)
print(result)  # [('queen', 0.88)]
```

## Advantages Over One-Hot Encoding

| Aspect | One-Hot | Embeddings |
|--------|---------|------------|
| **Dimensionality** | 10,000-100,000+ | 100-300 |
| **Sparsity** | 99.99% zeros | Dense (no zeros) |
| **Semantic similarity** | None | Yes |
| **Analogies** | No | Yes (vector arithmetic) |
| **Transfer learning** | No | Yes (pre-trained) |
| **Memory** | O(V²) | O(V × d), d << V |
| **OOV handling** | Poor | Better (FastText subwords) |

**Example:**

```python
# One-hot: All pairs equally dissimilar
similarity(cat, dog) = 0
similarity(cat, car) = 0
similarity(cat, royalty) = 0

# Embeddings: Captures semantic relationships
similarity(cat, dog) = 0.82      # Both animals
similarity(cat, car) = 0.15      # Unrelated
similarity(cat, royalty) = 0.08  # Unrelated
```

## Practical Example: Sentiment Analysis with Embeddings

```python
import numpy as np
from tensorflow import keras

# Sample data
texts = ["I love this movie", "terrible film", "amazing experience", "worst ever"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=10)

# Model with learned embeddings
model = keras.Sequential([
    keras.layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    # Each word → 32-dimensional dense vector (learned during training)
    
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, np.array(labels), epochs=50, verbose=0)

# Predict
test_text = ["I love this"]
test_seq = tokenizer.texts_to_sequences(test_text)
test_X = pad_sequences(test_seq, maxlen=10)
prediction = model.predict(test_X, verbose=0)
print(f"Prediction: {'Positive' if prediction > 0.5 else 'Negative'}")
```

With embeddings, the model learns that "love" and "amazing" are semantically similar (both positive), enabling better generalization.

## Key Takeaways

1. **Embeddings are dense, low-dimensional vector representations**—typically 100-300 dimensions vs. 10,000+ for one-hot
2. **Semantic similarity**: Similar words have similar vectors (measured by cosine similarity)
3. **Learned from data**: Trained on large corpora using distributional hypothesis ("words in similar contexts have similar meanings")
4. **Vector arithmetic captures relationships**: "king" - "man" + "woman" ≈ "queen"
5. **Transfer learning**: Pre-trained embeddings (Word2Vec, GloVe, FastText) accelerate training
6. **Embedding space**: Words cluster by semantic category; directions encode relationships (gender, tense, etc.)
7. **Vastly superior to one-hot**: Efficient, semantic, generalizable
8. **Next step**: Understanding how embeddings are trained (Word2Vec—next topic)

## External Resources

- [Word2Vec Tutorial](https://arxiv.org/abs/1301.3781) - Original Word2Vec paper (Mikolov et al., Google)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) - Stanford's GloVe project
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Visual explanation by Jay Alammar

