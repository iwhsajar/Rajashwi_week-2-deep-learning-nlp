# Word2Vec: Intuition and Vector Arithmetic

## Learning Objectives
- Understand the two Word2Vec architectures: Skip-gram and CBOW (Continuous Bag of Words)
- Learn how Word2Vec learns word relationships from context
- Explore vector arithmetic and semantic analogies
- Recognize Word2Vec's impact on modern NLP

## Why This Matters

Word2Vec revolutionized NLP in 2013 by demonstrating that simple neural networks could learn rich semantic relationships from unlabeled text. Understanding Word2Vec matters because:

- **Foundation of modern NLP**: Influenced BERT, GPT, and all transformer models
- **Transfer learning pioneer**: Showed that pre-training on large corpora enables few-shot learning
- **Semantic understanding**: First widely-adopted method to capture meaning through vector arithmetic
- **Production systems**: Still used in recommendation systems, search, and content understanding
- **Conceptual clarity**: Word2Vec's simplicity makes it perfect for understanding embedding principles

As you build your NLP foundation this week, Word2Vec represents the breakthrough that made deep learning for text practical. The intuitions you develop here will carry forward to RNNs (Thursday) and beyond.

## The Core Idea

### Learning from Context

**Key insight**: A word's meaning is determined by the words around it (distributional hypothesis).

```
"The cat sat on the mat."
"The dog sat on the mat."

"cat" and "dog" appear in similar contexts → should have similar embeddings
```

Word2Vec trains embeddings by predicting context words from target words (or vice versa).

### Two Architectures

1. **Skip-gram**: Given a target word, predict surrounding context words
2. **CBOW (Continuous Bag of Words)**: Given context words, predict the target word

## Skip-Gram Architecture

### How It Works

**Objective**: Given a word, predict its context (surrounding words).

**Example:**

```
Sentence: "The quick brown fox jumps over the lazy dog"
Window size: 2 (2 words before and after)

Target word: "fox"
Context (within window): ["quick", "brown", "jumps", "over"]

Skip-gram task: Given "fox", predict "quick", "brown", "jumps", "over"
```

### Training Process

**Step 1**: Slide a window over the corpus

```
Sentence: [The, quick, brown, fox, jumps, over, the, lazy, dog]

Window 1 (center="quick", context within ±2):
  Target: "quick" → Context: ["The", "brown", "fox"]

Window 2 (center="brown", context within ±2):
  Target: "brown" → Context: ["The", "quick", "fox", "jumps"]

Window 3 (center="fox", context within ±2):
  Target: "fox" → Context: ["quick", "brown", "jumps", "over"]

... continue for entire corpus
```

**Step 2**: For each (target, context) pair, train the model

```
Input: "fox" (as index, e.g., 2547)
Output: Probability distribution over vocabulary for context words

Model tries to maximize:
P(context word | target word)
```

### The Neural Network

**Architecture (simplified):**

```
Input layer:    One-hot vector of target word (vocab_size)
                     ↓
Hidden layer:   Embedding matrix (vocab_size × embedding_dim)
                     ↓
                Word embedding (embedding_dim)
                     ↓
Output layer:   Softmax over vocabulary (vocab_size)
                     ↓
                Probability distribution over context words
```

**Example:**

```python
# Simplified Skip-gram (conceptual)
vocab_size = 10000
embedding_dim = 100

# Input: "fox" (index 2547)
target_word_idx = 2547
target_one_hot = np.zeros(vocab_size)
target_one_hot[2547] = 1  # One-hot encoding

# Embedding layer (this is what we want to learn!)
embedding_matrix = np.random.randn(vocab_size, embedding_dim)
word_embedding = embedding_matrix[target_word_idx]  # Shape: (100,)

# Output layer: Predict context words
output_weights = np.random.randn(embedding_dim, vocab_size)
logits = np.dot(word_embedding, output_weights)
probabilities = softmax(logits)  # Shape: (vocab_size,)

# High probabilities for context words: "quick", "brown", "jumps", "over"
# Low probabilities for unrelated words: "pizza", "quantum", "zebra"
```

**Training objective**: Adjust `embedding_matrix` so that words appearing in similar contexts get similar embeddings.

### Intuition: Why Skip-Gram Works

```
Training example 1: "cat" → predict ["the", "sat", "on", "mat"]
Training example 2: "dog" → predict ["the", "sat", "on", "mat"]

Since "cat" and "dog" predict similar context words, the model learns similar embeddings for them!
```

## CBOW (Continuous Bag of Words) Architecture

### How It Works

**Objective**: Given context words, predict the target word (reverse of Skip-gram).

**Example:**

```
Sentence: "The quick brown fox jumps over the lazy dog"
Window size: 2

Context: ["quick", "brown", "jumps", "over"]
Target: "fox"

CBOW task: Given ["quick", "brown", "jumps", "over"], predict "fox"
```

### The Neural Network

**Architecture (simplified):**

```
Input layer:    One-hot vectors of context words (multiple)
                     ↓
Hidden layer:   Embedding matrix (shared for all context words)
                     ↓
                Average of context word embeddings
                     ↓
Output layer:   Softmax over vocabulary
                     ↓
                Predicted target word
```

**Example:**

```python
# Simplified CBOW (conceptual)
context_words = ["quick", "brown", "jumps", "over"]  # Indices: [42, 158, 391, 572]
target_word = "fox"  # Index: 2547

# Get embeddings for each context word
context_embeddings = [embedding_matrix[42], embedding_matrix[158], 
                      embedding_matrix[391], embedding_matrix[572]]

# Average context embeddings
avg_context = np.mean(context_embeddings, axis=0)  # Shape: (100,)

# Predict target word
logits = np.dot(avg_context, output_weights)
probabilities = softmax(logits)

# High probability for "fox" (index 2547)
```

### Intuition: Why CBOW Works

```
Example 1: ["the", "sat", "on", "mat"] → predict "cat"
Example 2: ["the", "sat", "on", "mat"] → predict "dog"

The model learns that ["the", "sat", "on", "mat"] context can map to both "cat" and "dog"
→ assigns similar embeddings to "cat" and "dog"
```

## Skip-Gram vs. CBOW

| Aspect | Skip-Gram | CBOW |
|--------|-----------|------|
| **Task** | Target → Context | Context → Target |
| **Input** | Single word | Multiple words (averaged) |
| **Training speed** | Slower | Faster |
| **Rare words** | Better | Worse |
| **Use case** | Larger datasets, rare words | Smaller datasets, frequent words |
| **Default choice** | Yes (generally preferred) | No |

**Rule of thumb**: Use Skip-gram unless you have a small dataset or computational constraints.

## Word2Vec Training in Practice

### Efficient Training Techniques

**1. Negative Sampling**

Problem: Computing softmax over 100,000-word vocabulary is expensive.

Solution: Instead of predicting all context words, distinguish true context words from random "negative" samples.

```python
# For target word "fox" with context word "brown"

Positive example: ("fox", "brown") → label = 1 (true context pair)

Negative examples (randomly sampled):
("fox", "pizza") → label = 0
("fox", "quantum") → label = 0
("fox", "zebra") → label = 0
("fox", "Antarctica") → label = 0

# Train binary classifier: Is this pair (target, context) real or random?
```

This reduces complexity from O(vocab_size) to O(num_negative_samples), typically 5-20.

**2. Subsampling Frequent Words**

Problem: Words like "the", "a", "is" appear constantly but provide little semantic information.

Solution: Randomly skip very frequent words during training.

```python
# Probability of keeping a word
def keep_probability(word_frequency, threshold=1e-5):
    return (np.sqrt(word_frequency / threshold) + 1) * (threshold / word_frequency)

# Frequent words like "the" (frequency = 0.07) → low probability of keeping
# Rare words like "quantum" (frequency = 0.00001) → high probability of keeping
```

### Using Pre-trained Word2Vec

```python
# Using Gensim library
from gensim.models import Word2Vec, KeyedVectors

# Load pre-trained Google News Word2Vec (3 million words, 300 dimensions)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Get embedding for a word
king_vec = model['king']
print(king_vec.shape)  # (300,)

# Find similar words
similar_to_king = model.most_similar('king', topn=5)
print(similar_to_king)
# [('queen', 0.651), ('monarch', 0.623), ('prince', 0.594), ('crown', 0.582), ('kings', 0.571)]
```

## Vector Arithmetic: The Magic of Word2Vec

### Classic Example: King - Man + Woman = Queen

```python
from gensim.models import KeyedVectors

# Load Word2Vec model
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Vector arithmetic
result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(result)
# [('queen', 0.71)]
```

**How it works:**

```
vector(king) - vector(man) + vector(woman) ≈ vector(queen)

Intuition:
- vector(king) - vector(man) ≈ "royalty" direction (removing maleness)
- Add vector(woman) ≈ royalty + femaleness
- Result is close to vector(queen)
```

### More Analogy Examples

**Geography:**

```python
# Paris - France + Italy ≈ Rome
result = model.most_similar(positive=['Paris', 'Italy'], negative=['France'], topn=1)
print(result)
# [('Rome', 0.68)]
```

**Verb Tense:**

```python
# walking - walk + swim ≈ swimming
result = model.most_similar(positive=['walking', 'swim'], negative=['walk'], topn=1)
print(result)
# [('swimming', 0.74)]
```

**Comparatives:**

```python
# good - better + bad ≈ worse
result = model.most_similar(positive=['good', 'worse'], negative=['better'], topn=1)
print(result)
# [('bad', 0.72)]
```

**Company - CEO:**

```python
# Microsoft - Bill_Gates + Apple ≈ Steve_Jobs
result = model.most_similar(positive=['Microsoft', 'Steve_Jobs'], negative=['Bill_Gates'], topn=1)
print(result)
# [('Apple', 0.65)]
```

### Why Vector Arithmetic Works

Word2Vec learns **consistent directions** in embedding space:

```
Gender direction:
king → queen (same direction as) man → woman, actor → actress

Country capital direction:
France → Paris (same direction as) Italy → Rome, Germany → Berlin

Verb tense direction:
walk → walked (same direction as) jump → jumped, run → ran
```

These directions are emergent properties—not explicitly programmed, but learned from data!

## Visualizing Word2Vec Embeddings

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Select words to visualize
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess',
         'dog', 'cat', 'puppy', 'kitten',
         'car', 'truck', 'vehicle', 'automobile']

# Get their embeddings
word_vectors = np.array([model[word] for word in words])

# Reduce to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(word_vectors)

# Plot
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.annotate(word, (x, y), fontsize=12)

plt.title('Word2Vec Embeddings (2D projection)')
plt.show()

# Words cluster by semantic category:
# - Royal terms: king, queen, prince, princess
# - Animals: dog, cat, puppy, kitten
# - Vehicles: car, truck, vehicle, automobile
```

## Training Your Own Word2Vec Model

```python
from gensim.models import Word2Vec

# Sample corpus (list of tokenized sentences)
corpus = [
    ['I', 'love', 'natural', 'language', 'processing'],
    ['Word', 'embeddings', 'are', 'amazing'],
    ['Deep', 'learning', 'revolutionized', 'NLP'],
    ['I', 'love', 'deep', 'learning']
]

# Train Word2Vec model
model = Word2Vec(
    sentences=corpus,
    vector_size=100,      # Embedding dimensionality
    window=5,             # Context window size (±5 words)
    min_count=1,          # Ignore words with frequency < min_count
    workers=4,            # Parallel training
    sg=1                  # 1 = Skip-gram, 0 = CBOW
)

# Get embedding
love_vec = model.wv['love']
print(love_vec.shape)  # (100,)

# Find similar words
similar = model.wv.most_similar('love', topn=3)
print(similar)
# Based on this tiny corpus, results won't be meaningful
# (Word2Vec needs millions of words to learn good embeddings)
```

**Realistic use**: Train on domain-specific corpus (medical texts, legal documents, social media) to capture domain vocabulary.

## Limitations of Word2Vec

### 1. Single Vector per Word

```
"bank" → [0.1, -0.3, 0.7, ...]

"I went to the bank to deposit money."  (financial institution)
"We sat by the river bank."             (land alongside water)

Same embedding for both senses!
```

**Solution**: Contextualized embeddings (ELMo, BERT) - covered in future weeks.

### 2. Out-of-Vocabulary Words

```python
# If "smartphone" wasn't in training corpus:
smartphone_vec = model['smartphone']  # KeyError!

# Workaround: Use subword embeddings (FastText)
```

### 3. No Sentence-Level Understanding

Word2Vec embeds individual words, not sentences:

```
"The movie was not good" → embeddings for ["The", "movie", "was", "not", "good"]
# Averaging these loses negation ("not good" ≠ average of "not" + "good")
```

**Solution**: Sentence encoders (Universal Sentence Encoder, Sentence-BERT).

## Key Takeaways

1. **Word2Vec learns word embeddings from context**—words in similar contexts get similar vectors
2. **Two architectures**: Skip-gram (target → context) and CBOW (context → target)
3. **Skip-gram** predicts context from target word; **CBOW** predicts target from context
4. **Vector arithmetic captures analogies**: "king" - "man" + "woman" ≈ "queen"
5. **Efficient training**: Negative sampling and subsampling make training on billions of words feasible
6. **Pre-trained models** (Google News, Wikipedia) enable transfer learning
7. **Limitations**: Single vector per word (no context), OOV words, no sentence understanding
8. **Impact**: Word2Vec revolutionized NLP and influenced all modern language models

## External Resources

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) - Original Word2Vec paper (Mikolov et al.)
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) - Visual explanation by Jay Alammar
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) - Detailed mathematical walkthrough

