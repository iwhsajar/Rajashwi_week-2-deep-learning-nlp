# Tokenization and Text Processing

## Learning Objectives
- Understand what tokenization is and why it's the foundation of NLP
- Learn different tokenization strategies: word-level, character-level, and subword-level
- Explore Byte Pair Encoding (BPE) and other subword tokenization methods
- Master text cleaning and preprocessing pipelines

## Why This Matters

Tokenization is the first critical step in every NLP pipeline. The quality of your tokenization directly impacts model performance:

- **Model input**: Neural networks can't process raw text—tokenization converts text into processable units
- **Vocabulary size**: Tokenization strategy determines model complexity and memory requirements
- **Handling rare words**: Subword tokenization (BPE) enables handling of previously unseen words
- **Language-specific challenges**: Different languages require different tokenization approaches
- **Production systems**: Efficient tokenization is crucial for real-time applications

As you build your NLP foundation this week, understanding tokenization is essential. Poor tokenization leads to information loss, excessive vocabulary sizes, and degraded model performance—regardless of how sophisticated your neural architecture is.

## What is Tokenization?

### Definition

**Tokenization** is the process of breaking text into smaller units called **tokens**. These tokens are the basic building blocks that NLP models process.

**Simple example:**

```
Text: "I love natural language processing!"
Tokens: ["I", "love", "natural", "language", "processing", "!"]
```

### Why Tokenize?

Neural networks operate on numbers, not raw text. Tokenization is the first step in converting text to numerical form:

```
Text → Tokens → Numbers → Embeddings → Neural Network
```

The choice of tokenization strategy affects:
- **Vocabulary size**: How many unique tokens exist
- **Semantic granularity**: Whether tokens carry meaning
- **Handling of rare words**: Can the model process words it hasn't seen?

## Text Preprocessing: Cleaning Before Tokenization

Before tokenization, text often needs cleaning:

### Common Preprocessing Steps

```python
import re
import string

def preprocess_text(text):
    # 1. Lowercase (reduces vocabulary size)
    text = text.lower()
    # "Hello World" → "hello world"
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # "Check out http://example.com" → "Check out "
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # "<p>Hello</p>" → "Hello"
    
    # 4. Remove special characters (optional, depends on task)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # "Hello, world!" → "Hello world"
    
    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # "Hello    world" → "Hello world"
    
    return text

# Example
text = "  Hello, World! Visit http://example.com  "
cleaned = preprocess_text(text)
print(cleaned)  # "hello world visit"
```

### When NOT to Remove Punctuation

For some tasks, punctuation carries meaning:

```
# Sentiment analysis
"This is great!" (positive, emphatic)
"This is great" (positive, neutral)

# Question classification
"What time is it?" (question)
"What time is it" (ambiguous without '?')
```

**Rule of thumb**: Clean only what truly adds noise for your specific task.

## Tokenization Strategies

### 1. Word-Level Tokenization

Split text into words (most intuitive approach).

**Simple whitespace splitting:**

```python
text = "I love natural language processing"
tokens = text.split()
print(tokens)
# ['I', 'love', 'natural', 'language', 'processing']
```

**Problems with simple splitting:**

```python
text = "Don't split contractions incorrectly!"
tokens = text.split()
print(tokens)
# ['Don't', 'split', 'contractions', 'incorrectly!']
# Issues: "Don't" kept intact (good), but "incorrectly!" includes punctuation (bad)
```

**Better word tokenization with NLTK:**

```python
import nltk
nltk.download('punkt')

text = "Don't split contractions incorrectly!"
tokens = nltk.word_tokenize(text)
print(tokens)
# ['Do', "n't", 'split', 'contractions', 'incorrectly', '!']
# Handles contractions and separates punctuation
```

**Using spaCy (production-grade):**

```python
import spacy

nlp = spacy.load("en_core_web_sm")
text = "Don't split contractions incorrectly!"
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)
# ['Do', "n't", 'split', 'contractions', 'incorrectly', '!']
```

**Advantages of word-level tokenization:**
- Intuitive and human-readable
- Tokens often have clear semantic meaning
- Works well for languages with clear word boundaries (English, Spanish)

**Disadvantages:**
- Large vocabulary size (100,000+ unique words in English)
- Can't handle out-of-vocabulary (OOV) words
  - Model sees "running" during training
  - Encounters "jogging" during inference → treats as unknown
- Struggles with morphologically rich languages (Finnish, Turkish)
- Compound words create issues (German: "Donaudampfschifffahrtsgesellschaft" = Danube steamship company)

### 2. Character-Level Tokenization

Split text into individual characters.

```python
text = "Hello!"
tokens = list(text)
print(tokens)
# ['H', 'e', 'l', 'l', 'o', '!']
```

**Advantages:**
- Extremely small vocabulary (26 letters + punctuation + digits ≈ 100 tokens)
- No out-of-vocabulary problem (can handle any word)
- Works across all languages
- Handles typos and misspellings gracefully

**Disadvantages:**
- Sequences become very long
  - "natural language processing" → 27 characters (vs. 3 words)
- Individual characters carry little semantic meaning
  - Hard for model to learn that "c-a-t" relates to felines
- Computationally expensive (longer sequences to process)

**When to use:**
- Tasks involving character-level patterns (spelling correction, text generation)
- Languages without clear word boundaries (Chinese, Japanese—though subword is better)
- Handling noisy text (social media with typos)

### 3. Subword Tokenization (Best of Both Worlds)

Split text into meaningful subword units—more granular than words, more semantic than characters.

**Example:**

```python
# Word-level
"unhappiness" → ["unhappiness"]  # If "unhappiness" is in vocabulary
"unhappiness" → ["<UNK>"]        # If not in vocabulary

# Subword-level
"unhappiness" → ["un", "happiness"]
# Even if "unhappiness" is unseen, model understands "un" (negation) and "happiness"
```

**Advantages:**
- Moderate vocabulary size (typically 10,000-50,000 tokens)
- Handles rare and unseen words through composition
- Works across languages
- Captures morphological structure

**Disadvantages:**
- Less intuitive than word-level
- Requires training a tokenizer on your corpus

#### Byte Pair Encoding (BPE)

The most popular subword tokenization algorithm.

**Algorithm Overview:**

1. Start with character-level vocabulary
2. Find the most frequent pair of adjacent tokens
3. Merge that pair into a new token
4. Repeat for desired vocabulary size

**Example:**

```
Corpus: "low low low lower lower newest newest newest newest newest"

Initial vocabulary (characters): {l, o, w, e, r, n, s, t, <space>}

Iteration 1: Most frequent pair = ("e", "s") → merge to "es"
Vocabulary: {l, o, w, e, r, n, s, t, es, <space>}

Iteration 2: Most frequent pair = ("es", "t") → merge to "est"
Vocabulary: {l, o, w, e, r, n, s, t, es, est, <space>}

Iteration 3: Most frequent pair = ("l", "o") → merge to "lo"
Vocabulary: {l, o, w, e, r, n, s, t, es, est, lo, <space>}

Iteration 4: Most frequent pair = ("lo", "w") → merge to "low"
Vocabulary: {l, o, w, e, r, n, s, t, es, est, lo, low, <space>}

Result:
"lowest" → ["low", "est"]
"newer" → ["n", "e", "w", "er"]
```

**Implementation with Hugging Face Tokenizers:**

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize BPE tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()

# Train on corpus
trainer = BpeTrainer(vocab_size=1000, special_tokens=["<UNK>", "<PAD>"])
corpus = ["This is a sample corpus", "We train the tokenizer on this corpus"]
tokenizer.train_from_iterator(corpus, trainer)

# Tokenize new text
output = tokenizer.encode("This tokenizer handles unseen words")
print(output.tokens)
# Subword tokens like ["This", "token", "izer", "handles", ...]
```

**Real-world example (GPT-2 uses BPE):**

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "unhappiness"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['un', 'happiness'] or similar subword split
```

#### WordPiece (used by BERT)

Similar to BPE but uses likelihood-based scoring instead of frequency.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "unhappiness"
tokens = tokenizer.tokenize(text)
print(tokens)
# ['un', '##hap', '##pi', '##ness']
# '##' indicates continuation of previous token
```

#### SentencePiece (language-agnostic)

Treats text as raw byte stream, works without pre-tokenization (ideal for languages without spaces).

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    '--input=corpus.txt --model_prefix=m --vocab_size=5000'
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('m.model')

tokens = sp.encode_as_pieces("unhappiness")
print(tokens)
# ['▁un', 'happiness'] (▁ represents space)
```

## Building a Vocabulary

After tokenization, create a mapping from tokens to integers:

```python
from collections import Counter

# Sample corpus
corpus = [
    "I love NLP",
    "NLP is amazing",
    "I love machine learning"
]

# Tokenize
all_tokens = []
for sentence in corpus:
    tokens = sentence.lower().split()
    all_tokens.extend(tokens)

# Count frequencies
token_counts = Counter(all_tokens)
print(token_counts)
# Counter({'i': 2, 'love': 2, 'nlp': 2, 'is': 1, 'amazing': 1, 'machine': 1, 'learning': 1})

# Create vocabulary (sorted by frequency)
vocab = {token: idx + 2 for idx, (token, count) in enumerate(token_counts.most_common())}
vocab['<PAD>'] = 0  # Padding token
vocab['<UNK>'] = 1  # Unknown token

print(vocab)
# {'<PAD>': 0, '<UNK>': 1, 'i': 2, 'love': 3, 'nlp': 4, 'is': 5, 'amazing': 6, 'machine': 7, 'learning': 8}
```

### Special Tokens

Most vocabularies include special tokens:

- `<PAD>`: Padding token (to make sequences same length)
- `<UNK>` or `[UNK]`: Unknown token (for out-of-vocabulary words)
- `<BOS>` or `[CLS]`: Beginning of sequence
- `<EOS>` or `[SEP]`: End of sequence / separator
- `<MASK>`: Masked token (for masked language modeling)

## Numericalization: Tokens to Numbers

Convert tokens to integer indices using the vocabulary:

```python
def tokenize_and_numericalize(text, vocab):
    tokens = text.lower().split()
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return indices

text = "I love deep learning"
indices = tokenize_and_numericalize(text, vocab)
print(indices)
# [2, 3, 1, 8]
# "I"=2, "love"=3, "deep"=<UNK>=1, "learning"=8
```

## Padding and Truncation

Neural networks require fixed-length inputs. Use padding and truncation:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Variable-length sequences
sequences = [
    [2, 3, 4],           # Length 3
    [2, 3, 4, 5, 6],     # Length 5
    [2]                  # Length 1
]

# Pad to same length
padded = pad_sequences(sequences, maxlen=5, padding='post', value=0)
print(padded)
# [[2, 3, 4, 0, 0],
#  [2, 3, 4, 5, 6],
#  [2, 0, 0, 0, 0]]

# Truncate if too long
long_seq = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
truncated = pad_sequences(long_seq, maxlen=5, truncating='post')
print(truncated)
# [[1, 2, 3, 4, 5]]  # Last 5 tokens kept (or first 5 if truncating='pre')
```

## Complete Preprocessing Pipeline

```python
import re
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TextPreprocessor:
    def __init__(self, vocab_size=10000, max_length=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
        return text
    
    def fit(self, texts):
        """Build vocabulary from corpus"""
        all_tokens = []
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            all_tokens.extend(tokens)
        
        # Get most common tokens
        token_counts = Counter(all_tokens)
        most_common = token_counts.most_common(self.vocab_size - 2)  # Reserve 2 for special tokens
        
        for idx, (token, count) in enumerate(most_common):
            self.vocab[token] = idx + 2  # Start from 2 (0=PAD, 1=UNK)
    
    def transform(self, texts):
        """Convert texts to padded sequences"""
        sequences = []
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            indices = [self.vocab.get(token, 1) for token in tokens]  # 1 = <UNK>
            sequences.append(indices)
        
        # Pad sequences
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', value=0)
        return padded

# Usage
corpus = [
    "I love natural language processing!",
    "Deep learning is amazing for NLP.",
    "Transformers revolutionized NLP."
]

preprocessor = TextPreprocessor(vocab_size=100, max_length=10)
preprocessor.fit(corpus)
processed = preprocessor.transform(["I love NLP and deep learning"])
print(processed)
# [[2, 3, 4, 5, 6, 7, 0, 0, 0, 0]] (padded to length 10)
```

## Key Takeaways

1. **Tokenization breaks text into processable units**—the foundation of all NLP pipelines
2. **Three main strategies**: Word-level (intuitive), character-level (no OOV), subword (balanced)
3. **Subword tokenization (BPE, WordPiece, SentencePiece)** is the modern standard—handles rare words while maintaining semantic meaning
4. **Preprocessing matters**: Lowercasing, URL removal, and cleaning reduce noise
5. **Vocabulary maps tokens to integers**—neural networks need numbers, not strings
6. **Special tokens** (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`) handle padding, unknown words, and sequence boundaries
7. **Padding and truncation** create fixed-length sequences required by neural networks
8. **Complete pipeline**: Clean → tokenize → build vocab → numericalize → pad → ready for neural network

## External Resources

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index) - Modern, fast tokenization library
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - Original BPE paper
- [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226) - SentencePiece paper

