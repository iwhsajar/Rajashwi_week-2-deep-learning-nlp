# Natural Language Processing Basics

## Learning Objectives
- Understand what Natural Language Processing (NLP) is and its core applications
- Learn common NLP tasks: classification, generation, translation, and information extraction
- Recognize the unique challenges of working with text data
- Connect NLP fundamentals to this week's deep learning focus

## Why This Matters

Natural Language Processing is one of the most transformative applications of deep learning, powering technologies you use every day:

- **Search engines**: Understanding queries and ranking relevant results
- **Virtual assistants**: Siri, Alexa, Google Assistant interpreting spoken commands
- **Translation**: Real-time language translation across the globe
- **Content moderation**: Detecting harmful content on social platforms
- **Customer service**: Chatbots handling support inquiries 24/7
- **Healthcare**: Extracting insights from medical records
- **Finance**: Sentiment analysis for market predictions

As you transition from general deep learning fundamentals to NLP this week, you're entering one of AI's most active and impactful domains. Text is everywhere, and the ability to process it automatically creates enormous business and social value.

## What is Natural Language Processing?

### Definition

**Natural Language Processing (NLP)** is the field of AI focused on enabling computers to understand, interpret, and generate human language.

"Natural language" refers to languages humans speak and write (English, Spanish, Mandarin, etc.), as opposed to programming languages or formal mathematical notation.

### The Core Challenge

Human language is incredibly complex:
- **Ambiguity**: "I saw her duck" (Did she bend down? Did she own a duck?)
- **Context-dependence**: "That's sick!" (positive in slang, negative in medical context)
- **Nuance**: Sarcasm, humor, implied meaning
- **Variability**: Dialects, slang, grammatical errors, abbreviations
- **Infinite creativity**: Humans constantly create novel sentences never seen before

Unlike structured data (numbers in spreadsheets), language is messy, ambiguous, and deeply tied to human knowledge and culture.

### Why Deep Learning for NLP?

Traditional NLP relied on hand-crafted rules:
- Regular expressions for pattern matching
- Grammar parsers
- Manually designed features

**Limitations:**
- Rules are brittle and language-specific
- Can't handle variability and ambiguity
- Require expert linguistic knowledge

**Deep Learning Revolution:**
- Learn patterns automatically from data
- Handle ambiguity through probabilistic modeling
- Scale to billions of words of text
- Transfer knowledge across languages and tasks

This week, you'll learn how neural networks (RNNs, LSTMs) process sequences of words to solve NLP problems.

## Common NLP Tasks

### 1. Text Classification

Assign a category or label to a piece of text.

**Examples:**
- **Sentiment analysis**: Is this movie review positive, negative, or neutral?
- **Spam detection**: Is this email spam or legitimate?
- **Topic categorization**: Is this news article about sports, politics, or technology?
- **Intent detection**: What does the user want? (book flight, check weather, set alarm)

**Input/Output:**

```
Input: "This movie was absolutely terrible. Waste of money."
Output: NEGATIVE

Input: "Best pizza I've ever had! Will definitely come back."
Output: POSITIVE
```

**How it works with deep learning:**
- Convert text to numerical representation (embeddings)
- Pass through neural network
- Output layer predicts class probabilities

### 2. Named Entity Recognition (NER)

Identify and classify named entities (people, organizations, locations, dates, etc.) in text.

**Example:**

```
Input: "Apple Inc. announced a new product in Cupertino on March 15th."

Output:
- Apple Inc. → ORGANIZATION
- Cupertino → LOCATION
- March 15th → DATE
```

**Applications:**
- Information extraction from documents
- Building knowledge graphs
- Medical record processing (identify drugs, diseases, symptoms)
- Resume parsing (extract names, companies, skills)

### 3. Machine Translation

Translate text from one language to another.

**Example:**

```
Input (English): "How are you today?"
Output (Spanish): "¿Cómo estás hoy?"
```

**Modern approach:**
- Encoder-decoder architectures (you'll learn about RNNs/LSTMs this week)
- Attention mechanisms
- Transformer models (BERT, GPT)

**Challenges:**
- Idioms don't translate literally
- Word order varies across languages
- Grammatical gender and agreement
- Cultural context

### 4. Text Generation

Generate human-like text based on a prompt or context.

**Examples:**
- **Autocomplete**: "The weather today is..." → "sunny and warm"
- **Chatbots**: Responding to customer questions
- **Creative writing**: Story generation, poetry
- **Code generation**: GitHub Copilot, coding assistants

**Input/Output:**

```
Input: "Once upon a time, there was a"
Output: "Once upon a time, there was a brave knight who lived in a distant kingdom."
```

### 5. Question Answering

Given a question and context, generate or extract the answer.

**Example:**

```
Context: "The Eiffel Tower is located in Paris, France. It was completed in 1889."
Question: "When was the Eiffel Tower completed?"
Answer: "1889"
```

**Types:**
- **Extractive**: Answer is a span of text from the context
- **Generative**: Answer is synthesized (may not appear verbatim in context)

### 6. Summarization

Condense long documents into shorter summaries.

**Example:**

```
Input: [10-page research paper]
Output: "This paper proposes a new neural architecture for sentiment analysis. 
The model achieves 95% accuracy on benchmark datasets, outperforming previous 
state-of-the-art by 3%. Key innovations include multi-head attention and 
hierarchical encoding."
```

**Types:**
- **Extractive**: Select important sentences from original text
- **Abstractive**: Generate new summary text (more human-like)

## Challenges Unique to Text Data

### 1. Variable Length Inputs

Unlike images (fixed 224×224 pixels), text varies wildly:
- Tweets: 280 characters max
- Emails: 100-1000 words
- Books: 100,000+ words

**Deep learning challenge**: Neural networks expect fixed-size inputs.

**Solution**: Padding, truncation, or sequential processing (RNNs, which you'll learn Thursday).

### 2. Discrete Symbols (Words)

Text is made of discrete units (words, characters), not continuous numbers.

**Challenge**: Can't directly feed words into neural networks.

**Solution**: Convert words to numbers through:
- **One-hot encoding** (Wednesday: today's topic later)
- **Word embeddings** (Wednesday: dense vector representations)

### 3. Ambiguity and Context

The same word means different things in different contexts:

```
"Apple released a new phone."   → Apple = company
"I ate an apple for lunch."     → apple = fruit

"The bank is closed."           → bank = financial institution
"We sat by the river bank."     → bank = land alongside water
```

**Solution**: Contextual representations (RNNs, LSTMs capture surrounding context).

### 4. Long-Range Dependencies

Meaning can depend on words far apart:

```
"The woman who lives next door and works as a doctor at the hospital downtown is coming to dinner."
```

Subject: "The woman"
Verb: "is coming" (11 words later)

**Challenge**: Model must remember information across long sequences.

**Solution**: LSTMs (you'll learn Thursday) designed specifically for this.

### 5. Lack of Labeled Data (for some tasks)

Creating labeled NLP datasets is expensive:
- Sentiment labeling requires human annotators
- Translation requires bilingual experts
- Medical NLP requires domain experts

**Solution**: 
- Transfer learning (pre-train on large unlabeled corpora, fine-tune on small labeled datasets)
- Semi-supervised and self-supervised learning

### 6. Domain-Specific Language

Models trained on general text struggle with specialized domains:
- Medical jargon
- Legal language
- Social media slang
- Technical documentation

**Solution**: Domain-specific fine-tuning or specialized datasets.

## The NLP Pipeline

A typical NLP system involves several steps:

### Step 1: Text Preprocessing

```python
# Raw text
text = "  I LOVE this!!! Best product ever... https://example.com  "

# Preprocessing steps:
# - Lowercasing
text = text.lower()  # "  i love this!!! best product ever... https://example.com  "

# - Remove URLs, special characters
import re
text = re.sub(r'http\S+', '', text)  # Remove URLs
text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
# Result: "i love this best product ever"

# - Tokenization (split into words) - covered later today
tokens = text.split()  # ['i', 'love', 'this', 'best', 'product', 'ever']
```

### Step 2: Tokenization

Break text into units (words, subwords, characters).

```python
# Word-level
"I love NLP" → ["I", "love", "NLP"]

# Character-level
"I love NLP" → ["I", " ", "l", "o", "v", "e", " ", "N", "L", "P"]

# Subword-level (handles rare words better)
"unhappiness" → ["un", "happiness"]
```

You'll explore tokenization in depth later today.

### Step 3: Numericalization

Convert tokens to numbers.

```python
# Build vocabulary
vocab = {"i": 1, "love": 2, "nlp": 3, "this": 4}

# Convert sentence
"I love NLP" → [1, 2, 3]
```

### Step 4: Embedding

Map numbers to dense vectors (you'll learn about embeddings later today).

```python
# Each word becomes a dense vector
"love" → [0.2, -0.5, 0.8, 0.1, ...]  (e.g., 100-dimensional vector)
```

### Step 5: Model Processing

Feed embeddings through neural network (RNN, LSTM, Transformer).

### Step 6: Output

Prediction, generated text, extracted entities, etc.

## NLP in This Week's Context

**Monday (completed)**: You learned about general neural network visualization and training strategies.

**Tuesday (completed)**: You mastered backpropagation and gradient descent—essential for training NLP models.

**Wednesday (today)**: You're learning NLP foundations—tokenization, encoding, embeddings.

**Thursday (upcoming)**: You'll learn RNNs and LSTMs—architectures designed for sequential data like text.

**Friday (upcoming)**: Model persistence and optimization—deploying NLP models to production.

This progression takes you from general deep learning to specialized sequential modeling for language.

## Practical Example: Sentiment Analysis

```python
import tensorflow as tf
from tensorflow import keras

# Sample data (preprocessed)
texts = ["I love this", "terrible product", "amazing experience", "worst purchase ever"]
labels = [1, 0, 1, 0]  # 1 = positive, 0 = negative

# Tokenization (simplified)
tokenizer = keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# Result: [[1, 2, 3], [4, 5], [6, 7], [8, 9, 10]]

# Padding (make all sequences same length)
X = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=10)

# Model (simple baseline)
model = keras.Sequential([
    keras.layers.Embedding(input_dim=1000, output_dim=32, input_length=10),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=10)

# Predict on new text
new_text = ["I really enjoyed it"]
new_seq = tokenizer.texts_to_sequences(new_text)
new_X = keras.preprocessing.sequence.pad_sequences(new_seq, maxlen=10)
prediction = model.predict(new_X)
print("Positive" if prediction > 0.5 else "Negative")
```

This is a simplified example—you'll build more sophisticated models later this week.

## Key Takeaways

1. **NLP enables computers to understand and generate human language**—one of AI's most impactful applications
2. **Common tasks**: Classification, named entity recognition, translation, generation, question answering, summarization
3. **Text presents unique challenges**: Variable length, discrete symbols, ambiguity, context-dependence, long-range dependencies
4. **Deep learning revolutionized NLP** by learning patterns automatically from data
5. **The NLP pipeline**: Preprocessing → tokenization → numericalization → embedding → model → output
6. **This week's progression**: General deep learning → NLP fundamentals → Sequential models (RNNs/LSTMs) → Production deployment
7. **Real-world impact**: Search, translation, chatbots, content moderation, healthcare, finance—NLP is everywhere

## External Resources

- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) - Comprehensive course materials
- [Natural Language Processing with Python (NLTK Book)](https://www.nltk.org/book/) - Practical introduction to NLP
- [Hugging Face NLP Course](https://huggingface.co/course/chapter1/1) - Modern NLP with transformers (preview of advanced topics)

