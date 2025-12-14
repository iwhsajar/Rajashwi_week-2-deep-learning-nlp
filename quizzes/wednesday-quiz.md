# Weekly Knowledge Check: NLP Fundamentals, Tokenization, and Embeddings (Wednesday)

## Part 1: Multiple Choice

### 1. What is the main problem with one-hot encoding for words?

- [ ] A) It's computationally too fast
- [ ] B) It treats all word pairs as equally dissimilar
- [ ] C) It can't represent rare words
- [ ] D) It requires labeled data

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) It treats all word pairs as equally dissimilar

**Explanation:** In one-hot encoding, every word is orthogonal to every other word. The cosine similarity between any two different one-hot vectors is 0. "Cat" and "dog" are just as dissimilar as "cat" and "refrigerator", which doesn't reflect semantic relationships.

- **Why others are wrong:**
  - A) One-hot is actually computationally expensive for large vocabularies
  - C) One-hot can represent any word in the vocabulary
  - D) One-hot encoding doesn't require labels
</details>

---

### 2. What does the equation "king - man + woman = ?" approximately equal in word embeddings?

- [ ] A) queen
- [ ] B) prince
- [ ] C) princess
- [ ] D) royalty

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) queen

**Explanation:** Word embeddings encode semantic relationships as directions in vector space. "King - man" removes the "male" component, leaving "royalty". Adding "woman" gives "royalty + female", which is closest to "queen". This demonstrates vector arithmetic capturing analogies.

- **Why others are wrong:**
  - B) Prince is male royalty (child)
  - C) Princess is female royalty (child), not quite equivalent
  - D) Royalty is abstract, not a person
</details>

---

### 3. In Skip-gram Word2Vec, what is the training task?

- [ ] A) Given a sentence, predict the next sentence
- [ ] B) Given context words, predict the target word
- [ ] C) Given a word, predict its definition
- [ ] D) Given a target word, predict the context words

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Given a target word, predict the context words

**Explanation:** Skip-gram takes a single target word and tries to predict the surrounding context words. For "fox" in "the quick brown fox jumps", it predicts "quick", "brown", "jumps". Words appearing in similar contexts get similar embeddings.

- **Why others are wrong:**
  - A) That's sentence-level prediction
  - B) That's CBOW (Continuous Bag of Words), the opposite task
  - D) Word2Vec doesn't use definitions
</details>

---

### 4. What is the typical dimensionality of word embeddings?

- [ ] A) 2-10 dimensions
- [ ] B) 10,000-50,000 dimensions
- [ ] C) 1 dimension per word
- [ ] D) 100-300 dimensions

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) 100-300 dimensions

**Explanation:** Word embeddings typically use 100-300 dimensions, enough to capture semantic nuances while remaining computationally efficient. Word2Vec uses 300 dimensions (Google News), and GloVe offers 50-300 dimensional versions.

- **Why others are wrong:**
  - A) Too few dimensions to capture semantic complexity
  - B) That's one-hot encoding dimensionality, not embeddings
  - C) That describes indices, not embeddings
</details>

---

### 5. What is "negative sampling" in Word2Vec training?

- [ ] A) Removing negative words from the vocabulary
- [ ] B) Penalizing negative embeddings
- [ ] C) Training on randomly sampled word pairs that don't appear together
- [ ] D) Sampling from a negative distribution

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Training on randomly sampled word pairs that don't appear together

**Explanation:** Instead of computing softmax over the entire vocabulary, the model learns to distinguish true context pairs from randomly sampled "negative" pairs. This reduces computational complexity from O(vocab_size) to O(num_negative_samples), typically 5-20.

- **Why others are wrong:**
  - A) Negative sampling doesn't remove words
  - B) It's not about penalizing embeddings
  - D) "Negative distribution" is not a standard term
</details>

---

### 6. What does CBOW stand for in Word2Vec?

- [ ] A) Continuous Bag of Words
- [ ] B) Contextual Bag of Words
- [ ] C) Complete Bag of Words
- [ ] D) Concatenated Bag of Words

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Continuous Bag of Words

**Explanation:** CBOW stands for Continuous Bag of Words. It's called "continuous" because it uses continuous (dense) vector representations. The "bag of words" refers to averaging context word embeddings without considering word order.

- **Why others are wrong:**
  - A, C, D) These are not the standard acronym expansions
</details>

---

### 7. What is the output shape of a Keras Embedding layer with input_dim=10000, output_dim=100, and input_length=50?

- [ ] A) (batch_size, 100)
- [ ] B) (batch_size, 10000, 100)
- [ ] C) (batch_size, 50)
- [ ] D) (batch_size, 50, 100)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) (batch_size, 50, 100)

**Explanation:** The Embedding layer converts each word index to a dense vector. With input_length=50 words and output_dim=100 dimensions per word, the output shape is (batch_size, 50, 100) - 50 embeddings of 100 dimensions each.

- **Why others are wrong:**
  - A) Missing the sequence length dimension
  - B) 10000 is vocabulary size, not output dimension
  - C) Missing the embedding dimension
</details>

---

### 8. What is the cosine similarity of two one-hot vectors representing different words?

- [ ] A) 0.0
- [ ] B) 0.5
- [ ] C) 1.0
- [ ] D) -1.0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) 0.0

**Explanation:** One-hot vectors for different words are orthogonal - they have 1 in different positions and 0 elsewhere. The dot product of orthogonal vectors is 0, so cosine similarity is 0. This shows why one-hot can't capture semantic similarity.

- **Why others are wrong:**
  - B) 0.5 would require some overlap
  - C) 1.0 means identical vectors
  - D) -1.0 means opposite direction (rare in NLP)
</details>

---

### 9. Which of the following is NOT a common NLP task?

- [ ] A) Sentiment analysis
- [ ] B) Named entity recognition
- [ ] C) Image segmentation
- [ ] D) Machine translation

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Image segmentation

**Explanation:** Image segmentation is a computer vision task, not an NLP task. Sentiment analysis, named entity recognition, and machine translation all process text data.

- **Why others are wrong:**
  - A) Sentiment analysis classifies text as positive/negative
  - B) NER identifies entities (people, places, dates) in text
  - D) Translation converts text between languages
</details>

---

### 10. What is the distributional hypothesis in NLP?

- [ ] A) Words should be distributed evenly in a corpus
- [ ] B) All words have uniform probabilities
- [ ] C) Words with similar contexts have similar meanings
- [ ] D) Rare words should be removed

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Words with similar contexts have similar meanings

**Explanation:** The distributional hypothesis states: "You shall know a word by the company it keeps." Words appearing in similar contexts (e.g., "cat" and "dog" both appear near "pet", "fur", "animal") have similar meanings and should have similar embeddings.

- **Why others are wrong:**
  - A) Word distribution is not about even spacing
  - C) Word probabilities follow Zipf's law (very uneven)
  - D) Rare word handling is a separate concern
</details>

---

### 11. What advantage do subword tokenizers (like BPE) have over word-level tokenizers?

- [ ] A) They are faster to train
- [ ] B) They can handle out-of-vocabulary words by breaking them into known subwords
- [ ] C) They use less memory
- [ ] D) They produce shorter sequences

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) They can handle out-of-vocabulary words by breaking them into known subwords

**Explanation:** BPE (Byte Pair Encoding) breaks unknown words into subword units. "Unhappiness" becomes ["un", "happiness"], both of which are known. This allows handling rare or new words without expanding the vocabulary.

- **Why others are wrong:**
  - A) Subword tokenization can be slower
  - B) Memory usage depends on implementation
  - D) Subword tokenization often produces longer sequences
</details>

---

### 12. In word embeddings, what does "dense" representation mean?

- [ ] A) The vectors are compressed
- [ ] B) The vectors are stored in a database
- [ ] C) The vectors use integer values
- [ ] D) Most or all vector elements are non-zero

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Most or all vector elements are non-zero

**Explanation:** Dense embeddings have values in every dimension, typically ranging from -1 to 1. In contrast, one-hot vectors are "sparse" (99.99% zeros). Dense representations are more memory-efficient and capture semantic information.

- **Why others are wrong:**
  - A) Dense refers to non-zero values, not compression
  - C) Storage method is unrelated to density
  - D) Embeddings use float values, not integers
</details>

---

### 13. What is the primary purpose of tokenization in NLP?

- [ ] A) To break text into smaller units (words, subwords, or characters)
- [ ] B) To encrypt the text
- [ ] C) To translate text to another language
- [ ] D) To classify the text into categories

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) To break text into smaller units (words, subwords, or characters)

**Explanation:** Tokenization splits raw text into processable units. "I love NLP" becomes ["I", "love", "NLP"] at word level. This is the first step in most NLP pipelines before numericalization and embedding.

- **Why others are wrong:**
  - A) Tokenization is not encryption
  - C) Translation is a downstream task
  - D) Classification is a downstream task
</details>

---

### 14. Pre-trained word embeddings like Word2Vec or GloVe enable what type of learning?

- [ ] A) Reinforcement learning
- [ ] B) Supervised learning
- [ ] C) Transfer learning
- [ ] D) Unsupervised classification

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Transfer learning

**Explanation:** Pre-trained embeddings capture general language knowledge from massive corpora. You can transfer this knowledge to your specific task (sentiment analysis, NER, etc.) without training embeddings from scratch, even with limited data.

- **Why others are wrong:**
  - A) Reinforcement learning uses reward signals
  - B) Supervised learning requires labels for every sample
  - D) "Unsupervised classification" is contradictory
</details>

---

### 15. What challenge does the word "bank" present for traditional word embeddings?

- [ ] A) It's too short
- [ ] B) It's a stop word
- [ ] C) It has multiple meanings (polysemy) but gets one vector
- [ ] D) It contains special characters

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) It has multiple meanings (polysemy) but gets one vector

**Explanation:** "Bank" means financial institution OR riverbank. Word2Vec/GloVe assign one vector per word, conflating different senses. Contextualized embeddings (BERT, ELMo) solve this by producing different vectors based on context.

- **Why others are wrong:**
  - A) Word length doesn't cause problems
  - C) "Bank" is not a stop word (like "the", "a", "is")
  - D) "Bank" has no special characters
</details>

---

## Part 2: True/False

### 16. Skip-gram and CBOW perform the same task but in opposite directions.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** Skip-gram predicts context words from a target word. CBOW predicts the target word from context words. They're inverse tasks, both resulting in learned word embeddings from co-occurrence patterns.
</details>

---

### 17. Word2Vec embeddings assign different vectors to the same word in different contexts.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Word2Vec assigns one fixed vector per word, regardless of context. "Bank" gets the same embedding whether it means "financial institution" or "river bank". Contextualized embeddings (ELMo, BERT) solve this limitation.
</details>

---

### 18. The Keras Embedding layer can be initialized with pre-trained Word2Vec or GloVe vectors.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** You can load pre-trained embeddings into a Keras Embedding layer using the `weights` parameter: `Embedding(vocab_size, dim, weights=[matrix], trainable=False)`. Setting `trainable=False` freezes the pre-trained weights.
</details>

---

### 19. In NLP, longer text sequences are always better for model performance.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Longer sequences increase computational cost and can introduce noise. Many models truncate to a maximum length (e.g., 512 tokens). The optimal length depends on the task and available computational resources.
</details>

---

### 20. Word embeddings capture semantic relationships that one-hot encoding cannot.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** Embeddings place similar words near each other in vector space (cat similar to dog). One-hot encoding treats all words as equally dissimilar (orthogonal vectors), missing semantic relationships entirely.
</details>

---

## Part 3: Code Prediction

### 21. What does this tokenization produce?

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(["I love NLP", "I love deep learning"])
sequences = tokenizer.texts_to_sequences(["I love"])
print(sequences)
```

- [ ] A) [['I', 'love']]
- [ ] B) [[1, 2]]
- [ ] C) [[0, 1]]
- [ ] D) [1, 2]

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) [[1, 2]]

**Explanation:** The Tokenizer assigns integer indices starting from 1 (0 is reserved for padding). "I" appears most frequently and gets index 1, "love" gets index 2. The output is a list of sequences, each sequence is a list of integers.

- **Why others are wrong:**
  - A) texts_to_sequences returns integers, not strings
  - C) Indexing starts at 1, not 0
  - D) Result should be nested list (list of sequences)
</details>

---

### 22. What is the output dimension?

```python
embedding_layer = keras.layers.Embedding(
    input_dim=5000,    # vocabulary size
    output_dim=128,    # embedding dimension
    input_length=100   # max sequence length
)
# Input shape: (batch_size, 100)
```

- [ ] A) (batch_size, 5000)
- [ ] B) (batch_size, 128)
- [ ] C) (batch_size, 100, 128)
- [ ] D) (batch_size, 5000, 128)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) (batch_size, 100, 128)

**Explanation:** Each of the 100 input word indices is converted to a 128-dimensional embedding vector. Output shape is (batch, sequence_length, embedding_dim) = (batch_size, 100, 128).
</details>

---

### 23. What will this cosine similarity calculation return?

```python
import numpy as np

vec_a = np.array([1, 0, 0])
vec_b = np.array([0, 1, 0])
dot_product = np.dot(vec_a, vec_b)
similarity = dot_product / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
print(similarity)
```

- [ ] A) 1.0
- [ ] B) 0.5
- [ ] C) 0.0
- [ ] D) -1.0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 0.0

**Explanation:** Vectors [1,0,0] and [0,1,0] are orthogonal (perpendicular). Their dot product is 0, making cosine similarity 0. This represents no semantic relationship, like one-hot encoded words.
</details>

---

### 24. What does this Word2Vec analogy query compute?

```python
model.most_similar(positive=['Paris', 'Japan'], negative=['France'], topn=1)
```

- [ ] A) Cities similar to Paris
- [ ] B) Paris + Japan - France (expected: Tokyo)
- [ ] C) Countries similar to France
- [ ] D) Paris - Japan + France

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Paris + Japan - France (expected: Tokyo)

**Explanation:** This computes `vector(Paris) + vector(Japan) - vector(France)`. Semantically: "Paris is to France as ? is to Japan" - the answer is Tokyo (capital city relationship).
</details>

---

### 25. What is the vocabulary size after running this code?

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["cat dog", "cat bird", "dog bird mouse"])
vocab_size = len(tokenizer.word_index)
print(vocab_size)
```

- [ ] A) 3
- [ ] B) 4
- [ ] C) 5
- [ ] D) 8

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 4

**Explanation:** Unique words: "cat", "dog", "bird", "mouse" = 4 words. Duplicates are counted once. The word_index dictionary has one entry per unique word.
</details>

---

### 26. What does pad_sequences do to a sequence that's shorter than maxlen?

```python
from keras.preprocessing.sequence import pad_sequences
sequences = [[1, 2, 3]]
padded = pad_sequences(sequences, maxlen=5, padding='pre')
print(padded)
```

- [ ] A) [[1, 2, 3, 0, 0]]
- [ ] B) [[0, 0, 1, 2, 3]]
- [ ] C) [[1, 2, 3, 1, 2]]
- [ ] D) [[1, 2, 3]]

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) [[0, 0, 1, 2, 3]]

**Explanation:** With `padding='pre'`, zeros are added at the beginning. The sequence [1, 2, 3] becomes [0, 0, 1, 2, 3] to reach maxlen=5. Pre-padding is preferred for RNNs so the final hidden state captures actual content.
</details>

---

## Part 4: Fill-in-the-Blank

### 27. The formula for cosine similarity between vectors A and B is: cos_sim = (A dot B) / (||A|| * ||B||). Cosine similarity of 1.0 means the vectors point in the _______ direction.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** same

**Explanation:** Cosine similarity of 1.0 means the vectors are perfectly aligned (same direction). 0.0 means orthogonal (no relationship). -1.0 means opposite directions.
</details>

---

### 28. The distributional hypothesis states: "You shall know a word by the _______ it keeps."

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** company (or "context")

**Explanation:** The distributional hypothesis is the foundation of Word2Vec. Words appearing in similar contexts have similar meanings. "Cat" and "dog" appear in similar sentences, so they get similar embeddings.
</details>

---

### 29. In tokenization, BPE stands for Byte Pair _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Encoding

**Explanation:** Byte Pair Encoding (BPE) is a subword tokenization algorithm. It starts with characters and iteratively merges the most frequent pairs. This handles rare words by breaking them into known subwords.
</details>

---

### 30. Word embeddings are called "dense" because all vector elements are _______, unlike sparse one-hot vectors.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** non-zero (or "filled", "used", "meaningful")

**Explanation:** Dense embeddings have values in every dimension, typically ranging from -1 to 1. One-hot vectors are sparse (99.99% zeros). Dense representations are more efficient and capture semantic information.
</details>

---

### 31. To use pre-trained GloVe embeddings in Keras, you create an embedding _______ and pass it to the weights parameter.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** matrix

**Explanation:** You load GloVe vectors into a numpy matrix where each row corresponds to a word in your vocabulary. This matrix is passed to `Embedding(weights=[embedding_matrix])`.
</details>

---

### 32. In the NLP pipeline, the step that converts text into individual words or subwords is called _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** tokenization

**Explanation:** Tokenization is the process of breaking text into tokens (words, subwords, or characters). "I love NLP" tokenizes to ["I", "love", "NLP"]. This precedes numericalization and embedding.
</details>

---

## Part 5: Scenario-Based Questions

### 33. You're building a sentiment analysis model for product reviews. You have limited training data (1,000 samples). What embedding strategy should you use?

- [ ] A) Train embeddings from scratch with random initialization
- [ ] B) Use pre-trained embeddings (Word2Vec or GloVe) with fine-tuning
- [ ] C) Use one-hot encoding
- [ ] D) Skip embeddings entirely

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Use pre-trained embeddings (Word2Vec or GloVe) with fine-tuning

**Explanation:** With limited data, learning good embeddings from scratch is impossible. Pre-trained embeddings provide semantic knowledge from massive corpora, enabling transfer learning. Fine-tuning adapts them to your domain.

- **Why others are wrong:**
  - A) 1,000 samples is far too few to learn meaningful embeddings
  - C) One-hot lacks semantic information
  - D) Embeddings are essential for text models
</details>

---

### 34. Your Word2Vec model gives unexpected results for "king - man + woman". What's the most likely cause?

- [ ] A) The corpus was too large
- [ ] B) The training corpus didn't contain enough examples of these words in appropriate contexts
- [ ] C) Word2Vec can't handle analogies
- [ ] D) The embedding dimension was too high

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The training corpus didn't contain enough examples of these words in appropriate contexts

**Explanation:** Analogy quality depends on having sufficient training data with words used in their expected contexts. A small or domain-specific corpus may not capture the gender relationship between king/queen/man/woman.

- **Why others are wrong:**
  - A) Larger corpus usually helps
  - C) Word2Vec famously handles analogies well
  - D) Higher dimensions usually improve quality
</details>

---

### 35. You need to process social media text with many misspellings and slang. Which tokenization approach is best?

- [ ] A) Word-level tokenization with a dictionary lookup
- [ ] B) Character-level tokenization
- [ ] C) Subword tokenization (BPE or WordPiece)
- [ ] D) Only use sentences that pass spell-check

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Subword tokenization (BPE or WordPiece)

**Explanation:** Subword tokenization handles out-of-vocabulary words by breaking them into known subwords. "cooolll" becomes ["coo", "ol", "ll"], allowing the model to process it. Word-level would fail on misspellings.

- **Why others are wrong:**
  - A) Misspellings won't be in the dictionary
  - B) Character-level loses word semantics
  - D) You'd lose most social media data
</details>

---

### 36. Your embedding layer has input_dim=50000 and output_dim=300. About how many trainable parameters does it have?

- [ ] A) 50,000
- [ ] B) 15,000,000
- [ ] C) 300
- [ ] D) 50,300

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 15,000,000

**Explanation:** Embedding layer parameters = vocabulary_size x embedding_dimension = 50,000 x 300 = 15,000,000. Each word has a 300-dimensional vector, and all values are trainable.

- **Why others are wrong:**
  - A) That's just vocabulary size
  - C) That's just embedding dimension
  - D) That's the sum, not the product
</details>

---

### 37. You want to find words similar to "machine_learning" in a Word2Vec model but get a KeyError. What's the issue?

- [ ] A) Word2Vec doesn't support multi-word phrases
- [ ] B) The phrase may not be in the vocabulary (case sensitivity or underscore formatting)
- [ ] C) Word2Vec can only handle single letters
- [ ] D) The model is corrupted

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The phrase may not be in the vocabulary (case sensitivity or underscore formatting)

**Explanation:** Word2Vec models are vocabulary-specific. If "machine_learning" wasn't in the training corpus (or was formatted differently like "Machine_Learning"), you'll get a KeyError. Try checking exact formatting and case.

- **Why others are wrong:**
  - A) Some Word2Vec models include phrases
  - C) Word2Vec works with words
  - D) KeyError indicates missing vocabulary word
</details>

---

## Bonus Questions

### 38. Why is cosine similarity preferred over Euclidean distance for comparing word embeddings?

- [ ] A) Cosine similarity is faster to compute
- [ ] B) Cosine similarity measures angle (direction), ignoring magnitude, focusing on semantic relationship
- [ ] C) Euclidean distance doesn't work in high dimensions
- [ ] D) Cosine similarity always gives higher scores

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Cosine similarity measures angle (direction), ignoring magnitude, focusing on semantic relationship

**Explanation:** Cosine similarity measures the angle between vectors, independent of their length. Two vectors pointing in the same direction have similarity 1.0 regardless of magnitude. This captures semantic similarity better than raw distance.

- **Why others are wrong:**
  - A) Computation cost is similar
  - C) Euclidean distance works in any dimension
  - D) Cosine similarity ranges from -1 to 1
</details>

---

### 39. What is the "curse of dimensionality" as it relates to one-hot encoding?

- [ ] A) High-dimensional vectors are slow to process
- [ ] B) Memory requirements grow exponentially
- [ ] C) As vocabulary grows, vectors become extremely sparse and distances become less meaningful
- [ ] D) One-hot vectors can't be normalized

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) As vocabulary grows, vectors become extremely sparse and distances become less meaningful

**Explanation:** With 100,000 word vocabulary, each one-hot vector has 99,999 zeros. All word pairs are equidistant, making distance metrics meaningless. Dense embeddings (100-300 dims) avoid this while preserving semantic information.

- **Why others are wrong:**
  - A) Sparse vectors can be processed efficiently
  - B) Memory grows linearly, not exponentially
  - D) One-hot vectors can be normalized
</details>

---

### 40. GloVe (Global Vectors) differs from Word2Vec primarily in that GloVe:

- [ ] A) Uses much larger embeddings
- [ ] B) Is trained on global co-occurrence statistics, not sliding windows
- [ ] C) Only works for English text
- [ ] D) Produces sparse vectors

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Is trained on global co-occurrence statistics, not sliding windows

**Explanation:** Word2Vec uses local context windows (predict nearby words). GloVe builds a global word co-occurrence matrix first, then factorizes it. Both produce dense embeddings, but GloVe leverages corpus-wide statistics.

- **Why others are wrong:**
  - A) Both use similar dimensions (50-300)
  - C) Both work for any language
  - D) Both produce dense vectors
</details>

---

*Quiz generated by Practice Quiz Agent for Week 2: Deep Learning & NLP - Wednesday Content*

