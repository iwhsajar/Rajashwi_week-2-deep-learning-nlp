# Weekly Technical Assessment: Deep Learning & NLP

> **Week 2 Comprehensive Assessment**
> **Time Limit:** 90 minutes
> **Total Points:** 100
> **Passing Score:** 70%

---

## Section A: Conceptual Understanding (30 Points)

*Answer each question concisely. Focus on demonstrating understanding, not length.*

### A1. Deep Learning Visualization & Autoencoders (6 points)

**A1a.** (2 pts) What is TensorBoard and what are two key metrics you would track during training?

<details>
<summary>Model Answer</summary>

**TensorBoard** is a visualization toolkit for TensorFlow that helps monitor and debug neural network training.

**Key metrics to track:**
1. **Loss curves** (training and validation) - to detect overfitting
2. **Accuracy/Performance metrics** - to measure model improvement
3. **Learning rate** - to verify scheduling is working
4. **Gradient distributions** - to detect vanishing/exploding gradients
</details>

**A1b.** (2 pts) Describe the architecture of an autoencoder. What is the purpose of the bottleneck layer?

<details>
<summary>Model Answer</summary>

**Autoencoder architecture:**
1. **Encoder:** Compresses input into lower-dimensional representation
2. **Bottleneck (Latent Space):** Compressed representation
3. **Decoder:** Reconstructs original input from bottleneck

**Purpose of bottleneck:**
- Forces the network to learn efficient compression
- Captures essential features while discarding noise
- The narrower the bottleneck, the more compression required
</details>

**A1c.** (2 pts) Name two practical applications of autoencoders.

<details>
<summary>Model Answer</summary>

**Applications:**
1. **Dimensionality reduction** - Alternative to PCA, captures non-linear relationships
2. **Denoising** - Training on noisy inputs to produce clean outputs
3. **Anomaly detection** - High reconstruction error indicates anomalies
4. **Feature learning** - Latent representations for other tasks
5. **Image compression** - Learning compact image representations
</details>

---

### A2. Backpropagation & Optimization (8 points)

**A2a.** (3 pts) Explain the backpropagation algorithm in three steps. Why is the chain rule essential?

<details>
<summary>Model Answer</summary>

**Three steps of backpropagation:**
1. **Forward pass:** Compute predictions layer by layer
2. **Calculate loss:** Compare predictions to true labels
3. **Backward pass:** Compute gradients working backward from output to input

**Why chain rule is essential:**
- Neural networks are compositions of functions: f(g(h(x)))
- Chain rule allows computing gradients through nested functions:
  - dL/dw = dL/da * da/dz * dz/dw
- Without chain rule, we couldn't efficiently compute how each weight affects the final loss
</details>

**A2b.** (2 pts) Compare SGD with momentum to Adam optimizer. When might you choose each?

<details>
<summary>Model Answer</summary>

| Aspect | SGD + Momentum | Adam |
|--------|----------------|------|
| Learning rate | Fixed | Adaptive per-parameter |
| Momentum | Single momentum term | First and second moment estimates |
| Tuning | Requires careful LR selection | Works well with defaults |
| Convergence | Can find better minima | Faster initial convergence |

**When to choose:**
- **SGD + momentum:** Fine-tuning, when generalization matters most, research showing final performance matters
- **Adam:** Quick prototyping, default choice, training from scratch
</details>

**A2c.** (3 pts) What is batch normalization? List two benefits it provides during training.

<details>
<summary>Model Answer</summary>

**Batch normalization** normalizes layer inputs to have zero mean and unit variance using batch statistics, then scales and shifts with learnable parameters.

**Formula:**
- Normalize: (x - mean) / sqrt(variance + epsilon)
- Scale and shift: gamma * x_normalized + beta

**Benefits:**
1. **Faster training:** Allows higher learning rates without instability
2. **Reduces internal covariate shift:** Layer inputs remain stable across training
3. **Regularization effect:** Batch statistics add noise, reducing overfitting
4. **Less sensitive to initialization:** More forgiving of weight initialization choices
</details>

---

### A3. Natural Language Processing Fundamentals (8 points)

**A3a.** (2 pts) What is tokenization? Explain the difference between word-level and subword tokenization.

<details>
<summary>Model Answer</summary>

**Tokenization** is breaking text into smaller units (tokens) for processing.

**Word-level:** Splits on whitespace/punctuation
- "Hello world" -> ["Hello", "world"]
- Pro: Simple, interpretable
- Con: Large vocabulary, OOV (out-of-vocabulary) issues

**Subword:** Breaks words into smaller pieces (BPE, WordPiece)
- "unhappiness" -> ["un", "happiness"] or ["un", "happy", "ness"]
- Pro: Handles rare/unseen words, smaller vocabulary
- Con: Less interpretable, tokens may not be meaningful alone
</details>

**A3b.** (3 pts) Compare one-hot encoding to word embeddings. Why are embeddings preferred for NLP?

<details>
<summary>Model Answer</summary>

| Aspect | One-Hot Encoding | Word Embeddings |
|--------|------------------|-----------------|
| Dimensions | Vocabulary size (10,000+) | Fixed (50-300) |
| Values | Binary (0 or 1) | Continuous real numbers |
| Memory | Sparse, wasteful | Dense, efficient |
| Semantics | No similarity captured | Similar words = similar vectors |

**Why embeddings preferred:**
1. **Dimensionality:** Much smaller vectors
2. **Semantic meaning:** Captures relationships (king-man+woman=queen)
3. **Generalization:** Similar words share similar representations
4. **Transfer learning:** Pre-trained embeddings (Word2Vec, GloVe) available
</details>

**A3c.** (3 pts) Explain the Word2Vec Skip-gram model. How does it learn word relationships?

<details>
<summary>Model Answer</summary>

**Skip-gram model:** Given a target word, predict surrounding context words.

**How it works:**
1. Input: One-hot encoded target word
2. Hidden layer: Word embedding (what we want to learn)
3. Output: Probability distribution over vocabulary for context words

**How it learns relationships:**
- Words appearing in similar contexts get similar embeddings
- Training optimizes: maximize P(context | target)
- Result: Words like "king" and "queen" share similar embeddings because they appear in similar contexts

**Vector arithmetic emerges:**
- king - man + woman = queen
- paris - france + italy = rome
</details>

---

### A4. Sequential Models (8 points)

**A4a.** (3 pts) What problem do RNNs solve that feedforward networks cannot? What is the "vanishing gradient problem"?

<details>
<summary>Model Answer</summary>

**RNNs solve:**
- Processing sequential data of variable length
- Maintaining memory of previous inputs through hidden state
- Capturing temporal dependencies and patterns

**Vanishing gradient problem:**
- During backpropagation through time, gradients are multiplied repeatedly
- With values < 1, gradients shrink exponentially
- Result: Early time steps have near-zero gradients
- Effect: Network can't learn long-term dependencies

**Mathematical insight:**
- dL/dh_0 = dL/dh_n * (product of dh_t/dh_{t-1} terms)
- If each term is < 1, product approaches 0
</details>

**A4b.** (3 pts) How does LSTM address the vanishing gradient problem? Name the three gates and their purposes.

<details>
<summary>Model Answer</summary>

**How LSTM addresses vanishing gradients:**
- Cell state provides a gradient "highway" through time
- Uses additive operations (not multiplicative) to update cell state
- Gates use sigmoid to control information flow

**Three gates:**
1. **Forget gate (f_t):** Decides what to discard from cell state
   - f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)

2. **Input gate (i_t):** Decides what new information to store
   - i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)

3. **Output gate (o_t):** Decides what to output from cell state
   - o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)

**Cell state update:**
- C_t = f_t * C_{t-1} + i_t * tanh(W_C * [h_{t-1}, x_t] + b_C)
</details>

**A4c.** (2 pts) What is sequence padding and why is masking important when using padded sequences?

<details>
<summary>Model Answer</summary>

**Sequence padding:** Adding zeros to sequences to make them equal length for batch processing.

```python
# [1, 2, 3] and [4, 5] become:
# [1, 2, 3] and [4, 5, 0]  (post-padding)
```

**Why masking is important:**
- Without masking, padded zeros affect:
  - RNN hidden state computations
  - Loss calculation
  - Gradient updates
- With masking:
  - Model knows to ignore padded positions
  - Only real tokens contribute to learning
  - More accurate sequence modeling
</details>

---

## Section B: Practical Application (40 Points)

### B1. Autoencoder Architecture (10 points)

Design an autoencoder for compressing 784-dimensional MNIST images (28x28 flattened) to a 32-dimensional latent space.

**B1a.** (4 pts) Write the encoder architecture. Include layer sizes, activations, and justify your choices.

<details>
<summary>Model Answer</summary>

```python
encoder = Sequential([
    # Input: 784 (28x28 flattened)
    Dense(256, activation='relu', input_shape=(784,)),
    # Gradual compression
    Dense(128, activation='relu'),
    # Bottleneck
    Dense(32, activation='relu')  # or 'linear' for continuous latent
])
```

**Justification:**
- **Gradual compression (784->256->128->32):** Progressive reduction helps preserve information
- **ReLU activation:** Non-linearity allows learning complex patterns; computationally efficient
- **Symmetric reduction:** Each layer roughly halves dimensions
- **Final layer:** ReLU for bounded positive values, or linear if negative values are meaningful
</details>

**B1b.** (4 pts) Write the decoder architecture to reconstruct the original image.

<details>
<summary>Model Answer</summary>

```python
decoder = Sequential([
    # Input: 32 (latent space)
    Dense(128, activation='relu', input_shape=(32,)),
    # Mirror the encoder
    Dense(256, activation='relu'),
    # Output: reconstruct original dimensions
    Dense(784, activation='sigmoid')  # For normalized pixel values [0,1]
])
```

**Justification:**
- **Symmetric to encoder:** 32->128->256->784 mirrors compression path
- **Sigmoid output:** MNIST pixels are normalized [0,1], sigmoid produces valid range
- **Same hidden dimensions:** Balanced capacity for encoding and decoding
</details>

**B1c.** (2 pts) What loss function should you use for this autoencoder? Why?

<details>
<summary>Model Answer</summary>

**Loss function:** Binary Cross-Entropy or Mean Squared Error

**Binary Cross-Entropy:**
```python
loss = 'binary_crossentropy'
```
- Treats each pixel as probability
- Works well with sigmoid output
- Standard choice for reconstruction tasks with normalized data

**Mean Squared Error:**
```python
loss = 'mse'
```
- Measures pixel-wise reconstruction error
- Simpler interpretation
- Can work with linear output layer

**Recommendation:** Binary cross-entropy with sigmoid output for MNIST, as it treats reconstruction probabilistically.
</details>

---

### B2. RNN/LSTM Forward Pass (15 points)

Consider a simple RNN processing a sequence of 3 time steps.

**Given:**
- Input sequence: x = [[0.5], [1.0], [0.3]] (3 time steps, 1 feature)
- Hidden state weights: W_h = [[0.8]]
- Input weights: W_x = [[0.5]]
- Bias: b = [0.1]
- Activation: tanh
- Initial hidden state: h_0 = [0]

**B2a.** (6 pts) Calculate the hidden state at each time step (h_1, h_2, h_3).

<details>
<summary>Model Answer</summary>

**RNN formula:** h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)

**Time step 1 (t=1):**
```
z_1 = W_h * h_0 + W_x * x_1 + b
z_1 = 0.8 * 0 + 0.5 * 0.5 + 0.1
z_1 = 0 + 0.25 + 0.1 = 0.35

h_1 = tanh(0.35) = 0.337
```

**Time step 2 (t=2):**
```
z_2 = W_h * h_1 + W_x * x_2 + b
z_2 = 0.8 * 0.337 + 0.5 * 1.0 + 0.1
z_2 = 0.270 + 0.5 + 0.1 = 0.870

h_2 = tanh(0.870) = 0.702
```

**Time step 3 (t=3):**
```
z_3 = W_h * h_2 + W_x * x_3 + b
z_3 = 0.8 * 0.702 + 0.5 * 0.3 + 0.1
z_3 = 0.562 + 0.15 + 0.1 = 0.812

h_3 = tanh(0.812) = 0.670
```

**Results:** h_1 = 0.337, h_2 = 0.702, h_3 = 0.670
</details>

**B2b.** (4 pts) If this were a sequence classification task (e.g., sentiment), which hidden state would you use and why?

<details>
<summary>Model Answer</summary>

**Answer:** Use **h_3** (the final hidden state)

**Why:**
- Final hidden state h_3 has seen all inputs (x_1, x_2, x_3)
- Contains accumulated information from entire sequence
- Standard practice: pass final hidden state to Dense layer for classification

**Implementation:**
```python
# Keras LSTM returns only final state by default
lstm = LSTM(64)  # Returns h_T only

# Or explicitly:
lstm = LSTM(64, return_sequences=False)  # Returns final h_T
```

**Alternative - All states:**
```python
lstm = LSTM(64, return_sequences=True)  # Returns h_1, h_2, ..., h_T
# Then apply GlobalAveragePooling1D or Attention
```
</details>

**B2c.** (5 pts) Explain why the vanishing gradient problem would make this RNN struggle to learn if the sequence had 100 time steps instead of 3.

<details>
<summary>Model Answer</summary>

**The vanishing gradient problem with 100 time steps:**

**Mathematical explanation:**
- Gradient w.r.t. h_1 involves: dL/dh_1 = dL/dh_100 * (dh_100/dh_99) * (dh_99/dh_98) * ... * (dh_2/dh_1)
- Each term dh_t/dh_{t-1} involves derivative of tanh and W_h
- tanh derivative range: (0, 1], often < 1
- Product of 99 terms each < 1 approaches 0 exponentially

**Numerical example:**
- If average gradient factor = 0.9
- After 100 steps: 0.9^99 = 0.00003 (practically zero)

**Consequences:**
- Gradients for early time steps become negligible
- Weights affecting early inputs barely update
- Network "forgets" long-term dependencies
- Cannot learn patterns spanning many time steps

**LSTM solution:**
- Cell state provides additive path (not multiplicative)
- Gates control information flow without destroying gradients
- Can learn dependencies 100+ time steps apart
</details>

---

### B3. Model Training Pipeline (15 points)

You're training an LSTM for sentiment analysis. Here's a partial training setup:

```python
model = Sequential([
    Embedding(10000, 128, mask_zero=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**B3a.** (5 pts) Add early stopping and model checkpoint callbacks. Explain each parameter choice.

<details>
<summary>Model Answer</summary>

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor='val_loss',      # Monitor validation loss
        patience=5,              # Wait 5 epochs for improvement
        restore_best_weights=True,  # Return to best epoch's weights
        verbose=1                # Print when stopping
    ),
    
    ModelCheckpoint(
        'best_sentiment_model.keras',  # Save path
        monitor='val_loss',            # Metric to monitor
        save_best_only=True,           # Only save improvements
        mode='min',                    # Lower loss is better
        verbose=1
    )
]
```

**Parameter explanations:**
- **monitor='val_loss':** Track validation (not training) to detect overfitting
- **patience=5:** Allow temporary validation dips; don't stop too early
- **restore_best_weights=True:** Final model has best generalization
- **save_best_only=True:** Don't save inferior models
- **mode='min':** Loss should decrease (use 'max' for accuracy)
</details>

**B3b.** (5 pts) The model overfits after epoch 8. Add two regularization techniques to the model.

<details>
<summary>Model Answer</summary>

```python
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

model = Sequential([
    Embedding(10000, 128, mask_zero=True),
    
    # Regularization 1: LSTM with recurrent dropout
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    
    Dense(64, activation='relu', 
          kernel_regularizer=l2(0.01)),  # Regularization 2: L2
    
    Dropout(0.5),  # Regularization 3: Dropout after dense
    
    Dense(1, activation='sigmoid')
])
```

**Techniques added:**
1. **LSTM dropout (0.3):** Drops 30% of input connections each step
2. **Recurrent dropout (0.3):** Drops 30% of recurrent connections
3. **L2 regularization (0.01):** Penalizes large weights in Dense layer
4. **Dropout (0.5):** Randomly zeros 50% of Dense outputs during training

**Why these work:**
- Dropout prevents co-adaptation of neurons
- L2 keeps weights small, improving generalization
- Forces network to learn redundant representations
</details>

**B3c.** (5 pts) After training, how would you save and load this model for inference? Show the code.

<details>
<summary>Model Answer</summary>

**Saving the model:**
```python
# Option 1: Save complete model (recommended)
model.save('sentiment_model.keras')

# Option 2: SavedModel format (for TF Serving)
model.save('sentiment_savedmodel', save_format='tf')

# Option 3: Weights only (need architecture separately)
model.save_weights('sentiment_weights.weights.h5')
```

**Loading for inference:**
```python
from tensorflow.keras.models import load_model

# Load complete model
loaded_model = load_model('sentiment_model.keras')

# Verify it works
predictions = loaded_model.predict(test_data)
```

**For weights-only approach:**
```python
# Must rebuild architecture first
new_model = build_sentiment_model()  # Same architecture
new_model.load_weights('sentiment_weights.weights.h5')
```

**Best practices:**
- Save tokenizer too (pickle or json)
- Version your models with timestamps
- Test loaded model before deploying
</details>

---

## Section C: Debugging and Analysis (20 Points)

### C1. Training Curve Analysis (10 points)

Given these training metrics for a text classification LSTM:

```
Epoch 1:  train_loss=0.69, val_loss=0.68, train_acc=0.52, val_acc=0.53
Epoch 5:  train_loss=0.45, val_loss=0.48, train_acc=0.78, val_acc=0.76
Epoch 10: train_loss=0.22, val_loss=0.52, train_acc=0.91, val_acc=0.79
Epoch 15: train_loss=0.08, val_loss=0.75, train_acc=0.98, val_acc=0.77
Epoch 20: train_loss=0.03, val_loss=1.05, train_acc=0.99, val_acc=0.74
```

**C1a.** (4 pts) Describe what is happening at each phase of training (epochs 1-5, 5-10, 10-20).

<details>
<summary>Model Answer</summary>

**Epochs 1-5: Healthy Learning**
- Both losses decreasing together (0.69->0.45, 0.68->0.48)
- Both accuracies increasing together
- Small gap between train and validation metrics
- Model is learning generalizable patterns

**Epochs 5-10: Beginning of Overfitting**
- Training loss continues dropping (0.45->0.22)
- Validation loss starts increasing (0.48->0.52)
- Train accuracy climbing (78%->91%) but val accuracy plateauing (76%->79%)
- Gap widening indicates memorization beginning

**Epochs 10-20: Severe Overfitting**
- Training loss near zero (0.22->0.03)
- Validation loss doubling (0.52->1.05)
- Train accuracy near 100%, val accuracy dropping (79%->74%)
- Model completely memorizing training data
- No longer generalizing to new data
</details>

**C1b.** (3 pts) At which epoch should training have stopped? What metric indicates this?

<details>
<summary>Model Answer</summary>

**Stop at epoch 5** (or possibly between 5-8)

**Indicator:** Validation loss at minimum (0.48)

**Evidence:**
- Epoch 5: val_loss = 0.48 (lowest)
- Epoch 5: val_acc = 0.76 (good, before decline)
- After epoch 5: val_loss increases monotonically
- Training beyond this point hurts generalization

**Implementation:**
```python
EarlyStopping(
    monitor='val_loss',
    patience=3,  # Would stop around epoch 8
    restore_best_weights=True  # Returns to epoch 5 weights
)
```
</details>

**C1c.** (3 pts) List three techniques to improve this model's generalization.

<details>
<summary>Model Answer</summary>

**Techniques to improve generalization:**

1. **Early Stopping with Patience:**
```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

2. **Dropout Regularization:**
```python
LSTM(64, dropout=0.3, recurrent_dropout=0.3)
model.add(Dropout(0.5))  # After Dense layers
```

3. **L2 Weight Regularization:**
```python
Dense(64, kernel_regularizer=l2(0.01))
LSTM(64, kernel_regularizer=l2(0.01))
```

4. **Data Augmentation (for text):**
- Back-translation (translate to another language and back)
- Synonym replacement
- Random word deletion/insertion

5. **Reduce Model Complexity:**
- Fewer LSTM units (64 -> 32)
- Fewer Dense layers or neurons
</details>

---

### C2. Code Debugging (10 points)

Identify and fix the errors in this NLP model:

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Task: Sentiment classification (binary) on movie reviews
# Vocabulary size: 10000, Max sequence length: 200

model = Sequential([
    Embedding(10000, 64, input_length=200),
    LSTM(128, return_sequences=True),
    Dense(64, activation='relu'),
    Dense(1, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

# Training
model.fit(X_train, y_train, epochs=10)
```

**C2a.** (6 pts) Identify at least three errors and explain why each is problematic.

<details>
<summary>Model Answer</summary>

**Error 1: `return_sequences=True` with Dense layer**
- LSTM outputs 3D tensor (batch, timesteps, features)
- Dense expects 2D input (batch, features)
- Will cause dimension mismatch error
- **Fix:** Use `return_sequences=False` for classification

**Error 2: `softmax` for binary classification**
- Softmax with 1 output neuron always outputs 1.0
- Softmax is for multi-class (2+ classes)
- **Fix:** Use `sigmoid` for binary classification

**Error 3: `categorical_crossentropy` for binary task**
- categorical_crossentropy expects one-hot encoded labels
- Binary classification should use `binary_crossentropy`
- **Fix:** Use `loss='binary_crossentropy'`

**Error 4: No validation data**
- Can't monitor overfitting
- No way to implement early stopping
- **Fix:** Add `validation_split=0.2` or `validation_data`

**Error 5: No masking for variable-length sequences**
- If sequences are padded, model learns from padding
- **Fix:** Add `mask_zero=True` to Embedding layer
</details>

**C2b.** (4 pts) Write the corrected code.

<details>
<summary>Model Answer</summary>

```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Task: Sentiment classification (binary) on movie reviews
# Vocabulary size: 10000, Max sequence length: 200

model = Sequential([
    # Added mask_zero for padded sequences
    Embedding(10000, 64, input_length=200, mask_zero=True),
    
    # Changed return_sequences to False for classification
    LSTM(128, return_sequences=False, dropout=0.2),
    
    Dense(64, activation='relu'),
    Dropout(0.5),  # Added regularization
    
    # Changed to sigmoid for binary classification
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Correct loss for binary
    metrics=['accuracy']  # Added metrics
)

# Training with validation
early_stop = EarlyStopping(monitor='val_loss', patience=3, 
                           restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=10,
    validation_split=0.2,  # Added validation
    callbacks=[early_stop]
)
```
</details>

---

## Section D: Short Essay (10 Points)

Choose ONE of the following questions. Write a clear, structured response (150-250 words).

### Option 1: The Evolution of Text Representation

Trace the evolution from one-hot encoding to word embeddings to contextual embeddings (like those from LSTMs). Explain why each advancement was necessary and what problem it solved.

<details>
<summary>Model Answer</summary>

**The Evolution of Text Representation**

**One-Hot Encoding (Baseline):**
Text processing began with one-hot encoding, representing each word as a sparse binary vector. While simple, this approach suffered from critical limitations: vocabulary-sized dimensions (10,000+ features), no semantic relationships (all words equidistant), and massive memory requirements for large vocabularies.

**Static Word Embeddings (Word2Vec, GloVe):**
Word2Vec revolutionized NLP by learning dense, low-dimensional vectors (50-300 dimensions) where semantic relationships emerged naturally. "King - man + woman = queen" demonstrated that meaning could be captured mathematically. However, these embeddings were **static**: each word had one fixed representation regardless of context. "Bank" meant the same whether discussing finance or rivers.

**Contextual Embeddings (LSTMs, Transformers):**
LSTMs addressed the context problem by producing word representations that depend on surrounding words. The hidden state at each position incorporates previous context, so "bank" in "river bank" differs from "bank account." Each word's representation is dynamic, computed fresh for each sentence.

**Why This Progression Mattered:**
1. Dimensionality reduction (one-hot -> embeddings)
2. Semantic similarity (embeddings capture meaning)
3. Context sensitivity (LSTM/transformers understand usage)

This evolution enabled modern NLP: sentiment analysis, machine translation, and language generation all depend on contextual understanding that static representations cannot provide.
</details>

---

### Option 2: Why Deep Learning Needs Regularization

Explain why neural networks are prone to overfitting and describe how dropout and early stopping work as complementary regularization strategies.

<details>
<summary>Model Answer</summary>

**Why Deep Learning Needs Regularization**

**The Overfitting Problem:**
Neural networks are powerful function approximators with millions of parameters. This capacity is both their strength and weakness: given enough parameters, networks can memorize training data instead of learning generalizable patterns. A model with more parameters than training examples can achieve zero training loss while failing on new data.

**Dropout: Training-Time Regularization**
Dropout randomly sets a fraction of neurons to zero during each training batch (typically 20-50%). This prevents **co-adaptation**: neurons cannot rely on specific other neurons always being present. The network must learn redundant representations, making predictions robust to missing features. Mathematically, dropout approximates ensemble learning, training many "sub-networks" simultaneously.

**Early Stopping: Implicit Regularization**
Early stopping monitors validation performance and halts training when generalization stops improving. Initially, both training and validation error decrease. As overfitting begins, validation error rises while training error continues falling. Stopping at the inflection point captures the model at peak generalization. Early stopping implicitly constrains the optimization, preventing the network from fully fitting training data.

**Complementary Strategies:**
Dropout works during training (modifying what the network learns), while early stopping works on training duration (when to stop). Together, dropout ensures robust features while early stopping ensures we don't overfit to training noise. Modern practice uses both: dropout forces generalizable representations, early stopping selects the optimal training point.
</details>

---

## Answer Key Summary

| Section | Points | Passing (70%) |
|---------|--------|---------------|
| A: Conceptual | 30 | 21 |
| B: Practical | 40 | 28 |
| C: Debugging | 20 | 14 |
| D: Essay | 10 | 7 |
| **Total** | **100** | **70** |

---

*Weekly Technical Assessment generated by Quality Assurance Agent*
*Week 2: Deep Learning & NLP - From Gradients to Language Models*

