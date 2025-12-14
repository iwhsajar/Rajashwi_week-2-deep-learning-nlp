# Interview Questions: Week 2 - Deep Learning & NLP

> **Purpose:** This question bank prepares trainees for technical interviews on deep learning fundamentals and natural language processing. Questions are categorized by difficulty following the 70-25-5 distribution.
>
> **Self-Quiz Instructions:** Read the question, think of your answer including the **Keywords**, then reveal the solution to compare.

---

## Beginner (Foundational) - 70%

### Q1: What is TensorBoard and why is it useful for training neural networks?

**Keywords:** Visualization, Metrics, Debugging, Training, Callbacks

<details>
<summary>Click to Reveal Answer</summary>

TensorBoard is a visualization toolkit provided by TensorFlow that helps you understand, debug, and optimize neural network training.

**Key capabilities:**
- **Real-time metric visualization:** Track loss and accuracy during training
- **Training curve comparison:** Compare experiments side-by-side
- **Model graph visualization:** Understand network architecture
- **Histogram tracking:** Monitor weight and gradient distributions

You integrate TensorBoard using a callback: `TensorBoard(log_dir='logs')` passed to `model.fit()`.
</details>

---

### Q2: What is an autoencoder and what are its main components?

**Keywords:** Encoder, Decoder, Latent Space, Bottleneck, Reconstruction

<details>
<summary>Click to Reveal Answer</summary>

An **autoencoder** is a neural network trained to reconstruct its input, learning efficient compressed representations in the process.

**Main components:**
1. **Encoder:** Compresses input into a lower-dimensional latent space
2. **Bottleneck (Latent Space):** Compressed representation of the input
3. **Decoder:** Reconstructs the original input from the latent representation

**Use cases:** Dimensionality reduction, denoising, anomaly detection, feature learning.

The network learns by minimizing **reconstruction loss** (difference between input and output).
</details>

---

### Q3: What is backpropagation and how does it enable neural networks to learn?

**Keywords:** Chain Rule, Gradients, Error Signal, Weight Updates, Backward Pass

<details>
<summary>Click to Reveal Answer</summary>

**Backpropagation** (backward propagation of errors) is the algorithm that computes gradients for training neural networks.

**How it works:**
1. **Forward pass:** Compute predictions layer by layer
2. **Calculate loss:** Compare predictions to true labels
3. **Backward pass:** Propagate error backward using the **chain rule**
4. **Compute gradients:** Calculate how each weight contributed to the error
5. **Update weights:** Adjust weights in the direction that reduces error

The chain rule allows efficient gradient computation through many layers:
dL/dw = dL/da * da/dz * dz/dw
</details>

---

### Q4: Explain the difference between SGD, Adam, and RMSprop optimizers.

**Keywords:** Learning Rate, Momentum, Adaptive, Gradient, Convergence

<details>
<summary>Click to Reveal Answer</summary>

| Optimizer | Description | Key Feature |
|-----------|-------------|-------------|
| **SGD** | Stochastic Gradient Descent | Uses fixed learning rate; simple but slow |
| **RMSprop** | Root Mean Square Propagation | Adapts learning rate per-parameter based on recent gradient magnitudes |
| **Adam** | Adaptive Moment Estimation | Combines momentum AND adaptive learning rates |

**Adam** is often the default choice because it:
- Adapts learning rates for each parameter
- Uses momentum to smooth updates
- Handles sparse gradients well
- Requires less hyperparameter tuning

**When to use each:**
- SGD + momentum: Fine-tuning, when you want more control
- Adam: General-purpose, good default
- RMSprop: RNNs and non-stationary problems
</details>

---

### Q5: What is batch normalization and why is it used?

**Keywords:** Internal Covariate Shift, Normalization, Training Speed, Stability, Scale

<details>
<summary>Click to Reveal Answer</summary>

**Batch Normalization** normalizes layer inputs to have zero mean and unit variance during training, stabilizing the learning process.

**How it works:**
1. Compute batch mean and variance
2. Normalize: (x - mean) / sqrt(variance + epsilon)
3. Scale and shift with learnable parameters (gamma, beta)

**Benefits:**
- **Faster training:** Allows higher learning rates
- **Reduced internal covariate shift:** Layer inputs remain stable
- **Regularization effect:** Adds slight noise (batch statistics)
- **Less sensitivity to initialization:** More forgiving weight initialization

Place batch normalization layers **after Dense/Conv layers** but **before activation** functions (though both approaches work).
</details>

---

### Q6: What is tokenization in NLP and why is it necessary?

**Keywords:** Text Processing, Vocabulary, Splitting, Tokens, Preprocessing

<details>
<summary>Click to Reveal Answer</summary>

**Tokenization** is the process of breaking text into smaller units (tokens) for processing by NLP models.

**Types of tokenization:**
- **Word-level:** Splits on whitespace/punctuation ("Hello world" -> ["Hello", "world"])
- **Subword:** Breaks words into smaller pieces (BPE, WordPiece)
- **Character-level:** Each character is a token

**Why necessary:**
- Neural networks need numerical input, not raw text
- Creates a **vocabulary** of known tokens
- Each token maps to an integer (token ID)
- Enables consistent text representation

**Challenge:** Out-of-vocabulary (OOV) words must be handled with special tokens or subword methods.
</details>

---

### Q7: What is one-hot encoding and what are its limitations for text?

**Keywords:** Sparse Vector, Vocabulary Size, Dimensionality, Binary, Semantic

<details>
<summary>Click to Reveal Answer</summary>

**One-hot encoding** represents each word as a binary vector with a 1 in the position corresponding to that word's index and 0s elsewhere.

**Example (vocabulary of 5 words):**
- "cat" = [1, 0, 0, 0, 0]
- "dog" = [0, 1, 0, 0, 0]

**Limitations:**
1. **Dimensionality explosion:** Vector size equals vocabulary size (10,000+ words = 10,000+ dimensions)
2. **Sparse representation:** Mostly zeros, memory-inefficient
3. **No semantic meaning:** "cat" and "kitten" are equally distant as "cat" and "airplane"
4. **No similarity:** Dot product between any two different words is always 0

**Solution:** Use dense embeddings instead (Word2Vec, Keras Embedding layer).
</details>

---

### Q8: What is a word embedding and how does it differ from one-hot encoding?

**Keywords:** Dense Vector, Semantic Similarity, Learned Representation, Dimensions, Context

<details>
<summary>Click to Reveal Answer</summary>

**Word embeddings** are dense, low-dimensional vector representations of words that capture semantic meaning.

| Aspect | One-Hot Encoding | Word Embeddings |
|--------|------------------|-----------------|
| **Dimensions** | Vocabulary size (sparse) | Fixed small size (50-300) |
| **Values** | Binary (0 or 1) | Continuous real numbers |
| **Similarity** | All words equidistant | Similar words = close vectors |
| **Semantic meaning** | None | Captures relationships |

**Key properties of embeddings:**
- Similar words have similar vectors (cosine similarity)
- Support vector arithmetic: king - man + woman = queen
- Can be pre-trained (Word2Vec, GloVe) or learned during training (Keras Embedding layer)

Embeddings transform discrete words into a continuous space where relationships are meaningful.
</details>

---

### Q9: What is an RNN and what problem does it solve that feedforward networks cannot?

**Keywords:** Sequential Data, Hidden State, Memory, Time Steps, Temporal

<details>
<summary>Click to Reveal Answer</summary>

A **Recurrent Neural Network (RNN)** is designed to process sequential data by maintaining a **hidden state** that carries information across time steps.

**Problem with feedforward networks:**
- Fixed input size
- No memory of previous inputs
- Cannot capture temporal dependencies

**How RNNs solve this:**
1. Process inputs one at a time (time steps)
2. Hidden state updated at each step: h_t = f(h_{t-1}, x_t)
3. Hidden state acts as **memory** of previous inputs
4. Same weights shared across all time steps (parameter efficiency)

**Use cases:** Text generation, sentiment analysis, machine translation, time series prediction.

**Limitation:** Vanishing gradient problem makes learning long-term dependencies difficult.
</details>

---

### Q10: What is an LSTM and how does it solve the vanishing gradient problem?

**Keywords:** Gates, Cell State, Long-term Memory, Forget Gate, Input Gate

<details>
<summary>Click to Reveal Answer</summary>

**LSTM (Long Short-Term Memory)** is an RNN variant designed to learn long-term dependencies through a gating mechanism.

**Key components:**
1. **Cell state (C_t):** Long-term memory highway that can carry information unchanged
2. **Forget gate:** Decides what to discard from cell state
3. **Input gate:** Decides what new information to add
4. **Output gate:** Decides what to output from current state

**How it solves vanishing gradients:**
- Cell state provides a **gradient highway** through time
- Gradients can flow unchanged through cell state (no repeated multiplication)
- Gates use sigmoid (0-1) to selectively pass or block information
- Additive operations (not multiplicative) preserve gradients

**Result:** LSTMs can effectively learn dependencies hundreds of time steps apart.
</details>

---

### Q11: How do you handle variable-length sequences in Keras?

**Keywords:** Padding, Masking, pad_sequences, Mask Layer, Batch Processing

<details>
<summary>Click to Reveal Answer</summary>

Variable-length sequences must be standardized for batch processing in neural networks.

**Padding:**
```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(sequences, maxlen=100, padding='post')
```
- Adds zeros to make all sequences equal length
- `padding='pre'` or `padding='post'` controls where zeros are added
- `truncating` parameter handles sequences longer than maxlen

**Masking:**
- Tells the model to ignore padded positions
- Enable with `mask_zero=True` in Embedding layer
- Or use explicit `Masking` layer

**Why masking matters:**
- Without masking, padded zeros affect loss and gradients
- Masking ensures model only learns from real data
- Essential for accurate sequence modeling
</details>

---

### Q12: What is the purpose of model checkpointing during training?

**Keywords:** Saving, Best Model, Resume Training, ModelCheckpoint, Callback

<details>
<summary>Click to Reveal Answer</summary>

**Model checkpointing** saves model weights during training, allowing you to:
- Recover from training interruptions
- Keep the best-performing model
- Resume training from a saved state

**ModelCheckpoint callback:**
```python
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min'
)
```

**Key parameters:**
- `monitor`: Metric to track (val_loss, val_accuracy)
- `save_best_only`: Only save when metric improves
- `mode`: 'min' for loss, 'max' for accuracy
- `save_weights_only`: Save weights vs full model

**Best practice:** Always checkpoint based on validation metrics, not training metrics.
</details>

---

### Q13: What is early stopping and why is it important?

**Keywords:** Overfitting, Patience, Validation, Restore Best, Generalization

<details>
<summary>Click to Reveal Answer</summary>

**Early stopping** is a regularization technique that stops training when validation performance stops improving.

**How it works:**
1. Monitor validation metric each epoch
2. Stop if no improvement for `patience` epochs
3. Optionally restore weights from best epoch

**Implementation:**
```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

**Why important:**
- **Prevents overfitting:** Stops before model memorizes training data
- **Saves compute:** Avoids unnecessary training epochs
- **Automatic:** No need to manually determine optimal epochs

**Key insight:** The model checkpoint at the best epoch often outperforms the final epoch's model.
</details>

---

### Q14: What is dropout and how does it prevent overfitting?

**Keywords:** Regularization, Random, Neurons, Training, Ensemble

<details>
<summary>Click to Reveal Answer</summary>

**Dropout** is a regularization technique that randomly "drops" (sets to zero) a fraction of neurons during training.

**How it works:**
- During training: Each neuron has probability p (e.g., 0.5) of being temporarily removed
- During inference: All neurons used, but outputs scaled by (1-p)
- Forces network to not rely on any single feature

**Why it prevents overfitting:**
1. **Ensemble effect:** Each forward pass trains a different "sub-network"
2. **Redundancy:** Network learns multiple paths to solve problems
3. **Co-adaptation:** Prevents neurons from over-relying on each other

**Usage:**
```python
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # 50% dropout
```

**Typical values:** 0.2-0.5 for hidden layers, lower for input layers.
</details>

---

## Intermediate (Application/Scenario) - 25%

### Q15: You're training a model and notice training loss keeps decreasing but validation loss increases after epoch 10. What is happening and how do you fix it?

**Keywords:** Overfitting, Regularization, Early Stopping, Dropout, Data Augmentation

**Hint:** The diverging curves are a classic diagnostic pattern.

<details>
<summary>Click to Reveal Answer</summary>

**Diagnosis:** This is classic **overfitting** - the model is memorizing training data instead of learning generalizable patterns.

**Evidence:**
- Training loss decreasing = model fits training data better
- Validation loss increasing = model performs worse on unseen data
- Growing gap = memorization, not generalization

**Solutions:**

1. **Early Stopping:**
```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```

2. **Add Dropout:**
```python
model.add(Dropout(0.3))  # After dense layers
```

3. **L2 Regularization:**
```python
Dense(128, kernel_regularizer=l2(0.01))
```

4. **Reduce Model Complexity:** Fewer layers or neurons

5. **Data Augmentation:** Generate more varied training examples

**Best practice:** Use the model from epoch 10 (best validation) via `restore_best_weights=True`.
</details>

---

### Q16: You're building a sentiment classifier and your text data has variable lengths (10 to 500 words). How do you handle this?

**Keywords:** Padding, Truncation, Masking, max_length, RNN/LSTM

**Hint:** Consider what batch processing requires.

<details>
<summary>Click to Reveal Answer</summary>

**Problem:** Neural networks require fixed-size inputs, but text lengths vary.

**Solution approach:**

1. **Choose a max length:**
   - Analyze length distribution
   - Choose a value that covers most examples (e.g., 95th percentile)
   - Balance between coverage and memory

2. **Pad and truncate:**
```python
padded = pad_sequences(sequences, 
                       maxlen=200,
                       padding='post',    # Add zeros at end
                       truncating='post') # Cut long sequences at end
```

3. **Enable masking:**
```python
Embedding(vocab_size, 128, mask_zero=True)
```

4. **Consider truncation strategy:**
   - `post`: Keep beginning (often contains key info)
   - `pre`: Keep end (for recent context)

**Why masking:**
- Without masking, padded zeros affect RNN hidden state
- Masking tells model to ignore padded positions
- Critical for accurate sequence modeling
</details>

---

### Q17: Your autoencoder reconstructs MNIST digits well but the latent space isn't useful for visualization. What might be wrong and how would you improve it?

**Keywords:** Latent Dimension, Regularization, Variational, Bottleneck, Clustering

**Hint:** The bottleneck size matters for what the latent space captures.

<details>
<summary>Click to Reveal Answer</summary>

**Potential issues and solutions:**

1. **Bottleneck too large:**
   - If latent dimension is too high, encoder may not learn compressed representations
   - **Fix:** Reduce latent dimension (try 2-32 for visualization)

2. **No structure in latent space:**
   - Standard autoencoders don't guarantee organized latent spaces
   - **Fix:** Use **Variational Autoencoder (VAE)** which adds regularization:
     - KL divergence loss forces latent space toward normal distribution
     - Creates smooth, interpolatable latent space

3. **Insufficient bottleneck constraint:**
   - Add **L1/L2 regularization** on latent layer
   - Encourages sparse, more interpretable representations

4. **Architecture issues:**
   - Too powerful decoder can reconstruct from arbitrary latent vectors
   - **Fix:** Balance encoder/decoder capacity

**Visualization tips:**
- Use t-SNE or PCA on latent vectors
- Color by digit class to verify clustering
- Well-structured latent space should show digit clusters
</details>

---

### Q18: You're training a text generation model and the output is repetitive ("the the the..."). What could cause this and how do you fix it?

**Keywords:** Temperature, Sampling, Greedy, Top-k, Probability Distribution

**Hint:** Think about how the model chooses the next word.

<details>
<summary>Click to Reveal Answer</summary>

**Cause:** The model is likely using **greedy decoding** - always picking the most probable next token.

**Why this causes repetition:**
- If "the" is most probable, model picks it
- Given "the", "the" might again be most probable
- Creates degenerate loops

**Solutions:**

1. **Temperature sampling:**
```python
logits = logits / temperature  # temperature > 1 = more random
probs = softmax(logits)
next_token = np.random.choice(vocab, p=probs)
```
   - Higher temperature (1.0-1.5) = more diversity
   - Lower temperature (0.5-0.8) = more focused

2. **Top-k sampling:**
   - Only consider k most probable tokens
   - Sample from this subset

3. **Top-p (nucleus) sampling:**
   - Consider tokens that sum to probability p
   - Dynamically adjusts candidate set size

4. **Training improvements:**
   - More training data
   - Increase model capacity
   - Add dropout to prevent overfitting to common patterns
</details>

---

### Q19: You need to compare different LSTM architectures for a sequence classification task. What experiment would you design?

**Keywords:** Controlled Experiment, Hyperparameters, Validation, Metrics, Baseline

**Hint:** Think about controlling variables and fair comparison.

<details>
<summary>Click to Reveal Answer</summary>

**Experiment design:**

1. **Establish baseline:**
   - Simple LSTM: 1 layer, 64 units
   - Record accuracy, F1, training time

2. **Variables to test:**
   - Number of LSTM layers (1, 2, 3)
   - Hidden units (32, 64, 128, 256)
   - Bidirectional vs unidirectional
   - Dropout rate (0.2, 0.3, 0.5)
   - GRU vs LSTM

3. **Control conditions:**
   - Same train/val/test split
   - Same embedding dimension and pretrained embeddings
   - Same optimizer (Adam) and learning rate
   - Same batch size and early stopping criteria

4. **Evaluation:**
   - Primary: Validation accuracy or F1
   - Secondary: Training time, model size
   - Use multiple random seeds (3-5) and report mean +/- std

5. **Analysis:**
   - Learning curves (TensorBoard)
   - Confusion matrices
   - Per-class performance

**Report template:**
| Architecture | Val Accuracy | F1 | Training Time | Parameters |
</details>

---

### Q20: Your Word2Vec model gives unexpected results for "king - man + woman". How would you debug this?

**Keywords:** Training Data, Vocabulary, Vector Quality, Cosine Similarity, Evaluation

**Hint:** The quality of embeddings depends heavily on training data.

<details>
<summary>Click to Reveal Answer</summary>

**Debugging steps:**

1. **Check vocabulary:**
   - Are "king", "man", "woman", "queen" in vocabulary?
   - Words must appear frequently enough in training data

2. **Verify training data:**
   - Sufficient corpus size? (Need millions of words for quality)
   - Relevant domain? (Wikipedia vs tweets produce different embeddings)
   - Clean data? (Proper tokenization, lowercase consistency)

3. **Examine hyperparameters:**
   - Embedding dimension (100-300 typical)
   - Window size (context words to consider)
   - Minimum word frequency threshold
   - Training epochs

4. **Test basic relationships:**
```python
model.most_similar('king')  # Should return royalty-related words
model.similarity('king', 'queen')  # Should be high
model.similarity('king', 'apple')  # Should be low
```

5. **Consider pretrained:**
   - If corpus is small, use pretrained embeddings (GloVe, word2vec-google-news)
   - Fine-tune on domain-specific data if needed

6. **Evaluation:**
   - Use analogy benchmarks
   - Visualize with t-SNE to check clustering
</details>

---

## Advanced (Deep Dive/System Design) - 5%

### Q21: Explain what happens inside an LSTM cell step-by-step. Include the math for each gate.

**Keywords:** Forget Gate, Input Gate, Output Gate, Cell State, Sigmoid, Tanh

<details>
<summary>Click to Reveal Answer</summary>

**LSTM cell computation at time step t:**

Given: input x_t, previous hidden state h_{t-1}, previous cell state C_{t-1}

**1. Forget Gate (what to remove from cell state):**
```
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
```
- Output: values between 0 (forget) and 1 (keep)

**2. Input Gate (what new info to store):**
```
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
C_tilde = tanh(W_C * [h_{t-1}, x_t] + b_C)
```
- i_t: which values to update
- C_tilde: candidate values to add

**3. Update Cell State:**
```
C_t = f_t * C_{t-1} + i_t * C_tilde
```
- Forget old info: f_t * C_{t-1}
- Add new info: i_t * C_tilde

**4. Output Gate (what to output):**
```
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```
- o_t: which parts of cell state to output
- h_t: filtered cell state through tanh

**Key insight:** Cell state C_t flows through with only element-wise operations (addition, multiplication), preserving gradients over long sequences.
</details>

---

### Q22: Design a training pipeline for a production NLP model that handles checkpointing, early stopping, learning rate scheduling, and logging.

**Keywords:** Callbacks, TensorBoard, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

<details>
<summary>Click to Reveal Answer</summary>

**Production training pipeline:**

```python
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger, TerminateOnNaN
)
from datetime import datetime

# Create versioned experiment directory
experiment_name = f"sentiment_lstm_{datetime.now():%Y%m%d_%H%M%S}"
log_dir = f"experiments/{experiment_name}"

callbacks = [
    # 1. Save best model
    ModelCheckpoint(
        f"{log_dir}/best_model.keras",
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    
    # 2. Save periodic checkpoints (for resume)
    ModelCheckpoint(
        f"{log_dir}/checkpoint_epoch_{{epoch:02d}}.keras",
        save_freq='epoch',
        save_best_only=False
    ),
    
    # 3. Early stopping
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    
    # 4. Learning rate reduction
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # 5. TensorBoard logging
    TensorBoard(
        log_dir=f"{log_dir}/tensorboard",
        histogram_freq=1,
        profile_batch=0
    ),
    
    # 6. CSV logging for analysis
    CSVLogger(f"{log_dir}/training_log.csv"),
    
    # 7. Safety: terminate on NaN
    TerminateOnNaN()
]

# Train with all callbacks
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=callbacks
)

# Save final artifacts
model.save(f"{log_dir}/final_model.keras")
with open(f"{log_dir}/config.json", 'w') as f:
    json.dump(model_config, f)
```

**Pipeline features:**
- Version control via timestamped directories
- Both best and periodic checkpoints
- Automatic learning rate decay
- Comprehensive logging (TensorBoard + CSV)
- NaN detection for training stability
</details>

---

### Q23: How would you implement a custom training loop in TensorFlow/Keras that gives you more control than model.fit()?

**Keywords:** GradientTape, Custom Loop, Manual Updates, Metrics, Flexibility

<details>
<summary>Click to Reveal Answer</summary>

**Custom training loop implementation:**

```python
import tensorflow as tf

# Model, optimizer, loss function
model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Metrics
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

@tf.function  # Compile for performance
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    # Compute and apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    train_loss.update_state(loss)
    train_acc.update_state(y, predictions)
    return loss

@tf.function
def val_step(x, y):
    predictions = model(x, training=False)
    loss = loss_fn(y, predictions)
    val_loss.update_state(loss)
    val_acc.update_state(y, predictions)

# Training loop
for epoch in range(epochs):
    # Reset metrics
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    
    # Training
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    # Validation
    for x_batch, y_batch in val_dataset:
        val_step(x_batch, y_batch)
    
    # Print progress
    print(f"Epoch {epoch+1}: "
          f"Loss={train_loss.result():.4f}, Acc={train_acc.result():.4f}, "
          f"Val_Loss={val_loss.result():.4f}, Val_Acc={val_acc.result():.4f}")
    
    # Custom logic: gradient clipping, custom LR scheduling, etc.
```

**When to use custom loops:**
- Gradient accumulation for large effective batch sizes
- Multi-GPU custom distribution strategies
- GANs with alternating generator/discriminator updates
- Custom gradient clipping or modification
- Research and experimental architectures
</details>

---

## Bonus Interview Scenarios

### Q24: Walk me through how you would build an end-to-end text classification system.

**Keywords:** Preprocessing, Embeddings, Architecture, Training, Deployment

<details>
<summary>Click to Reveal Answer</summary>

**End-to-end text classification pipeline:**

**1. Data Preparation:**
- Clean text: lowercase, remove special characters, handle contractions
- Split: train (70%), validation (15%), test (15%)
- Analyze: class distribution, text length distribution

**2. Text Preprocessing:**
```python
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
train_padded = pad_sequences(train_sequences, maxlen=200, padding='post')
```

**3. Architecture Choice:**
- **Simple/Fast:** Embedding + GlobalAveragePooling + Dense
- **Sequence-aware:** Embedding + LSTM/BiLSTM + Dense
- **State-of-art:** Pretrained transformer (BERT)

**4. Model Implementation:**
```python
model = Sequential([
    Embedding(vocab_size, 128, mask_zero=True),
    Bidirectional(LSTM(64, dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

**5. Training:**
- Use callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Monitor validation metrics
- Address class imbalance if present

**6. Evaluation:**
- Confusion matrix
- Per-class precision, recall, F1
- Error analysis on misclassified examples

**7. Deployment:**
- Save model and tokenizer
- Create inference pipeline
- Handle edge cases (empty text, unknown tokens)
</details>

---

### Q25: Explain the trade-offs between different model saving formats in TensorFlow/Keras.

**Keywords:** H5, SavedModel, Weights, Portability, Serving

<details>
<summary>Click to Reveal Answer</summary>

**Model saving formats comparison:**

| Format | Command | Contents | Use Case |
|--------|---------|----------|----------|
| **HDF5 (.h5/.keras)** | `model.save('model.h5')` | Architecture + weights + optimizer | Development, prototyping |
| **SavedModel** | `model.save('saved_model/')` | Full serialization + signatures | Production, TF Serving |
| **Weights only** | `model.save_weights('weights.h5')` | Just weights | Transfer learning, fine-tuning |

**Trade-offs:**

**HDF5 (.h5 / .keras):**
- Pros: Single file, familiar format, Keras-native
- Cons: May lose custom objects, less portable outside Python
- Best for: Development, model iteration

**SavedModel (directory):**
- Pros: Language-agnostic, TensorFlow Serving compatible, captures full graph
- Cons: Directory structure, larger size
- Best for: Production deployment, TF Serving, TF Lite conversion

**Weights only:**
- Pros: Smallest, can apply to different architectures
- Cons: Need matching model architecture to reload
- Best for: Transfer learning, ensemble models

**Recommendation:**
- Use `.keras` format during development
- Export to SavedModel for production deployment
- Save weights separately for transfer learning experiments
</details>

---

*Interview Question Bank generated by Quality Assurance Agent for Week 2: Deep Learning & NLP*
*Distribution: 14 Beginner (56%), 6 Intermediate (24%), 3 Advanced (12%), 2 Bonus (8%)*

