# Weekly Knowledge Check: RNNs, LSTMs, and Sequential Data (Thursday)

## Part 1: Multiple Choice

### 1. What is the main limitation of basic RNNs that LSTMs solve?

- [ ] A) RNNs suffer from vanishing gradients on long sequences
- [ ] B) RNNs can't process sequences
- [ ] C) RNNs are too slow
- [ ] D) RNNs can't use embeddings

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) RNNs suffer from vanishing gradients on long sequences

**Explanation:** In RNNs, gradients must propagate through many time steps during backpropagation. Each step multiplies the gradient, causing it to shrink exponentially. After ~10-20 steps, gradients vanish, preventing learning of long-term dependencies.

- **Why others are wrong:**
  - B) RNNs do process sequences, that's their purpose
  - C) Speed isn't their fundamental limitation
  - D) RNNs work well with embeddings
</details>

---

### 2. How many gates does an LSTM cell have?

- [ ] A) 1
- [ ] B) 3
- [ ] C) 2
- [ ] D) 4

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) 3

**Explanation:** LSTMs have three gates: (1) Forget gate - what to remove from cell state, (2) Input gate - what new information to add, (3) Output gate - what to output from cell state. These use sigmoid activation to produce values between 0 and 1.

- **Why others are wrong:**
  - A) One gate would be too simple
  - C) GRUs have 2 gates, not LSTMs
  - D) LSTMs have 3 gates, not 4
</details>

---

### 3. What is the difference between cell state (C) and hidden state (h) in LSTMs?

- [ ] A) They are the same thing
- [ ] B) Cell state stores weights; hidden state stores activations
- [ ] C) Cell state is long-term memory; hidden state is the output
- [ ] D) Hidden state is long-term memory; cell state is the output

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Cell state is long-term memory; hidden state is the output

**Explanation:** The cell state (C) acts as a "memory highway" where information can flow unchanged across many time steps. The hidden state (h) is a filtered version produced by the output gate and serves as the LSTM's output at each time step.

- **Why others are wrong:**
  - A) They serve different purposes
  - B) Both store activations, not weights
  - D) It's the opposite
</details>

---

### 4. When stacking LSTM layers, what parameter must be set to True for all but the last LSTM?

- [ ] A) `trainable`
- [ ] B) `stateful`
- [ ] C) `bidirectional`
- [ ] D) `return_sequences`

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) `return_sequences`

**Explanation:** By default, LSTM returns only the final hidden state. When stacking LSTMs, intermediate layers need `return_sequences=True` to output the full sequence, which the next LSTM layer expects as input.

- **Why others are wrong:**
  - A) trainable affects weight updates, not output shape
  - B) stateful maintains state across batches
  - C) bidirectional is a separate wrapper
</details>

---

### 5. How many gates does a GRU (Gated Recurrent Unit) have?

- [ ] A) 2
- [ ] B) 1
- [ ] C) 3
- [ ] D) 4

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) 2

**Explanation:** GRU has two gates: reset gate and update gate. This is simpler than LSTM's three gates, making GRU faster to train with often similar performance.

- **Why others are wrong:**
  - A) One gate is too simple
  - C) 3 gates is for LSTMs
  - D) 4 gates is more than any standard architecture
</details>

---

### 6. What is the purpose of masking in RNN/LSTM models?

- [ ] A) To ignore padded positions during computation
- [ ] B) To hide the output from the loss function
- [ ] C) To encrypt the model weights
- [ ] D) To reduce model size

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) To ignore padded positions during computation

**Explanation:** When sequences are padded, the padding tokens are meaningless. Masking (`mask_zero=True`) tells the RNN/LSTM to skip these positions during forward pass and gradient computation.

- **Why others are wrong:**
  - B) Masking affects computation, not loss visibility
  - C) Masking is not encryption
  - D) Masking doesn't change model size
</details>

---

### 7. What does the hidden state in an RNN represent?

- [ ] A) The model's weights
- [ ] B) The input embeddings
- [ ] C) The loss function value
- [ ] D) The network's memory at each time step

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) The network's memory at each time step

**Explanation:** The hidden state captures information from previous time steps. It's passed from one step to the next, allowing the network to "remember" what it has seen earlier in the sequence.

- **Why others are wrong:**
  - A) Weights are separate from hidden state
  - C) Embeddings are inputs, not hidden state
  - D) Loss is computed at the end, not per time step
</details>

---

### 8. Why do LSTMs use sigmoid activation for their gates?

- [ ] A) Sigmoid outputs values between 0 and 1, acting as "switches"
- [ ] B) Sigmoid is faster to compute
- [ ] C) Sigmoid prevents overfitting
- [ ] D) Sigmoid produces larger gradients

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Sigmoid outputs values between 0 and 1, acting as "switches"

**Explanation:** Sigmoid outputs values in [0, 1], which represent how much information to pass through. 0 means "completely forget/block" and 1 means "completely keep/pass". This is perfect for gating mechanisms.

- **Why others are wrong:**
  - B) Tanh is equally fast
  - C) Regularization prevents overfitting, not activation choice
  - D) Sigmoid actually has smaller gradients than ReLU
</details>

---

### 9. What is backpropagation through time (BPTT)?

- [ ] A) A method to speed up training
- [ ] B) A type of RNN architecture
- [ ] C) Applying backpropagation to unfolded RNN sequences
- [ ] D) An alternative to gradient descent

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Applying backpropagation to unfolded RNN sequences

**Explanation:** BPTT unfolds the RNN across time steps and applies standard backpropagation. Gradients are computed for each time step and propagated backward through the sequence.

- **Why others are wrong:**
  - A) BPTT can actually be slow for long sequences
  - C) It's an algorithm, not an architecture
  - D) BPTT uses gradient descent
</details>

---

### 10. What problem can occur when gradients become extremely large during RNN training?

- [ ] A) Vanishing gradients
- [ ] B) Dead neurons
- [ ] C) Underfitting
- [ ] D) Exploding gradients

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Exploding gradients

**Explanation:** When gradients multiply through many time steps and become very large, training becomes unstable. The loss oscillates wildly or diverges to infinity. Gradient clipping is the standard solution.

- **Why others are wrong:**
  - A) Vanishing is when gradients become too small
  - C) Dead neurons are a ReLU-specific issue
  - D) Underfitting is about model capacity
</details>

---

### 11. What does this code output?

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences = [[1, 2], [3, 4, 5, 6]]
padded = pad_sequences(sequences, maxlen=4, padding='pre')
print(padded[0])
```

- [ ] A) [1, 2, 0, 0]
- [ ] B) [0, 0, 1, 2]
- [ ] C) [1, 2, 3, 4]
- [ ] D) [0, 1, 2, 0]

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) [0, 0, 1, 2]

**Explanation:** With `padding='pre'`, zeros are added at the beginning. The first sequence [1, 2] becomes [0, 0, 1, 2] to reach maxlen=4.

- **Why others are wrong:**
  - A) That's post-padding
  - C) That's the second sequence
  - D) That's neither pre nor post padding
</details>

---

### 12. What is the output shape of this model for a batch of 32 sequences?

```python
model = keras.Sequential([
    layers.Embedding(10000, 128, input_length=50),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32)
])
```

- [ ] A) (32, 64)
- [ ] B) (32, 50, 32)
- [ ] C) (32, 50, 64)
- [ ] D) (32, 32)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) (32, 32)

**Explanation:** The first LSTM returns sequences (50, 64). The second LSTM (without return_sequences) returns only the final state (32,). For batch_size=32, output is (32, 32).

- **Why others are wrong:**
  - A) That would require 64 units in final layer
  - B) That would require return_sequences=True on second LSTM
  - C) That's the output of the first LSTM
</details>

---

### 13. Bidirectional LSTMs are suitable for which type of task?

- [ ] A) Text classification and named entity recognition
- [ ] B) Real-time speech recognition
- [ ] C) Text generation (predicting next word)
- [ ] D) Online translation with streaming input

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Text classification and named entity recognition

**Explanation:** Bidirectional LSTMs process sequences in both directions, requiring the entire sequence. They're perfect for classification where you have the complete input. Text generation needs to work left-to-right without seeing future tokens.

- **Why others are wrong:**
  - A) Generation can't see future words
  - B) Real-time processing needs low latency, can't wait for full sequence
  - D) Online translation needs to start before input finishes
</details>

---

### 14. What is the primary reason LSTMs use addition in cell state updates instead of multiplication?

- [ ] A) Addition is faster to compute
- [ ] B) Addition uses less memory
- [ ] C) Addition allows gradients to flow without vanishing
- [ ] D) Addition produces larger outputs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Addition allows gradients to flow without vanishing

**Explanation:** The cell state update `C_t = f_t * C_{t-1} + i_t * C_tilde` uses addition. This allows gradients to flow backward through the cell state without the multiplicative shrinking that causes vanishing gradients in basic RNNs.

- **Why others are wrong:**
  - A) Computational cost is similar
  - C) Memory usage is similar
  - D) Output magnitude isn't the goal
</details>

---

### 15. Which statement about pre-padding vs. post-padding is correct for LSTM classification?

- [ ] A) Post-padding is always preferred
- [ ] B) Padding type doesn't affect results
- [ ] C) Neither type should be used
- [ ] D) Pre-padding is generally preferred so the final state captures real content

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Pre-padding is generally preferred so the final state captures real content

**Explanation:** In classification, the final hidden state is used for prediction. With pre-padding, actual content is processed last, so the final hidden state captures real information. With post-padding, the RNN processes padding after content, potentially overwriting useful information.

- **Why others are wrong:**
  - A) Pre-padding is usually better for classification
  - C) Padding type does affect results
  - D) Padding is necessary for batch processing
</details>

---

## Part 2: True/False

### 16. Bidirectional LSTMs are suitable for text generation tasks.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Bidirectional LSTMs process sequences in both directions, requiring the entire sequence. In text generation, you generate tokens one at a time without seeing future tokens. Bidirectional LSTMs are great for classification but not for generation.
</details>

---

### 17. LSTMs solve vanishing gradients because the cell state uses addition instead of multiplication for updates.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** The cell state update `C_t = f_t * C_{t-1} + i_t * C_tilde` uses addition. This allows gradients to flow backward through the cell state without the multiplicative shrinking that causes vanishing gradients in basic RNNs.
</details>

---

### 18. GRUs have more parameters than LSTMs because they have more gates.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** GRUs have 2 gates (reset and update) while LSTMs have 3 gates. This means GRUs have fewer parameters and train faster, though sometimes with similar performance.
</details>

---

### 19. The forget gate in an LSTM decides what old information to keep in the cell state.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** The forget gate produces values between 0 and 1 for each element of the cell state. Values close to 1 mean "keep this information," while values close to 0 mean "forget this information."
</details>

---

### 20. RNNs naturally handle variable-length sequences without any modifications.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** RNNs process sequences one element at a time, maintaining hidden state. This means they can handle any sequence length. However, for batch processing, sequences are often padded to the same length.
</details>

---

## Part 3: Code Prediction

### 21. What is the output shape after this layer?

```python
layers.LSTM(64, return_sequences=True, input_shape=(100, 128))
# Input shape: (batch_size, 100, 128)
```

- [ ] A) (batch_size, 64)
- [ ] B) (batch_size, 100, 64)
- [ ] C) (batch_size, 100, 128)
- [ ] D) (batch_size, 64, 100)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (batch_size, 100, 64)

**Explanation:** With `return_sequences=True`, the LSTM outputs the hidden state at every time step. The output shape is (batch, sequence_length, units) = (batch, 100, 64).
</details>

---

### 22. What happens if you stack LSTMs without `return_sequences=True` on the first layer?

```python
model = keras.Sequential([
    layers.LSTM(64),                    # Missing return_sequences=True
    layers.LSTM(32),
])
```

- [ ] A) Model trains normally
- [ ] B) Error: Second LSTM expects sequences but receives a single vector
- [ ] C) The model runs slower
- [ ] D) Output dimension doubles

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Error: Second LSTM expects sequences but receives a single vector

**Explanation:** Without `return_sequences=True`, the first LSTM outputs only the final state (shape: batch, 64). The second LSTM expects sequences (batch, timesteps, features), causing a shape mismatch error.
</details>

---

### 23. How many trainable parameters are in this LSTM layer approximately?

```python
layers.LSTM(32, input_shape=(50, 64))
# Input: 64-dimensional, Output: 32 units
```

- [ ] A) ~2,000
- [ ] B) ~12,000
- [ ] C) ~32
- [ ] D) ~64

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) ~12,000

**Explanation:** LSTM has 4 gates, each with weights for input and hidden state. Parameters = 4 * ((input_dim * units) + (units * units) + units) = 4 * (64*32 + 32*32 + 32) = 4 * (2048 + 1024 + 32) = 4 * 3104 = 12,416.
</details>

---

### 24. What does `recurrent_dropout=0.2` do in an LSTM layer?

```python
layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
```

- [ ] A) Drops 20% of input connections
- [ ] B) Drops 20% of recurrent connections (between time steps)
- [ ] C) Drops 20% of output connections
- [ ] D) Drops 20% of the hidden units permanently

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Drops 20% of recurrent connections (between time steps)

**Explanation:** `recurrent_dropout` applies dropout to the recurrent connections (h_t to h_{t+1}). `dropout` applies to input connections (x_t to h_t). Both help prevent overfitting in sequential models.
</details>

---

### 25. What will this Bidirectional layer output shape be?

```python
layers.Bidirectional(layers.LSTM(64))
# Input: (batch, 50, 100)
```

- [ ] A) (batch, 64)
- [ ] B) (batch, 128)
- [ ] C) (batch, 50, 64)
- [ ] D) (batch, 50, 128)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (batch, 128)

**Explanation:** Bidirectional runs two LSTMs (forward and backward), each producing 64-dim output. By default, outputs are concatenated: 64 + 64 = 128. Without return_sequences, only final states are returned.
</details>

---

### 26. What is the purpose of this callback during LSTM training?

```python
keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

- [ ] A) Speeds up training by skipping validation
- [ ] B) Stops training when validation loss stops improving and restores best weights
- [ ] C) Increases learning rate when loss plateaus
- [ ] D) Saves the model after every epoch

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Stops training when validation loss stops improving and restores best weights

**Explanation:** EarlyStopping monitors validation loss, waits 5 epochs without improvement, then stops training. `restore_best_weights=True` ensures the model has weights from the best epoch, not the final epoch.
</details>

---

## Part 4: Fill-in-the-Blank

### 27. The LSTM forget gate uses _______ activation to produce values between 0 and 1.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** sigmoid

**Explanation:** All LSTM gates (forget, input, output) use sigmoid activation. Sigmoid outputs values between 0 and 1, which act as "switches" controlling information flow. 0 means "completely forget" and 1 means "completely keep".
</details>

---

### 28. For sequences of varying lengths, we typically _______ shorter sequences and _______ longer sequences.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** pad and truncate

**Explanation:** Neural networks require fixed-size inputs. Shorter sequences are padded with zeros (typically pre-padding). Longer sequences are truncated to the maximum length. The `pad_sequences` function handles both.
</details>

---

### 29. The _______ dropout in LSTM applies dropout to the recurrent connections between time steps.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** recurrent (or "recurrent_dropout")

**Explanation:** LSTM has two dropout parameters: `dropout` applies to input connections, and `recurrent_dropout` applies to recurrent connections (between time steps). Recurrent dropout helps prevent overfitting in sequential models.
</details>

---

### 30. RNNs maintain a _______ state that carries information from previous time steps.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** hidden

**Explanation:** The hidden state is the RNN's memory, updated at each time step and passed to the next. It allows the network to use information from earlier in the sequence when processing current input.
</details>

---

### 31. To prevent exploding gradients in RNNs, we use gradient _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** clipping

**Explanation:** Gradient clipping limits gradient magnitude during training. If gradients exceed a threshold, they're scaled down. This prevents exploding gradients from destabilizing training while preserving gradient direction.
</details>

---

### 32. GRU stands for Gated _______ Unit.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Recurrent

**Explanation:** GRU (Gated Recurrent Unit) is a simplified alternative to LSTM with only 2 gates. It combines the forget and input gates into an update gate, using fewer parameters while achieving similar performance.
</details>

---

## Part 5: Scenario-Based Questions

### 33. You're building a sentiment classifier for movie reviews. Some reviews are 500 words, others are 50 words. How should you handle this?

- [ ] A) Only use reviews with exactly 100 words
- [ ] B) Pad shorter reviews and truncate longer ones to a fixed length
- [ ] C) Ignore the length difference and feed sequences directly
- [ ] D) Train separate models for short and long reviews

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Pad shorter reviews and truncate longer ones to a fixed length

**Explanation:** For batch processing, all sequences must have the same length. Use `pad_sequences(sequences, maxlen=200)` to truncate long sequences and pad short ones. Choose maxlen based on dataset analysis (e.g., 95th percentile length).

- **Why others are wrong:**
  - A) You'd lose most of your data
  - C) Batch processing requires fixed lengths
  - D) Unnecessarily complex
</details>

---

### 34. Your LSTM model achieves 95% training accuracy but only 60% validation accuracy. What should you try?

- [ ] A) Add more LSTM layers
- [ ] B) Increase the number of LSTM units
- [ ] C) Add dropout and recurrent_dropout regularization
- [ ] D) Train for more epochs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Add dropout and recurrent_dropout regularization

**Explanation:** Large gap between training and validation accuracy indicates overfitting. Dropout regularization randomly disables connections during training, forcing the network to learn more robust features.

- **Why others are wrong:**
  - A) More layers would increase overfitting
  - B) More units would increase overfitting
  - D) More epochs would worsen overfitting
</details>

---

### 35. You're building a character-level text generator. Should you use Bidirectional LSTM?

- [ ] A) Yes, bidirectional is always better
- [ ] B) No, text generation must predict left-to-right without seeing future characters
- [ ] C) Yes, but only for the first layer
- [ ] D) Only if the text is in English

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) No, text generation must predict left-to-right without seeing future characters

**Explanation:** Text generation predicts the next token based only on previous tokens. Bidirectional processing would "cheat" by looking at future tokens that don't exist during generation. Use unidirectional LSTM for generation tasks.

- **Why others are wrong:**
  - A) Bidirectional isn't suitable for generation
  - C) Even first layer can't see future during generation
  - D) Language doesn't matter for this architectural decision
</details>

---

### 36. Your RNN-based model trains very slowly because you're processing sequences of length 1000. What can you try?

- [ ] A) Use SimpleRNN instead of LSTM
- [ ] B) Use truncated backpropagation through time (TBPTT)
- [ ] C) Increase the learning rate to 1.0
- [ ] D) Remove all regularization

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Use truncated backpropagation through time (TBPTT)

**Explanation:** TBPTT limits how far back gradients propagate, trading off long-term learning for faster training. For very long sequences, you can also consider attention mechanisms or chunking the sequence.

- **Why others are wrong:**
  - A) SimpleRNN would have worse performance on long sequences
  - C) Learning rate 1.0 would cause training instability
  - D) Removing regularization doesn't speed up training significantly
</details>

---

### 37. You want to classify text where word order in both directions matters equally (e.g., named entity recognition). Which architecture is best?

- [ ] A) Unidirectional LSTM
- [ ] B) Bidirectional LSTM
- [ ] C) Simple feedforward network
- [ ] D) Autoencoder

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Bidirectional LSTM

**Explanation:** Bidirectional LSTM processes the sequence in both directions, capturing context from both past and future words. For NER, knowing both "New" before and "City" after helps identify "York" as part of a location entity.

- **Why others are wrong:**
  - A) Misses backward context
  - C) Can't capture sequential dependencies
  - D) Autoencoders are for reconstruction, not classification
</details>

---

## Bonus Questions

### 38. What is the "constant error carousel" in LSTMs?

- [ ] A) A type of learning rate schedule
- [ ] B) The cell state pathway that allows gradients to flow unchanged across time steps
- [ ] C) A regularization technique
- [ ] D) The output gate mechanism

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The cell state pathway that allows gradients to flow unchanged across time steps

**Explanation:** The cell state acts as a "highway" where information (and gradients) can flow across many time steps with minimal transformation. The additive updates (rather than multiplicative) prevent gradient vanishing, enabling learning of long-term dependencies.

- **Why others are wrong:**
  - A) It's an architectural feature, not a training technique
  - C) It's about gradient flow, not regularization
  - D) Output gate is separate from the cell state pathway
</details>

---

### 39. Why might you choose GRU over LSTM for a specific task?

- [ ] A) GRU always performs better
- [ ] B) GRU is newer and more advanced
- [ ] C) GRU has fewer parameters and trains faster with often similar performance
- [ ] D) GRU handles longer sequences than LSTM

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) GRU has fewer parameters and trains faster with often similar performance

**Explanation:** GRU has 2 gates (vs. LSTM's 3), resulting in fewer parameters and faster training. For many tasks, GRU achieves similar accuracy to LSTM. It's a good choice when training speed matters or you have limited data.

- **Why others are wrong:**
  - A) Performance depends on the task
  - B) LSTM (1997) and GRU (2014) are both well-established
  - D) Both handle long sequences similarly well
</details>

---

### 40. In the context of sequence-to-sequence models, what is a "teacher forcing" training strategy?

- [ ] A) Using human experts to label data
- [ ] B) Feeding ground truth outputs as inputs to decoder during training
- [ ] C) Training multiple teachers on different data subsets
- [ ] D) Forcing the model to use specific attention patterns

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Feeding ground truth outputs as inputs to decoder during training

**Explanation:** In teacher forcing, the decoder receives the correct previous token (from training data) rather than its own prediction. This speeds up training but can cause exposure bias - the model never learns to recover from its own mistakes.

- **Why others are wrong:**
  - A) Teacher forcing is about training procedure, not labeling
  - C) It's not about ensemble training
  - D) It's not related to attention mechanisms
</details>

---

*Quiz generated by Practice Quiz Agent for Week 2: Deep Learning & NLP - Thursday Content*

