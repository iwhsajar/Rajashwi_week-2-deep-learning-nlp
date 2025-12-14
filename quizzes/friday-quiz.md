# Weekly Knowledge Check: Model Persistence, Callbacks, and Regularization (Friday)

## Part 1: Multiple Choice

### 1. What is the difference between model.save('model.h5') and model.save_weights('weights.h5')?

- [ ] A) .h5 saves everything; _weights saves only learned parameters
- [ ] B) No difference
- [ ] C) .h5 saves weights; _weights saves architecture
- [ ] D) .h5 is newer; _weights is deprecated

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) .h5 saves everything; _weights saves only learned parameters

**Explanation:** `model.save()` saves the complete model: architecture, weights, optimizer state, and training configuration. `model.save_weights()` saves only the learned weights. With weights-only, you must recreate the architecture in code before loading.

- **Why others are wrong:**
  - B) They save different things
  - C) It's the opposite
  - D) Both are actively supported
</details>

---

### 2. Which model format is recommended by TensorFlow for production deployment?

- [ ] A) H5 format
- [ ] B) JSON format
- [ ] C) SavedModel format
- [ ] D) Pickle format

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) SavedModel format

**Explanation:** SavedModel is TensorFlow's native format, optimized for TensorFlow Serving and production deployment. It includes architecture, weights, optimizer state, and is platform-independent.

- **Why others are wrong:**
  - A) H5 is legacy, still works but not recommended for new projects
  - C) JSON only stores architecture, not weights
  - D) Pickle is not a TensorFlow format
</details>

---

### 3. What does the `patience` parameter in EarlyStopping control?

- [ ] A) How many epochs to wait before stopping when metric stops improving
- [ ] B) The learning rate decay
- [ ] C) How many layers to train
- [ ] D) The batch size

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) How many epochs to wait before stopping when metric stops improving

**Explanation:** Patience determines how many epochs without improvement the callback waits before stopping training. For example, patience=10 means training stops if the monitored metric hasn't improved for 10 consecutive epochs.

- **Why others are wrong:**
  - A) Patience doesn't affect layer training
  - B) That's ReduceLROnPlateau's job
  - D) Batch size is set in model.fit()
</details>

---

### 4. Which regularization technique randomly disables neurons during training?

- [ ] A) L1 regularization
- [ ] B) L2 regularization
- [ ] C) Batch normalization
- [ ] D) Dropout

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Dropout

**Explanation:** Dropout randomly sets a fraction of neurons to zero during each training step. This prevents co-adaptation, forcing the network to learn more robust features. During inference, all neurons are active.

- **Why others are wrong:**
  - A) L1 penalizes absolute weight values
  - B) L2 penalizes squared weight values
  - C) Batch normalization normalizes activations
</details>

---

### 5. What does `save_best_only=True` in ModelCheckpoint accomplish?

- [ ] A) Saves after every epoch
- [ ] B) Saves only the first epoch
- [ ] C) Saves only the last epoch
- [ ] D) Saves only when the monitored metric improves

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Saves only when the monitored metric improves

**Explanation:** With `save_best_only=True`, the checkpoint is saved only when the monitored metric (e.g., val_loss) shows improvement. This saves disk space and ensures you always have the best model.

- **Why others are wrong:**
  - A) That would be save_best_only=False
  - C/D) Saving is based on metric improvement, not epoch number
</details>

---

### 6. What does L2 regularization penalize?

- [ ] A) Absolute value of weights
- [ ] B) Number of non-zero weights
- [ ] C) Squared value of weights
- [ ] D) Activation magnitudes

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Squared value of weights

**Explanation:** L2 regularization adds `lambda * sum(w^2)` to the loss. This penalizes large weights, encouraging smaller, more distributed weights. It's also called "weight decay" because it shrinks weights toward zero.

- **Why others are wrong:**
  - A) L1 regularization penalizes absolute values
  - B) L1 encourages sparsity, not L2
  - D) Activation regularization is a separate technique
</details>

---

### 7. What is the typical dropout rate for Dense layers?

- [ ] A) 0.5
- [ ] B) 0.01
- [ ] C) 0.9
- [ ] D) 1.0

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) 0.5

**Explanation:** A dropout rate of 0.5 (50%) is standard for Dense layers. This means half the neurons are randomly deactivated during each training step. Lower rates (0.2-0.3) are common for LSTM recurrent dropout.

- **Why others are wrong:**
  - A) 0.01 is too low to have much effect
  - C/D) 0.9-1.0 would drop too many neurons, causing underfitting
</details>

---

### 8. When loading a model with custom layers, what must you provide?

- [ ] A) Just the file path
- [ ] B) The original training data
- [ ] C) The custom_objects dictionary mapping layer names to classes
- [ ] D) The optimizer configuration

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) The custom_objects dictionary mapping layer names to classes

**Explanation:** Custom layers, losses, or metrics must be registered when loading: `load_model('model.h5', custom_objects={'CustomLayer': CustomLayer})`. Without this, you'll get "Unknown layer" error.

- **Why others are wrong:**
  - A) Custom objects need explicit registration
  - C) Training data isn't needed for loading
  - D) Optimizer state is saved in the model file
</details>

---

### 9. What triggers ReduceLROnPlateau to reduce the learning rate?

- [ ] A) After every epoch
- [ ] B) When training loss becomes zero
- [ ] C) Randomly during training
- [ ] D) When the monitored metric stops improving for patience epochs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) When the monitored metric stops improving for patience epochs

**Explanation:** ReduceLROnPlateau monitors a metric (e.g., val_loss) and reduces the learning rate when it plateaus. If no improvement for `patience` epochs, the learning rate is multiplied by `factor` (e.g., 0.5).

- **Why others are wrong:**
  - A) That would reduce LR too aggressively
  - C) Zero loss doesn't trigger LR reduction
  - D) It's based on metric improvement, not random
</details>

---

### 10. What callback should you use to log metrics to a CSV file?

- [ ] A) TensorBoard
- [ ] B) EarlyStopping
- [ ] C) CSVLogger
- [ ] D) TerminateOnNaN

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) CSVLogger

**Explanation:** CSVLogger writes training metrics (loss, accuracy, etc.) to a CSV file after each epoch. This is useful for post-training analysis in spreadsheets or plotting tools.

- **Why others are wrong:**
  - A) TensorBoard logs to binary format for visualization
  - B) EarlyStopping stops training, doesn't log
  - D) TerminateOnNaN stops training on NaN loss
</details>

---

### 11. If training stops at epoch 50 but the best validation loss was at epoch 35, what happens with `restore_best_weights=True`?

- [ ] A) Model has weights from epoch 50
- [ ] B) Model has random weights
- [ ] C) Training continues from epoch 35
- [ ] D) Model has weights from epoch 35

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Model has weights from epoch 35

**Explanation:** With `restore_best_weights=True`, the model's weights are automatically restored to the epoch with the best monitored metric (epoch 35 in this case), not the final epoch.

- **Why others are wrong:**
  - A) That's what happens without restore_best_weights
  - B) Weights are never randomized during training
  - C) Training doesn't resume, weights are restored after stopping
</details>

---

### 12. What is overfitting characterized by?

- [ ] A) High training error, high validation error
- [ ] B) Low training error, low validation error
- [ ] C) Low training error, high validation error
- [ ] D) High training error, low validation error

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Low training error, high validation error

**Explanation:** Overfitting means the model performs well on training data but poorly on validation/test data. It has memorized training examples instead of learning generalizable patterns.

- **Why others are wrong:**
  - A) Both high = underfitting
  - B) Both low = good generalization
  - D) This pattern is very rare
</details>

---

### 13. What should you do BEFORE saving a model to production?

- [ ] A) Train for as many epochs as possible
- [ ] B) Remove all regularization
- [ ] C) Verify model performance on test data
- [ ] D) Delete validation data

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Verify model performance on test data

**Explanation:** Before deploying, evaluate the model on held-out test data to ensure it generalizes well. This final check catches overfitting that might not be visible in validation metrics.

- **Why others are wrong:**
  - A) More epochs can cause overfitting
  - B) Regularization helps generalization
  - D) Never delete data
</details>

---

### 14. What is the purpose of the `factor` parameter in ReduceLROnPlateau?

- [ ] A) How many epochs to wait
- [ ] B) Minimum learning rate
- [ ] C) Maximum learning rate
- [ ] D) Multiplier for learning rate reduction (e.g., 0.5 means halve the LR)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Multiplier for learning rate reduction (e.g., 0.5 means halve the LR)

**Explanation:** When triggered, the learning rate is multiplied by `factor`. For example, factor=0.5 halves the learning rate from 0.001 to 0.0005. Factor=0.1 reduces to 10% of previous value.

- **Why others are wrong:**
  - A) That's `patience`
  - C) That's `min_lr`
  - D) There's no max_lr parameter
</details>

---

### 15. What does `kernel_regularizer=regularizers.l2(0.01)` add to the loss function?

- [ ] A) 0.01 * sum(|weights|)
- [ ] B) 0.01 * max(weights)
- [ ] C) 0.01 * sum(weights^2)
- [ ] D) 0.01 * mean(weights)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 0.01 * sum(weights^2)

**Explanation:** L2 regularization adds a term to the loss proportional to the sum of squared weights. This penalizes large weights, encouraging the model to use smaller, more distributed weights.

- **Why others are wrong:**
  - A) That's L1 regularization
  - C/D) These are not standard regularization formulas
</details>

---

## Part 2: True/False

### 16. EarlyStopping requires validation data to function properly.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** EarlyStopping monitors validation metrics (val_loss, val_accuracy) to decide when to stop. Without validation data, there's nothing to monitor, and the callback can't detect overfitting.
</details>

---

### 17. Dropout is applied during both training and inference.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Dropout is only applied during training. During inference (prediction), all neurons are active, and outputs are scaled appropriately. This ensures consistent predictions.
</details>

---

### 18. SavedModel format stores the model as a single file.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** SavedModel format creates a directory structure containing saved_model.pb, variables folder, and assets folder. It's not a single file like H5, but it can be zipped for distribution.
</details>

---

### 19. L1 regularization encourages sparse weight matrices (many zeros).

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** L1 regularization penalizes the absolute value of weights, which tends to push small weights exactly to zero. This creates sparse models where many weights are exactly 0, effectively performing feature selection.
</details>

---

### 20. More training data is the most effective way to combat overfitting.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** More diverse training examples provide better coverage of the input space, helping the model learn generalizable patterns rather than memorizing specific examples. Regularization techniques are workarounds when more data isn't available.
</details>

---

## Part 3: Code Prediction

### 21. What does this code save?

```python
model.save('my_model.h5')
```

- [ ] A) Only the model weights
- [ ] B) Only the model architecture
- [ ] C) Complete model (architecture, weights, optimizer state, configuration)
- [ ] D) Only the training history

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Complete model (architecture, weights, optimizer state, configuration)

**Explanation:** `model.save()` with .h5 extension saves everything needed to reconstruct and continue training the model: architecture, learned weights, optimizer state, and compilation configuration.
</details>

---

### 22. What is the output when training with this callback?

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# Training: val_loss decreases for 20 epochs, then increases for 5 epochs
```

- [ ] A) Training stops at epoch 25 with best weights from epoch 20
- [ ] B) Training continues until max epochs
- [ ] C) Training stops immediately
- [ ] D) Error occurs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Training stops at epoch 25 with best weights from epoch 20

**Explanation:** After epoch 20 (best), val_loss increases for 5 epochs (patience=5). Training stops at epoch 25, and weights are restored to epoch 20 (the best epoch).
</details>

---

### 23. What happens with this regularization?

```python
layers.Dense(128, kernel_regularizer=regularizers.l2(0.1))
```

- [ ] A) Strong regularization, may underfit
- [ ] B) No regularization effect
- [ ] C) Model trains faster
- [ ] D) Model uses more memory

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Strong regularization, may underfit

**Explanation:** L2 regularization of 0.1 is quite strong (typical values are 0.01-0.001). This heavily penalizes large weights, which may constrain the model too much and cause underfitting.
</details>

---

### 24. What does this callback combination do?

```python
callbacks = [
    ModelCheckpoint('best.h5', save_best_only=True),
    EarlyStopping(patience=10, restore_best_weights=True)
]
```

- [ ] A) Saves every epoch and stops early
- [ ] B) Saves best model to file AND stops early with best weights in memory
- [ ] C) Only saves weights, not full model
- [ ] D) Runs indefinitely

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Saves best model to file AND stops early with best weights in memory

**Explanation:** This combination: (1) ModelCheckpoint saves the best model to disk, (2) EarlyStopping stops training after 10 epochs without improvement and restores best weights. Both model file and in-memory model have best weights.
</details>

---

### 25. What does this dropout configuration do?

```python
layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)
```

- [ ] A) Drops 40% of all connections
- [ ] B) Drops 20% of input connections and 20% of recurrent connections separately
- [ ] C) Drops 20% of output neurons
- [ ] D) Doubles the training time

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Drops 20% of input connections and 20% of recurrent connections separately

**Explanation:** `dropout` applies to input-to-hidden connections (x_t to h_t), while `recurrent_dropout` applies to hidden-to-hidden connections (h_t to h_{t+1}). Both are applied independently.
</details>

---

### 26. What will this verification code check?

```python
loaded_model = keras.models.load_model('model.h5')
predictions = loaded_model.predict(X_test[:10])
assert predictions.shape[0] == 10
```

- [ ] A) That the model has 10 layers
- [ ] B) That predictions are returned for all 10 test samples
- [ ] C) That training took 10 epochs
- [ ] D) That the model has 10 outputs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) That predictions are returned for all 10 test samples

**Explanation:** This verifies that the loaded model can make predictions and that the output has the expected shape (10 predictions for 10 input samples). It's a basic integrity check for loaded models.
</details>

---

## Part 4: Fill-in-the-Blank

### 27. To save only the model weights without architecture, use `model.save________('weights.h5')`.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** _weights (or "_weights")

**Explanation:** `model.save_weights('weights.h5')` saves only the learned parameters. To load, you must first recreate the exact same architecture, then call `model.load_weights('weights.h5')`.
</details>

---

### 28. The _______ callback stops training when the loss becomes NaN.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** TerminateOnNaN

**Explanation:** TerminateOnNaN immediately stops training if the loss becomes NaN (not a number), which typically indicates exploding gradients or numerical instability.
</details>

---

### 29. L2 regularization is also known as _______ regularization or weight decay.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Ridge

**Explanation:** L2 regularization has multiple names: L2 regularization (from the L2 norm), Ridge regularization (from statistics), or weight decay (describing its effect of decaying weights toward zero).
</details>

---

### 30. When monitoring validation accuracy for early stopping, set `mode='______'` because higher accuracy is better.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** max

**Explanation:** For metrics where higher is better (accuracy, F1), set mode='max'. For metrics where lower is better (loss), set mode='min'. Mode='auto' infers from the metric name.
</details>

---

### 31. To use pre-trained GloVe embeddings without updating them during training, set `trainable=______` in the Embedding layer.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Setting `trainable=False` freezes the embedding weights, using the pre-trained values as-is. Set `trainable=True` to fine-tune the embeddings for your specific task.
</details>

---

### 32. The sign of _______ is that training loss keeps decreasing while validation loss increases.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** overfitting

**Explanation:** Overfitting is characterized by diverging training and validation curves: training loss decreases (model memorizing training data) while validation loss increases (model failing to generalize).
</details>

---

## Part 5: Scenario-Based Questions

### 33. Your model achieves 98% training accuracy but only 65% test accuracy. Which regularization technique should you try FIRST?

- [ ] A) Remove all layers except output
- [ ] B) Add dropout layers (0.5) and L2 regularization
- [ ] C) Increase model complexity
- [ ] D) Train for more epochs

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Add dropout layers (0.5) and L2 regularization

**Explanation:** The 33% gap indicates severe overfitting. Dropout forces the network to learn robust features by randomly disabling neurons. L2 regularization penalizes large weights. Both help generalization.

- **Why others are wrong:**
  - A) Too drastic, may cause underfitting
  - C) More complexity would increase overfitting
  - D) More epochs would worsen overfitting
</details>

---

### 34. You're deploying a model to TensorFlow Serving. Which save format should you use?

- [ ] A) H5 format (.h5)
- [ ] B) SavedModel format (directory)
- [ ] C) Weights only (.h5)
- [ ] D) JSON + weights separately

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) SavedModel format (directory)

**Explanation:** SavedModel is TensorFlow's native format, optimized for TensorFlow Serving. It includes all necessary information for serving predictions without needing the original Python code.

- **Why others are wrong:**
  - A) H5 works but isn't optimized for serving
  - C) Weights-only requires recreating architecture
  - D) Separate files are harder to manage
</details>

---

### 35. Your training plateaus at epoch 30 (val_loss stops decreasing). What callback would help?

- [ ] A) EarlyStopping with patience=5
- [ ] B) ReduceLROnPlateau with patience=5
- [ ] C) ModelCheckpoint saving every epoch
- [ ] D) TensorBoard

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) ReduceLROnPlateau with patience=5

**Explanation:** When training plateaus, reducing the learning rate can help the model fine-tune and escape the plateau. ReduceLROnPlateau automatically reduces LR when the metric stops improving.

- **Why others are wrong:**
  - A) EarlyStopping would stop training, not help escape plateau
  - C) Checkpointing saves models, doesn't affect training
  - D) TensorBoard visualizes, doesn't affect training
</details>

---

### 36. You want to save the tokenizer alongside your text classification model for production. How should you do this?

- [ ] A) The tokenizer is automatically saved with model.save()
- [ ] B) Save tokenizer separately using pickle
- [ ] C) Tokenizers don't need to be saved
- [ ] D) Rebuild tokenizer from vocabulary each time

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Save tokenizer separately using pickle

**Explanation:** The tokenizer is not part of the Keras model and must be saved separately. Use `pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))` and load with `pickle.load()` before inference.

- **Why others are wrong:**
  - A) model.save() doesn't include tokenizer
  - C) You need the same tokenizer to preprocess new text
  - D) Rebuilding requires original training texts
</details>

---

### 37. Your validation loss is noisy (fluctuates a lot). How should you configure EarlyStopping?

- [ ] A) Set patience=2
- [ ] B) Set patience=15 and min_delta=0.001
- [ ] C) Remove EarlyStopping entirely
- [ ] D) Monitor training loss instead

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Set patience=15 and min_delta=0.001

**Explanation:** With noisy validation metrics, use higher patience to avoid premature stopping due to random fluctuations. min_delta ensures only meaningful improvements (>0.001) reset the patience counter.

- **Why others are wrong:**
  - A) Low patience would cause premature stopping
  - C) You'd lose protection against overfitting
  - D) Training loss can't detect overfitting
</details>

---

## Bonus Questions

### 38. What is data augmentation in the context of overfitting prevention?

- [ ] A) Collecting more real data
- [ ] B) Creating variations of existing data to artificially increase dataset size
- [ ] C) Removing outliers from the data
- [ ] D) Normalizing data to [0, 1]

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Creating variations of existing data to artificially increase dataset size

**Explanation:** Data augmentation creates new training examples by applying transformations (rotation, flip for images; synonym replacement for text). This effectively increases dataset size and diversity without collecting new data.

- **Why others are wrong:**
  - A) That's data collection, not augmentation
  - C) That's data cleaning
  - D) That's normalization
</details>

---

### 39. Why should you verify model integrity after loading?

- [ ] A) To check if the file exists
- [ ] B) To ensure predictions match the original model and weights loaded correctly
- [ ] C) To measure loading speed
- [ ] D) To check file size

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To ensure predictions match the original model and weights loaded correctly

**Explanation:** Verification catches issues like architecture mismatches, missing custom objects, or corrupted files. Compare predictions on test samples: if they match, the model loaded correctly.

- **Why others are wrong:**
  - A) File existence doesn't ensure correctness
  - C) Loading speed isn't about integrity
  - D) File size doesn't verify content
</details>

---

### 40. In ensemble methods for reducing overfitting, what is "bagging"?

- [ ] A) Training one model multiple times
- [ ] B) Training multiple models on random subsets of data and averaging predictions
- [ ] C) Removing features from the data
- [ ] D) Using only the best features

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Training multiple models on random subsets of data and averaging predictions

**Explanation:** Bagging (Bootstrap Aggregating) trains multiple models on different random samples of the training data. Averaging their predictions reduces variance and overfitting. Random Forest is a famous bagging algorithm.

- **Why others are wrong:**
  - A) That's training with different seeds
  - C) That's feature selection
  - D) That's also feature selection
</details>

---

*Quiz generated by Practice Quiz Agent for Week 2: Deep Learning & NLP - Friday Content*

