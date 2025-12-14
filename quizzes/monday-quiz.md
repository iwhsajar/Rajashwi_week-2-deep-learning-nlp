# Weekly Knowledge Check: Deep Learning Fundamentals (Monday)

## Part 1: Multiple Choice

### 1. What is the default port for TensorBoard's web interface?

- [ ] A) 6006
- [ ] B) 8080
- [ ] C) 5000
- [ ] D) 3000

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) 6006

**Explanation:** TensorBoard starts on port 6006 by default. You access it at `http://localhost:6006`. If port 6006 is already in use, you can specify a different port with `tensorboard --logdir=logs --port=6007`.

- **Why others are wrong:**
  - B) 8080 is commonly used for web servers/proxies
  - C) 5000 is Flask's default port
  - D) 3000 is commonly used by React development servers
</details>

---

### 2. What does `histogram_freq=1` do in the TensorBoard callback?

- [ ] A) Logs weight distribution histograms every epoch
- [ ] B) Logs histograms once at the start of training
- [ ] C) Creates 1 histogram per layer
- [ ] D) Limits histogram bins to 1

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Logs weight distribution histograms every epoch

**Explanation:** The `histogram_freq` parameter specifies how often (in epochs) to log weight and bias histograms. Setting `histogram_freq=1` means histograms are logged after every epoch, allowing you to visualize how weights evolve during training.

- **Why others are wrong:**
  - A) It logs continuously, not just at the start
  - C) Multiple histograms are created per layer (weights, biases)
  - D) Bin count is determined separately, not by this parameter
</details>

---

### 3. What is the purpose of the "bottleneck" layer in an autoencoder?

- [ ] A) To increase computational efficiency
- [ ] B) To prevent overfitting
- [ ] C) To force the network to learn a compressed representation
- [ ] D) To add noise to the input

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) To force the network to learn a compressed representation

**Explanation:** The bottleneck is a layer with fewer dimensions than the input. By forcing all information through this narrow layer, the network must learn which features are most important for reconstruction. This creates a compressed, meaningful representation of the input data.

- **Why others are wrong:**
  - A) The bottleneck actually adds computational steps, not efficiency
  - B) Overfitting prevention uses techniques like dropout, not bottleneck design
  - D) Denoising autoencoders add noise to inputs, but the bottleneck serves a different purpose
</details>

---

### 4. In an autoencoder, what is the relationship between input and output during training?

- [ ] A) Input is an image, output is a class label
- [ ] B) Input is raw data, output is normalized data
- [ ] C) Input is text, output is a translation
- [ ] D) Input equals output (reconstruct the input)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Input equals output (reconstruct the input)

**Explanation:** Autoencoders are trained with `model.fit(x_train, x_train)` - the input and target are the same! The goal is to reconstruct the input after compression through the bottleneck. This is unsupervised learning because no labels are needed.

- **Why others are wrong:**
  - A) That describes classification, not autoencoders
  - C) Normalization is preprocessing, not the autoencoder's task
  - D) That describes sequence-to-sequence translation models
</details>

---

### 5. Which loss function is commonly used for autoencoders with image data normalized to [0, 1]?

- [ ] A) Sparse categorical crossentropy
- [ ] B) Hinge loss
- [ ] C) Binary crossentropy or Mean Squared Error
- [ ] D) Cosine similarity

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Binary crossentropy or Mean Squared Error

**Explanation:** For images normalized to [0, 1], binary crossentropy or MSE are commonly used. Binary crossentropy works because each pixel can be treated as an independent probability. MSE measures the pixel-wise reconstruction error directly.

- **Why others are wrong:**
  - A) Sparse categorical crossentropy is for multi-class classification with integer labels
  - B) Hinge loss is for SVM-style classifiers
  - D) Cosine similarity measures vector direction, not reconstruction quality
</details>

---

### 6. What does `write_graph=True` do in the TensorBoard callback?

- [ ] A) Writes the loss graph to a file
- [ ] B) Creates a bar graph of accuracy
- [ ] C) Enables visualization of the model's computational graph
- [ ] D) Saves the training history as a graph image

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Enables visualization of the model's computational graph

**Explanation:** Setting `write_graph=True` allows TensorBoard to visualize the model's architecture as a computational graph, showing layer connections, tensor shapes, and data flow.

- **Why others are wrong:**
  - A) Loss values are logged as scalars, not as a "graph file"
  - C) Accuracy is shown in the Scalars dashboard, not created by this parameter
  - D) Training history is plotted dynamically, not saved as an image by this parameter
</details>

---

### 7. How many input neurons does an MLP need for 28x28 pixel MNIST images?

- [ ] A) 28
- [ ] B) 56
- [ ] C) 784
- [ ] D) 1

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 784

**Explanation:** MLPs require flattened input, so 28x28 = 784 pixels become 784 input neurons, one per pixel. The 2D image structure is converted to a 1D vector before feeding into the network.

- **Why others are wrong:**
  - A) 28 is just one dimension of the image
  - B) 56 is just 28+28, not the total pixels
  - D) 1 would mean treating the entire image as a single value
</details>

---

### 8. What activation function should be used in the output layer for multi-class classification?

- [ ] A) ReLU
- [ ] B) Sigmoid
- [ ] C) Tanh
- [ ] D) Softmax

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) Softmax

**Explanation:** Softmax converts raw output scores (logits) into a probability distribution where all outputs are between 0 and 1 and sum to 1. This is essential for multi-class classification where you need to choose one class.

- **Why others are wrong:**
  - A) ReLU is for hidden layers, not classification output
  - B) Sigmoid is for binary classification (2 classes)
  - C) Tanh outputs values between -1 and 1, not probabilities
</details>

---

### 9. Which TensorBoard dashboard shows training and validation loss over epochs?

- [ ] A) Scalars Dashboard
- [ ] B) Histograms Dashboard
- [ ] C) Graphs Dashboard
- [ ] D) Images Dashboard

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Scalars Dashboard

**Explanation:** The Scalars Dashboard tracks single numerical values over time, including training loss, validation loss, accuracy, and any custom metrics you log.

- **Why others are wrong:**
  - A) Graphs Dashboard shows model architecture
  - B) Histograms Dashboard shows weight distributions
  - D) Images Dashboard displays image data
</details>

---

### 10. What type of learning do autoencoders perform?

- [ ] A) Supervised learning
- [ ] B) Reinforcement learning
- [ ] C) Unsupervised learning
- [ ] D) Semi-supervised learning

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Unsupervised learning

**Explanation:** Autoencoders learn without labeled data. The "label" is simply the input itself - the model learns to reconstruct its input through compression, requiring no external labels or annotations.

- **Why others are wrong:**
  - A) Supervised learning requires explicit labels for each input
  - C) Reinforcement learning uses reward signals from environment interaction
  - D) Semi-supervised uses a mix of labeled and unlabeled data
</details>

---

### 11. In the formula for a Dense layer, y = activation(Wx + b), what does W represent?

- [ ] A) The bias terms
- [ ] B) The input values
- [ ] C) The activation function
- [ ] D) The weight matrix

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) The weight matrix

**Explanation:** W is the weight matrix containing learnable parameters that transform inputs. Each weight connects an input neuron to an output neuron, and these weights are updated during training via backpropagation.

- **Why others are wrong:**
  - A) Bias terms are represented by 'b'
  - C) Input values are represented by 'x'
  - D) The activation function is applied to the result of Wx + b
</details>

---

### 12. What is a denoising autoencoder trained to do?

- [ ] A) Add noise to images
- [ ] B) Classify noisy images
- [ ] C) Reconstruct clean data from noisy input
- [ ] D) Compress noise into a smaller representation

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Reconstruct clean data from noisy input

**Explanation:** A denoising autoencoder receives corrupted (noisy) input but is trained to output the clean, original data. This forces the network to learn robust features that ignore noise.

- **Why others are wrong:**
  - A) Noise is added as a preprocessing step, not by the autoencoder output
  - B) Denoising autoencoders reconstruct, not classify
  - D) The goal is to remove noise, not compress it
</details>

---

### 13. Why do we normalize MNIST pixel values to [0, 1] range?

- [ ] A) To reduce storage space
- [ ] B) To make training faster and more stable
- [ ] C) To increase image resolution
- [ ] D) To convert grayscale to RGB

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To make training faster and more stable

**Explanation:** Normalizing inputs to a consistent range helps neural networks train more effectively. Large input values (0-255) can cause unstable gradients, while normalized values (0-1) keep activations and gradients in a well-behaved range.

- **Why others are wrong:**
  - A) Normalization doesn't affect storage (same data type)
  - C) Normalization doesn't change image dimensions or resolution
  - D) Normalization is about value range, not color channels
</details>

---

### 14. What happens during the encoding phase of an autoencoder?

- [ ] A) Data is progressively compressed through hidden layers
- [ ] B) Data is expanded to higher dimensions
- [ ] C) Labels are assigned to data
- [ ] D) Data is shuffled for training

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) Data is progressively compressed through hidden layers

**Explanation:** The encoder takes high-dimensional input and progressively reduces its dimensionality through hidden layers until reaching the bottleneck (latent representation). This compression forces the network to learn essential features.

- **Why others are wrong:**
  - A) Expansion happens in the decoder phase
  - C) Autoencoders don't assign labels (unsupervised)
  - D) Shuffling is a data preprocessing step, not part of encoding
</details>

---

### 15. What is the purpose of `validation_split=0.2` in model.fit()?

- [ ] A) To use 20% of training data for validation without training on it
- [ ] B) To train on 20% of the data
- [ ] C) To split the model into 20% of its layers
- [ ] D) To reduce learning rate by 20%

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** A) To use 20% of training data for validation without training on it

**Explanation:** `validation_split=0.2` reserves the last 20% of the training data for validation. The model evaluates on this held-out portion after each epoch but never trains on it, helping detect overfitting.

- **Why others are wrong:**
  - B) It still trains on 80% of the data
  - C) Model architecture is unaffected
  - D) Learning rate is controlled by the optimizer, not this parameter
</details>

---

## Part 2: True/False

### 16. TensorBoard requires your training code to finish before you can view visualizations.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** TensorBoard runs independently of your training code. You can launch TensorBoard and view training metrics in real-time as your model trains. This is one of its most powerful features for monitoring long training runs.
</details>

---

### 17. Autoencoders require labeled data for training.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** Autoencoders are unsupervised learners. They reconstruct their input without needing labels. The "label" is simply the input itself, making autoencoders useful when labeled data is scarce.
</details>

---

### 18. The encoder and decoder in an autoencoder must have the exact same architecture (mirrored).

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** While symmetric architectures are common, encoder and decoder don't need to be identical mirrors. The key constraint is that the encoder compresses to the latent dimension and the decoder expands back to the original dimension.
</details>

---

### 19. A typical MLP achieves 97-98% accuracy on MNIST with just 2-3 hidden layers.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** True

**Explanation:** MNIST is a relatively simple classification problem. A well-designed MLP with 2-3 hidden layers and proper training can achieve 97-98% accuracy. This serves as a baseline before moving to more complex architectures like CNNs.
</details>

---

### 20. The training set should be the smallest portion of your data split.

- [ ] True
- [ ] False

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** False

**Explanation:** The training set should be the largest portion (typically 60-80%). Models learn from training data, so they need sufficient examples. Validation and test sets are smaller (10-20% each).
</details>

---

## Part 3: Code Prediction

### 21. What does this code create?

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(784, activation='sigmoid')
])
```

- [ ] A) A regression model
- [ ] B) An MNIST classifier
- [ ] C) An RNN for sequence data
- [ ] D) An autoencoder with 32-dimensional bottleneck

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** D) An autoencoder with 32-dimensional bottleneck

**Explanation:** This architecture compresses 784 dimensions down to 32 (encoder: 784->128->32), then expands back to 784 (decoder: 32->128->784). The sigmoid output produces values in [0, 1], typical for image reconstruction.

- **Why others are wrong:**
  - A) Regression would have fewer output neurons
  - B) A classifier would have 10 outputs with softmax
  - C) RNNs use recurrent layers, not just Dense layers
</details>

---

### 22. What is the output shape of this layer?

```python
layers.Dense(64, activation='relu', input_shape=(784,))
```

- [ ] A) (None, 784)
- [ ] B) (None, 64)
- [ ] C) (64, 784)
- [ ] D) (784, 64)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (None, 64)

**Explanation:** A Dense layer with 64 units transforms inputs to 64-dimensional outputs. The 'None' represents the batch dimension, which is flexible. Input was 784, but output is determined by the number of units (64).
</details>

---

### 23. How many trainable parameters does this layer have?

```python
layers.Dense(128, input_shape=(784,))
```

- [ ] A) 784
- [ ] B) 128
- [ ] C) 100,480
- [ ] D) 912

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) 100,480

**Explanation:** Parameters = (inputs x outputs) + biases = (784 x 128) + 128 = 100,352 + 128 = 100,480. Each input connects to each output (weights), plus one bias per output neuron.
</details>

---

### 24. What happens when you run this training code?

```python
model.fit(x_train, x_train, epochs=10)
```

- [ ] A) Error: labels cannot equal inputs
- [ ] B) Model trains as a classifier
- [ ] C) Model trains as an autoencoder (reconstructs input)
- [ ] D) Model trains with no labels

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Model trains as an autoencoder (reconstructs input)

**Explanation:** When input and target are the same (`x_train, x_train`), the model learns to reconstruct its input. This is the defining characteristic of autoencoder training - unsupervised learning through reconstruction.
</details>

---

### 25. What does this code log to TensorBoard?

```python
tensorboard_callback = TensorBoard(
    log_dir='logs',
    histogram_freq=1,
    write_graph=True
)
```

- [ ] A) Only loss values
- [ ] B) Weight histograms every epoch and model graph
- [ ] C) Only the model architecture
- [ ] D) Only weight histograms

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Weight histograms every epoch and model graph

**Explanation:** `histogram_freq=1` logs weight/bias histograms every epoch, and `write_graph=True` logs the model architecture. Default scalar metrics (loss, accuracy) are always logged.
</details>

---

### 26. What will predictions contain after this code?

```python
predictions = model.predict(x_test[:5])
print(predictions.shape)
```
(Assume model has 10 output neurons with softmax)

- [ ] A) (5,)
- [ ] B) (5, 10)
- [ ] C) (10,)
- [ ] D) (5, 1)

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) (5, 10)

**Explanation:** With 5 input samples and 10 output neurons (softmax for 10 classes), predictions shape is (5, 10). Each row contains 10 probability values that sum to 1.
</details>

---

## Part 4: Fill-in-the-Blank

### 27. To launch TensorBoard from the command line pointing to a logs directory, you use the command: `tensorboard --logdir=_______`

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** logs (or "logs/fit", or the path to your log directory)

**Explanation:** The `--logdir` argument tells TensorBoard where to find the log files. It should point to the directory containing your experiment logs.
</details>

---

### 28. In an autoencoder, the compressed middle layer is called the _______ layer or latent representation.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** bottleneck (or "latent", "code", "encoding")

**Explanation:** The bottleneck layer is the narrowest point in the autoencoder where data is most compressed. This forces the network to learn the most important features for reconstruction.
</details>

---

### 29. To normalize MNIST pixel values from [0, 255] to [0, 1], you divide by _______.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** 255 (or 255.0)

**Explanation:** MNIST pixels range from 0-255. Dividing by 255 scales values to [0, 1]: `x_train = x_train.astype('float32') / 255.0`
</details>

---

### 30. For multi-class classification with one-hot encoded labels, you should use _______ as the loss function.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** categorical_crossentropy

**Explanation:** Categorical crossentropy is designed for multi-class classification with one-hot encoded labels. For integer labels, use `sparse_categorical_crossentropy` instead.
</details>

---

### 31. The TensorBoard dashboard that shows how weights and biases change over training is called the _______ dashboard.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** Histograms (or "Distributions")

**Explanation:** The Histograms dashboard shows distributions of weights, biases, and activations across training epochs, helping you identify vanishing/exploding gradients or dead neurons.
</details>

---

### 32. In data splitting, the _______ set is used only once for final evaluation after all development is complete.

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** test

**Explanation:** The test set should be used only once at the end to get an unbiased estimate of model performance. Using it during development leads to overfitting to the test set.
</details>

---

## Part 5: Scenario-Based Questions

### 33. You trained an autoencoder on normal factory sensor data. When you process new sensor readings, one sample has reconstruction error 10x higher than average. What should you conclude?

- [ ] A) The autoencoder failed to learn properly
- [ ] B) The sample is likely an anomaly requiring investigation
- [ ] C) The sample should be added to training data
- [ ] D) The model needs more epochs of training

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The sample is likely an anomaly requiring investigation

**Explanation:** Autoencoders trained on normal data learn to reconstruct normal patterns well. Anomalous inputs don't match learned patterns, resulting in high reconstruction error. This is the principle behind autoencoder-based anomaly detection.

- **Why others are wrong:**
  - A) High error on anomalies is expected behavior, not failure
  - C) Adding anomalies to training would defeat the purpose
  - D) More training wouldn't help if the sample is genuinely abnormal
</details>

---

### 34. Your TensorBoard shows training loss decreasing but validation loss increasing after epoch 10. What is happening?

- [ ] A) The model is underfitting
- [ ] B) The model is overfitting
- [ ] C) The learning rate is too low
- [ ] D) The data is corrupted

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The model is overfitting

**Explanation:** When training loss decreases but validation loss increases, the model is memorizing training data rather than learning generalizable patterns. This divergence is the classic signature of overfitting.

- **Why others are wrong:**
  - A) Underfitting shows high loss on BOTH training and validation
  - C) Low learning rate would slow convergence, not cause this pattern
  - D) Corrupted data would affect both training and validation
</details>

---

### 35. You want to compare three different model architectures in TensorBoard. How should you organize your logs?

- [ ] A) Use the same log directory for all three
- [ ] B) Use separate subdirectories within a parent log directory
- [ ] C) Use completely unrelated directories
- [ ] D) Log only the best model

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Use separate subdirectories within a parent log directory

**Explanation:** TensorBoard can compare multiple runs when they're in subdirectories of the same parent: `logs/model_A`, `logs/model_B`, `logs/model_C`. Point TensorBoard to `logs/` to see all three overlaid.

- **Why others are wrong:**
  - A) Same directory would overwrite or mix logs
  - C) Unrelated directories can't be easily compared
  - D) Comparing requires logging all models
</details>

---

### 36. Your MNIST classifier gets 95% training accuracy but only 70% test accuracy. What should you try first?

- [ ] A) Add more layers to the network
- [ ] B) Increase the number of epochs
- [ ] C) Add dropout regularization
- [ ] D) Use a larger dataset

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** C) Add dropout regularization

**Explanation:** The large gap between training (95%) and test (70%) accuracy indicates overfitting. Dropout randomly disables neurons during training, forcing the network to learn more robust features and reducing overfitting.

- **Why others are wrong:**
  - A) More layers would increase overfitting risk
  - B) More epochs would worsen overfitting
  - D) MNIST is already large enough; regularization is the immediate solution
</details>

---

### 37. You're building an autoencoder for image compression. Your latent dimension is 2, and reconstructions are blurry. What should you try?

- [ ] A) Reduce the latent dimension to 1
- [ ] B) Increase the latent dimension (e.g., to 32 or 64)
- [ ] C) Remove the decoder
- [ ] D) Use fewer hidden layers

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Increase the latent dimension (e.g., to 32 or 64)

**Explanation:** A 2-dimensional bottleneck is too restrictive for most images - too much information is lost. Increasing the latent dimension gives the network more capacity to preserve important features for reconstruction.

- **Why others are wrong:**
  - A) Reducing dimension would lose even more information
  - C) The decoder is required for reconstruction
  - D) Fewer hidden layers would reduce learning capacity
</details>

---

## Bonus Questions

### 38. What is the relationship between autoencoders and PCA (Principal Component Analysis)?

- [ ] A) They are completely unrelated techniques
- [ ] B) Autoencoders with linear activations approximate PCA
- [ ] C) PCA is always superior to autoencoders
- [ ] D) Autoencoders can only work with image data

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) Autoencoders with linear activations approximate PCA

**Explanation:** A linear autoencoder (no non-linear activations) learns to extract the same principal components as PCA. However, autoencoders with non-linear activations can capture more complex, non-linear relationships in data.

- **Why others are wrong:**
  - A) Both perform dimensionality reduction
  - C) Autoencoders can capture non-linear patterns that PCA cannot
  - D) Autoencoders work with any data type
</details>

---

### 39. Why is it important to use timestamps in TensorBoard log directory names?

- [ ] A) To make the logs load faster
- [ ] B) To prevent overwriting previous experiments and enable comparison
- [ ] C) TensorBoard requires timestamps
- [ ] D) To reduce storage space

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) To prevent overwriting previous experiments and enable comparison

**Explanation:** Using timestamps like `logs/fit/20231215-143022` creates unique directories for each run, preserving all experiment results and allowing side-by-side comparison in TensorBoard.

- **Why others are wrong:**
  - A) Timestamps don't affect loading speed
  - C) TensorBoard doesn't require timestamps, it's a best practice
  - D) Timestamps add to directory names, not reduce storage
</details>

---

### 40. In the context of MNIST classification, what does the term "decision boundary" refer to?

- [ ] A) The line separating training and test data
- [ ] B) The hyperplanes/surfaces separating different digit classes in feature space
- [ ] C) The maximum number of epochs for training
- [ ] D) The threshold for accepting a prediction

<details>
<summary><b>Click for Solution</b></summary>

**Correct Answer:** B) The hyperplanes/surfaces separating different digit classes in feature space

**Explanation:** Decision boundaries are the surfaces learned by the classifier that separate regions of feature space belonging to different classes. For MNIST, these are complex surfaces in 784-dimensional space that separate digit classes 0-9.

- **Why others are wrong:**
  - A) That's data splitting, not decision boundaries
  - C) Epoch count is a training hyperparameter
  - D) That's a classification threshold, related but different concept
</details>

---

*Quiz generated by Practice Quiz Agent for Week 2: Deep Learning & NLP - Monday Content*

