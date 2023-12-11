# Electrocardiogram_2023


This repository contains the solution of Reto 2023-2 Deep Learning for classifying electrocardiogram (ECG) signals. The model is a 1D convolutional neural network (CNN) with three layers.

## Model Architecture
 **Input Layer**:

 *  The input layer receives a 1D signal with 2049 elements representing an electrocardiogram (ECG).
**Convolutional Layers:**

* Three convolutional layers:
    * Each layer uses 32, 64, and 128 filters with a kernel size of 3, respectively.
    * These layers extract local features from the input signal by performing convolutions.
    * The "relu" activation function is applied after each convolution to introduce non-linearity and help the model learn complex patterns.
    * Max pooling layers with a pool size of 2 are used after each convolutional layer to reduce the dimensionality of the feature maps and control overfitting.
**Flatten Layer**:

* The flattened layer converts the output of the last convolutional layer into a 1D vector.


**Dense Layers:**

* Two dense layers:
    * The first dense layer has 256 units and the second has 128 units.
    * These layers learn more complex relationships between the extracted features.
    * The "relu" activation function is applied after each dense layer to introduce non-linearity.
    * Dropout layers with a rate of 0.5 are used after each dense layer to prevent overfitting and improve generalization.


**Output Layer:**

The output layer has 3 units and uses the "softmax" activation function.
This layer predicts the probability of the input signal belonging to one of the three classes: normal, arrhythmia, or noise.

## Mathematical explanation
    
### Convolutional Layers:

$$
y_i = \sum_{j=1}^{f^2} w_j x_{i+j-1} + b
$$
where:

* $y_i$ is the output value at position i of the feature map

* $w_j$ is the weight of the $j$th element in the filter

* $x_{i+j−1}$ is the input value at position $i+j−1$

* $b$ is the bias of the filter

### ReLU Activation Function:

$$
f(x) = \max(0, x)
$$

### Pooling Layers:

$$
y_i = \text{pool}(x_{i_1}, x_{i_2}, ..., x_{i_n})
$$
where 
$y_i$  is the output value at position i of the downsampled feature map and $x_{i_1}, x_{i_2}, ..., x_{i_n}$  are the input values within the pooling window.

### Flatten Layer:

Converts multi-dimensional feature maps into a single 1D vector.

### Dense Layers:

$$
y = W^T x + b
$$
where:

y is the output vector
W is the weight matrix
x is the input vector
b is the bias vector

### Dropout:

Randomly drops out neurons during training.

### Softmax Output Layer:

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$
where:

$p_i$  is the probability of the input belonging to class i

$z_i$ is the unnormalized log probability for class i

## Model Architecture Justification


The proposed CNN model architecture with three convolutional layers and two dense layers is a highly suitable choice for the ECG signal classification challenge. While previous models with only two convolutional layers achieved 85% accuracy, this model seeks to improve upon that performance by leveraging a deeper architecture and addressing the issue of overfitting.


### Justification for Specific Architecture Elements

#### **3 Convolutional Layers:**

*   Increased Feature Extraction: The additional convolutional layer allows the model to extract a richer set of features from the ECG signals. This is critical for identifying subtle differences between normal, arrhythmia, and noise signals.
*   Improved Learning Capacity: A deeper architecture with more layers allows the model to learn more complex relationships between the features and the class labels. This is especially important for capturing the intricate patterns in ECG signals.
*   Resilience to Overfitting: The use of pooling layers after each convolutional layer helps reduce the dimensionality of the feature maps, making the model less prone to overfitting and more generalizable to unseen data.

#### **Dense Layers and Dropout:**

*    High-level Feature Representation: The dense layers learn higher-level representations of the extracted features, leading to a more robust classification. These layers can capture complex relationships between features that might be missed by convolutional layers alone.
*   Overfitting Prevention: Dropout layers help prevent overfitting by randomly dropping out neurons during training. This forces the model to learn more robust features, which are less likely to overfit to the training data.

#### **Optimizer and Learning Rate:**

* Efficient Optimization: The Adam optimizer is a well-established choice for training deep learning models. It is known for its efficient convergence and ability to handle large datasets.
* Optimal Learning Rate: Choosing an appropriate learning rate is crucial for model training. The provided learning rate likely allows the model to learn quickly while preventing divergence.

#### **Supporting References:**

A Survey on Deep Learning for ECG Signal Classification: https://arxiv.org/abs/2304.02577
Efficient Deep Learning for ECG Signal Classification: https://arxiv.org/abs/2204.04420
Deep Learning for ECG Signal Classification: A Survey: https://arxiv.org/abs/2305.18592
Benefits over Previous Models:

While previous models achieved 85% accuracy with two convolutional layers, they were prone to overfitting. This suggests that they might not generalize well to unseen data. The proposed model with three convolutional layers and dropout layers is expected to deliver improved performance by:

Extracting richer features
Learning more complex relationships between features
Being less prone to overfitting
This ultimately leads to a more robust and accurate model for ECG signal classification.

## Training Procedure

The model was trained on a dataset of 116,953 ECG signals, divided into a training set of 93,562 signals and a validation set of 23,391 signals. The Adam optimizer was used with the categorical cross-entropy loss function. The model was trained for 10 epochs.

## Evaluation
### Metrics
Several metrics were used to evaluate the model's performance:

 * Accuracy: Overall classification accuracy.
* Precision: Ratio of correctly predicted positive cases to total predicted positive cases.
* Recall: Ratio of correctly predicted positive cases to actual positive cases.

### Cross-validation
To assess the model's generalizability and prevent overfitting, 5-fold cross-validation was employed. The dataset was split into five equally sized folds, and the model was trained and evaluated five times, each time using a different fold for validation and the remaining four folds for training.

#### Mathematical Explanation of Cross-validation:

Cross-validation involves splitting the data into K folds. Each fold is used as a validation set once, while the remaining K-1 folds are used for training. The performance of the model is averaged across all K folds, providing a more robust estimate of generalization error compared to a single train-test split.

### Mean and Standard Deviation:

Mean: The average of all values in a set. Calculated by summing all values and dividing by the number of values.
Standard Deviation: A measure of the dispersion of data points from the mean. A lower standard deviation indicates that the data points are closer to the mean, while a higher standard deviation indicates that the data points are more spread out.
Interpretation of Mean and Standard Deviation in Cross-validation:

The mean provides an overall estimate of the model's performance on unseen data.
The standard deviation indicates the variability in the model's performance across different folds. A lower standard deviation suggests that the model performs consistently across different subsets of the data, while a higher standard deviation indicates that the model's performance is more sensitive to the specific training data used.

## Conclusion

The proposed deep learning model demonstrates promising results for ECG classification, achieving high accuracy and consistency across cross-validation. This suggests its potential for accurate and reliable classification of ECG signals in real-world applications.

