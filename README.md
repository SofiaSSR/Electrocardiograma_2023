# Electrocardiograma_2023


This repository contains the solution of Reto 2023-2 Deep Learning for classifying electrocardiogram (ECG) signals. The model is a 1D convolutional neural network (CNN) with three layers.

## Architecture
The CNN architecture is as follows:

Input: (2049, 1)

Conv1D(32, 3, activation='relu')
MaxPool1D(2)

Conv1D(64, 3, activation='relu')
MaxPool1D(2)

Flatten()
Dense(128, activation='relu')
Dense(3, activation='softmax')


The first layer is a convolutional layer with 32 filters of size 3. The convolution operation extracts local features from the signal. The max pooling layer reduces the dimensionality of the signal while preserving the most important features.

The second layer is another convolutional layer with 64 filters of size 3. The third layer is a flattening layer that converts the signal into a vector of features.

The final two layers are dense layers that learn the relationships between the features and the classes of ECG signals. The output layer uses the softmax activation function to generate a probability for each class.

Mathematical explanation
The convolutional operation can be expressed mathematically as follows:

y = f(x * w + b)
where:

y is the output of the convolution operation
x is the input signal
w is the filter
b is the bias
The max pooling operation can be expressed mathematically as follows:

y = max(x_1, x_2, ..., x_n)
where:

y is the output of the max pooling operation
x_1, x_2, ..., x_n are the values in the input signal
The softmax activation function can be expressed mathematically as follows:

y_i = e^i / sum(e^i)
where:

y_i is the output of the softmax activation function for class i
i is the class index
Experiments
The model was trained on a dataset of 116,953 ECG signals. The dataset was divided into a training set of 100,000 signals and a test set of 16,953 signals.

The model was trained for 10 epochs using the Adam optimizer. The loss function used was the categorical cross-entropy loss function.


Conclusion
The proposed model is a simple and effective approach to ECG classification. The model achieved high accuracy on a challenging dataset.

Additional comments:


The model could be improved by experimenting with different hyperparameters, such as the number of filters, the size of the filters, and the number of epochs.

The model could also be improved by using a different loss function, such as the F1 loss function.

The model could also be improved by using data augmentation techniques, such as adding noise or flipping the signals.

