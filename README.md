# Detecting_Anomaly_in_the_ECG_Data_Using_Autoencoders.
# AutoEncoder:

AutoEncoder is an unsupervised Artificial Neural Network that attempts to encode the data by compressing it into the lower dimensions (bottleneck layer or code) and then decoding the data to reconstruct the original input. The bottleneck layer (or code) holds the compressed representation of the input data. The number of hidden units in the code is called code size.

In this project, I am implementing Autoencoders for anomaly detection in the ECG data using Tensorflow.

The encoder of the model consists of four layers that encode the data into lower dimensions. The decoder of the model consists of four layers that reconstruct the input data.

The model is compiled with Mean Squared Logarithmic loss and Adam optimizer. The model is then trained with 20 epochs with a batch size of 512.

The reconstruction errors are considered to be anomaly scores. The threshold is then calculated by summing the mean and standard deviation of the reconstruction errors. The reconstruction errors above this threshold are considered to be anomalies. We can further fine-tune the model by leveraging the Keras-tuner.
