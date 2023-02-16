# Digits recognition
###### A small neural network project which was done for practice.

![alt text](https://github.com/Averlex/digits-recognition/blob/master/nn_work_res.png "The neural network results")

Description
-----------

A simple implementation of a 3-layer perceptron built on Python 3.10 using NumPy, SciPy and Matplotlib. Project runs from the shell, no additional GUI features are presented.

Configuration
-------------
1. Input nodes: 784  
2. Output nodes: 10  
3. Hidden nodes: 200
4. Learning coefficient: 0.1
5. Number of epochs: 5
  
Sigmoid function was choisen as the activation function.  
List items 3 - 5 may be changed at runtime.

Datasets
--------

All tests were performed over MNIST dataset. Any custom datasets will also work fine, the only requirement is that images should be 28x28 pixels and pre-converted to .csv format.
The training results may be written in a log file. It's also fair that there is a possibility to load any training results which were saved previously to perform a "quick learn" procedure.

Additional functionality
------------------------

Additionaly, two more functions for reverse queries over the network were implemented. The main (and only purpose) was to better understand how the neural network "sees" the dataset. After running some tests it became clear that lower network efficiency provides a clearer picture of the digit as an output.

Files contents
--------------
* **_mnist_test.csv_** and **_mnist_train.csv_**: default datasets for testing and training the NN. Full MNIST dataset, 10k and 60k records respectively.
* _**mnist_test_10.csv**_ and **_mnist_train_100.csv_**: shortened versions of the datasets mentioned above. Both were very useful duting debug process.
* **_wih.csv_** and **_who.csv_**: auto-generated files containing the training results. Both created and used on user's demand only. 
