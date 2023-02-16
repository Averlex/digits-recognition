# A main module. Contains the main interactive functions, coordinates the executive sequence
# It's possible to save the training results as well as load them from a file later
# Options 5 and 6 from the menu are implemented only with intent of understanding
# how the neural network "sees" the numbers itself
# Alexey Chukhutin, February 2023

import numpy as np
import matplotlib.pyplot as plt
from neural import *
from neural_actions import *


# Printing the welcoming menu
def print_menu():
    print("Choose an option from list below:")
    print("1. Configure the parameters")
    print("2. Train the network")
    print("3. Test the network")
    print("4. Recognize 10 digits")
    print("5. Show the entered digit")
    print("6. Show 10 digits")
    print("0. Exit")

    return


# Getting the digit for reverse querying
def get_digit():
    dig = -1
    dig = int(input("Enter the digit: "))

    return dig


# Showing the result of reverse query on plot
def show_digit(inputs, target):
    plt.yticks([])
    plt.imshow(inputs, cmap='Greys', interpolation='None')
    plt.title("Single digit reverse query result", pad=20)
    ax = plt.subplot()
    ax.xaxis.set_major_locator(plt.FixedLocator([14]))
    ax.set_xticklabels([str(target)], fontsize=15)
    plt.show()

    return


# Showing a set of digits on the screen
# inputs: array to show
# targets: array of actual digits
def show_some_digits(inputs, targets, text):
    arr_to_show = np.concatenate(inputs, axis=1, dtype=float)

    plt.title("Digits interpretation", pad=50)
    plt.xlabel(text, labelpad=10)
    plt.ylabel("")
    plt.yticks([])
    ax = plt.subplot()
    ax.xaxis.set_major_locator(plt.FixedLocator([x for x in range(14, 784, 28)]))
    ax.xaxis.set_major_formatter(plt.FixedFormatter([str(xval) for xval in targets]))

    plt.imshow(arr_to_show, cmap='Greys', interpolation='None')

    plt.show()
    return


# Initializing the default network scale and learning rate
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

# Total amount of epochs (learning cycles)
epoch_num = 5

# Setting up the paths for source data files
test_file = "mnist_test.csv"
train_file = "mnist_train_100.csv"

# Creating the new class object
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, epoch_num)

ch = -1
set_res = False
while ch != 0:
    print_menu()
    try:
        ch = int(input("Your option: "))
    except ValueError:
        ch = -1
    print()

    if set_res:
        n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, epoch_num)
        set_res = False

    # Neural network custom parameters settings
    if ch == 1:
        res = configure(hidden_nodes, learning_rate, epoch_num)
        if res is not None:
            hidden_nodes = res[0]
            learning_rate = res[1]
            epoch_num = res[2]
            set_res = True
    # Training the network (or importing preset values)
    elif ch == 2:
        if not quick_train(n):
            train_nn(n, train_file)
            save_logs(n)
    # Testing the network on a default dataset (10k rows)
    elif ch == 3:
        test_nn(n, test_file)
    # Showing the results of the network on 10 random records from the test file
    elif ch == 4:
        arr_res = ten_random_queries(n, test_file)
        source_records = arr_res[0]
        results = arr_res[1]
        show_some_digits(source_records, results, "Recognized digits")
    # Showing the digit the way the network "sees" it
    elif ch == 5:
        digit = get_digit()
        print(digit, type(digit))
        if 0 <= digit <= 9:
            arr = proc_digit(digit)
            res = n.reverse_query(arr)
            res = decode_rev_query(res)
            show_digit(res, digit)
        else:
            print("The number is incorrect! Please, try again.\n")
    # Showing a set of 10 digits
    elif ch == 6:
        res = get_some_digits(n)
        tmp = [i for i in range(10)]
        show_some_digits(res, tmp, "Source digits")
